# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=5",
#   "diffusers",
#   "accelerate",
#   "numpy",
# ]
# ///
"""Qwen-Image-2.1's text encoder and VAE decoder, behind a Unix socket.

The transformer runs in Rust on the same GPU; these two do not, so they run
here and the engine calls them in the order the GPU is used: encode, denoise,
decode. One request at a time, one connection, no batching.

The wire is little-endian and self-delimiting, raw bytes only:

    request   u32 kind | u32 payload_len | payload
    response  u32 kind | u32 status | u32 payload_len | payload

    PING     1  -                                      -> u32 text_dim
    ENCODE   2  u32 prompt_len | utf8 prompt           -> u32 text_len | text_len*text_dim bf16
    DECODE   3  u32 height | u32 width | u32 rows | rows*64 bf16
                                                       -> u32 height | u32 width | height*width*3 u8
    SHUTDOWN 4  -                                      -> (the process exits)

`status` is 0 for ok and 1 for a request that failed, whose payload is then the
message. A failed request leaves the connection up: the engine aborts that one
generation and serves the next.

Both halves follow the reference pipeline (`QwenImage21Pipeline`) where the
numbers come from: the prompt template and the system-token drop of
`_get_qwen_prompt_embeds`, and the unpack, denormalize and clamp of
`_decode_latents`. The embeddings are the text tower's last hidden state
*before* its final norm, which the reference takes by hooking the norm to
return its own input.
"""

import argparse
import signal
import socket
import stat
import struct
import sys
import traceback
from pathlib import Path

import torch

SYS = "Comprehend and analyze the provided prompt."
# The raw template, not `apply_chat_template` on the whole prompt: the two
# tokenize differently and the checkpoint expects this one.
TEMPLATE = (
    f"<|im_start|>system\n{SYS}<|im_end|>\n"
    f"<|im_start|>user\n{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

PING, ENCODE, DECODE, SHUTDOWN = 1, 2, 3, 4
LATENT_DIM = 64
VAE_SCALE = 16
# The engine's text budget; a longer prompt is refused, not truncated.
MAX_TEXT_TOKENS = 2048
# A request larger than this is a framing error, not something to allocate for.
MAX_REQUEST = 8 << 20


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("the engine closed the connection")
        buf += chunk
    return bytes(buf)


def read_u32(conn: socket.socket) -> int:
    return struct.unpack("<I", read_exact(conn, 4))[0]


def send(conn: socket.socket, kind: int, status: int, payload: bytes) -> None:
    conn.sendall(struct.pack("<III", kind, status, len(payload)))
    conn.sendall(payload)


def bf16_bytes(tensor: torch.Tensor) -> bytes:
    """bf16 as little-endian bytes: numpy has no bf16, the bit pattern is the payload."""
    wide = tensor.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy()
    return wide.astype("<u2", copy=False).tobytes()


class Encoder:
    """Qwen3-VL's text tower, read before its final norm."""

    def __init__(self, directory: Path, device: torch.device):
        from transformers import Qwen3VLProcessor

        self.directory = directory
        self.device = device
        self.model = None
        self.dim = 4096
        self.processor = Qwen3VLProcessor.from_pretrained(directory / "processor", local_files_only=True)
        system = [{"role": "system", "content": [{"type": "text", "text": SYS}]}]
        self.drop_idx = len(self.processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
        log(f"text encoder: deferred load, dropping {self.drop_idx} system tokens")

    def _ensure_model(self):
        if self.model is None:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = (
                Qwen3VLForConditionalGeneration.from_pretrained(
                    self.directory / "text_encoder", dtype=torch.bfloat16, local_files_only=True,
                    device_map=f"cuda:{self.device.index}",
                )
                .eval()
            )
            self.dim = self.model.config.get_text_config().hidden_size
            log(f"text encoder loaded: dim {self.dim}")

    def offload(self):
        if self.model is not None:
            self.model.to("cpu")
            torch.cuda.empty_cache()

    def reload(self):
        if self.model is not None:
            self.model.to(self.device)

    def encode(self, prompt: str) -> bytes:
        self._ensure_model()
        text = TEMPLATE.format(prompt if prompt else " ")
        inputs = self.processor(text=[text], padding=True, padding_side="left", return_tensors="pt").to(self.device)
        mask = inputs.attention_mask.bool()
        tokens = int(mask.sum()) - self.drop_idx
        if tokens < 1 or tokens > MAX_TEXT_TOKENS:
            raise ValueError(f"the prompt is {tokens} text tokens, the engine takes 1..{MAX_TEXT_TOKENS}")

        forward = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "output_hidden_states": True,
        }
        # transformers>=5 computes multimodal RoPE from the token type ids the
        # processor emits alongside input_ids.
        if getattr(inputs, "mm_token_type_ids", None) is not None:
            forward["mm_token_type_ids"] = inputs.mm_token_type_ids
        text_model = getattr(self.model.model, "language_model", self.model.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            with torch.no_grad():
                hidden = self.model(**forward).hidden_states[-1]
        finally:
            handle.remove()
        # One prompt, so the mask selects its tokens in order; the leading
        # system tokens are the template's, not the prompt's.
        selected = hidden[mask][self.drop_idx :]
        result = struct.pack("<I", selected.shape[0]) + bf16_bytes(selected)
        del inputs, hidden, selected
        torch.cuda.empty_cache()
        return result


class Decoder:
    """The VAE in float32, which is the dtype the reference decodes in."""

    def __init__(self, directory: Path, device: torch.device):
        from diffusers import AutoencoderKLQwenImage21

        self.device = device
        self.vae = (
            AutoencoderKLQwenImage21.from_pretrained(
                directory / "vae", torch_dtype=torch.float32, local_files_only=True,
                device_map=f"cuda:{device.index}",
            )
            .eval()
        )
        self.vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256)
        self.channels = self.vae.config.z_dim
        if self.channels != LATENT_DIM:
            raise ValueError(f"the vae has {self.channels} latent channels, a token is {LATENT_DIM}")
        # The checkpoint's per-channel latent statistics; the transformer's
        # latents are normalized by them and the VAE reads them back.
        mean, std = self.vae.config.latents_mean, self.vae.config.latents_std
        if len(mean) != self.channels or len(std) != self.channels:
            raise ValueError(f"vae config has {len(mean)} means and {len(std)} stds for {self.channels} channels")
        shape = (1, self.channels, 1, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(shape)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(shape)
        log(f"vae: {self.channels} latent channels, {VAE_SCALE}x spatial")

    def decode(self, height: int, width: int, payload: bytes) -> bytes:
        rows, extra = divmod(len(payload), self.channels * 2)
        if extra:
            raise ValueError(f"{len(payload)} latent bytes is not a multiple of {self.channels * 2}")
        # `_unpack_latents`: a token is one 16x16 cell, and the grid is rounded
        # down to even on both sides.
        grid_h, grid_w = 2 * (height // (2 * VAE_SCALE)), 2 * (width // (2 * VAE_SCALE))
        if grid_h < 1 or grid_w < 1 or rows != grid_h * grid_w:
            raise ValueError(f"{height}x{width} wants {grid_h * grid_w} latent rows, got {rows}")
        latent = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).view(1, rows, self.channels)
        z = latent.transpose(1, 2).reshape(1, self.channels, 1, grid_h, grid_w)
        z = z.to(device=self.device, dtype=torch.float32) * self.std + self.mean
        with torch.no_grad():
            # (1, channels, 1, H, W), already clamped to [-1, 1] by the VAE.
            image = self.vae.decode(z, return_dict=False)[0][0, :3, 0]
        rgb = ((image / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous()
        out_h, out_w = rgb.shape[:2]
        return struct.pack("<II", out_h, out_w) + rgb.cpu().numpy().tobytes()


def respond(kind: int, payload: bytes, encoder: Encoder, decoder: Decoder) -> bytes:
    if kind == PING:
        return struct.pack("<I", encoder.dim)
    if kind == ENCODE:
        (length,) = struct.unpack("<I", payload[:4])
        if len(payload) != 4 + length:
            raise ValueError(f"prompt of {length} bytes, payload is {len(payload)}")
        return encoder.encode(payload[4:].decode("utf-8"))
    if kind == DECODE:
        height, width, rows = struct.unpack("<III", payload[:12])
        if len(payload) != 12 + rows * LATENT_DIM * 2:
            raise ValueError(f"{rows} latent rows, payload is {len(payload)} bytes")
        return decoder.decode(height, width, payload[12:])
    raise ValueError(f"unknown request kind {kind}")


def session(conn: socket.socket, encoder: Encoder, decoder: Decoder) -> bool:
    """One connection's requests, in order. False once the engine said shutdown."""
    while True:
        try:
            kind = read_u32(conn)
            length = read_u32(conn)
            if length > MAX_REQUEST:
                raise ValueError(f"a {length} byte request is over the {MAX_REQUEST} byte limit")
            payload = read_exact(conn, length)
        except ConnectionError:
            return True  # the engine went away; the next one may still come
        except ValueError as e:
            # A bad length leaves the stream unsynchronizable: drop this
            # connection, keep the process for the next one.
            log(f"bad request frame: {e}")
            return True
        if kind == SHUTDOWN:
            return False
        try:
            body = respond(kind, payload, encoder, decoder)
            status = 0
        except Exception as e:  # one failed generation is not a failed process
            log(f"request {kind} failed: {e}\n{traceback.format_exc()}")
            body, status = str(e).encode("utf-8"), 1
        send(conn, kind, status, body)


def bind(path: Path) -> socket.socket:
    if path.exists():
        if not stat.S_ISSOCK(path.stat().st_mode):
            raise ValueError(f"{path} exists and is not a socket")
        path.unlink()  # left behind by a sidecar that died
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    return listener


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen-Image-2.1 text encoder and VAE sidecar")
    ap.add_argument("--model-dir", type=Path, required=True, help="the diffusers pipeline directory")
    ap.add_argument("--device", type=int, default=0, help="CUDA ordinal, shared with the transformer")
    ap.add_argument("--socket", type=Path, required=True, help="Unix socket to listen on")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    device = torch.device(f"cuda:{args.device}")
    log(f"loading from {args.model_dir} on {device}")
    encoder = Encoder(args.model_dir, device)
    decoder = Decoder(args.model_dir, device)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    listener = bind(args.socket)
    log(f"listening on {args.socket}")
    print("READY", flush=True)
    try:
        while True:
            conn, _ = listener.accept()
            with conn:
                if not session(conn, encoder, decoder):
                    break
    finally:
        listener.close()
        args.socket.unlink(missing_ok=True)
        log("stopped")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        log(traceback.format_exc())
        sys.exit(1)
