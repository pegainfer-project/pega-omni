"""Drives a running pega-omni image server with the official `openai` Python SDK.

The image counterpart of `openai_sdk_check.py`: if the SDK's stock
`images.generate` works unchanged, a client written for OpenAI works too.

    uv run --with openai tools/openai_images_check.py --base-url http://127.0.0.1:8001/v1 --model pega-omni-sim-image

Any image engine: `--size` must be one the engine serves, `--extra` is passed through.
"""

import argparse
import base64
import json
import struct

import openai

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == PNG_MAGIC and data[12:16] == b"IHDR", data[:16]
    return struct.unpack(">II", data[16:24])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="pega-omni-sim-image")
    ap.add_argument("--size", default="96x64")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--extra", type=json.loads, default={}, help="the request's `extra` object, as JSON")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    client = openai.OpenAI(base_url=args.base_url, api_key="unused", timeout=args.timeout)
    width, height = (int(x) for x in args.size.split("x"))

    resp = client.images.generate(
        model=args.model,
        prompt="A lighthouse on a cliff at dawn.",
        n=args.n,
        size=args.size,
        response_format="b64_json",
        extra_body={"extra": args.extra},
    )
    assert len(resp.data) == args.n, len(resp.data)
    for item in resp.data:
        assert png_size(base64.b64decode(item.b64_json)) == (width, height)

    try:
        client.images.generate(model=args.model, prompt="x", size="7x7")
        raise AssertionError("unserved size accepted")
    except openai.BadRequestError as e:
        assert e.body["param"] == "size", e.body

    try:
        client.images.generate(model=args.model, prompt="x", response_format="url")
        raise AssertionError("url accepted")
    except openai.BadRequestError as e:
        assert e.body["param"] == "response_format", e.body

    try:
        client.images.generate(model="gpt-image-1", prompt="x")
        raise AssertionError("unknown model accepted")
    except openai.NotFoundError as e:
        assert e.body["code"] == "model_not_found", e.body

    assert [m.id for m in client.models.list()] == [args.model]
    print(f"openai {openai.__version__}: b64 png pictures, errors and models all conform")


if __name__ == "__main__":
    main()
