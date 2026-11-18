# Qwen3-TTS Base reference recordings

These tools record the official 1.7B Base implementation for the subsequent
Rust/CUDA voice-clone work. They do not add Base support to the server.
The existing [CustomVoice golden workflow](qwen3-tts.md#correctness) remains
available through the same `golden.py` command.

## Download the fixtures

The [v1 fixture archive](https://github.com/Mrtroll486/pega-omni/releases/download/qwen3-tts-base-fixtures-v1/qwen3-tts-base-fixtures-v1.tar.gz)
contains the twelve Safetensors, eight prepared WAVs, and `manifest.json`.
`ATTRIBUTION.md` maps the WAVs and the embedded `ref_pcm` tensors to their
LibriSpeech and AISHELL-3 sources and documents how the recordings were processed.

```bash
tar -xzf qwen3-tts-base-fixtures-v1.tar.gz
(cd qwen3-tts-base-fixtures-v1 && sha256sum -c SHA256SUMS)
uv run tools/qwen3_tts/record_cases.py \
  --manifest qwen3-tts-base-fixtures-v1/prepared/manifest.json \
  --out qwen3-tts-base-fixtures-v1/fixtures --diagnostics --check-only
```

## Inputs

`tools/qwen3_tts/cases.json` selects the source audio, corpus transcripts and
generation text. Source roots are supplied by the caller; no local absolute
paths are part of the case definitions.

| Case | Corpus recording | Reference duration |
|---|---|---:|
| `en-short` | LibriSpeech dev-clean `1272-128104-0000` | 5.855 s |
| `en-long` | LibriSpeech dev-clean `1272-128104-0002` | 12.485 s |
| `cn-short` | AISHELL-3 `SSB00090025` | 5.328 s |
| `cn-long` | AISHELL-3 `SSB02730077` + `SSB02730036` | 18.608 s |

Download [LibriSpeech dev-clean](https://www.openslr.org/12/) (CC BY 4.0)
and the selected files from [AISHELL-3 at f20d5db4](https://huggingface.co/datasets/AISHELL/AISHELL-3/tree/f20d5db4a31fe779ef07bb1af4ea92da5c786622)
(Apache-2.0 according to its dataset card). The English root contains
`LibriSpeech/dev-clean/`; the Chinese root contains `train/wav/` and
`train/content.txt`. Exact paths and transcript provenance are in the cases
file; corpus labels are retained as supplied.
The Chinese long reference concatenates two complete recordings without
silence or fades; it is a context-boundary case, not a natural quality sample.

```bash
uv run tools/qwen3_tts/prepare.py \
  --en-root /data/EN_speech --cn-root /data/CN_speech \
  --out /data/base-inputs
```

Preparation reads the originals without modifying them, concatenates before
resampling, and writes mono 24 kHz PCM16 WAVs plus `manifest.json`.
NumPy 2.5.4, SciPy 1.18.1 and SoundFile 0.14.0 are pinned in the script.
The manifest records conversion versions and parameters, source paths and
sample counts. It also prepares a distinct same-speaker recording and a
different-speaker recording per language for listening comparisons.

## Record and check

Use `Qwen/Qwen3-TTS-12Hz-1.7B-Base` at revision
`fd4b254389122332181a7c3db7f27e918eec64e3`, including its `speech_tokenizer/`.
The recorder pins qwen-tts to `022e286b98fbec7e1e916cb940cdf532cd9f488e`
and torch/torchaudio to 2.11.0 with CUDA 13.0 wheels. It uses BF16, SDPA,
seed 0 and the checkpoint's generation defaults. Each fixture records the
GPU name and compute capability, PyTorch's CUDA build version, the installed
versions of torch, transformers and qwen-tts, and the effective generation
arguments. Use the released fixture bytes for replay because sampling can
vary across GPUs even with the same seed.

```bash
CUDA_VISIBLE_DEVICES=0 uv run tools/qwen3_tts/record_cases.py \
  --manifest /data/base-inputs/manifest.json \
  --model /models/Qwen3-TTS-12Hz-1.7B-Base \
  --out /data/base-fixtures --diagnostics
```

The four cases run in `xvector` and `icl` modes with
`non_streaming_mode=True`: eight main fixtures. `--diagnostics` also runs the
two short references in both modes with the False layout: twelve fixtures
in total. The layout flag controls the model's text track, not HTTP audio
streaming. A separate process records each case and releases its GPU memory
before the next one starts.

The matrix checks EOS, generated-code/logit counts, finite tensors, codec
latent alignment, exact waveform lengths and crop identity. Long ICL cases
must exceed 300 combined reference and generated frames. Every False-layout
diagnostic must contain more than one trailing text embedding.

Outputs include each fixture's command and log, `summary.json` with tensor
shapes and recorded boundaries, and WAVs of the short-reference outputs for
listening. Keep audio and large fixtures outside Git. To recheck fixture
metadata and boundaries and regenerate the summary and listening WAVs on the
CPU, use `--check-only` without `--model`.

## Fixture contract

Safetensors stores floating-point observations as FP32 and IDs as int64.
Here `T` is generated frames, `R` is reference frames, `L` is prefill tokens,
and `Q` is trailing text tokens. The 1.7B model uses hidden width 2048 and
16 code groups.

| Tensor | Shape | Meaning |
|---|---|---|
| `text_ids` | `[n]` | Wrapped target text IDs |
| `ref_pcm` | `[N]` | Prepared 24 kHz reference samples; Base only |
| `speaker_mel` | `[M, 128]` | FP32 mel before the speaker-encoder cast |
| `speaker_mel_input` | `[M, 128]` | Actual BF16 encoder-input values, saved as FP32 |
| `speaker_embedding` | `[2048]` | Speaker-encoder output values, saved as FP32 |
| `ref_text_ids` | `[n_ref]` | Wrapped reference transcript IDs; ICL only |
| `ref_codec_latent` | `[R, 512]` | Continuous input to the codec encoder's `quantizer.encode`; ICL only |
| `ref_codes` | `[R, 16]` | Reference audio codes; ICL only |
| `prefill_embeds` | `[L, 2048]` | Actual talker prefill input values |
| `attention_mask` | `[L]` | Prefill attention mask |
| `trailing_text_hidden` | `[Q, 2048]` | Text embeddings consumed during decode |
| `tts_pad_embed` | `[1, 2048]` | Text-track padding embedding |
| `talker_logits` | `[T+1, 3072]` | Raw codebook-0 logits, including the final EOS step |
| `predictor_logits` | `[T, 15, 2048]` | Raw remaining-codebook logits |
| `codes` | `[T, 16]` | Generated frames for teacher-forced replay |
| `wav_full` | `[(R+T)*1920]` | Continuous decode before cropping; Base only |
| `wav` | `[T*1920]` | Generated waveform after cropping |

Raw logits precede sampling processors. Replay uses the saved codes to keep
the model history fixed while comparing numerical boundaries.
Each row of `ref_codec_latent` corresponds to the same frame in `ref_codes`,
so codec encoder tests can compare the continuous values before checking
discrete codebook choices.
Base ICL decodes the **complete** reference and generated codes together
through the official underlying decoder, then crops at `R * 1920` samples.
X-vector-only has no reference-code prefix. The public decoder chunks at
300 frames and is not used as the continuous Base waveform oracle.
Metadata records `codec_boundaries` (frame counts, samples per frame and
crop start/end), `text_track` lengths and effective generation arguments.

## Recorded results

2026-11-18, RTX 3090 (sm86), torch 2.11.0+cu130, CUDA 13.0, transformers 4.57.3,
qwen-tts 0.1.1, BF16 SDPA, seed 0. Sampling used temperature 0.9, top-k 50 and
top-p 1.0 for both talker and predictor, talker repetition penalty 1.05 and a
limit of 8192 new tokens. All twelve recordings reached EOS naturally and
passed the CPU recheck. Two preparation runs produced byte-identical WAVs and
manifests.

| Case | Mode | Non-streaming | Reference + generated frames | Prefill / trailing tokens |
|---|---|---|---:|---:|
| en-short | xvector | True | 0 + 235 = 235 | 82 / 1 |
| en-short | icl | True | 74 + 226 = 300 | 185 / 1 |
| en-short | xvector | False | 0 + 238 = 238 | 10 / 71 |
| en-short | icl | False | 74 + 233 = 307 | 84 / 26 |
| en-long | xvector | True | 0 + 229 = 229 | 71 / 1 |
| en-long | icl | True | 157 + 217 = 374 | 278 / 1 |
| cn-short | xvector | True | 0 + 322 = 322 | 79 / 1 |
| cn-short | icl | True | 67 + 269 = 336 | 158 / 1 |
| cn-short | xvector | False | 0 + 301 = 301 | 10 / 68 |
| cn-short | icl | False | 67 + 283 = 350 | 77 / 13 |
| cn-long | xvector | True | 0 + 253 = 253 | 72 / 1 |
| cn-long | icl | True | 233 + 245 = 478 | 332 / 1 |

The recorder rejected missing reference audio and a generation capped before
EOS. The CustomVoice tensors matched the previous recording exactly, and the
sm86 Rust golden passed with 34.59 dB streaming SNR and 34.19 dB with a lagging
consumer.

Manual listening covered the four prepared references, four speaker-control
recordings, and eight short-reference outputs. No obvious errors were heard.
These WAVs were byte-identical to the ones from the v1 recording run.

## Fixture checksums

```text
424b9ffb666c9279ba750520f39531329274c17deea2c2959feabe8388823bf8  fixtures/cn-long-icl-true.safetensors
4f6cbc8ec906cf051fc61acbbdd732dd3b2466b56852d17f50ca1ec41ae25e75  fixtures/cn-long-xvector-true.safetensors
dc5ed58889d93070b524589e4e149123a1cc9c663818d897d8a72179bfc9f162  fixtures/cn-short-icl-false.safetensors
bb5273faf697ad3b7d9be9b6521ff74c2983ec603a96e9d038b9b040479ac2f3  fixtures/cn-short-icl-true.safetensors
126027faa8dfe8512414c3015399a8b440e1bd7ef287e2f88c13d410bf339792  fixtures/cn-short-xvector-false.safetensors
3a8d283e307955a914d146a83ac2f56f9f9be157ff0c8ac2db9b58f0f281a467  fixtures/cn-short-xvector-true.safetensors
d4135a82bd7131ac76efa23bc941ccefbb6ca69f3e40cba8b1fdcc05122de9ff  fixtures/en-long-icl-true.safetensors
0c9f65b89ff90bd31cf6d52b2ecc9c97b2633a757a9314915e6d6244da38d2a9  fixtures/en-long-xvector-true.safetensors
cf9b5054dfb729b0a6926872fb4889d7c504ae60105b394306f1f1eca748a147  fixtures/en-short-icl-false.safetensors
b48ec585440b21d4a26f764bfe1da39beade16ea9c45b1e8cd4cee08739b16d6  fixtures/en-short-icl-true.safetensors
2c8ea655c865c9649bb4c8a1003fd8883efbfd0efdf3d95ccdad9bbafadd86d9  fixtures/en-short-xvector-false.safetensors
c5891ddf2e48bc75a092c1b98cf86c0e1fa03db7c921b06de8f6130ccda0e9ea  fixtures/en-short-xvector-true.safetensors
37734c135ffb3ebd01cddec9501a364a6b91c38e0d23262735575a9380a32633  prepared/cn-different.wav
1b5106d10a9192884bac0726ef9b4af2db81b1d26f0458325dc7b0270097b5bc  prepared/cn-heldout.wav
037dd7786372afa3b3c601a87d2aa871ab335a2a03b459c484c622a06897d719  prepared/cn-long.wav
985c8eb7b3ac9ac3ec723576cc83882890f4bc4cd7317501fa94f46a8db679f4  prepared/cn-short.wav
863083098efb02d38f58f0c4426edbaa290a97b87bb7b0a1c7f3ac8cafcd84cf  prepared/en-different.wav
5d38a6c8e792b8ac3327c024affc6ef043eed5b7637215e633761233297888dd  prepared/en-heldout.wav
aed0806fd6d22f87a1fc3b3251fed39b5f6bcd2a711d517bda0397d24a5504be  prepared/en-long.wav
eb8dec02ded295532d0c066bed27272a5b17dcd5e7a64ac6afd366b4ab21a05f  prepared/en-short.wav
13bb5ba9b281c171f7736824f7cbf2b8eb6ac441b336b11cf07c20ccc4baa85b  prepared/manifest.json
```
