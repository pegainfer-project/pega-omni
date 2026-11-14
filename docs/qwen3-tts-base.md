# Qwen3-TTS Base reference recordings

These tools record the official 1.7B Base implementation for the subsequent
Rust/CUDA voice-clone work. They do not add Base support to the server.
The existing [CustomVoice golden workflow](qwen3-tts.md#correctness) remains
available through the same `golden.py` command.

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
different-speaker recording per language for the quality controls.

## Record and check

Use `Qwen/Qwen3-TTS-12Hz-1.7B-Base` at revision
`fd4b254389122332181a7c3db7f27e918eec64e3`, including its `speech_tokenizer/`.
The recorder pins qwen-tts to `022e286b98fbec7e1e916cb940cdf532cd9f488e`
and torch/torchaudio to 2.11.0 with CUDA 13.0 wheels. It uses BF16, SDPA,
seed 0 and the checkpoint's generation defaults. Each fixture records the
effective generation arguments; a seed alone does not guarantee identical
sampling across GPUs.

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

The matrix checks EOS, generated-code/logit counts, finite tensors, exact
waveform lengths and crop identity. Long ICL cases must actually exceed
300 combined reference/generated frames; every False-layout diagnostic must
contain more than one trailing text embedding. Missing coverage fails the
run rather than silently truncating or padding a recording.

Outputs include each fixture's command and log, `summary.json` with tensor
shapes and recorded boundaries, short-reference WAVs, and `quality-input.json`.
Keep audio and large fixtures outside Git. To recheck recordings without a GPU,
repeat the command with `--check-only` to recheck fixture metadata and
recorded boundaries before rewriting the summary and quality input.

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
Base ICL decodes the **complete** reference and generated codes together
through the official underlying decoder, then crops at `R * 1920` samples.
X-vector-only has no reference-code prefix. The public decoder chunks at
300 frames and is not used as the continuous Base waveform oracle.
Metadata records `codec_boundaries` (frame counts, samples per frame and
crop start/end), `text_track` lengths and effective generation arguments.

## Paired layout diagnostic

The independent models are `openai/whisper-small` at
`973afd24965f72e36ca33b3055d56a652f456b4d` and
`microsoft/wavlm-base-plus-sv` at `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`.
Download only their inference files:

```bash
uvx --from huggingface_hub hf download openai/whisper-small \
  --revision 973afd24965f72e36ca33b3055d56a652f456b4d --local-dir /models/whisper-small \
  --include model.safetensors '*.json' merges.txt
uvx --from huggingface_hub hf download microsoft/wavlm-base-plus-sv \
  --revision feb593a6c23c1cc3d9510425c29b0a14d2b07b1e --local-dir /models/wavlm-base-plus-sv \
  --include pytorch_model.bin config.json preprocessor_config.json
uv run tools/qwen3_tts/quality.py --manifest /data/base-fixtures/quality-input.json \
  --asr-model /models/whisper-small --speaker-model /models/wavlm-base-plus-sv \
  --device cuda:0 --out /data/base-fixtures/quality-results.json
```

The script requires paired True/False text layouts. Its report contains model
directory names, effective ASR generation settings, per-case target text,
transcripts, English WER / Chinese CER, speaker cosines and controls.
Scoring applies NFKC and casefold, then replaces `’` with ASCII `'`.
English uses ASCII alphanumeric words with internal apostrophes; Chinese
counts Unicode letters and numbers individually, ignoring punctuation and
whitespace. It does not expand numbers or convert simplified/traditional
characters.

ASR is greedy; waveforms must fit its 30-second short-form path.
WavLM is independent of the Qwen speaker encoder, but its VoxCeleb training
and English-tagged model card do not establish a universal cosine threshold,
especially for Chinese.
These few cases diagnose layout differences; they are not a quality gate.

## Recorded results

2026-11-14, RTX 3090 (sm86), torch/torchaudio 2.11.0+cu130,
transformers 4.57.3, BF16 SDPA, seed 0. These are correctness runs on shared
GPUs. Sampling used temperature 0.9, top-k 50, top-p 1.0 for both talker and
predictor, talker repetition penalty 1.05 and a limit of 8192 new tokens.
All twelve recordings reached EOS naturally. Two preparation runs produced
byte-identical WAVs and manifests; the complete CPU recheck passed.

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

All waveform lengths and exact reference-frame crops passed. The recorder
also rejected missing reference inputs and a real generation capped before
EOS without writing a fixture. The six existing CustomVoice tensors matched
the previous local fixture exactly; the rebuilt sm86 Rust golden passed its
unchanged thresholds (SNR 34.59 / 34.19 dB).

The eight short-reference outputs were also scored with the pinned independent
models above. Cosine compares generated speech with the reference recording.

| Language | Mode | True WER / CER | False WER / CER | True cosine | False cosine |
|---|---|---:|---:|---:|---:|
| English | x-vector-only | 0 / 65 | 0 / 65 | 0.8574 | 0.8143 |
| English | ICL | 0 / 65 | 0 / 65 | 0.9547 | 0.9081 |
| Chinese | x-vector-only | 27 / 102 | 27 / 102 | 0.9656 | 0.9669 |
| Chinese | ICL | 27 / 102 | 0 / 102 | 0.9690 | 0.9677 |

The error columns count edits / reference words or characters. In each of the
three Chinese outputs with 27 edits, 26 were traditional/simplified character
variants and one was the recognizer's `晴` → `勤` substitution. The script
does not convert character variants, so its raw CER is not a pronunciation
error rate or evidence of a True-layout regression.

Real-audio same/different-speaker cosine controls were 0.9724 / 0.8046 for
English and 0.9524 / 0.7631 for Chinese. English x-vector-only output scored
0.8584 / 0.8389 against the different speaker in True / False layouts,
showing weak speaker discrimination in this sample. Chinese verifier scores
remain uncalibrated. These data do not establish a general layout-quality
ranking; the primary fixtures use True to capture the initial complete-text
prompt, while False fixtures retain the decode-time text path.

Manual listening covered the four prepared references, four speaker-control
recordings, and eight short-reference outputs. No obvious errors were reported
in this listening check.
