"""CPU checks for the continuous codec oracle, with no model download."""

import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from golden import decode_continuous


@unittest.skipIf(torch is None, "requires the golden recorder's torch dependency")
class ContinuousDecodeTests(unittest.TestCase):
    def test_all_frames_reach_the_decoder_in_one_call(self):
        # 366 crosses the public API's 300-frame boundary; 17 covers short audio.
        for frames in (17, 366):
            with self.subTest(frames=frames):
                codes = torch.arange(frames * 16).reshape(frames, 16)
                calls = []

                def decoder(batch):
                    self.assertFalse(torch.is_grad_enabled())
                    calls.append(batch.clone())
                    return torch.arange(frames * 1920, dtype=torch.bfloat16)[None, None]

                tokenizer = SimpleNamespace(
                    device="cpu",
                    get_model_type=lambda: "qwen3_tts_tokenizer_12hz",
                    model=SimpleNamespace(
                        decoder=decoder,
                        decode_upsample_rate=1920,
                        get_output_sample_rate=lambda: 24000,
                    ),
                )
                wav, rate = decode_continuous(tokenizer, codes)
                self.assertEqual(len(calls), 1)
                self.assertTrue(torch.equal(calls[0], codes.T[None]))
                self.assertEqual(wav.shape, (frames * 1920,))
                self.assertEqual(wav.dtype, torch.float32)
                self.assertEqual(wav.device.type, "cpu")
                self.assertEqual(rate, 24000)

    def test_incomplete_decoder_output_is_rejected(self):
        tokenizer = SimpleNamespace(
            device="cpu",
            get_model_type=lambda: "qwen3_tts_tokenizer_12hz",
            model=SimpleNamespace(
                decoder=lambda _: torch.zeros(1, 1, 1920),
                decode_upsample_rate=1920,
                get_output_sample_rate=lambda: 24000,
            ),
        )
        with self.assertRaisesRegex(ValueError, "expected 3840"):
            decode_continuous(tokenizer, torch.zeros(2, 16, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
