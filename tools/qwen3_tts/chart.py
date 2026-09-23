"""The README chart: median time to first audio and audio throughput per
concurrency, pega-omni against vLLM-Omni, from two `vs_vllm_omni.sh bench`
result directories. The headline ratios are computed, never typed.

    chart.py <vllm-omni results> <pega-omni results> <out.png>
"""

import json
import sys
from pathlib import Path

import cairosvg

C = [1, 8, 16, 64]
W, H = 960, 540
BG, FG, DIM, PEGA, VLLM, GRID = "#141b2b", "#ffffff", "#8b95ab", "#7aa2ff", "#465068", "#232c40"
FONT = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""


def load(results: Path) -> dict:
    runs = [json.loads((results / f"c{c}.json").read_text()) for c in C]
    return {
        "ttfp": [r["median_audio_ttfp_ms"] for r in runs],
        "tput": [r["audio_throughput"] for r in runs],
    }


def panel(x0: int, title: str, pega: list, vllm: list, ratio) -> list:
    top, bot, pw, bw = 190, 440, 400, 34
    peak = max(pega + vllm)
    gw = pw / len(C)
    out = [
        f'<text x="{x0}" y="160" fill="{FG}" font-size="18" font-weight="600">{title}</text>',
        f'<line x1="{x0}" y1="{bot}" x2="{x0 + pw}" y2="{bot}" stroke="{GRID}" stroke-width="2"/>',
    ]
    for i, c in enumerate(C):
        cx = x0 + gw * i + gw / 2
        for j, (v, col, ours) in enumerate(((pega[i], PEGA, True), (vllm[i], VLLM, False))):
            h = max(3, (bot - top) * v / peak)
            x = cx - bw - 2 + j * (bw + 4)
            out.append(f'<rect x="{x:.1f}" y="{bot - h:.1f}" width="{bw}" height="{h:.1f}" rx="5" fill="{col}"/>')
            out.append(
                f'<text x="{x + bw / 2:.1f}" y="{bot - h - 8:.1f}" fill="{FG if ours else DIM}" font-size="13" '
                f'font-weight="{600 if ours else 400}" text-anchor="middle">{v:.0f}</text>'
            )
        out.append(f'<text x="{cx:.1f}" y="{bot + 24}" fill="{DIM}" font-size="14" text-anchor="middle">c={c}</text>')
        out.append(
            f'<text x="{cx:.1f}" y="{bot + 46}" fill="{PEGA}" font-size="14" font-weight="700" '
            f'text-anchor="middle">{ratio(pega[i], vllm[i]):.1f}×</text>'
        )
    return out


def svg(vllm: dict, pega: dict) -> str:
    faster = max(v / p for p, v in zip(pega["ttfp"], vllm["ttfp"]))
    more = pega["tput"][-1] / vllm["tput"][-1]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="pega-omni vs vLLM-Omni, Qwen3-TTS on one GB300" {FONT}>',
        f'<rect width="{W}" height="{H}" rx="24" fill="{BG}"/>',
        f'<text x="48" y="74" fill="{FG}" font-size="32" font-weight="700"><tspan fill="{PEGA}">{faster:.1f}×</tspan> '
        f'faster first audio. <tspan fill="{PEGA}">{more:.1f}×</tspan> the throughput.</text>',
        f'<text x="48" y="106" fill="{DIM}" font-size="16">Qwen3-TTS-12Hz-1.7B · one GB300 · '
        f"vs vLLM-Omni at its best single-GPU setup, on its own benchmark</text>",
    ]
    out += panel(48, "Time to first audio, ms ↓", pega["ttfp"], vllm["ttfp"], lambda p, v: v / p)
    out += panel(512, "Audio seconds per second ↑", pega["tput"], vllm["tput"], lambda p, v: p / v)
    for x, col, name in ((48, PEGA, "pega-omni"), (178, VLLM, "vLLM-Omni 0.30.0rc1")):
        out.append(f'<rect x="{x}" y="{H - 40}" width="14" height="14" rx="3" fill="{col}"/>')
        out.append(f'<text x="{x + 22}" y="{H - 28}" fill="{FG}" font-size="14">{name}</text>')
    return "\n".join(out + ["</svg>"]) + "\n"


if __name__ == "__main__":
    vllm_dir, pega_dir, png = map(Path, sys.argv[1:4])
    cairosvg.svg2png(bytestring=svg(load(vllm_dir), load(pega_dir)).encode(), write_to=str(png), scale=2)
    print(png)
