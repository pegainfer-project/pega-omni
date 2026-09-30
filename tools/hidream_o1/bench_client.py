"""vLLM-Omni's diffusion benchmark client, sending the OpenAI request pega-omni accepts.

Its `/v1/images/generations` backend puts vLLM-Omni's own `num_inference_steps`
(50 by default, 2 for warmups) and `seed` at the top level of every request.
pega-omni serves the distilled checkpoint at its fixed 28 steps, takes `seed` in
`extra`, and refuses unknown fields, so this drops those two keys and changes
nothing else: same prompts, sizes, concurrency, timing and metrics.

    python bench_client.py <vllm-omni checkout> <diffusion_benchmark_serving.py arguments...>
"""

import dataclasses
import importlib
import runpy
import sys

root = sys.argv[1]
script_dir = f"{root}/benchmarks/diffusion"
sys.path[:0] = [script_dir, root]

backends = importlib.import_module("backends")

original = backends.async_request_openai_image_generations


async def openai_request(request, session, pbar=None):
    return await original(dataclasses.replace(request, num_inference_steps=None, seed=None), session, pbar)


endpoint = "/v1/images/generations"
backends.backends_function_mapping["2i"][endpoint] = (openai_request, endpoint)
sys.argv = [f"{script_dir}/diffusion_benchmark_serving.py", *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
