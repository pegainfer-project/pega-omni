import os, sys, time
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_text_to_image_prompt, get_model_class_name

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image-2.1-Turbo")
omni = Omni(model=MODEL, trust_remote_code=True)
mcls = get_model_class_name(omni)
print("model class:", mcls)

def gen():
    pd = {"prompt": "A lighthouse on a cliff at dawn.", "negative_prompt": ""}
    pd = build_text_to_image_prompt(mcls, pd, height=1024, width=1024)
    params = OmniDiffusionSamplingParams(height=1024, width=1024, seed=42,
                                         num_inference_steps=8, true_cfg_scale=1.0)
    t0 = time.perf_counter()
    out = omni.generate(pd, sampling_params_list=[params])
    return time.perf_counter() - t0

gen()  # warmup
times = [gen() for _ in range(3)]
print("VLLM-OMNI L20 8-step times:", ["%.3f" % t for t in times])
print("VLLM-OMNI mean %.3f s/image" % (sum(times)/len(times)))