import os

from huggingface_hub import snapshot_download
from transformers.utils import move_cache

MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
MODEL_DIR = "/home/alexm/models/trt/Meta-Llama-3-70B-Instruct"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors 
    )
    move_cache()


if __name__ == "__main__":
    main()
