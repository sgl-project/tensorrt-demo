import json
import argparse
import subprocess


def main(args):
    with open(args.model_config, "r") as f:
        config = json.load(f)

    size_params = "--max_batch_size={} --max_input_len={} --max_output_len={}".format(
        config["max_batch_size"], config["max_input_len"],
        config["max_output_len"])

    extra_params = "--paged_kv_cache=enable --remove_input_padding=enable"
    cmd = (
        "trtllm-build {} --checkpoint_dir {} --output_dir={} --tp_size={} --workers={}  --gemm_plugin={} --gpt_attention_plugin={}"
        .format(extra_params, config["model_trt_ckpt"],
                config["model_trt_engine"], config["tp_size"],
                config["tp_size"], config["dtype"], config["dtype"]))

    print("Running command: {}".format(cmd))
    subprocess.run(cmd, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, default=None)

    args = parser.parse_args()

    main(args)
