## TensorRT-Demo

First, clone the [tensorrt-demo](https://github.com/neuralmagic/tensorrt-demo) repository:

```bash
git clone git@github.com:neuralmagic/tensorrt-demo.git
cd tensorrt-demo
export tensorrt_demo_dir=`pwd`

```

Then, clone the [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend) repository:

```bash
git clone git@github.com:triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
export tensorrtllm_backend_dir=`pwd`
git lfs install
```

Ensure that the version of [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend) is set to [r24.04](https://github.com/triton-inference-server/tensorrtllm_backend/tree/r24.04):

```bash
git fetch --all
git checkout -b r24.04 -t origin/r24.04

git submodule update --init --recursive
```

Copy **triton_model_repo** directory from tensorrt-demo to tensorrtllm_backend: 

```bash
cp -r ${tensorrt_demo_dir}/triton_model_repo ${tensorrtllm_backend_dir}/
```

Start **trt-llm-triton** docker:

```bash
export models_dir=$HOME/models
docker run -it -d --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus all -v ${tensorrtllm_backend_dir}:/tensorrtllm_backend  -v $HOME/models:/models -v ${tensorrt_demo_dir}:/root/tensorrt-demo --name triton_server nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3 bash

docker exec -it triton_server /bin/bash
```

Set model params. Modify *model_type* and *model_name* to point to your model, and modify the model dtype/tp_size/max_batch_size etc... based on your requirements:

```bash
export models_dir=/models

export model_type=llama
export model_name=Meta-Llama-3-70B-Instruct

export model_dtype=float16
export model_tp_size=2

export max_batch_size=256
export max_input_len=2048
export max_output_len=1024

export model_path=${models_dir}/${model_name}
export trt_model_path=${models_dir}/${model_name}-trt-ckpt
export trt_engine_path=${models_dir}/${model_name}-trt-engine
```

Convert hugging face checkpoint to TRT checkpoint:

```bash
cd /tensorrtllm_backend
cd ./tensorrt_llm/examples/${model_type}

python3 convert_checkpoint.py \
    --model_dir ${model_path} \
    --dtype ${model_dtype} \
    --tp_size ${model_tp_size} \
    --output_dir ${trt_model_path} \
```

Compile TRT checkpoint to TRT engine:

```bash     
trtllm-build \
    --checkpoint_dir=${trt_model_path} \
    --gpt_attention_plugin=${model_dtype} \
    --gemm_plugin=${model_dtype} \
    --remove_input_padding=enable \
    --paged_kv_cache=enable \
    --tp_size=${model_tp_size} \
    --max_batch_size=${max_batch_size} \
    --max_input_len=${max_input_len} \
    --max_output_len=${max_output_len} \
    --max_num_tokens=${max_output_len} \
    --opt_num_tokens=${max_output_len} \
    --output_dir=${trt_engine_path} \
```

Copy the generated TRT engine to *triton_model_repo* as follows:

```bash     
cd /tensorrtllm_backend/triton_model_repo
cp -r ${trt_engine_path}/* ./tensorrt_llm/1
```

Modify **triton_model_repo** config files as follows:
1. Modify **ensemble/config.pbtxt**: 

| Param | Value |
| ----- | ----- |
| `max_batch_size` | Set to the value of **${max_batch_size}**  |

2. Modify **preprocessing/config.pbtxt**: 

| Param | Value |
| ----- | ----- |
| `max_batch_size` | Set to the value of **${max_batch_size}**  |
| `tokenizer_dir` | Set to the value of **${model_path}**  |

3. Modify **postprocessing/config.pbtxt**: 

| Param | Value |
| ----- | ----- |
| `max_batch_size` | Set to the value of **${max_batch_size}**  |
| `tokenizer_dir` | Set to the value of **${model_path}**  |

4. Modify **tensorrt_llm/config.pbtxt**: 

| Param | Value |
| ----- | ----- |
| `max_batch_size` | Set to the value of **${max_batch_size}**  |
| `decoupled` | Ensure it is set to **true** (to allow generate_stream)  |
| `gpt_model_type` | Ensure it is using **inflight_fused_batching** to allow continuous batching of requests  |
| `batch_scheduler_policy` | Ensure it is using **max_utilization** to batch requests as much as possible  |
| `kv_cache_free_gpu_mem_fraction` | Ensure it is set to **0.9**. This value indicates the maximum fraction of GPU memory (after loading the model) that may be used for KV cache.  |


4. Modify **tensorrt_llm_bls/config.pbtxt**: 

| Param | Value |
| ----- | ----- |
| `max_batch_size` | Set to the value of **${max_batch_size}**  |
| `decoupled` | Ensure it is set to **true** (to allow generate_stream)  |

Start Triton server:

```bash
cd /tensorrtllm_backend
python3 scripts/launch_triton_server.py --world_size=${model_tp_size} --model_repo=/tensorrtllm_backend/triton_model_repo
```

Ensure that the triton-server is loaded correctly by checking that the model parts are in READY state, like in this output:

```bash
I0530 15:11:18.363912 56200 server.cc:677] 
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| ensemble         | 1       | READY  |
| postprocessing   | 1       | READY  |
| preprocessing    | 1       | READY  |
| tensorrt_llm     | 1       | READY  |
| tensorrt_llm_bls | 1       | READY  |
+------------------+---------+--------+

I0530 15:11:18.675865 56200 metrics.cc:877] Collecting metrics for GPU 0: NVIDIA A100-SXM4-80GB
```

At this point, triton-server is running inside the docker container, so we can exit the docker or go to another terminal to run the client.

For client benchmarking, we are using [benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) from the [vLLM](https://github.com/vllm-project/vllm) repository.

First, clone the vLLM repository and install the package (a clean virtualenv is recommended here):

```bash
git clone git@github.com:vllm-project/vllm.git
cd vllm
export vllm_dir=`pwd`
pip install -e .
```

Now, we can run benchmark_serving.py to benchmark the triton-server:

```bash
cd ${vllm_dir}
cd benchmarks

# This is the same model from above
export model_name=Meta-Llama-3-70B-Instruct

# Modify --sonnet-input-len, --sonnet-prefix-len, --sonnet-output-len and --request-rate based on your requirements 
python benchmark_serving.py --backend tensorrt-llm --endpoint /v2/models/ensemble/generate_stream  --host 0.0.0.0 --port 8000 --model $HOME/models/${model_name} --num-prompts 100 --save-result --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 512 --sonnet-prefix-len 256 --sonnet-output-len 256 --request-rate 1
```

To run a vLLM server, we need first to match its **--gpu-memory-utilization** parameter with triton's **--kv_cache_free_gpu_mem_fraction**. Above, we have set **--kv_cache_free_gpu_mem_fraction=0.9**, however, it is not the same as vLLM's default **--gpu-memory-utilization=0.9**, since triton's parameter is relating to the fraction of GPU memory that we have **after loading the model** (where in vLLM it is before loading the model). Therefore, the right --gpu-memory-utilization for vLLM would be computed as *((GPU_TOTAL_MEMORY - MODEL_MEMORY) \* 0.9 + MODEL_MEMORY) / GPU_TOTAL_MEMORY*. For LLama3 70B FP16 with *MODEL_MEMORY=68296MB*, and A100 GPU with *GPU_TOTAL_MEMORY=81920MB*, we get *((81920-68296)\*0.9 + 68296) / 81920 = 0.9833*, so we need to use **--gpu-memory-utilization=0.9833** in this case.

```bash
cd ${vllm_dir}

# These are the same model params from above (that were used inside the docker container)
export model_name=Meta-Llama-3-70B-Instruct
export model_tp_size=2
export model_dtype=float16
export max_input_len=2048
export vllm_gpu_memory_utilization=0.9833 

# Run server
python3 vllm/entrypoints/openai/api_server.py --model $HOME/models/${model_name} --max-model-len ${max_input_len} --disable-log-requests --enforce-eager --tensor-parallel-size ${model_tp_size} --dtype=${model_dtype} --port 8888 --gpu-memory-utilization ${vllm_gpu_memory_utilization}
```

Run benchmark_serving.py to benchmark the vllm-server:

```bash
cd ${vllm_dir}
cd benchmarks

export model_name=Meta-Llama-3-70B-Instruct

# Modify --sonnet-input-len, --sonnet-prefix-len, --sonnet-output-len and --request-rate based on your requirements 
python benchmark_serving.py --backend vllm --host localhost --port 8888 --endpoint /v1/completions --model $HOME/models/${model_name} --num-prompts 100 --save-result --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 512 --sonnet-prefix-len 256 --sonnet-output-len 256 --request-rate 1
```