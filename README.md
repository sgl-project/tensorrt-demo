## TensorRT-Demo

### Prerequisite

```bash
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update

sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

sudo apt install nvidia-container-toolkit -y

sudo systemctl daemon-reload && sudo systemctl restart docker
```

```bash
docker pull nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3
docker run --shm-size 32g --gpus all -itd $(docker image ls | grep 24.06 | awk '{print $3}') /bin/bash
```

After starting Docker, we can use `docker exec -it $(docker ps | grep nvidia | awk '{print $1}') /bin/bash`.

### Examples

#### Meta-Llama-3-8B-Instruct
```bash
# setup token
export HF_TOKEN=hf_xxx

# install huggingface-cli
pip3 install "huggingface_hub[cli]"

# setup dir
cd /root && mkdir -p /root/Meta-Llama-3-8B-Instruct

# download model
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --exclude "*.bin" --exclude "*.pth" --exclude "*.pt" --local-dir /root/Meta-Llama-3-8B-Instruct

# install deps
pip3 install datasets==2.14.6 evaluate~=0.4.1 rouge_score~=0.1.2 sentencepiece~=0.1.99 --extra-index-url https://pypi.nvidia.com

# convert checkpoint
wget https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.10.0/examples/llama/convert_checkpoint.py
python3 convert_checkpoint.py --model_dir /root/Meta-Llama-3-8B-Instruct --dtype float16 --tp_size 1 --output_dir work_dir

# build engine
trtllm-build --checkpoint_dir=work_dir --output_dir=trt_dir --gpt_attention_plugin=float16 --gemm_plugin=float16 --remove_input_padding=enable --paged_kv_cache=enable --use_paged_context_fmha enable --multiple_profiles enable --tp_size=1 --max_batch_size=512 --max_input_len=8192 --max_num_tokens=8192

# setup dir
mkdir -p /tensorrtllm_backend
cd /tensorrtllm_backend && git clone --depth=1 --single-branch https://github.com/sgl-project/tensorrt-demo
cp -r /root/trt_dir/* tensorrt-demo/triton_model_repo/tensorrt_llm/1/
wget https://raw.githubusercontent.com/triton-inference-server/tensorrtllm_backend/v0.10.0/scripts/launch_triton_server.py
mv /tensorrtllm_backend/tensorrt-demo/* /tensorrtllm_backend

# run server
ulimit -n 65535 && python3 launch_triton_server.py --world_size=1 --model_repo=/tensorrtllm_backend/triton_model_repo
```

#### Meta-Llama-3-70B-Instruct
```bash
# setup token
export HF_TOKEN=hf_xxx

# install huggingface-cli
pip3 install "huggingface_hub[cli]"

# setup dir
cd /root && mkdir -p /root/Meta-Llama-3-70B-Instruct

# download model
huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --exclude "*.bin" --exclude "*.pth" --exclude "*.pt" --local-dir /root/Meta-Llama-3-70B-Instruct

# install deps
pip3 install datasets==2.14.6 evaluate~=0.4.1 rouge_score~=0.1.2 sentencepiece~=0.1.99 --extra-index-url https://pypi.nvidia.com

# convert checkpoint
wget https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.10.0/examples/llama/convert_checkpoint.py
python3 convert_checkpoint.py --model_dir /root/Meta-Llama-3-70B-Instruct --dtype float16 --tp_size 8 --output_dir work_dir

# build engine
trtllm-build --checkpoint_dir=work_dir --output_dir=trt_dir --gpt_attention_plugin=float16 --gemm_plugin=float16 --remove_input_padding=enable --paged_kv_cache=enable --use_paged_context_fmha enable --multiple_profiles enable --tp_size=8 --max_batch_size=512 --max_input_len=8192 --max_num_tokens=8192 --workers 8

# setup dir
mkdir -p /tensorrtllm_backend
cd /tensorrtllm_backend && git clone --depth=1 --single-branch https://github.com/sgl-project/tensorrt-demo
cp -r /root/trt_dir/* tensorrt-demo/triton_model_repo/tensorrt_llm/1/
wget https://raw.githubusercontent.com/sgl-project/tensorrt-demo/main/launch_triton_server.py
mv /tensorrtllm_backend/tensorrt-demo/* /tensorrtllm_backend

sed -i 's|/root/Meta-Llama-3-8B-Instruct|/root/Meta-Llama-3-70B-Instruct|g' triton_model_repo/preprocessing/config.pbtxt
sed -i 's|/root/Meta-Llama-3-8B-Instruct|/root/Meta-Llama-3-70B-Instruct|g' triton_model_repo/postprocessing/config.pbtxt

# run server
ulimit -n 65535 && python3 launch_triton_server.py --world_size=8 --model_repo=/tensorrtllm_backend/triton_model_repo
```

#### Meta-Llama-3-70B-Instruct FP8

```bash
# setup token
export HF_TOKEN=hf_xxx

# install huggingface-cli
pip3 install "huggingface_hub[cli]"

# setup dir
cd /root && mkdir -p /root/Meta-Llama-3-70B-Instruct

# download model
huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --exclude "*.bin" --exclude "*.pth" --exclude "*.pt" --local-dir /root/Meta-Llama-3-70B-Instruct

# install deps
pip3 install datasets==2.14.6 evaluate~=0.4.1 rouge_score~=0.1.2 sentencepiece~=0.1.99 nemo-toolkit[all]<=1.20.0,>=1.18.0 transformers_stream_generator==0.0.4 tiktoken mpmath==1.3.0 --extra-index-url https://pypi.nvidia.com
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-modelopt

# convert checkpoint
wget https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.10.0/examples/quantization/quantize.py
python3 quantize.py --model_dir /root/Meta-Llama-3-70B-Instruct --dtype float16 --qformat fp8 --kv_cache_dtype fp8 --calib_size 512 --tp_size 8 --output_dir work_dir

# build engine
trtllm-build --checkpoint_dir=work_dir --output_dir=trt_dir --gpt_attention_plugin=float16 --gemm_plugin=float16 --remove_input_padding=enable --paged_kv_cache=enable --use_paged_context_fmha enable --multiple_profiles enable --tp_size=8 --max_batch_size=512 --max_input_len=8192 --max_num_tokens=8192 --workers 8 --strongly_typed 

# setup dir
mkdir -p /tensorrtllm_backend
cd /tensorrtllm_backend && git clone --depth=1 --single-branch https://github.com/sgl-project/tensorrt-demo
cp -r /root/trt_dir/* tensorrt-demo/triton_model_repo/tensorrt_llm/1/
wget https://raw.githubusercontent.com/sgl-project/tensorrt-demo/main/launch_triton_server.py
mv /tensorrtllm_backend/tensorrt-demo/* /tensorrtllm_backend

sed -i 's|/root/Meta-Llama-3-8B-Instruct|/root/Meta-Llama-3-70B-Instruct|g' triton_model_repo/preprocessing/config.pbtxt
sed -i 's|/root/Meta-Llama-3-8B-Instruct|/root/Meta-Llama-3-70B-Instruct|g' triton_model_repo/postprocessing/config.pbtxt

# run server
ulimit -n 65535 && python3 launch_triton_server.py --world_size=8 --model_repo=/tensorrtllm_backend/triton_model_repo
```
