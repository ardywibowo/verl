
source "private/keys.sh"

pip install torch torchvision
pip install flash-attn --no-build-isolation

apt-get update
apt-get install cuda-toolkit-12-4
apt-get install libcudnn9-cuda-12 libcudnn9-dev-cuda-12

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex

pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
pip install --no-deps megatron-core==0.12.0rc3
pip install vllm

pip install -r requirements.txt