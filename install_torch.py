import subprocess
import sys
import platform

def get_cuda_version():
    try:
        out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL).decode()
        return True
    except Exception:
        return False

def install_torch(cuda: bool):
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if cuda:
        # Default to CUDA 11.8. Change to cu121 if needed
        pip_cmd += [
            "torch==2.2.1+cu118",
            "torchvision==0.17.1+cu118",
            "-f", "https://download.pytorch.org/whl/torch_stable.html"
        ]
    else:
        pip_cmd += ["torch", "torchvision"]

    subprocess.run(pip_cmd)

if __name__ == "__main__":
    has_cuda = get_cuda_version()
    print(f"{'✅ CUDA detected' if has_cuda else '❌ CUDA not detected'} — installing appropriate PyTorch version.")
    install_torch(cuda=has_cuda)
