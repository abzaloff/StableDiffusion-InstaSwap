import subprocess
import os, sys
from typing import Any
import pkg_resources
from tqdm import tqdm
import urllib.request
from packaging import version as pv

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        model_path = os.path.abspath("models")


BASE_PATH = os.path.dirname(os.path.realpath(__file__))

req_file = os.path.join(BASE_PATH, "requirements.txt")

models_dir = os.path.join(models_path, "insightface")


model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)
model_tmp_path = model_path + ".part"
model_min_size = 100 * 1024 * 1024

def pip_install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])

def pip_uninstall(*args):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *args])

def is_installed (
        package: str, version: str | None = None, strict: bool = True
):
    has_package = None
    try:
        has_package = pkg_resources.get_distribution(package)
        if has_package is not None:
            installed_version = has_package.version
            if (installed_version != version and strict == True) or (pv.parse(installed_version) < pv.parse(version) and strict == False):
                return False
            else:
                return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def is_valid_swapper_model(path):
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < model_min_size:
            print(f"Face swap model is too small: {path}", flush=True)
            return False
        import onnx
        onnx.load(path)
        return True
    except Exception as e:
        print(f"Face swap model validation failed: {e}", flush=True)
        return False

def download_and_validate_swapper_model():
    if os.path.exists(model_tmp_path):
        print(f"Removing incomplete previous download: {model_tmp_path}", flush=True)
        os.remove(model_tmp_path)

    print(f"Downloading face swap model to temporary file: {model_tmp_path}", flush=True)
    print("This can take several minutes. Do not close Forge until the download finishes.", flush=True)
    download(model_url, model_tmp_path)

    print("Download finished, verifying face swap model...", flush=True)
    if not is_valid_swapper_model(model_tmp_path):
        try:
            os.remove(model_tmp_path)
        except OSError:
            pass
        raise RuntimeError("Downloaded face swap model is invalid")

    os.replace(model_tmp_path, model_path)
    print(f"Face swap model has been downloaded and verified: {model_path}", flush=True)

def ensure_swapper_model_works():
    if os.path.exists(model_path) and is_valid_swapper_model(model_path):
        return True

    if os.path.exists(model_path):
        print("Face swap model is broken and will be downloaded again", flush=True)
    else:
        print("Face swap model is missing and will be downloaded", flush=True)

    download_and_validate_swapper_model()
    return False

def download(url, path):
    with urllib.request.urlopen(url) as request:
        total = int(request.headers.get('Content-Length', 0))
        downloaded = 0
        next_progress = 50 * 1024 * 1024
        with open(path, "wb") as file:
            with tqdm(total=total, desc=f"Downloading {os.path.basename(path)}", unit='B', unit_scale=True, unit_divisor=1024) as progress:
                while True:
                    chunk = request.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    downloaded += len(chunk)
                    progress.update(len(chunk))
                    if downloaded >= next_progress:
                        if total > 0:
                            print(f"Download progress: {downloaded // (1024 * 1024)} MB / {total // (1024 * 1024)} MB", flush=True)
                        else:
                            print(f"Download progress: {downloaded // (1024 * 1024)} MB", flush=True)
                        next_progress += 50 * 1024 * 1024

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

last_device = None
first_run = False
available_devices = ["CPU", "CUDA"]

try:
    last_device_log = os.path.join(BASE_PATH, "last_device.txt")
    with open(last_device_log) as f:
        last_device = f.readline().strip()
    if last_device not in available_devices:
        last_device = None
except:
    last_device = "CPU"
    first_run = True
    with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
        txt.write(last_device)

with open(req_file) as file:
    install_count = 0
    ort = "onnxruntime-gpu"
    import torch
    try:
        if torch.cuda.is_available():
            if first_run or last_device is None:
                last_device = "CUDA"
        elif torch.backends.mps.is_available() or hasattr(torch,'dml'):
            ort = "onnxruntime"
            if first_run:
                pip_uninstall("onnxruntime", "onnxruntime-gpu")
            if last_device == "CUDA" or last_device is None:
                last_device = "CPU"
        else:
            if last_device == "CUDA" or last_device is None:
                last_device = "CPU"
        with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
            txt.write(last_device)
        if not is_installed(ort,"1.16.1",False):
            install_count += 1
            pip_install(ort, "-U")
    except Exception as e:
        print(e)
        print(f"\nERROR: Failed to install {ort} - InstaSwap won't start")
        raise e
    strict = True
    for package in file:
        package_version = None
        try:
            package = package.strip()
            if "==" in package:
                package_version = package.split('==')[1]
            elif ">=" in package:
                package_version = package.split('>=')[1]
                strict = False
            if not is_installed(package,package_version,strict):
                install_count += 1
                pip_install(package)
        except Exception as e:
            print(e)
            print(f"\nERROR: Failed to install {package} - InstaSwap won't start")
            raise e
    if not ensure_swapper_model_works():
        install_count += 1
    if install_count > 0:
        print(f"""
        +---------------------------------+
        --- PLEASE, RESTART the Server! ---
        +---------------------------------+
        """)
