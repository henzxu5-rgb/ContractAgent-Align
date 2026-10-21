"""Isolated cloud runtime, resumable model download, environment receipt."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import re
import urllib.request
import zipfile
import email
import argparse
from concurrent.futures import ThreadPoolExecutor
from .download import fetch_ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated", action="store_true", help="Do not inherit the image's torch/torchvision packages")
    parser.add_argument("--cuda-variant", choices=("cu118", "cu128"), default="cu118")
    parser.add_argument("--domestic-mirrors", action="store_true", help="Use domestic transport while verifying original wheel SHA256")
    parser.add_argument("--package-workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.package_workers <= 4: raise ValueError("Package concurrency must be between 1 and 4")
    project = Path(__file__).resolve().parent.parent
    env = project / ".venv"
    python = env / "bin/python"
    if not python.exists():
        command = [sys.executable, "-m", "venv"]
        if not args.isolated: command.append("--system-site-packages")
        subprocess.run([*command, str(env)], check=True)
    else:
        # Reuse existing CUDA 11 runtime packages, but install all pinned ML overrides in this venv.
        cfg = env / "pyvenv.cfg"
        if args.isolated:
            cfg.write_text(cfg.read_text().replace("include-system-site-packages = true", "include-system-site-packages = false"))
        else:
            cfg.write_text(cfg.read_text().replace("include-system-site-packages = false", "include-system-site-packages = true"))
    lock = project / "contract_agent/requirements-train.txt"
    receipt = project / "environment_receipt.json"
    if not receipt.exists():
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        wheel_name = f"torch-2.7.1+{args.cuda_variant}-{tag}-{tag}-manylinux_2_28_x86_64.whl"
        index = urllib.request.urlopen(f"https://download.pytorch.org/whl/{args.cuda_variant}/torch/", timeout=60).read().decode()
        links = re.findall(r'href="([^"]+)"', index)
        encoded_name = wheel_name.replace("+", "%2B")
        link = next(x for x in links if encoded_name.lower() in x.lower())
        wheel_url, checksum = link.split("#sha256=")
        if wheel_url.startswith("/"): wheel_url = "https://download.pytorch.org" + wheel_url
        if args.domestic_mirrors:
            wheel_url = f"https://mirrors.aliyun.com/pytorch-wheels/{args.cuda_variant}/" + encoded_name
        wheel = project / "contract_agent/artifacts/cache" / wheel_name
        fetch_ranges(wheel_url, wheel, checksum, workers=12)
        # Large CUDA wheels are also fetched with validated parallel ranges.
        cuda_wheels = []
        from packaging.requirements import Requirement
        with zipfile.ZipFile(wheel) as archive:
            metadata_path = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA") and n.startswith("torch-"))
            requires = email.message_from_bytes(archive.read(metadata_path)).get_all("Requires-Dist")
        pinned_large = []
        for raw_req in requires:
            req = Requirement(raw_req)
            if req.marker and not req.marker.evaluate(): continue
            if not (req.name.startswith("nvidia-") or req.name == "triton"): continue
            version = next(s.version for s in req.specifier if s.operator == "==")
            pinned_large.append((req.name, version))
        pinned_large.append(("bitsandbytes", "0.47.0"))
        def fetch_package(spec):
            package, version = spec
            metadata = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=60))
            info = next(x for x in metadata["urls"] if "x86_64" in x["filename"] and "manylinux" in x["filename"] and (tag in x["filename"] if package == "triton" else True))
            target = wheel.parent / info["filename"]
            source = info["url"]
            if args.domestic_mirrors:
                source = "https://mirrors.aliyun.com/pypi/packages/" + source.split("/packages/", 1)[1]
            fetch_ranges(source, target, info["digests"]["sha256"], workers=12)
            return str(target)
        with ThreadPoolExecutor(max_workers=args.package_workers) as pool:
            cuda_wheels = list(pool.map(fetch_package, pinned_large))
        subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir", str(wheel), *cuda_wheels,
                        "--index-url", "https://mirrors.aliyun.com/pypi/simple"], check=True, timeout=1200)
        subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir", "-r", str(lock),
                        "--index-url", "https://mirrors.aliyun.com/pypi/simple"], check=True, timeout=2400)
        code = "import torch,transformers,trl,peft,bitsandbytes,json; x=torch.randn(16,16,device='cuda',dtype=torch.float16); y=x@x; assert torch.isfinite(y).all(); print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'trl':trl.__version__,'peft':peft.__version__,'bnb':bitsandbytes.__version__,'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'memory':torch.cuda.get_device_properties(0).total_memory,'fp16_matmul_verified':True}))"
        output = subprocess.check_output([str(python), "-c", code], text=True)
        receipt.write_text(output, encoding="utf-8")
        print(output, flush=True)
    # Download exact reviewed revision, not an unpinned ModelScope master.
    print("ENVIRONMENT_READY; model weights are downloaded and verified by the separate download_model job", flush=True)


if __name__ == "__main__": main()
