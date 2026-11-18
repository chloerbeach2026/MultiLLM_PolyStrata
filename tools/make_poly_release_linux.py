#!/usr/bin/env python3
"""Build and package the Linux x86-64 CUDA 13 strata-poly release.

The first supported target is SM75 (Tesla T4 / RTX 20). The build uses the
same source fingerprint and BUILD.json contract as setup.py's prebuilt loader.
The NVIDIA driver is a host dependency: the CUDA stub is used only to link and
is never included in the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run_checked(args: list[str], *, timeout: int = 60) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{' '.join(args)} failed ({p.returncode}):\n{p.stdout[-12000:]}")
    return p.stdout


def require_elf(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"required build output is missing: {path}")
    header = run_checked(["readelf", "-h", str(path)])
    if "ELF64" not in header or "Advanced Micro Devices X86-64" not in header:
        raise RuntimeError(f"{path.name} is not a Linux x86-64 ELF binary")
    deps = run_checked(["ldd", str(path)])
    missing = []
    for line in deps.splitlines():
        if "not found" not in line:
            continue
        if "libcuda.so.1" in line:
            continue
        missing.append(line.strip())
    if missing:
        raise RuntimeError(f"{path.name} has unresolved runtime dependencies:\n" + "\n".join(missing))
    needed = run_checked(["readelf", "-d", str(path)])
    if "libcuda.so.1" not in needed:
        raise RuntimeError(f"{path.name} does not declare the expected host NVIDIA driver dependency")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--archs", default="75",
                        help="semicolon- or comma-separated CUDA SM targets (default: 75; T4)")
    args = parser.parse_args()

    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise SystemExit("this release builder requires Linux x86-64")
    archs = sorted({int(x) for x in args.archs.replace(",", ";").split(";") if x.strip()})
    if not archs or any(x < 75 or x > 120 for x in archs):
        raise SystemExit(f"unsupported CUDA architecture list: {archs}; expected SM75 or newer")
    if 75 not in archs:
        raise SystemExit("the first project release must include SM75 for the Kaggle Tesla T4 target")

    dirty = run_checked(["git", "status", "--porcelain", "--ignored", "--",
                         *setup.ENGINE_SOURCES, *setup.VISION_SOURCES])
    if dirty.strip():
        raise SystemExit("engine sources are not a clean commit:\n" + dirty)

    nvcc, cuda_v = setup.find_nvcc()
    if not nvcc or cuda_v != (13, 0):
        raise SystemExit(f"CUDA Toolkit 13.0 is required for the pinned runtime wheels; found {cuda_v}")
    cuda_root = Path(nvcc).resolve().parent.parent
    stub_candidates = [
        cuda_root / "targets" / "x86_64-linux" / "lib" / "stubs",
        cuda_root / "lib64" / "stubs",
    ]
    stub_dir = next((p for p in stub_candidates if (p / "libcuda.so").is_file()), None)
    if stub_dir is None:
        raise SystemExit("CUDA driver stub libcuda.so was not found under the selected toolkit")
    # CMake must discover CUDA::cuda_driver. This is a link-time stub only;
    # the real libcuda.so.1 is always supplied by the Kaggle host driver.
    os.environ["LIBRARY_PATH"] = os.pathsep.join(
        [str(stub_dir), os.environ.get("LIBRARY_PATH", "")]).rstrip(os.pathsep)
    os.environ["CMAKE_LIBRARY_PATH"] = str(stub_dir)

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    poly_build = out / "build-poly"
    vision_build = out / "build-vision"
    llama = setup.get_llama_cpp()
    vision_llama = setup.get_llama_cpp(vision=True)
    arch_arg = ";".join(f"{a}-real" for a in archs)
    common = [
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch_arg}",
        f"-DCUDAToolkit_ROOT={cuda_root}",
        f"-DCMAKE_LIBRARY_PATH={stub_dir}",
        "-DSTRATA_PORTABLE=ON",
    ]

    setup.cmake_build(
        ROOT, poly_build, "strata-poly",
        ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
         "-DSTRATA_MMQ_KQUANTS=ON", "-DSTRATA_Q6K_EXPERTS=ON",
         f"-DSTRATA_GGML_DIR={llama}", *common],
        None, "",
    )
    setup.cmake_build(
        ROOT / "tools" / "vision", vision_build, "strata-vision",
        [f"-DLLAMA_DIR={vision_llama}", "-DSTRATA_VISION_CUDA=ON", *common],
        None, "",
    )

    poly_exe = poly_build / setup.POLY_EXE
    vision_exe = vision_build / "bin" / setup.VEXE
    require_elf(poly_exe)
    require_elf(vision_exe)
    kq_candidates = list(poly_build.rglob(setup.KQ_LIB))
    if not kq_candidates:
        raise RuntimeError(f"{setup.KQ_LIB} was not produced; K-quant MMQ support is required")
    kq_lib = kq_candidates[0]
    require_elf(kq_lib)

    source = setup.source_hash(setup.ENGINE_SOURCES)
    vision_source = setup.source_hash(setup.VISION_SOURCES)
    commit = run_checked(["git", "rev-parse", "HEAD"]).strip()
    meta = {
        "source": "release",
        "platform": "linux-x64",
        "backend": "cuda",
        "version": setup.source_version(),
        "source_commit": commit,
        "archs": archs,
        "cuda": "13.0",
        "toolkit": 13,
        "src": source,
        "vision": "gpu",
        "vision_src": vision_source,
        "portable": True,
        "driver": "host NVIDIA driver (libcuda.so.1; not bundled)",
        "kq": setup.KQ_LIB,
    }

    asset = out / setup.poly_asset()
    with zipfile.ZipFile(asset, "w", compression=zipfile.ZIP_STORED) as z:
        z.write(poly_exe, setup.POLY_EXE)
        z.write(vision_exe, setup.VEXE)
        z.write(kq_lib, setup.KQ_LIB)
        z.writestr("BUILD.json", json.dumps(meta, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(asset) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"archive CRC check failed at {bad}")
        names = set(z.namelist())
        required = {setup.POLY_EXE, setup.VEXE, setup.KQ_LIB, "BUILD.json"}
        if not required <= names:
            raise RuntimeError(f"archive is missing files: {sorted(required - names)}")

    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text(f"{digest}  {asset.name}\n", encoding="ascii")
    tag = f"v{setup.source_version()}-linux-sm75-{commit[:12]}"
    (out / "release-tag.txt").write_text(tag + "\n", encoding="ascii")
    (out / "release-notes.md").write_text(
        f"Linux x86-64 CUDA 13.0 prebuilt for PolyStrata source {commit}.\n\n"
        f"- CUDA architectures: {', '.join(f'sm_{x}' for x in archs)}\n"
        f"- Includes {setup.POLY_EXE}, GPU {setup.VEXE}, and {setup.KQ_LIB}.\n"
        "- Built and link-checked in the CUDA 13.0 Linux build container.\n"
        "- libcuda.so.1 is intentionally not bundled; Kaggle's NVIDIA driver supplies it.\n"
        "- GPU inference smoke testing is separate from this compile/package workflow.\n",
        encoding="utf-8",
    )
    print(f"ASSET={asset}")
    print(f"SIZE={asset.stat().st_size}")
    print(f"SHA256={digest}")
    print(f"TAG={tag}")
    print(f"BUILD={json.dumps(meta, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
