#!/usr/bin/env python3
"""Cuenta instrucciones FP en heavy_cpu (PTX)."""

import subprocess
import re
import os
from pathlib import Path

WORK_DIR = Path(__file__).parent.resolve()

FILES = {
    "ej6.cu": "ej6_gpu.ptx",
    "ej6_cpu.cu": "ej6_cpu.ptx",
}

FP_PATTERNS = [
    r"sin\.approx\.f32",
    r"cos\.approx\.f32",
    r"mul\.f32",
    r"add\.f32",
    r"sub\.f32",
    r"fma\.rn\.f32",
]


def main():
    os.chdir(WORK_DIR)
    
    for src, ptx in FILES.items():
        if not Path(src).exists():
            continue
        
        # Compilar a PTX (silencioso)
        subprocess.run(["nvcc", "-arch=sm_75", "--ptx", src, "-o", ptx],
                       capture_output=True)
        
        if not Path(ptx).exists():
            print(f"{src}: error compilando")
            continue
        
        content = Path(ptx).read_text()
        
        print(f"\n{src} - Instrucciones FP:")
        total = 0
        for pattern in FP_PATTERNS:
            count = len(re.findall(pattern, content))
            if count > 0:
                print(f"  {pattern}: {count}")
                total += count
        print(f"  TOTAL: {total}")


if __name__ == "__main__":
    main()
