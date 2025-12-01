#!/usr/bin/env python3
"""
Script para compilar ej6.cu y ej6_cpu.cu, generar PTX/SASS y contar
instrucciones de punto flotante en la función/kernel heavy_cpu.

Uso:
    python analyze_flops.py

Requisitos:
    - nvcc en PATH (CUDA Toolkit instalado)
    - cuobjdump en PATH (viene con CUDA Toolkit)

Alternativas mejores para contar FLOPs:
    1. nvprof / Nsight Compute: herramientas de profiling de NVIDIA que
       miden directamente los FLOPs ejecutados en la GPU.
       Ejemplo: ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum ./ej6 256

    2. Análisis estático del código fuente: contar manualmente las operaciones
       en el bucle y multiplicar por N e iteraciones.
       En heavy_cpu: sinf (1), cosf (1), mul (2), add (1) = 5 ops/iter
       Total: N * 10000 * 5 FLOPs

    3. Usar PTX/SASS para ver instrucciones reales generadas por el compilador.
"""

import subprocess
import re
import os
import sys
from pathlib import Path

# Directorio de trabajo
WORK_DIR = Path(__file__).parent.resolve()

# Archivos a analizar
FILES = {
    "ej6_gpu": {"src": "ej6.cu", "is_cuda": True},
    "ej6_cpu": {"src": "ej6_cpu.cu", "is_cuda": False},
}

# Instrucciones FP a buscar en PTX/SASS
FP_INSTRUCTIONS_PTX = [
    r"sin\.approx",
    r"cos\.approx",
    r"mul\.f32",
    r"add\.f32",
    r"sub\.f32",
    r"div\.approx\.f32",
    r"fma\.rn\.f32",
    r"mad\.f32",
]

FP_INSTRUCTIONS_SASS = [
    r"FMUL",
    r"FADD",
    r"FFMA",
    r"FSIN",
    r"FCOS",
    r"MUFU",  # Multi-function unit (sin, cos, etc.)
]


def run_cmd(cmd, capture=True):
    """Ejecuta un comando y devuelve la salida."""
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture, text=True, cwd=WORK_DIR)
    if result.returncode != 0:
        print(f"    Error: {result.stderr}")
        return None
    return result.stdout if capture else ""


def compile_to_ptx(src_file, output_file):
    """Compila un archivo .cu a PTX."""
    cmd = ["nvcc", "-arch=sm_75", "--ptx", src_file, "-o", output_file]
    return run_cmd(cmd)


def compile_to_cubin(src_file, output_file):
    """Compila un archivo .cu a cubin para extraer SASS."""
    cmd = ["nvcc", "-arch=sm_75", "-cubin", src_file, "-o", output_file]
    return run_cmd(cmd)


def compile_binary(src_file, output_file):
    """Compila un archivo .cu a binario ejecutable."""
    cmd = ["nvcc", "-arch=sm_75", "-O2", src_file, "-o", output_file]
    return run_cmd(cmd)


def extract_sass(cubin_file):
    """Extrae el código SASS de un cubin usando cuobjdump."""
    cmd = ["cuobjdump", "-dis", cubin_file]
    return run_cmd(cmd)


def count_fp_instructions(text, patterns, func_name="heavy_cpu"):
    """Cuenta instrucciones FP en el texto dado."""
    # Buscar la función
    func_pattern = rf"(\.visible\s+\.entry\s+)?{func_name}"
    
    counts = {}
    total = 0
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count = len(matches)
        if count > 0:
            counts[pattern] = count
            total += count
    
    return counts, total


def analyze_ptx(ptx_file, func_name="heavy_cpu"):
    """Analiza un archivo PTX y cuenta instrucciones FP."""
    try:
        with open(ptx_file, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"    Archivo {ptx_file} no encontrado")
        return None, 0
    
    return count_fp_instructions(content, FP_INSTRUCTIONS_PTX, func_name)


def analyze_sass(sass_text, func_name="heavy_cpu"):
    """Analiza código SASS y cuenta instrucciones FP."""
    if not sass_text:
        return None, 0
    return count_fp_instructions(sass_text, FP_INSTRUCTIONS_SASS, func_name)


def estimate_flops_from_source():
    """Estima FLOPs basándose en análisis estático del código fuente."""
    print("\n" + "=" * 60)
    print("ESTIMACIÓN TEÓRICA DE FLOPs (análisis estático)")
    print("=" * 60)
    
    # Operaciones en el bucle interno de heavy_cpu:
    # x = sinf(x) * 1.00001f + cosf(x) * 0.99999f;
    # - sinf(x): 1 op (puede ser varias internas, pero contamos 1)
    # - cosf(x): 1 op
    # - mul (*1.00001f): 1 op
    # - mul (*0.99999f): 1 op  
    # - add (+): 1 op
    # Total: 5 operaciones FP por iteración
    
    ops_per_iter = 5
    inner_iters = 10000
    
    # ej6_cpu.cu usa N = 1000000
    n_cpu = 1_000_000
    # ej6.cu usa N = 100000000
    n_gpu = 100_000_000
    
    total_cpu = n_cpu * inner_iters * ops_per_iter
    total_gpu = n_gpu * inner_iters * ops_per_iter
    
    print(f"\nOperaciones por iteración interna: {ops_per_iter}")
    print(f"  - sinf: 1")
    print(f"  - cosf: 1")
    print(f"  - mul: 2")
    print(f"  - add: 1")
    print(f"\nIteraciones internas (j): {inner_iters:,}")
    
    print(f"\nej6_cpu.cu (N={n_cpu:,}):")
    print(f"  FLOPs totales: {total_cpu:,} ({total_cpu/1e9:.2f} GFLOPs)")
    
    print(f"\nej6.cu (N={n_gpu:,}):")
    print(f"  FLOPs totales: {total_gpu:,} ({total_gpu/1e12:.2f} TFLOPs)")


def main():
    print("=" * 60)
    print("ANÁLISIS DE INSTRUCCIONES DE PUNTO FLOTANTE")
    print("=" * 60)
    
    os.chdir(WORK_DIR)
    
    # Verificar que nvcc está disponible
    if run_cmd(["nvcc", "--version"]) is None:
        print("\nError: nvcc no está en PATH. Instala CUDA Toolkit o ejecuta en Colab.")
        print("\nMostrando estimación teórica de FLOPs:")
        estimate_flops_from_source()
        return
    
    for name, info in FILES.items():
        src = info["src"]
        is_cuda = info["is_cuda"]
        
        print(f"\n{'=' * 60}")
        print(f"Analizando: {src}")
        print("=" * 60)
        
        if not Path(WORK_DIR / src).exists():
            print(f"  Archivo {src} no encontrado, saltando...")
            continue
        
        # Compilar a PTX
        ptx_file = f"{name}.ptx"
        print(f"\n1. Compilando a PTX...")
        compile_to_ptx(src, ptx_file)
        
        if Path(WORK_DIR / ptx_file).exists():
            print(f"\n2. Analizando PTX ({ptx_file})...")
            counts, total = analyze_ptx(ptx_file)
            if counts:
                print(f"   Instrucciones FP encontradas:")
                for pattern, count in sorted(counts.items(), key=lambda x: -x[1]):
                    print(f"     {pattern}: {count}")
                print(f"   Total instrucciones FP en código: {total}")
        
        # Para código CUDA, también generar SASS
        if is_cuda:
            cubin_file = f"{name}.cubin"
            print(f"\n3. Compilando a CUBIN para SASS...")
            compile_to_cubin(src, cubin_file)
            
            if Path(WORK_DIR / cubin_file).exists():
                print(f"\n4. Extrayendo SASS...")
                sass = extract_sass(cubin_file)
                if sass:
                    counts, total = analyze_sass(sass)
                    if counts:
                        print(f"   Instrucciones FP en SASS:")
                        for pattern, count in sorted(counts.items(), key=lambda x: -x[1]):
                            print(f"     {pattern}: {count}")
                        print(f"   Total instrucciones FP en SASS: {total}")
        
        # Compilar binario
        bin_file = name
        print(f"\n5. Compilando binario ejecutable...")
        compile_binary(src, bin_file)
    
    # Mostrar estimación teórica
    estimate_flops_from_source()
    
    print("\n" + "=" * 60)
    print("MÉTODOS ALTERNATIVOS (más precisos)")
    print("=" * 60)
    print("""
1. Nsight Compute (ncu) - Profiling real de GPU:
   ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\\
                 sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\\
                 sm__sass_thread_inst_executed_op_ffma_pred_on.sum \\
       ./ej6 256

2. nvprof (legacy, para GPUs más antiguas):
   nvprof --metrics flop_count_sp ./ej6 256

3. NVIDIA Nsight Systems para timeline y análisis completo:
   nsys profile ./ej6 256
""")


if __name__ == "__main__":
    main()
