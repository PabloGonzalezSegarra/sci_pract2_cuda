#!/bin/bash
# Script para crear entorno virtual y ejecutar el análisis de FLOPs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias (si hubiera)
# pip install -r requirements.txt

# Ejecutar el script de análisis
echo "Ejecutando análisis de FLOPs..."
python analyze_flops.py

echo ""
echo "Para ejecutar manualmente:"
echo "  source venv/bin/activate"
echo "  python analyze_flops.py"
