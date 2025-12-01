#!/bin/bash
# Script para compilar el documento LaTeX
# Uso: ./compile.sh [clean]
# Los archivos auxiliares se guardan en build/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAIN_FILE="main"
BUILD_DIR="build"

# Crear carpeta build si no existe
mkdir -p "$BUILD_DIR"

# Función para limpiar archivos auxiliares
clean() {
    echo "Limpiando carpeta build..."
    rm -rf "$BUILD_DIR"
    echo "Limpieza completada."
}

# Si se pasa "clean" como argumento, solo limpiar
if [ "$1" = "clean" ]; then
    clean
    exit 0
fi

echo "=== Compilando $MAIN_FILE.tex ==="

# Primera pasada de pdflatex
echo "[1/4] Primera pasada de pdflatex..."
pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$MAIN_FILE.tex" > /dev/null

# Compilar bibliografía si existe references.bib
if [ -f "references.bib" ]; then
    echo "[2/4] Compilando bibliografía..."
    # bibtex necesita el .aux en el directorio actual o especificar ruta
    cp references.bib "$BUILD_DIR/"
    cd "$BUILD_DIR"
    bibtex "$MAIN_FILE" > /dev/null 2>&1 || true
    cd "$SCRIPT_DIR"
else
    echo "[2/4] No hay references.bib, saltando bibtex..."
fi

# Segunda pasada para resolver referencias
echo "[3/4] Segunda pasada de pdflatex..."
pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$MAIN_FILE.tex" > /dev/null

# Tercera pasada para asegurar referencias cruzadas
echo "[4/4] Tercera pasada de pdflatex..."
pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$MAIN_FILE.tex" > /dev/null

# Copiar el PDF final a la raíz
cp "$BUILD_DIR/$MAIN_FILE.pdf" .

echo ""
echo "=== Compilación completada ==="
echo "Archivo generado: $MAIN_FILE.pdf"

# Abrir el PDF (opcional, descomenta la línea de tu sistema)
# xdg-open "$MAIN_FILE.pdf" 2>/dev/null &  # Linux
# open "$MAIN_FILE.pdf" &                   # macOS
