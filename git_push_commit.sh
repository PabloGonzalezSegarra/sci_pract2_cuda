#!/bin/bash
set -euo pipefail

# Script sencillo para añadir, commitear y hacer push.
# Uso: ejecutar desde la raíz del repositorio o desde la carpeta del proyecto.

echo "Comprobando estado del repositorio..."
if [ -n "$(git status --porcelain)" ]; then
    echo "Cambios detectados. Añadiendo todos los cambios..."
    git add -A
    echo "Haciendo commit con mensaje 'commit'..."
    git commit -m "commit"
    echo "Haciendo push al remoto..."
    git push
    echo "Push completado."
else
    echo "No hay cambios para commitear. Nada que hacer."
fi
