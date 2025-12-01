#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Script para compilar documentos LaTeX                                    ║
# ║  Uso: ./compile.sh [clean|open|help]                                      ║
# ║  Los archivos auxiliares se guardan en build/                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# Colores y estilos
# ─────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Símbolos Unicode
CHECK="✔"
CROSS="✖"
ARROW="➜"
WARN="⚠"
INFO="ℹ"
GEAR="⚙"
DOC="📄"
CLEAN="🧹"
ROCKET="🚀"

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAIN_FILE="main"
BUILD_DIR="build"
LOG_FILE="$BUILD_DIR/$MAIN_FILE.log"

# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

print_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}${WHITE}LaTeX Compiler${NC}  ${DIM}— Práctica 2 CUDA${NC}                           ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    local step=$1
    local total=$2
    local msg=$3
    echo -e "${BLUE}${ARROW}${NC} ${DIM}[${step}/${total}]${NC} ${msg}"
}

print_success() {
    echo -e "  ${GREEN}${CHECK}${NC} $1"
}

print_error() {
    echo -e "  ${RED}${CROSS}${NC} $1"
}

print_warning() {
    echo -e "  ${YELLOW}${WARN}${NC} $1"
}

print_info() {
    echo -e "  ${CYAN}${INFO}${NC} $1"
}

print_separator() {
    echo -e "${GRAY}────────────────────────────────────────────────────────────────${NC}"
}

# Ejecutar pdflatex con manejo de errores
run_pdflatex() {
    local pass_name=$1
    local temp_log=$(mktemp)
    
    if pdflatex -interaction=nonstopmode -file-line-error -output-directory="$BUILD_DIR" "$MAIN_FILE.tex" > "$temp_log" 2>&1; then
        print_success "$pass_name completada"
        rm -f "$temp_log"
        return 0
    else
        print_error "$pass_name falló"
        echo ""
        print_separator
        echo -e "${RED}${BOLD}Errores encontrados:${NC}"
        print_separator
        # Extraer y mostrar errores relevantes
        grep -E "^!|^l\.[0-9]+|Error:|Warning:" "$temp_log" | head -20 | while read line; do
            if [[ "$line" == "!"* ]]; then
                echo -e "  ${RED}$line${NC}"
            elif [[ "$line" == *"Error"* ]]; then
                echo -e "  ${RED}$line${NC}"
            elif [[ "$line" == *"Warning"* ]]; then
                echo -e "  ${YELLOW}$line${NC}"
            else
                echo -e "  ${GRAY}$line${NC}"
            fi
        done
        print_separator
        echo ""
        print_info "Log completo en: ${BOLD}$LOG_FILE${NC}"
        cp "$temp_log" "$LOG_FILE"
        rm -f "$temp_log"
        return 1
    fi
}

# Función para limpiar archivos auxiliares
clean() {
    print_header
    echo -e "${CLEAN} ${BOLD}Limpiando archivos de compilación...${NC}"
    echo ""
    
    if [ -d "$BUILD_DIR" ]; then
        local count=$(find "$BUILD_DIR" -type f 2>/dev/null | wc -l)
        rm -rf "$BUILD_DIR"
        print_success "Eliminada carpeta ${BOLD}build/${NC} ($count archivos)"
    else
        print_warning "La carpeta ${BOLD}build/${NC} no existe"
    fi
    
    # Limpiar PDF de la raíz si existe
    if [ -f "$MAIN_FILE.pdf" ]; then
        rm -f "$MAIN_FILE.pdf"
        print_success "Eliminado ${BOLD}$MAIN_FILE.pdf${NC}"
    fi
    
    echo ""
    print_success "${GREEN}Limpieza completada${NC}"
    echo ""
}

# Función para abrir el PDF
open_pdf() {
    if [ -f "$MAIN_FILE.pdf" ]; then
        print_info "Abriendo ${BOLD}$MAIN_FILE.pdf${NC}..."
        if command -v xdg-open &> /dev/null; then
            xdg-open "$MAIN_FILE.pdf" 2>/dev/null &
        elif command -v open &> /dev/null; then
            open "$MAIN_FILE.pdf" &
        else
            print_warning "No se encontró un visor de PDF"
        fi
    else
        print_error "No existe ${BOLD}$MAIN_FILE.pdf${NC}. Compila primero."
    fi
}

# Función principal de compilación
compile() {
    print_header
    
    local start_time=$(date +%s.%N)
    
    echo -e "${ROCKET} ${BOLD}Compilando ${WHITE}$MAIN_FILE.tex${NC}"
    echo ""
    
    # Crear carpeta build si no existe
    mkdir -p "$BUILD_DIR"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Paso 1: Primera pasada de pdflatex
    # ─────────────────────────────────────────────────────────────────────────
    print_step 1 4 "Primera pasada de pdflatex"
    if ! run_pdflatex "Primera pasada"; then
        echo -e "\n${RED}${BOLD}${CROSS} Compilación fallida${NC}\n"
        exit 1
    fi
    
    # ─────────────────────────────────────────────────────────────────────────
    # Paso 2: Bibliografía
    # ─────────────────────────────────────────────────────────────────────────
    print_step 2 4 "Procesando bibliografía"
    if [ -f "references.bib" ]; then
        cp references.bib "$BUILD_DIR/"
        cd "$BUILD_DIR"
        if bibtex "$MAIN_FILE" > /dev/null 2>&1; then
            print_success "Bibliografía procesada"
        else
            print_warning "Advertencias en bibliografía (continuando...)"
        fi
        cd "$SCRIPT_DIR"
    else
        print_info "No hay ${BOLD}references.bib${NC}, saltando..."
    fi
    
    # ─────────────────────────────────────────────────────────────────────────
    # Paso 3: Segunda pasada
    # ─────────────────────────────────────────────────────────────────────────
    print_step 3 4 "Segunda pasada de pdflatex"
    if ! run_pdflatex "Segunda pasada"; then
        echo -e "\n${RED}${BOLD}${CROSS} Compilación fallida${NC}\n"
        exit 1
    fi
    
    # ─────────────────────────────────────────────────────────────────────────
    # Paso 4: Tercera pasada
    # ─────────────────────────────────────────────────────────────────────────
    print_step 4 4 "Tercera pasada de pdflatex"
    if ! run_pdflatex "Tercera pasada"; then
        echo -e "\n${RED}${BOLD}${CROSS} Compilación fallida${NC}\n"
        exit 1
    fi
    
    # ─────────────────────────────────────────────────────────────────────────
    # Copiar PDF final
    # ─────────────────────────────────────────────────────────────────────────
    echo ""
    if [ -f "$BUILD_DIR/$MAIN_FILE.pdf" ]; then
        cp "$BUILD_DIR/$MAIN_FILE.pdf" .
        local end_time=$(date +%s.%N)
        local elapsed=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "?")
        local pages=$(pdfinfo "$MAIN_FILE.pdf" 2>/dev/null | grep Pages | awk '{print $2}' || echo "?")
        local size=$(du -h "$MAIN_FILE.pdf" | cut -f1)
        
        print_separator
        echo ""
        echo -e "${GREEN}${BOLD}${CHECK} Compilación exitosa${NC}"
        echo ""
        echo -e "  ${DOC} ${BOLD}Archivo:${NC}  $MAIN_FILE.pdf"
        echo -e "  ${GEAR} ${BOLD}Páginas:${NC}  $pages"
        echo -e "  ${GEAR} ${BOLD}Tamaño:${NC}   $size"
        if [ "$elapsed" != "?" ]; then
            printf "  ${GEAR} ${BOLD}Tiempo:${NC}   %.2fs\n" "$elapsed"
        fi
        echo ""
        print_info "Usa ${BOLD}./compile.sh open${NC} para abrir el PDF"
        echo ""
    else
        print_error "No se generó el PDF"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────
case "$1" in
    clean)
        clean
        ;;
    open)
        open_pdf
        ;;
    help|-h|--help)
        print_header
        echo -e "${BOLD}Uso:${NC} ./compile.sh [comando]"
        echo ""
        echo -e "${BOLD}Comandos:${NC}"
        echo -e "  ${GREEN}(sin args)${NC}  Compila el documento LaTeX"
        echo -e "  ${GREEN}clean${NC}       Elimina archivos de compilación"
        echo -e "  ${GREEN}open${NC}        Abre el PDF generado"
        echo -e "  ${GREEN}help${NC}        Muestra esta ayuda"
        echo ""
        ;;
    *)
        compile
        ;;
esac
