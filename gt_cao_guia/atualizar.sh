#!/usr/bin/env bash
# Atualiza os scripts do GT no laboratorio, SEM baixar arquivo solto por link.
#
# Baixar raw.githubusercontent com wget/navegador ja gravou uma pagina de erro 503 de um
# proxy Varnish por cima do realsense_gt.py: o HTTP devolve 200 com corpo HTML e o arquivo
# e salvo como se fosse o script. O git confere o conteudo por SHA e falha alto em vez de
# gravar lixo - por isso aqui e clone/pull, nunca download direto.
#
#   ./atualizar.sh                  usa o destino padrao
#   ./atualizar.sh /outro/destino   usa outro
set -euo pipefail

REPO="https://github.com/mauro-mendes/rpi5-realsense.git"
ESPELHO="${HOME}/.cache/rpi5-realsense"
DESTINO="${1:-${HOME}/cuscobot_ws/src/utils_package/scripts}"
ARQUIVOS=(realsense_gt.py cenario.py cenario.json)

[ -d "$DESTINO" ] || { echo "[ERRO] destino nao existe: $DESTINO"; exit 1; }

if [ -d "$ESPELHO/.git" ]; then
  echo "[git] atualizando $ESPELHO"
  git -C "$ESPELHO" fetch --quiet origin main
  git -C "$ESPELHO" reset --quiet --hard origin/main
else
  echo "[git] clonando em $ESPELHO"
  git clone --quiet --depth 20 "$REPO" "$ESPELHO"
fi
echo "[git] commit $(git -C "$ESPELHO" log --oneline -1)"

ORIGEM="$ESPELHO/gt_cao_guia"
for f in "${ARQUIVOS[@]}"; do
  src="$ORIGEM/$f"
  [ -f "$src" ] || { echo "[ERRO] nao veio no repo: $f"; exit 1; }
  # sanidade: uma pagina de erro comeca com '<' e tem poucos bytes
  if head -c 1 "$src" | grep -q '<'; then
    echo "[ERRO] $f comeca com '<' - isso e HTML, nao codigo. Abortado."; exit 1
  fi
  case "$f" in
    *.py)   python3 -c "import ast,io,sys; ast.parse(io.open(sys.argv[1],encoding='utf-8').read())" "$src" \
              || { echo "[ERRO] $f nao compila. Abortado."; exit 1; } ;;
    *.json) python3 -c "import json,sys; json.load(open(sys.argv[1],encoding='utf-8'))" "$src" \
              || { echo "[ERRO] $f nao e JSON valido. Abortado."; exit 1; } ;;
  esac
  [ -f "$DESTINO/$f" ] && cp -f "$DESTINO/$f" "$DESTINO/$f.bak"
  cp -f "$src" "$DESTINO/$f"
  printf "  OK  %-18s %7d bytes -> %s\n" "$f" "$(wc -c < "$src")" "$DESTINO"
done

echo
echo "[versao] confira que o sha mudou:"
python3 "$DESTINO/realsense_gt.py" --check 2>&1 | grep -E "^\[versao\]|^sha:" || true
