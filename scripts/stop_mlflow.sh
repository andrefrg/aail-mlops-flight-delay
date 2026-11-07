#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PID_FILE="$PWD/mlflow.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Sem mlflow.pid. Este script so termina a instancia iniciada por start_mlflow.sh."
  exit 1
fi
pid=$(cat "$PID_FILE")
if [[ ! "$pid" =~ ^[0-9]+$ ]] || [[ "$pid" -le 1 ]]; then
  echo "PID invalido em $PID_FILE." >&2
  exit 1
fi
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f -- "$PID_FILE"
  echo "O processo ja nao esta em execucao."
  exit 0
fi

# Refuse to signal an unrelated process if an old PID has been reused.
args=$(ps -p "$pid" -o args=)
if [[ "$args" != *mlflow*server* ]]; then
  echo "O PID $pid nao corresponde a mlflow server. Nenhum processo foi terminado." >&2
  exit 1
fi

# Signal the parent so it can shut down its workers normally.
kill -TERM "$pid"
for ((i = 0; i < 30; i++)); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f -- "$PID_FILE"
    echo "Servidor MLflow terminado. Contas, permissoes e chave secreta preservadas para o proximo arranque."
    exit 0
  fi
  sleep 1
done
echo "O processo ainda existe apos 30 segundos. Consulta mlflow.log; PID mantido em mlflow.pid." >&2
exit 1
