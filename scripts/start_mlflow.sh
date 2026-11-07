#!/usr/bin/env bash
set -euo pipefail

# Keep the database, artifacts, log and PID beside these scripts.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PID_FILE="$PWD/mlflow.pid"
LOG_FILE="$PWD/mlflow.log"

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "O MLflow ja esta em execucao (PID $pid)."
    exit 1
  fi
  rm -f -- "$PID_FILE"
fi

# Override with MLFLOW_BIN=/path/to/venv/bin/mlflow if needed.
MLFLOW_BIN="${MLFLOW_BIN:-$HOME/mlflow-venv/bin/mlflow}"
if [[ ! -x "$MLFLOW_BIN" ]]; then
  MLFLOW_BIN=$(command -v mlflow || true)
fi
if [[ -z "$MLFLOW_BIN" ]]; then
  echo "MLflow nao encontrado. Ativa o ambiente virtual ou define MLFLOW_BIN." >&2
  exit 1
fi

# Use the same Python environment as the MLflow executable.
MLFLOW_PYTHON="$(dirname -- "$MLFLOW_BIN")/python"
if [[ ! -x "$MLFLOW_PYTHON" ]]; then
  echo "Python nao encontrado ao lado de $MLFLOW_BIN. Define MLFLOW_BIN com o caminho da venv." >&2
  exit 1
fi
if ! "$MLFLOW_PYTHON" -c 'import flask_wtf' >/dev/null 2>&1; then
  echo "Instala a autenticacao na venv: $MLFLOW_PYTHON -m pip install --upgrade 'mlflow[auth]'" >&2
  exit 1
fi

if ! command -v ss >/dev/null; then
  echo "Comando ss nao encontrado. Instala o pacote iproute2." >&2
  exit 1
fi
if [[ -n "$(ss -H -ltn 'sport = :8080')" ]]; then
  echo "A porta 8080 ja esta ocupada. O servidor nao foi iniciado." >&2
  exit 1
fi

# Persist the CSRF key across restarts; never include it in source control.
umask 077
SECRET_FILE="$PWD/.mlflow-secret-key"
if [[ ! -s "$SECRET_FILE" ]]; then
  "$MLFLOW_PYTHON" -c 'import secrets; print(secrets.token_hex(32))' > "$SECRET_FILE"
fi
chmod 600 "$SECRET_FILE"
export MLFLOW_FLASK_SERVER_SECRET_KEY
MLFLOW_FLASK_SERVER_SECRET_KEY=$(cat "$SECRET_FILE")
export MLFLOW_AUTH_CONFIG_PATH="$PWD/basic_auth.ini"
if [[ ! -f "$MLFLOW_AUTH_CONFIG_PATH" ]]; then
  echo "Falta basic_auth.ini. Copia esse ficheiro para a pasta dos scripts." >&2
  exit 1
fi
chmod 600 "$MLFLOW_AUTH_CONFIG_PATH"

nohup "$MLFLOW_BIN" server \
  --app-name basic-auth \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 1 \
  --allowed-hosts "10.17.0.253:8080,localhost:8080,127.0.0.1:8080" \
  --cors-allowed-origins "http://10.17.0.253:8080" \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./artifacts \
  >> "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f -- "$PID_FILE"
  echo "O MLflow terminou durante o arranque. Consulta $LOG_FILE." >&2
  tail -n 20 "$LOG_FILE" >&2
  exit 1
fi
echo "MLflow com autenticacao lancado (PID $pid): http://10.17.0.253:8080"
echo "Utilizador inicial: username. As contas e permissoes ficam em basic_auth.db."
echo "Logs: $LOG_FILE"
