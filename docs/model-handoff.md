# Entrega da parte 1 à equipa

## Ficheiros

- models/model.joblib — nome base model, serialização joblib; pipeline completo.
- models/model_metadata.json — inputs, classes, C, max_iter, hash e contexto de treino.
- reports/generated/baseline/cross_validation.csv — parâmetros e métricas de validação.
- reports/generated/baseline/holdout.json — métricas por classe e matriz de confusão.
- reports/generated/baseline/holdout_subgroups.csv — resultados por mês, companhia e origem.
- reports/tables/temporal_splits.csv — datas e contagens dos folds.
- configs/baseline.json — configuração usada.

## Carregar o modelo

O pacote flight_delay deste repositório tem de estar acessível no ambiente do
consumidor; contém o transformador serializado junto com o classificador.
As versões principais usadas estão declaradas em requirements-baseline.txt.
O colega 2 integra essas bibliotecas no ambiente reproduzível da equipa.

```python
import joblib
import pandas as pd

model = joblib.load("models/model.joblib")
classes = model.predict(pd.DataFrame(records))
```

Não ajustar novos encoders/imputadores na API. Estão no pipeline e foram aprendidos
no treino. Categorias novas são suportadas. FlightDate também pode chegar como
FL_DATE. Confirmar os outros nomes BTS com o avaliador.

O exemplo examples/classify_request.json demonstra os oito atributos brutos.
O helper Python flight_delay.models.inference.classify_payload permite verificar
localmente instances -> classifications. O endpoint HTTP está em scripts/app.py
e é exposto pelo serviço flask_app do Compose.

## Limites de responsabilidade

O Compose configura a API; registry, autenticação, VMs, CI/CD e deployment
continuam fora do âmbito. Os dados e modelos gerados ficam fora do Git. O
baseline é inicial e prevê quase sempre on_time; os resultados não demonstram
utilidade operacional.

Comandos locais:

- Treino/preparação/avaliação: python train.py
- Inferência local: python scripts/predict.py --input CAMINHO_JSON

Os dados preparados e os modelos ficam fora do Git. O baseline é inicial e prevê
quase sempre on_time; os resultados não demonstram utilidade operacional.

## Dockerfile de treino — entrega da parte 1

```bash
docker build -f Dockerfile.train -t aail-train .
mkdir -p data models reports
docker run --rm \
  -v "$(cd ./dataset && pwd):/data:ro" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/reports:/app/reports" \
  aail-train
```

Os três mounts de saída preservam os Parquet, o modelo e as métricas após a
remoção do container. --retrain pode ser acrescentado para repetir o ajuste.
O relatório em Markdown também fica em reports/generated/baseline/summary.md.
Este container só prepara, treina e avalia: não expõe portas nem arranca serviços.
Para executar treino e API com Compose, consulte a secção Docker Compose do
README. O serviço de treino é opcional e utiliza o perfil `training`.
