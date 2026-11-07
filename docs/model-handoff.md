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
localmente instances -> classifications. Não implementa um endpoint HTTP nem
substitui o trabalho do colega 4.

## Limites de responsabilidade

Esta contribuição não configura servidor MLflow, registry, autenticação, VMs,
Docker de serviço, CI/CD ou deployment. O colega 2 pode registar o ficheiro model e os
parâmetros/métricas exportados no tracking da equipa. O colega 4 carrega esse
pipeline no endpoint acordado. O colega 3 integra os comandos nos workflows.

Proposta de comandos para a equipa, ainda a acordar:

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
O colega 3 decide como executar estes comandos no workflow da equipa.
