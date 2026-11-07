# Parte 1 — dados, features e modelo

Esta contribuição cobre apenas:

1. Auditoria de qualidade e ausentes.
2. Seleção explícita de features pré-partida.
3. Preparação dos voos elegíveis e três classes.
4. Folds cronológicos e baseline da aula.
5. Avaliação e entrega do pipeline guardado como model.

O classificador é `LogisticRegression(max_iter=200, C=C)`, com C em
`[0.1, 1.0, 10.0]`. O pipeline inclui as transformações dos dados de voos.

Comandos propostos para acordar com o grupo, a partir da raiz:

```bash
.venv/bin/python train.py
.venv/bin/python scripts/predict.py --input examples/classify_request.json
```

O treino guarda o modelo em `models/model.joblib` e reutiliza o artefacto
quando este corresponde ao contexto registado. `train.py --retrain` repete os
ajustes e substitui resultados locais.

- `configs/feature_policy.csv`: justificação dos 109 campos e oito inputs iniciais.
- `configs/baseline.json`: preparação, amostra, datas dos folds e C.
- `reports/tables/temporal_splits.csv`: cortes e contagens por fold.
- `notebooks/01_data_audit.ipynb`: auditoria inicial.
- `notebooks/02_baseline_results.ipynb`: exploração de treino e resultados.
- `docs/model-handoff.md`: entrega aos colegas.
- `docs/baseline-results.md`: métricas e limitações reais.

O `train.py` regista `estimator`, `C` e a accuracy do teste no MLflow, na experiência
`voo-classification`, e publica o pipeline no Registry como `voos-model`.
Configuração e comandos: [docs/README_mlflow.md](docs/README_mlflow.md).
Para executar apenas localmente, usa `python train.py --no-tracking`.
API HTTP, Docker de serviço e provisionamento continuam fora desta contribuição.
Os CSV originais ficam fora do Git em ../dataset/. DVC continua adiado.

Os CSV originais ficam fora do Git. Por defeito, o treino local lê os ficheiros
de `./dataset/`, dentro do repositório. Para usar outra pasta, executa
`python train.py --data-dir CAMINHO`.

O repositório inclui uma API HTTP, Dockerfiles de treino e serviço, e um workflow
de CI. A integração do treino e da API com o MLflow e as restantes etapas de
CI/CD ainda estão em preparação. DVC continua adiado.

## Docker Compose

O Compose arranca a API em `http://localhost:8081`. A API carrega
`models/model.joblib` do diretório local `models/`; treine o modelo antes de a API
ser considerada saudável. O serviço de treino do Compose lê, por defeito,
`../dataset/`, fora do repositório. Para usar outra pasta, define `DATASET_DIR`.

```powershell
$env:DATASET_DIR = (Resolve-Path ..\dataset).Path
docker compose --profile training run --build --rm train
docker compose up --build
```

O serviço `train` só é iniciado explicitamente com o perfil `training`.
