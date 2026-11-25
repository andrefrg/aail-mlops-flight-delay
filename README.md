# Parte 1 — dados, features e modelo

Esta contribuição cobre apenas:

1. Auditoria de qualidade e ausentes.
2. Seleção explícita de features pré-partida.
3. Preparação dos voos elegíveis e três classes.
4. Folds cronológicos e baseline da aula.
5. Avaliação e entrega do pipeline guardado como model.

O treino compara 13 candidatos nos mesmos três folds temporais: regressões
logísticas, Random Forest, Extra Trees, SVM linear, CatBoost e uma MLP treinada
em PyTorch. Quatro candidatos usam features enriquecidas, calculadas
no modelo a partir dos mesmos oito inputs. Seleciona
pelo maior macro-F1 médio de validação e ajusta o vencedor para entrega. O teste
final não participa na escolha. O pipeline inclui as transformações dos voos.

Comandos propostos para acordar com o grupo, a partir da raiz:

```bash
.venv/bin/python train.py
.venv/bin/python scripts/predict.py --input examples/classify_request.json
```

O treino guarda o modelo em `models/model.joblib` e reutiliza o artefacto
quando este corresponde ao contexto registado. `train.py --retrain` repete os
ajustes e substitui resultados locais.

- `configs/feature_policy.csv`: justificação dos 109 campos e oito inputs iniciais.
- `configs/baseline.json`: preparação, amostra, folds, candidatos e conjunto de features.
- `configs/engineered_features.csv`: definição das 18 features adicionais.
- `docs/feature-engineering.md`: cálculo automático, casos limite e contexto recente experimental.
- `reports/tables/temporal_splits.csv`: cortes e contagens por fold.
- `data/processed/monthly/month=YYYY-MM/`: partições mensais; os folds são filtros por data.
- `experiments/README.md`: comparação isolada de modelos na validação temporal.
- `notebooks/01_data_audit.ipynb`: auditoria inicial.
- `notebooks/02_baseline_results.ipynb`: exploração de treino e resultados.
- `docs/model-handoff.md`: entrega aos colegas.
- `docs/automatic-model-selection.md`: regra de seleção, comandos e consulta no MLflow.
- `docs/baseline-results.md`: métricas e limitações reais.

O train.py regista uma execução principal de seleção e uma execução filha por
candidato no MLflow, na experiência voos-classification. Só o vencedor é
registado no Registry como voos-model. Parâmetros, métricas de validação/teste e
artefactos da comparação ficam disponíveis para consulta.
Configuração e comandos: [docs/README_mlflow.md](docs/README_mlflow.md).
Para executar apenas localmente, usa `python train.py --no-tracking`.
Os CSV originais ficam fora do Git. Por defeito, o treino local lê os ficheiros
de `./dataset/`, dentro do repositório. Para usar outra pasta, executa
`python train.py --data-dir CAMINHO`.

O repositório inclui uma API HTTP, Dockerfiles de treino e serviço, e um workflow
de CI. O treino regista modelos no MLflow; a integração da API com o MLflow e as
restantes etapas de CI/CD ainda estão em preparação.

## Docker Compose

O Compose arranca a API em `http://localhost:8081`. A API carrega
`models/model.joblib` do diretório local `models/`; treine o modelo antes de a API
ser considerada saudável. O serviço de treino do Compose lê, por defeito,
`./dataset/`, dentro do repositório. Para usar outra pasta, define `DATASET_DIR`.

```powershell
$env:DATASET_DIR = (Resolve-Path .\dataset).Path
docker compose --profile training run --build --rm train
docker compose up --build
```

O serviço `train` só é iniciado explicitamente com o perfil `training`.

## Treino no GitHub Actions

O workflow `2 - Continuous Delivery` arranca automaticamente após um CI
bem-sucedido na `main` ou numa PR criada a partir de uma branch deste repositório,
num runner self-hosted com o label `vm01`. O CI publica também as imagens dessas
PRs e passa ao delivery o SHA exato que foi construído. O delivery obtém os
dados pelo DVC, executa a imagem correspondente e regista o modelo no MLflow.
PRs de forks não iniciam o treino na VM. O workflow também pode ser iniciado
manualmente em Actions com `Run workflow`; nesse caso, indica o SHA completo de
um commit cuja imagem de treino já tenha sido publicada no GHCR.

O runner precisa de Docker, `python3` com suporte a `venv`, acesso por SSH ao
remote DVC e acesso ao servidor MLflow. No Debian, o suporte a ambientes virtuais
pode exigir o pacote `python3-venv`. Configura as variáveis `REGISTRY`, `MLFLOW_TRACKING_URI`,
`MLFLOW_EXPERIMENT_NAME` e `MLFLOW_MODEL_NAME` no GitHub; se o MLflow exigir
autenticação, configura também os secrets `MLFLOW_USERNAME` e `MLFLOW_PASSWORD`.
O acesso ao DVC tem de funcionar sem interação no runner. Os CSVs ocupam cerca
de 6 GB, além do espaço necessário para a cache e o treino.

Os testes E2E e a promoção ainda estão por integrar e validar.
