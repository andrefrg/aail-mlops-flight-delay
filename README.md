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

O modelo já treinado está em `models/model.joblib`. O treino reutiliza o artefacto
verificado; `train.py --retrain` repete os ajustes e substitui resultados locais.
Para outro ambiente, o colega responsável integra as bibliotecas declaradas em
requirements.txt e o pacote deste repositório.

- `configs/feature_policy.csv`: justificação dos 109 campos e oito inputs iniciais.
- `configs/baseline.json`: preparação, amostra, datas dos folds e C.
- `reports/tables/temporal_splits.csv`: cortes e contagens por fold.
- `notebooks/01_data_audit.ipynb`: auditoria inicial.
- `notebooks/02_baseline_results.ipynb`: exploração de treino e resultados.
- `docs/model-handoff.md`: entrega aos colegas.
- `docs/baseline-results.md`: métricas e limitações reais.

Não se implementam nesta contribuição API HTTP, Docker de serviço, CI/CD, provisionamento,
registry, autenticação ou registo no MLflow. São trabalho dos restantes colegas.
Os CSV originais ficam fora do Git em ./dataset/. DVC continua adiado.

A entrega inclui **Dockerfile.train**, apenas para preparação/treino/avaliação.
Os comandos de build e execução estão em docs/model-handoff.md. A construção e
publicação da imagem no workflow pertencem ao colega 3; o Docker da API pertence
ao colega 4. Não existe Compose nem Dockerfile de serviço nesta contribuição.
