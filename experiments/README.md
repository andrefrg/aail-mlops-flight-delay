# Comparação de modelos na validação temporal

Experiências separadas do baseline da aula. O script não substitui
models/model.joblib, não consulta o teste final, não publica no MLflow e não
altera o modelo já exportado. Os candidatos são lidos da mesma configuração
usada por train.py, evitando duas listas diferentes de modelos.

A partir da raiz, com as dependências de dados/modelo instaladas:

```bash
python experiments/compare_models.py
python experiments/compare_models.py --sample-per-month 30000
python -m pytest tests/test_model_experiments.py -v
```

Requer os dados preparados em data/processed/. Para os criar, executar
python train.py --phase prepare. Dados antigos podem ser reorganizados com
python scripts/migrate_prepared_layout.py, sem recalcular os Parquet.

## Protocolo

Os candidatos são fixados antes da execução em configs/baseline.json:
três regressões logísticas da aula, regressão logística com pesos, duas
Random Forests (60 árvores/profundidade 16 e 80 árvores/profundidade 24),
Extra Trees (80 árvores/profundidade 24) e LinearSVC. As florestas usam
min_samples_leaf=30 e n_jobs=2. O SVM usa C=0.1, pesos balanced e
max_iter=5000. Junta-se uma referência maioritária que não pode vencer.
Todos usam os mesmos oito inputs e a preparação do baseline. Imputação, escala,
encoding e pesos são aprendidos só no treino de cada fold.

Mantêm-se os três folds expansivos e os sete dias de intervalo de
configs/baseline.json. A triagem usa até 5 000 voos por mês; a confirmação usa
até 30 000. A seleção determinística depende do identificador da linha, não do
alvo. Cada candidato recebe as mesmas linhas. Validação usa todos os voos
preparados dos períodos pedidos. Nenhuma linha do teste final entra no ajuste.

A classificação usa a média aritmética do macro-F1 dos três folds, excluindo a
referência maioritária da seleção. Também são apresentados accuracy, balanced
accuracy, recall moderate/severe, precision severe, tempos, avisos de convergência
e tamanho serializado. Desvios padrão entre folds não são intervalos de confiança.
O tempo de avaliação inclui leitura e transformação; não mede latência da API.

## Outputs

Cada execução cria uma pasta nova em reports/generated/model_experiments/;
--output permite indicar outra localização e recusa sobrescrever uma existente.
summary.csv resume as métricas; fold_metrics.csv contém cortes e contagens;
os JSON por fold incluem matriz de confusão, parâmetros e hash das linhas de
treino. provenance.json identifica código, commit, manifesto, fontes, versões e
protocolo. selection.json é uma classificação de desenvolvimento, sem promoção.

Os resultados pequenos para revisão ficam em reports/tables/model_experiments_*.
A comparação alargada mais recente está em expanded-results.md;
a discussão histórica está em results.md. O treino principal já lê estes
candidatos e regista as experiências no MLflow. Este script é uma comparação
local de desenvolvimento: não regista nem promove modelos. A integração
na pipeline continua dependente do merge e dos workflows da equipa. O teste histórico já foi observado em experiências
anteriores e não deve ser apresentado como nunca visto.

Referências dos estimadores:
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
