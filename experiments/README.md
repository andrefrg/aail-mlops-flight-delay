# Comparação de modelos na validação temporal

Experiências separadas do baseline da aula. O script não substitui
models/model.joblib, não consulta o teste final, não publica no MLflow e não
altera a configuração usada por train.py.

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

Os candidatos são fixados antes da execução: referência maioritária,
LogisticRegression(C=0.1, max_iter=200), a mesma regressão com class_weight=balanced
e RandomForestClassifier com 60 árvores, profundidade máxima 16, mínimo de 30
exemplos por folha, balanced_subsample, seed=42 e duas tarefas paralelas.
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
A discussão está em results.md. Uma integração futura do candidato aprovado no
treino principal deve preservar o contrato da API, ser avaliada quanto a recursos
na VM e registar as experiências no MLflow. O merge desta contribuição não muda
o classificador de produção. O teste histórico já foi observado em experiências
anteriores e não deve ser apresentado como nunca visto.

Referências dos estimadores:
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
