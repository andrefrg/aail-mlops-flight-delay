# Seleção automática — resultados

Candidato selecionado: balanced_forest (RandomForestClassifier).
Critério: mean_validation_macro_f1; média aritmética dos folds, empate pela
ordem da configuração. A referência maioritária não é candidata à promoção.

Preparados 13,275,415 voos elegíveis; treino com até
30,000 por mês. Seleção pelos folds de desenvolvimento;
o teste final não escolhe o modelo. Pré-processamento ajustado só no treino.

O vencedor foi ajustado em 533,267 exemplos anteriores
a 2023-06-24 (exclusivo), sem refit posterior no teste.

## Avaliação no teste histórico

- Período: 2023-07-01 até 2024-01-01 (exclusivo).
- Voos: 3,463,444.
- Accuracy: 0.522962; maioria: 0.809812.
- Macro-F1: 0.364194; maioria: 0.298304.
- Recall severo: 48.4635%.
- Precision severa: 10.9683%.

O teste já foi observado em experiências anteriores: não é um novo teste nunca
visto. Não existem timestamps de publicação dos rótulos BTS; a avaliação é
cronológica retrospetiva. Maior macro-F1 não garante utilidade operacional.

O pipeline completo está em models/model.joblib. A configuração e os resultados
por fold permitem auditar a escolha. A publicação no MLflow identifica os
candidatos, o critério e a versão vencedora; staging/production seguem a pipeline
de testes existente. Não se compara automaticamente com execuções históricas
que usem outros dados ou protocolos de validação.
