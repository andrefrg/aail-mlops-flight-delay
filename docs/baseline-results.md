# Parte 1 — resultados do baseline

## Modelo e dados

Estimador da aula: LogisticRegression(max_iter=200, C=C), C em [0.1, 1.0, 10.0].
C escolhido pela accuracy média em três folds temporais: 0.1.
Os outros parâmetros do estimador são os valores por omissão da biblioteca.

Preparados 13,275,415 voos elegíveis. Durações previstas
inválidas convertidas em ausência: 5.
Treino com até 30,000 voos por mês; validação e teste usam
todos os voos preparados dos respetivos períodos. Os oito inputs pré-partida
são selecionados explicitamente. A imputação, escala e encoding são aprendidos
só no treino de cada fold. Há 7 dias de intervalo antes da avaliação.

O modelo final foi ajustado em 533,267 voos anteriores
a 2023-06-24 (exclusivo), com 115
iterações e 0 avisos de convergência.
Não foi reajustado no período de teste.

## Teste: julho a dezembro de 2023

- Voos: 3,463,444.
- Accuracy: 0.809781; referência maioritária: 0.809812.
- Macro-F1: 0.298480; referência: 0.298304.
- Recall severe_delay: 0.003664%.

O modelo prevê quase sempre on_time e praticamente não deteta a classe severa.
A accuracy elevada reflete sobretudo o desbalanceamento. Não se demonstrou
utilidade operacional. O teste já tinha sido observado em experiências anteriores;
C foi selecionado apenas nas janelas de validação. Não se afirma que seja um teste
novo nunca inspecionado. Não existem timestamps de publicação dos rótulos BTS:
a avaliação é cronológica retrospetiva, sem simular esse calendário.

## Entrega aos colegas

- models/model.joblib: pipeline completo, nome base model.
- models/model_metadata.json: esquema, classes, parâmetros, hashes e proveniência.
- reports/tables/: tabelas pequenas para o relatório.
- reports/generated/baseline/: avaliação por classe, matriz de confusão e subgrupos.
- docs/model-handoff.md: interface e comandos propostos para a equipa.

Resultados produzidos pela execução de preparação, treino e avaliação desta versão.

Esta parte inclui um Dockerfile de treino, mas não implementa servidor HTTP, Docker de serviço, workflows GitHub Actions,
autenticação, VMs, registry ou publicação. Essas responsabilidades pertencem aos
colegas 2, 3 e 4. Os resultados locais estão prontos para o colega 2 registar no
sistema de tracking da equipa. Os comandos de treino/inferência são uma proposta
a acordar em conjunto; não são apresentados como decisão já aprovada pelo grupo.
