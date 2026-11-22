# Comparação alargada de modelos — 22 de novembro de 2026

Foram comparados os oito candidatos de configs/baseline.json e uma referência
maioritária. Mantiveram-se os oito inputs, a amostra determinística de até
30 000 voos por mês e os três folds temporais expansivos com sete dias de
intervalo. A validação usa 4 910 865 voos; o treino usa 263 271, 353 859
e 443 254 voos nos respetivos folds. Não foi lido o teste de julho–dezembro
de 2023. Os CSV originais e o modelo exportado anteriormente não foram alterados.

## Resultados de validação

As métricas abaixo são médias aritméticas dos três folds, não métricas de teste.
O desvio padrão entre folds não constitui um intervalo de confiança.

| Candidato | Macro-F1 médio | Desvio padrão | Accuracy | Recall severo | Precision severa |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_linear_svc | 0.373248 | 0.014036 | 0.7468 | 0.1151 | 0.1689 |
| balanced_forest_deeper | 0.357409 | 0.017832 | 0.4926 | 0.4626 | 0.1083 |
| balanced_extra_trees | 0.353319 | 0.018245 | 0.4842 | 0.4844 | 0.1062 |
| balanced_forest | 0.351914 | 0.016988 | 0.4806 | 0.4785 | 0.1065 |
| balanced_logistic | 0.349736 | 0.011281 | 0.4723 | 0.4393 | 0.1053 |
| logistic_C10.0 | 0.293820 | 0.002963 | 0.7835 | 0.0004 | 0.2355 |
| logistic_C1.0 | 0.293591 | 0.002884 | 0.7835 | 0.0003 | 0.1589 |
| logistic_C0.1 | 0.293166 | 0.002409 | 0.7836 | 0.0001 | 0.0698 |
| majority | 0.292904 | 0.002221 | 0.7837 | 0.0000 | 0.0000 |

LinearSVC balanced foi o melhor em macro-F1 nos três folds e, por isso, ganha
pelo critério atual. A média aumentou de 0,351914 para 0,373248: +0,021334
(2,13 pontos percentuais). Não foi realizado um teste de significância.

Existe um compromisso importante: o recall severo médio diminuiu de 0,4785
na floresta anterior para 0,1151 no SVM. O recall moderado do SVM também é
baixo (0,0815). A accuracy mais alta não significa melhor deteção de atrasos.
A Random Forest mais profunda melhorou macro-F1 face à floresta anterior
em todos os folds, mantendo recall severo médio de 0,4626. Extra Trees obteve
0,353319 de macro-F1 e recall severo de 0,4844. Estas alternativas devem ser
discutidas em função dos objetivos operacionais, sem selecionar pelo teste final.

## Custo e limites

Não ocorreram avisos de convergência. As florestas usam duas tarefas paralelas;
a profundidade e o mínimo de exemplos por folha limitam o tamanho. Os tempos
e tamanhos em CSV foram medidos no computador local; não medem recursos ou
latência na VM. O SVM demorou em média 23,5 s por ajuste, a floresta anterior
8,5 s e a mais profunda 13,9 s. São tempos de ajuste, excluindo avaliação.

O script guarda resultados de desenvolvimento; não ajusta um vencedor final,
não substitui models/model.joblib e não promove aliases no Registry.
No treino principal, train.py compara agora estes oito candidatos e seleciona
o maior macro-F1 médio automaticamente. Após integração, essa execução precisa
de ser validada no ambiente Docker da equipa.

## Reprodução e evidência

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python experiments/compare_models.py --sample-per-month 30000
```

- reports/tables/expanded_models_validation.csv: resumo dos nove modelos, incluindo a referência.
- reports/tables/expanded_models_folds.csv: 27 avaliações, contagens e tempos.
- reports/tables/expanded_models_provenance.json: configuração, fontes, manifesto e versões.
- Resultados completos locais: reports/generated/model_experiments/expanded_candidates_20261122/.

Esta execução usou código local ainda não commitado na branch expand-model-candidates.
O manifesto e o hash do código executado estão na proveniência. Versões locais: numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, scipy 1.18.0, pyarrow 23.0.1, joblib 1.5.3.
O ambiente CI/Docker usa o scikit-learn 1.7.2; estes resultados locais usam
a versão indicada acima. A comparação foi feita com a mesma versão para todos
os candidatos, mas o resultado da pipeline deve ser confirmado no seu ambiente.

## MLflow

A comparação foi registada em voos-classification, na execução
[model-comparison-expanded](http://10.17.0.234:8080/#/experiments/1/runs/562c24d31ecf4afd9041a61810743055).
A execução principal e as nove filhas terminaram com estado FINISHED: oito
candidatos e uma referência. O tag selected=true identifica o SVM. Foram
publicados parâmetros, métricas e relatórios; não foram registadas novas versões
no Model Registry nem alterados aliases de staging/production. Este registo foi
feito separadamente após a execução local; compare_models.py continua sem
publicar no MLflow por predefinição.
