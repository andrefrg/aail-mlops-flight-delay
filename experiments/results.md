# Comparação inicial na validação temporal

A comparação usa os oito inputs e os três folds do baseline, com sete dias de
intervalo. Os quatro candidatos e os seus parâmetros foram fixados antes da
execução. A triagem original usou até 5 000 voos/mês; a confirmação e a reexecução
documentada aqui usam a amostra completa de até 30 000/mês. Não há promoção de
modelo ou avaliação do teste final neste script.

## Resultados da confirmação

| Modelo | Macro-F1 médio | Accuracy média | Recall severo médio | Precision severa média |
| --- | ---: | ---: | ---: | ---: |
| Logística da aula | 0.2932 | 78.36% | 0.01% | 6.98% |
| Logística balanced | 0.3498 | 47.24% | 43.88% | 10.53% |
| Random Forest balanced | 0.3519 | 48.06% | 47.85% | 10.65% |
| Referência maioritária | 0.2929 | 78.37% | 0.00% | 0.00% |

Médias aritméticas entre os três folds, não métricas do teste final. Foram usados
263 271, 353 859 e 443 254 exemplos no treino dos folds. A validação usa todos os
4 910 865 voos preparados de outubro de 2022 a junho de 2023. Os candidatos usam
exatamente as mesmas linhas em cada fold; os cortes são por data, não por pasta.

A floresta fica ligeiramente à frente em macro-F1 médio, mas a vantagem sobre a
logística balanced é pequena e depende sobretudo do terceiro fold. Desvios padrão
entre folds não demonstram significância estatística nem são intervalos de confiança.
Os pesos aumentam recall das classes raras à custa de accuracy e falsos positivos.
A precision severa de cerca de 10,6% limita a utilidade operacional. Não se definiu
ainda um critério de aceitação operacional com a equipa.

Os tempos e tamanhos estão nos CSV por fold. São medições locais, não consumo de
recursos da VM nem latência da API. O tamanho refere-se ao pipeline ajustado em
cada fold, não a um modelo final de produção. Não ocorreram avisos de convergência.
O teste de jul.–dez. 2023 já foi observado em experiências anteriores; não deve
ser apresentado como um teste nunca visto, embora não seja usado nesta comparação.

## Reprodução e rastreabilidade

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python experiments/compare_models.py --sample-per-month 30000
```

- reports/tables/model_experiments_validation.csv: resumo da confirmação.
- reports/tables/model_experiments_folds.csv: cortes, contagens e métricas por fold.
- validation_provenance.json: código, commit, fontes, manifesto, versões e protocolo.
- README.md: interface, candidatos e limites da experiência.

O código foi reposto e esta confirmação foi executada novamente antes do PR.
Os resultados das experiências permanecem separados do treino principal. O merge
não troca o classificador de produção nem publica estes candidatos no MLflow.
A integração do candidato aprovado e a medição de recursos na VM são etapas futuras.

Commit executado: 7ae90481c0ae92761edf951fc39a5c8b32a62d54.
Versões: numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, pyarrow 25.0.1, joblib 1.6.0.
Maior diferença absoluta nas métricas agregadas face à confirmação original: 0.
