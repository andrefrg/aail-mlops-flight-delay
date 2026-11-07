# Dados e baseline — apenas parte 1

## Entradas e preparação

Os CSV originais são ./dataset/flights_2022.csv e flights_2023.csv.
Os dados derivados ficam em data/processed/, com pastas train (2022), validation
(jan.–jun. 2023) e test (jul.–dez. 2023), subdivididas por mês.

Mantêm-se apenas Cancelled == 0, Diverted == 0 e ArrDelay numérico finito.
As classes são on_time (<15), moderate_delay ([15,60)) e severe_delay (>=60).
ArrDelay não entra nas features. Flags de cancelamento/desvio servem apenas
para filtrar a população. Datas inválidas são colocadas em quarentena.
Duração prevista/distância não positivas passam a ausência; a mediana é ajustada
só no treino do pipeline. As fontes originais não são modificadas.

A lista permitida tem FlightDate, Reporting_Airline, Origin, Dest, CRSDepTime,
CRSArrTime, CRSElapsedTime e Distance. Derivam-se calendário e horas previstas;
usa-se one-hot tolerante a categorias novas e escala numérica. Nenhuma coluna
de resultado entra automaticamente por estar presente na tabela.
A política documental dos restantes campos não demonstra inutilidade preditiva.

## Folds a testar

Três folds expansivos, com sete dias de intervalo antes da validação:

| Fold | Treino até, exclusivo | Validação |
| --- | --- | --- |
| 2022_Q4 | 2022-09-24 | 2022-10-01 até 2023-01-01, exclusivo |
| 2023_Q1 | 2022-12-25 | 2023-01-01 até 2023-04-01, exclusivo |
| 2023_Q2 | 2023-03-25 | 2023-04-01 até 2023-07-01, exclusivo |

O treino começa em 2022-01-01. No modelo final usa-se treino anterior a
2023-06-24 e teste entre 2023-07-01 e 2024-01-01, exclusivo.
Os folds são cortes por data, não divisão aleatória de linhas.
O CSV reports/tables/temporal_splits.csv inclui as contagens efetivas.
São decisões implementadas nesta proposta; os comandos e protocolo de integração
continuam a precisar de acordo do grupo.

## Modelo da aula

O material fornecido usa:

```python
for C in [0.1, 1.0, 10.0]:
    model = LogisticRegression(max_iter=200, C=C)
```

É esse o estimador da implementação. Não se usa SGD, class_weight balanced ou
rota no classificador. Adaptaram-se os inputs Iris para voos e a divisão aleatória
para os folds temporais do projeto. C é escolhido pela accuracy média na validação,
como métrica da aula; macro-F1, matriz de confusão e métricas por classe/subgrupo
são complementares. O teste não escolhe C.

O treino usa até 30 000 voos por mês, amostrados por prioridade determinística
independente do alvo. Validação e teste usam todos os voos preparados dos seus
períodos. Não se treinou em todos os 13 milhões de registos. Não há timestamps
históricos de publicação das labels BTS, pelo que esta avaliação é retrospetiva.
O período de teste já foi observado em experiências anteriores; não é um teste
novo nunca inspecionado.

## Comandos propostos

Num ambiente com as bibliotecas do projeto, a partir da raiz:

```bash
python train.py
python scripts/predict.py --input examples/classify_request.json
python -m unittest discover -s tests -v
```

--data-dir permite outra pasta de dados. --phase prepare, --phase train e
--phase evaluate permitem executar etapas separadas. --force-prepare reconstrói
os derivados. --retrain repete o treino e substitui os artefactos e resultados
locais; não cria backups nem faz publicação. Sem esse argumento reutiliza-se o
modelo existente quando coincide com o contexto registado.

O pipeline completo é models/model.joblib. O teste é feito nesse mesmo modelo;
não há reajuste posterior usando o teste. A parte 1 entrega também schema,
parâmetros e métricas locais. O colega 2 trata do ambiente e tracking, o 3 dos
workflows e o 4 do carregamento/validação HTTP, Docker e deployment.
