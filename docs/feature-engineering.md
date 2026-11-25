# Features adicionais e disponibilidade na previsão

## O que está integrado

Os CSV originais e os oito inputs da API não mudam. O modelo contém o
transformador FlightFeatures, que calcula as colunas adicionais em fit/transform.
Os candidatos com feature_set=engineered ativam 18 colunas adicionais, antes do
one-hot encoding. Os candidatos anteriores continuam com feature_set=baseline.

| Grupo | Features | Hipótese a avaliar nos folds |
| --- | --- | --- |
| Interações categóricas | route, airline_origin, origin_departure_hour | Captar diferenças por rota e operações de cada companhia/aeroporto/horário |
| Hora cíclica | departure_time_sin/cos, arrival_time_sin/cos | Preservar minutos e proximidade entre 23h59 e 00h01 |
| Calendário | weekday_sin/cos, year_day_sin/cos, is_weekend | Representar periodicidade semanal e anual, incluindo anos bissextos |
| Escalas e relação | log_distance, log_scheduled_duration, distance_per_scheduled_minute | Expor relações entre distância e duração prevista |
| Frequência no treino | origin_training_frequency, route_training_frequency, airline_origin_training_frequency | Representar a presença relativa de grupos na amostra de treino |

O rácio distância/duração não mede velocidade real: a duração prevista contém
outros componentes. As frequências também não medem tráfego da última hora;
são proporções na amostra de treino, com até 30 mil voos por mês, e não contagens
operacionais de todos os voos de um aeroporto.

As frequências são aprendidas apenas nas linhas de treino de cada fold, sem usar
y, ArrDelay ou DepDelay. São guardadas no próprio modelo. transform não altera
as tabelas com base nos pedidos ou dados de validação. Um grupo desconhecido
recebe frequência zero e é tratado pelo one-hot com handle_unknown=ignore.
Não há consulta de eventos, rede ou base de dados durante a previsão.

Valores de distância/duração não positivos ou ausentes continuam em falta e
recebem imputação aprendida apenas no treino. Horários em falta geram valores
cíclicos em falta; horários inválidos são rejeitados pelas mesmas regras da API.
2400 é representado por zero minutos no ciclo horário. O calendário usa FlightDate,
a data de serviço fornecida; não se infere uma data real de chegada a partir de HHMM.

## Comparação automática sem mudar o protocolo

configs/baseline.json contém os oito candidatos anteriores e três novos pares:

- engineered_linear_svc: mesmo classificador de balanced_linear_svc.
- engineered_forest_deeper: mesmo classificador de balanced_forest_deeper.
- engineered_extra_trees: mesmo classificador de balanced_extra_trees.

Só muda a representação das features nesses pares. Mantêm-se as mesmas linhas,
classes, três folds, intervalo de sete dias e seleção por macro-F1 médio.
A configuração atual também inclui CatBoost com features enriquecidas e uma
MLP com features de base. Os 13 candidatos e a referência maioritária totalizam
42 ajustes de validação, seguidos do ajuste final habitual. A referência fica
fora da seleção.
São hipóteses de melhoria; ainda não há resultados reais destas novas features.
O custo aumenta, sobretudo com as interações categóricas; mantêm-se limites de
profundidade/folhas e n_jobs=2 nas florestas. Não se convertem os inputs para uma
matriz one-hot densa.

feature_set é registado nos parâmetros de cada candidato no MLflow, nas linhas
de validação e nos metadados da seleção/modelo. A configuração completa acompanha
a execução. As métricas históricas dos oito candidatos sem estas features não
devem ser apresentadas como resultados desta alteração.

No workflow, a imagem nova executa train.py e faz a preparação/seleção habitual.
Os Parquet existentes continuam compatíveis: as features adicionais são derivadas
no modelo e não exigem novas colunas brutas. Localmente, um modelo antigo não é
reutilizado após esta alteração de código/configuração sem --retrain.

```bash
# Comparação de desenvolvimento, sem teste final nem publicação no MLflow:
python experiments/compare_models.py --sample-per-month 30000

# Fluxo principal: executar pela pipeline da equipa após revisão/PR.
# Não é necessário fazer dvc add/push: os CSV originais não mudaram.
```

Os novos testes ajustam modelos apenas com dados sintéticos. Os testes existentes
da API também podem carregar o modelo local já guardado. Não foram
executados treinos completos, publicados resultados no MLflow, alterados aliases,
substituídos modelos locais ou feitos commits/pushes.

## Médias recentes: módulo experimental separado

features/recent.py implementa recent_departure_features para janelas de 30 e 60
minutos por aeroporto. Produz média de DepDelay, fração de atrasos >=15 minutos,
contagem de atrasos válidos, contagem de eventos e indicador de cobertura por janela.
O módulo não faz parte dos candidatos configurados, da preparação dos CSV nem
do /classify atual.

O enunciado especifica atributos brutos disponíveis antes da partida e labels
disponibilizadas após cada ronda. Não fornece um feed de partidas recentes nem
timestamps de publicação de DepDelay. Calcular médias com os resultados completos
dos CSV e assumir disponibilidade imediata não reproduz esse contrato. O módulo
não inventa timestamps de disponibilidade nem um atraso de publicação fixo.

Para o integrar seria necessário acordar o instante da previsão, ter uma fonte
operacional ou um replay com disponibilidade comprovada, definir a atualização
do histórico e repetir a avaliação com a mesma política do serviço.

Interface do módulo:

```python
from flight_delay.features.recent import recent_departure_features

# events: flight_id, Origin, departed_at, available_at, DepDelay
# queries: flight_id, Origin, prediction_at
features = recent_departure_features(
    events,
    queries,
    coverage_start="2023-01-01T00:00:00Z",
    observed_through="2023-01-03T00:00:00Z",
)
```

flight_id deve identificar o mesmo voo de forma global e consistente nas duas
tabelas. A data/hora da previsão é fornecida pelo chamador, não assumida como a
hora real de partida. Todos os timestamps têm de incluir fuso/offset explícito;
as comparações são feitas em UTC. Não se pode passar FlightDate+HHMM sem resolver
datas reais, meia-noite e fusos horários.

### Regras e casos limite

- Janela [T-window, T): inclui o início e exclui eventos no instante T.
- Só entram eventos do mesmo aeroporto com available_at <= T.
- O próprio flight_id é sempre excluído, mesmo que tenha partido antecipadamente.
- Mantêm-se os atrasos negativos (partidas antecipadas); a fração atrasada usa >=15.
- Sem eventos, média/fração em falta e contagens zero, se o snapshot cobre T.
- Se há eventos com atrasos inválidos, event_count e valid_count distinguem-nos;
  valores não finitos não entram na média.
- Sem cobertura temporal para T, contagens em falta, nunca um falso zero de tráfego.
- Com início de histórico parcial, covered=0; as estatísticas dos eventos conhecidos
  podem existir, mas não são apresentadas como uma janela completa.
- coverage_start e observed_through são declarações explícitas da fonte sobre
  completude; não são inferidas da primeira/última partida. Uma fonte pode estar
  atualizada sem ter recebido qualquer partida nesse intervalo. A declaração
  de cobertura deve abranger todos os aeroportos consultados; um feed parcial
  por aeroporto não pode ser apresentado como um snapshot global completo.
- Duplicados exatos são removidos. Eventos diferentes com o mesmo flight_id são
  rejeitados; revisões exigem resolução de versões upstream antes de usar o módulo.
- Eventos finais de partida com available_at anterior à partida são rejeitados.
- Queries não dependem de outras queries do mesmo batch. Nenhum pedido de
  previsão é convertido implicitamente em evento observado.

Nos folds, uma extensão operacional teria de consultar apenas eventos disponíveis
para cada previsão. Só poderia usar eventos anteriores da validação se o cenário
simulasse previsões contínuas com atualização desse feed, sem refit do modelo.
Um snapshot de outcomes futuros não pode ser usado para inventar essa atualização.

## Testes

```bash
python -m pytest tests/test_feature_engineering.py tests/test_model_experiments.py -v
```

Os testes cobrem frequências aprendidas só no treino, independência de labels e
do batch, aeroportos desconhecidos, meia-noite, anos bissextos, valores ausentes,
serialização skops e o /classify com modelos enriquecidos. O módulo recente tem
casos de eventos futuros/recebidos tarde, exclusão do próprio voo, limites de janela,
histórico vazio/parcial/desatualizado, offsets, duplicados e atrasos inválidos.
O workflow de CI inclui estes testes; não foram alterados os jobs de deployment.

Referências:
- https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
- https://scikit-learn.org/stable/common_pitfalls.html
- Enunciado mlops_project_flight.v1.1.pdf, secções 2 e 3.
