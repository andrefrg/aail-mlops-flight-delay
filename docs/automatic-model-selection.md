# Seleção automática e consulta no MLflow

## O que significa melhor

O critério é explícito em configs/baseline.json: mean_validation_macro_f1.
Comparam-se as três regressões da aula (C=0.1, 1.0, 10.0), regressão com pesos
(C=0.1) e uma Random Forest com 60 árvores, profundidade máxima 16, mínimo de
30 exemplos por folha e duas tarefas paralelas. A comparação foi alargada com
Random Forest (80 árvores, profundidade 24), Extra Trees (80 árvores,
profundidade 24, class_weight=balanced) e LinearSVC (C=0.1, balanced,
max_iter=5000). As duas novas florestas mantêm min_samples_leaf=30 e n_jobs=2.
A comparação inclui agora três versões com feature_set=engineered: SVM linear,
floresta mais profunda e Extra Trees. O CatBoost usa as features enriquecidas,
categorias nativas e pesos por classe. A MLP usa as features de base, one-hot,
entropia cruzada ponderada e duas camadas densas de 64 e 32 neurónios. Treina
em PyTorch na CPU e guarda os pesos em arrays NumPy para a inferência.
Há 13 candidatos. Os oito inputs brutos e os folds mantêm-se. As versões
enriquecidas derivam as 18 colunas descritas em feature-engineering.md. A
referência maioritária acompanha a avaliação e fica fora da seleção.

Os candidatos recebem as mesmas linhas e os mesmos oito inputs em cada um dos
três folds expansivos, com sete dias de intervalo. Transformações e pesos são
aprendidos só no treino. Todos têm de apresentar uma métrica finita em cada fold;
comparações incompletas falham. A média tem o mesmo peso para cada fold. Um
empate exato é resolvido pela ordem da configuração, de forma determinística.

Depois da seleção ajusta-se um novo pipeline vencedor no histórico de
desenvolvimento permitido. Só depois se avalia o teste histórico. Não se escolhe
ou troca o vencedor com base no teste. Não há refit posterior usando esse teste.
O teste de jul.–dez. 2023 já foi observado antes e não é um período nunca visto.

Melhor significa melhor macro-F1 médio entre estes candidatos, neste protocolo.
Não significa melhor modelo possível, maior accuracy, ausência de falsos positivos
ou comparação automática com toda a história do Registry. O compromisso entre
precision, recall, estabilidade e custo continua a precisar de análise da equipa.

## Executar

Com as dependências do projeto e a partir da raiz:

```bash
# Preparação, comparação, ajuste final, teste e registo no MLflow.
python train.py --retrain

# O mesmo fluxo sem contactar o MLflow.
python train.py --no-tracking --retrain

# Registar artefactos existentes verificados, sem repetir o treino.
python train.py --phase register
```

A leitura predefinida é dataset/, com Parquet em data/processed/monthly/.
Se os dados já estão preparados e a assinatura da preparação coincide, não é
necessário reler os CSV. Os candidatos e a métrica não alteram essa assinatura.
Sem --retrain pode reutilizar-se um modelo compatível; no MLflow essa execução
fica marcada como reuse_existing, em vez de ser apresentada como treino novo.

O cliente lê MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME e as
credenciais MLFLOW_TRACKING_USERNAME/MLFLOW_TRACKING_PASSWORD do ambiente ou .env.
As definições atuais são http://10.17.0.234:8080, voos-classification e voos-model.
As credenciais permanecem fora do Git. O acesso exige ligação à rede/VPN do DEI.

## O que aparece no MLflow

Abrir http://10.17.0.234:8080 e autenticar com a conta da equipa.

1. Em Experiments, abrir voos-classification.
2. Abrir a execução selection-train mais recente. O parâmetro selected_variant
   identifica o vencedor e selection_metric documenta o critério.
3. Expandir as execuções filhas para ver os 13 candidatos e a referência.
   Cada candidato tem parâmetros, cv_macro_f1_mean/std, outras métricas médias
   e métricas de validação por fold. O tag selected=true identifica o vencedor.
4. Selecionar os candidatos dessa mesma execução e usar Compare. Ordenar por
   cv_macro_f1_mean em ordem decrescente e verificar também precision/recall.
5. Em Artifacts, consultar validation.json de cada candidato; a execução principal
   guarda configuração, manifesto, seleção, CSV de comparação e relatórios de teste.
6. Em Models, abrir voos-model para consultar a versão registada do vencedor,
   os tags estimator e selected_variant e a ligação à execução que a produziu.
   Os restantes candidatos têm runs de comparação no MLflow.

Os tags role=candidate, role=reference e role=selection distinguem os papéis.
Um filtro útil é tags.role = 'candidate'. Comparar candidatos da mesma execução;
se se compararem execuções diferentes, confirmar dados, folds, amostra e critério.
As métricas cv_* são de validação. test_* são do teste final; os nomes históricos
accuracy/macro_f1 na execução principal também representam o teste.

## Entrega à pipeline

models/model.joblib continua a conter preparação e classificador juntos; o
consumidor não precisa de escolher uma classe sklearn nem ajustar encoders.
model_metadata.json identifica o estimador, candidato e critério. C e max_iter
identificam também o SVM linear; ficam vazios quando não se aplicam ao vencedor.

O treino cria uma versão no Registry. Quando o workflow fornece COMMIT_SHA,
o alias desse commit aponta para a versão escolhida, tal como antes. Os testes
E2E e a promoção para staging/production continuam no workflow da equipa.
O script de treino não move staging/production nem escolhe uma versão apenas
por ser a mais recente. A promoção existente não é uma comparação de métricas
entre execuções históricas.

Random Forest usa a mesma serialização skops que o pipeline atual. A lista de
tipos permitidos inclui o armazenamento interno das árvores sklearn apenas para
modelos RandomForestClassifier e ExtraTreesClassifier produzidos pelo treino; não se autoriza
automaticamente qualquer tipo encontrado num ficheiro recebido.

Referências:
- https://mlflow.org/docs/latest/ml/tracking/tracking-api/
- https://mlflow.org/docs/latest/ml/model-registry/

## Validação realizada em 22 de novembro de 2026

Passaram 54 testes locais e 15 subtests. Um teste integrado com MLflow local
registou e voltou a carregar uma Random Forest em skops. A execução real nos
três folds selecionou balanced_forest (macro-F1 médio de validação 0,351914).
No teste histórico, macro-F1 foi 0,364194 e accuracy 0,522962; essas métricas
não participaram na escolha.

No servidor da equipa, a execução 5fe9047ec5d3497690593740b9f6ccc5 ficou FINISHED,
com cinco runs de candidatos e uma de referência. O vencedor foi registado como
voos-model, versão 6. O modelo descarregado dessa versão produziu as mesmas
previsões que o pipeline local no exemplo de dois voos. Os aliases existentes
mantiveram-se intactos. Isto valida o registo e carregamento, não constitui uma
promoção de produção nem comprova os workflows de deployment nas VMs.
