# Auditoria inicial dos dados

O notebook `notebooks/01_data_audit.ipynb` explica os objetivos, cálculos,
resultados e limitações. A implementação reutilizável está em
`src/flight_delay/data/audit.py`, com entrada CLI em `scripts/audit_data.py`.

## Ambiente local

Python 3.11 ou superior. Se já usares Anaconda com pandas, numpy, matplotlib
e Jupyter, podes usar esse ambiente e executar os testes abaixo; não é obrigatório
criar outro ambiente. Foi validado com pandas 2.2.3/Python 3.12 e pandas
3.0.3/Python 3.14. Para criar um ambiente separado, na raiz do repositório:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python -m ipykernel install --prefix .venv --name laia-audit --display-name "Python (LAIA audit)"
python scripts/audit_data.py
jupyter lab notebooks/01_data_audit.ipynb
```

Selecionar o kernel Python (LAIA audit). Usar python3.12 em vez de python3.11
se essa for a versão instalada. O Python de sistema 3.9 não é suficiente.
As dependências têm intervalos iniciais; as versões de Python/pandas/numpy
usadas em cada auditoria ficam em summary.json.

## Inputs e outputs

Por omissão, os inputs são `./dataset/flights_2022.csv` e
`./dataset/flights_2023.csv`, relativos ao repositório, independentemente da
pasta corrente no terminal. Para outro local:

```bash
python scripts/audit_data.py --data-dir /caminho/dataset
```

No notebook pode definir-se FLIGHT_DATA_DIR antes de abrir o kernel.
Não se alteram dados originais. Os resultados completos são guardados em
`reports/generated/data_audit/`, excluído do Git. Incluem schema, hashes,
qualidade, ausentes brutos/elegíveis, classes, cobertura, subgrupos, contagens
de chaves candidatas repetidas e resumo numérico de campos selecionados.
O notebook guarda saídas agregadas para leitura mesmo sem os dados; a sua
reexecução valida tamanho/mtime das fontes e hashes do código/política.
Os SHA-256 das fontes são calculados pelo script, não novamente pelo notebook.

A política autocontida em `configs/feature_policy.csv` inclui os 109 campos e
as justificações, adaptadas da análise documental local de 18/10/2026 e das
regras do enunciado presente na raiz. Os documentos externos originais não são
dependências da execução. Estar marcado como candidato não comprova importância preditiva nem
disponibilidade efetiva no contrato da API. Os 109 campos são revistos na tabela
feature_review.csv. Nenhum modelo é ajustado nem é produzido dataset limpo.

## Desempenho e duplicados

A leitura é em blocos de 100 000 registos (ajustável com --chunksize).
Os contadores exatos de cardinalidade crescem com os valores distintos.
As chaves candidatas são contadas numa base SQLite temporária em disco; exige
espaço livre e pode demorar. --skip-duplicates omite esse controlo e assinala-o.
Não se verifica igualdade exata de linhas completas e não se apagam duplicados.
Campos sem chave válida são reportados separadamente. O hash das fontes exige
uma passagem adicional. O script pode ser interrompido; resultados só devem
ser interpretados depois de uma execução concluída com summary.json atualizado.

## Testes

```bash
python -m unittest discover -s tests -v
python scripts/audit_data.py --max-chunks 1 --chunksize 1000 --skip-duplicates
```

O segundo comando é um smoke test com resultados em data_audit_smoke; não é
uma amostra representativa. O notebook recusa resumos parciais.

As contagens de qualidade podem abranger todos os dados, mas a escolha de
features/modelo não deve usar o holdout. O notebook mostra distribuições do
alvo apenas até junho de 2023 por omissão. Não confundir mudanças descritivas
com evidência de utilidade preditiva.
