# Servidor MLflow no Debian 12

## Integração com o treino

O `train.py` liga-se por omissão a `http://10.17.0.253:8080`, usa a experiência
`voo-classification` e publica o pipeline completo no Model Registry como `voos-model`.
O Registry usa o mesmo servidor do tracking.

No PowerShell, a partir da raiz do repositório:

```powershell
python -m pip install -r requirements.txt
# Preenche primeiro o utilizador e a palavra-passe no .env da raiz.
python train.py --retrain
```

O `train.py` carrega automaticamente o `.env` da raiz do repositório, mesmo quando
é executado a partir de outra pasta. As variáveis já definidas no ambiente têm
prioridade sobre o ficheiro. Preenche `MLFLOW_TRACKING_USERNAME` e
`MLFLOW_TRACKING_PASSWORD` com as credenciais da conta existente no servidor.
O `.env` está excluído do Git. Em Docker, passa as variáveis com `--env-file .env`.

Cada execução regista os parâmetros `estimator` e `C` do modelo escolhido.
As métricas enviadas para o MLflow são:

| Métrica | Origem |
| --- | --- |
| `accuracy` | Accuracy do teste final |
| `macro_f1` | F1 médio das três classes no teste, com pesos iguais |
| `balanced_accuracy` | Recall médio das três classes no teste |
| `recall_severe_delay` | Recall de atrasos severos no teste |
| `recall_moderate_delay` | Recall de atrasos moderados no teste |
| `precision_severe_delay` | Precisão dos atrasos severos no teste |
| `cv_accuracy_mean` | Média da accuracy nos folds de validação da variante escolhida |
| `cv_accuracy_std` | Desvio-padrão dessas accuracies, com `ddof=0` |

As métricas de teste são lidas da avaliação final; as de validação vêm de
`reports/generated/baseline/cross_validation.csv`, filtrado pela variante escolhida.
Não se misturam os resultados dos outros valores de C ou da referência maioritária.
Os valores são frações entre 0 e 1. O modelo inclui o código de `flight_delay`
necessário às suas transformações. Não se usa autologging.

- `--phase prepare`: prepara dados sem contactar o MLflow.
- `--phase train`: regista os parâmetros, as duas métricas de validação e o modelo,
  sem métricas de teste.
- `--phase evaluate`: regista os parâmetros, as oito métricas e o modelo já existente.
- `--phase register`: publica o modelo e as oito métricas já guardados, sem preparar dados,
  treinar ou avaliar novamente. Verifica os hashes do modelo, configuração e manifesto
  e identifica na run o código do treino original. Útil para repetir uma publicação
  que falhou depois de o treino terminar.
- Sem `--phase`: executa preparação, treino e avaliação numa única run.
- `--no-tracking`: mantém a execução local, sem ligação ao servidor.

O comportamento de reutilização mantém-se: se o modelo e a avaliação existentes
forem válidos, são reutilizados e publicados numa nova run/versão, sem novo ajuste.
Usa `--retrain` para treinar novamente. Como o projeto verifica um hash do código,
esta alteração exige um novo treino se existirem artefactos de uma versão anterior.
Uma falha de ligação ou autenticação é propagada; não se apresenta o registo como concluído.
Não se alteram automaticamente os aliases `staging` ou `production`.

O pipeline é publicado em formato `skops`, autorizando explicitamente apenas os tipos
adicionais `flight_delay.features.build.FlightFeatures` e `numpy.dtype`.

Esta pasta contém scripts para iniciar e terminar o servidor MLflow numa VM Debian 12. O servidor disponibiliza a interface web e a API para registar experiências, parâmetros, métricas e artefactos enviados pelas máquinas de treino.

## Configuração atual

| Opção                      | Valor                                   |
| ---------------------------- | --------------------------------------- |
| Endereço de acesso          | `http://10.17.0.253:8080`             |
| Interface de escuta          | `0.0.0.0` (todas as interfaces da VM) |
| Porta                        | `8080/TCP`                            |
| Workers                      | `1`                                   |
| Ambiente virtual predefinido | `~/mlflow-venv`                       |
| Metadados                    | SQLite, em `mlflow.db`                |
| Artefactos                   | Diretório `artifacts/`               |

Os scripts trabalham sempre na pasta onde estão guardados. Por isso, a base de dados, os artefactos, os logs e o ficheiro de PID ficam ao lado dos scripts, independentemente da pasta a partir da qual os executas.

## 1. Preparar a VM

Executar na VM, através de SSH:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip iproute2 procps

python3 -m venv ~/mlflow-venv
~/mlflow-venv/bin/python -m pip install --upgrade pip
~/mlflow-venv/bin/python -m pip install "mlflow[auth]"

mkdir -p ~/mlflow
```

Confirma a instalação e as opções disponíveis na versão instalada:

```bash
~/mlflow-venv/bin/mlflow --version
~/mlflow-venv/bin/mlflow server --help
```

Para reproduzir a instalação no futuro, regista as versões instaladas:

```bash
~/mlflow-venv/bin/python -m pip freeze > ~/mlflow/requirements-installed.txt
```

## 2. Transferir os ficheiros por SFTP

No PowerShell do computador local:

```powershell
sftp utilizador@10.17.0.253
```

Na sessão SFTP:

```text
lcd D:/LAIA/projeto/mlflow
cd /home/admin
put start_mlflow.sh
put stop_mlflow.sh
put basic_auth.ini
put README.md
bye
```

Substitui `utilizador` pelo utilizador real da VM. Se já tens uma instalação com `mlflow.db` e `artifacts/`, coloca os scripts nessa pasta para continuar a usar os mesmos dados.

## 3. Iniciar o servidor

Na VM:

```bash
bash start_mlflow.sh
```

O servidor arranca sempre com autenticação ativa. No primeiro arranque, cria o administrador `username` com a palavra-passe `passwordpassword`, configurados em `basic_auth.ini`. Não pede credenciais no terminal. Nos reinícios, reutiliza as contas e permissões guardadas em `basic_auth.db`.

O script gera automaticamente uma chave secreta em `.mlflow-secret-key` e reutiliza-a nos arranques seguintes.

O script procura primeiro `~/mlflow-venv/bin/mlflow`. Executar esse ficheiro usa diretamente o Python e as dependências da venv, sem ser necessário ativá-la. Se esse executável não existir, procura `mlflow` no `PATH`.

Para indicar outra venv:

```bash
MLFLOW_BIN=/caminho/da/venv/bin/mlflow bash start_mlflow.sh
```

O script verifica se existe um processo com o PID guardado e se a porta 8080 está ocupada antes de arrancar. Inicia o MLflow com `nohup`, guarda o PID principal em `mlflow.pid` e acrescenta a saída ao ficheiro `mlflow.log`.

Podes fechar a sessão SSH depois do arranque. O processo continua a executar, mas **não arranca automaticamente após reiniciar a VM nem é reiniciado automaticamente se falhar**. Para esse comportamento, será necessário configurar um serviço `systemd`.

## 4. Aceder ao MLflow

No browser do teu computador, abre:

```text
http://10.17.0.253:8080
```

O computador precisa de acesso à rede da VM, diretamente ou através de VPN, e a infraestrutura e a firewall precisam de permitir a porta TCP 8080.

`0.0.0.0` é o endereço de escuta do servidor. No browser e nos clientes usa-se o IP ou o nome DNS real da VM.

Num programa Python na máquina de treino, com o pacote `mlflow` instalado:

```python
import mlflow

mlflow.set_tracking_uri("http://10.17.0.253:8080")
mlflow.set_experiment("teste-ligacao")

# Define primeiro as credenciais no ambiente (ver a secção de autenticação).
with mlflow.start_run():
    mlflow.log_param("origem", "cliente-remoto")
    mlflow.log_metric("valor_teste", 1.0)
```

## 5. Consultar os logs e o estado

```bash
cd ~/mlflow

# Acompanhar os logs; Ctrl+C termina apenas esta visualização.
tail -f mlflow.log
```

Outras verificações:

```bash
# Ver o processo principal guardado pelo script.
ps -p "$(cat mlflow.pid)" -o pid,ppid,etime,rss,args

# Ver se a porta está em escuta.
ss -lptn 'sport = :8080'

# Verificar a resposta HTTP local, se curl estiver instalado.
curl -f http://127.0.0.1:8080/health
```

A mensagem de arranque do script indica que o processo ainda existia após três segundos; não garante que a API já esteja pronta. Consulta os logs ou o endpoint `/health` para confirmar.

## 6. Terminar ou reiniciar

Usa o mesmo utilizador que iniciou o servidor:

```bash
bash stop_mlflow.sh
```

O script lê `mlflow.pid`, verifica se o comando do processo corresponde a `mlflow server` e envia `SIGTERM` ao processo principal. Aguarda até 30 segundos e remove o ficheiro de PID quando o processo termina. Se continuar ativo, mantém o PID e pede para consultar os logs.

Para reiniciar, executa o arranque apenas depois de a paragem terminar com sucesso:

```bash
bash stop_mlflow.sh && bash start_mlflow.sh
```

Terminar apenas um worker pode fazer com que o processo principal crie outro worker com um PID diferente. Usa o script de paragem para terminar o processo principal.

O script só gere a instância cujo PID foi guardado pelo script de arranque. Uma instância antiga iniciada manualmente precisa de ser parada antes de usar estes scripts.

## Ficheiros e persistência

| Ficheiro ou pasta   | Finalidade                                                  |
| ------------------- | ----------------------------------------------------------- |
| `start_mlflow.sh` | Iniciar o servidor em segundo plano                         |
| `stop_mlflow.sh`  | Terminar a instância iniciada pelo script                  |
| `mlflow.db`       | Base de dados de experiências, execuções e metadados     |
| `artifacts/`      | Ficheiros enviados para o armazenamento local de artefactos |
| `mlflow.log`      | Saída e erros do servidor; cresce a cada execução        |
| `mlflow.pid`      | PID principal da instância gerida pelos scripts            |

Para uma cópia de segurança simples e consistente, para o servidor e copia `mlflow.db`, `artifacts/`, `basic_auth.db` e `.mlflow-secret-key` em conjunto. Não apagues esses dados ao atualizar os scripts. Configura rotação de logs se o servidor ficar ligado durante longos períodos.

## Alterar a configuração

Edita as opções em `start_mlflow.sh` com o servidor parado:

- `--port`: porta de acesso ao servidor. Se a alterares, atualiza também a verificação `ss`, os valores de `--allowed-hosts` e a mensagem com o URL no script.
- `--allowed-hosts`: nomes e endereços aceites no cabeçalho HTTP `Host`. Atualiza esta lista se mudares o IP ou passares a usar DNS. Esta opção não é uma lista de IPs de clientes autorizados.
- `--cors-allowed-origins`: origens aceites nos pedidos do browser. O script autoriza
  `http://10.17.0.253:8080`. Atualiza também este valor se mudares o endereço, porta
  ou protocolo de acesso à interface.
- `--workers`: quantidade de processos que atendem pedidos. Um worker reduz o consumo de RAM; mais workers podem ajudar com pedidos simultâneos, mas aumentam o consumo de recursos.
- `--backend-store-uri`: localização da base de dados. O caminho SQLite atual é relativo à pasta dos scripts.
- `--artifacts-destination`: destino dos artefactos servidos pelo MLflow. Alterar esta opção não migra automaticamente os artefactos existentes.

A configuração atual usa HTTP com autenticação. HTTP Basic não cifra as credenciais; configura HTTPS antes de expor o serviço à Internet.

## Autenticação com utilizador e palavra-passe

O arranque já inclui `--app-name basic-auth`, de acordo com a [documentação do MLflow](https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth/).

- Utilizador inicial: `username`
- Palavra-passe inicial: `passwordpassword`
- Configuração: `basic_auth.ini`
- Contas e permissões: `basic_auth.db`
- Chave secreta persistente: `.mlflow-secret-key`

### Ativar numa instalação existente

Transfere `start_mlflow.sh`, `stop_mlflow.sh` e `basic_auth.ini` para a pasta do servidor na VM. Depois executa:

```bash
cd ~/mlflow
# Se o servidor estiver em execução, termina-o primeiro:
bash stop_mlflow.sh
# Instala as dependências de autenticação, mantendo a versão do MLflow:
versao=$(~/mlflow-venv/bin/python -c 'import mlflow; print(mlflow.__version__)')
~/mlflow-venv/bin/python -m pip install "mlflow[auth]==$versao"
chmod 600 basic_auth.ini
bash start_mlflow.sh
```

Se o servidor já estiver parado, omite o comando de paragem. Abre `http://10.17.0.253:8080` e entra com as credenciais acima.

As credenciais de `basic_auth.ini` criam o administrador no primeiro arranque. Se já existir uma base de autenticação com uma conta criada, mantêm-se as credenciais dessa conta; alterar o INI não repõe a sua palavra-passe. Gere alterações de contas pela API de autenticação do MLflow.

O INI contém a palavra-passe em texto simples e está excluído do Git. O script restringe o acesso ao ficheiro com `chmod 600`. Depois do primeiro arranque bem-sucedido, podes remover a linha `admin_password` do INI: a conta fica guardada na base de autenticação.

### Parar e voltar a iniciar

```bash
bash stop_mlflow.sh
bash start_mlflow.sh
```

O script de paragem termina o processo sem pedir login. Mantém as contas, permissões e a chave secreta. O próximo arranque ativa novamente a autenticação sem perguntas no terminal. Não apagues `basic_auth.db` nem `.mlflow-secret-key` nos reinícios; inclui-os nas cópias de segurança.

### Autenticar as máquinas de treino

No PowerShell, antes de executar o treino:

```powershell
$env:MLFLOW_TRACKING_URI = "http://10.17.0.253:8080"
$env:MLFLOW_TRACKING_USERNAME = MLFLOW_USERNAME
$env:MLFLOW_TRACKING_PASSWORD = MLFLOW_PASSWORD
```

Em Linux:

```bash
export MLFLOW_TRACKING_URI=http://10.17.0.253:8080
export MLFLOW_TRACKING_USERNAME= MLFLOW_USERNAME
export MLFLOW_TRACKING_PASSWORD= MLFLOW_PASSWORD
```

O código Python pode continuar a usar `mlflow.set_tracking_uri("http://10.17.0.253:8080")`; o cliente lê as credenciais das variáveis de ambiente. Configura-as também em qualquer API que carregue modelos deste servidor.

### Confirmar que a autenticação funciona

Na VM, o primeiro pedido deve devolver `401`, e o segundo, com credenciais válidas, `200`:

```bash
curl -o /dev/null -s -w '%{http_code}\n' http://127.0.0.1:8080/api/2.0/mlflow/experiments/search -H 'Content-Type: application/json' -d '{}'
# O curl pede a palavra-passe:
curl -o /dev/null -s -w '%{http_code}\n' -u username http://127.0.0.1:8080/api/2.0/mlflow/experiments/search -H 'Content-Type: application/json' -d '{}'
```

O endpoint `/health` verifica a disponibilidade; não confirma a proteção dos recursos.

## Problemas comuns

### MLflow não encontrado

Confirma o caminho da venv ou indica `MLFLOW_BIN` explicitamente. Também podes ativar uma venv que disponibilize o comando no `PATH`:

```bash
source /caminho/da/venv/bin/activate
bash start_mlflow.sh
```

### Porta 8080 ocupada

Identifica o processo existente antes de iniciar outra instância:

```bash
sudo ss -lptn 'sport = :8080'
ps -eo pid,ppid,args --forest | grep -E 'mlflow|uvicorn|gunicorn'
```

Se for uma instância gerida por estes scripts, usa `bash stop_mlflow.sh`. Se tiver sido iniciada manualmente, identifica e termina o processo principal correspondente.

### Erros com caracteres `\r` ou `$'\r'`

Os scripts podem ter sido guardados no Windows com finais de linha CRLF. Converte-os para LF na VM:

```bash
sed -i 's/\r$//' start_mlflow.sh stop_mlflow.sh
```

### O browser não consegue aceder

Verifica primeiro os logs e a porta na VM. Se o acesso local funcionar, confirma o IP, a ligação de rede ou VPN e as regras de firewall. Uma resposta de rejeição do cabeçalho `Host` exige verificar `--allowed-hosts`.

Se a página abre, mas as pesquisas de runs/modelos devolvem 403 e os logs mostram
`Blocked cross-origin request from http://10.17.0.253:8080`, confirma que o arranque
inclui `--cors-allowed-origins "http://10.17.0.253:8080"`. A lista de hosts e a lista
de origens são configurações diferentes. Atualiza o script na VM, para e volta a
iniciar o servidor e recarrega a página. Este ajuste não altera contas ou permissões.

### Consumo de RAM elevado

Confirma se existem instâncias duplicadas ou vários workers. Este script usa um worker. A memória total apresentada pelo sistema também pode incluir cache do Linux; consulta a coluna `available` em `free -h`.

## Referência

- [Documentação de self-hosting do MLflow](https://mlflow.org/docs/latest/self-hosting/)
- Para confirmar as opções da versão efetivamente instalada, usa `mlflow server --help` dentro da venv.
