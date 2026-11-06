# Entrega 2A — Diagnóstico técnico do MLX-LM

Data: 2026-11-06. Base local: `f9c53528ca04e6456f20bbfa7119fb29d40a5175`.
Branch: `codex/feat-mlx-diagnostic`. Nenhum commit enviado ao remoto.

**Parecer:** a execução local e a extração de logits funcionaram. A pontuação
`label_next_token` é tecnicamente viável **para os rótulos sem espaço `A`, `B`,
`C`, `D`, a revisão e o template exatos testados**, com validação contextual
obrigatória. Recomenda-se normalizar os logits após promover a precisão, definir
um único caminho de inferência e não usar diretamente os `logprobs` bfloat16 da
geração para medir massa de probabilidade. O diagnóstico não aprova o backend
definitivo nem o piloto; **Entrega 2B aguarda aprovação do responsável**.

## 1. Escopo, ambiente e compatibilidade

Foram consultados README, arquitetura, roadmap e AGENTS antes do trabalho.
O diagnóstico está em [`diagnostics/mlx_labels.py`](../../diagnostics/mlx_labels.py),
fora de `src/benchmark`. Reutiliza o softmax puro existente, sem alterar acurácia,
entropia, schemas, runner ou configuração da fundação. Não há adaptador MMLU.

| Item | Observado |
| --- | --- |
| Hardware | Mac14,2, Apple M2, 8.589.934.592 bytes de memória unificada (8 GiB) |
| Sistema | macOS 15.7.4, build 24G517, arm64 |
| Python | 3.12.2 nativo arm64 |
| MLX / Metal | 0.32.4 / 0.32.4; dispositivo Apple M2 disponível |
| MLX-LM | 0.32.2 |
| Transformers / tokenizers | 5.22.1 / 0.23.3, usados para tokenização, sem backend PyTorch |
| Hugging Face Hub / safetensors | 2.4.0 / 0.8.0 |
| NumPy / PyYAML | 2.5.4 / 6.0.3 no ambiente isolado |
| pip | Atualizado de 24.0 para 26.3 somente no ambiente isolado |
| Dependências | `pip check`: `No broken requirements found.` |

A [documentação atual do MLX](https://ml-explore.github.io/mlx/build/html/install.html)
exige Apple Silicon, Python nativo >=3.10 e macOS >=14. A distribuição efetivamente
instalada foi a wheel `cp312`/`macosx_15_0_arm64`: a máquina atende também a esse
requisito mais específico. Os [metadados do MLX-LM 0.32.2](https://pypi.org/pypi/mlx-lm/0.32.2/json)
declaram Python >=3.11, MLX >=0.32.2 e Transformers >=5.7.0. Compatibilidade foi
verificada por instalação e execução, além da leitura documental.

As versões transitivas estão fixadas em
[`diagnostics/mlx-requirements.txt`](../../diagnostics/mlx-requirements.txt).
O ambiente da fundação `.venv/` e `pyproject.toml` foram preservados.
O ambiente de diagnóstico e os downloads ficam em `.cache/mlx-diagnostic/`,
ignorado pelo Git. A presença de Transformers como dependência do tokenizer não
representa implementação de um backend Transformers.

## 2. Recursos e modelo antes do download

Antes da instalação/download, `df -h .` informou **6,9 GiB disponíveis**, volume
com 97% de ocupação. `memory_pressure -Q` informou 33% de memória livre segundo
seu próprio indicador; `sysctl vm.swapusage` mostrou **612,25 MiB de swap em uso**.
Esses valores são observações globais pontuais, não reserva garantida para o modelo.

Os metadados públicos foram consultados antes de baixar os pesos:

- Modelo: [`mlx-community/Qwen2.5-0.5B-Instruct-bf16`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-bf16).
- Revisão fixada: `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`.
- `model.safetensors`: **988.097.734 bytes**, aproximadamente 0,92 GiB.
- SHA-256 dos pesos baixados: `39da1e4174f12953ad7ace17239da672aa2b4b9ba723a09835963484c5a05b41`, igual ao identificador SHA-256 informado no armazenamento LFS do modelo.
- Parâmetros carregados: todos `bfloat16`, 988.065.536 bytes, sem quantização adicional.
- SHA-256 de `tokenizer.json`: `a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf`.
- SHA-256 de `tokenizer_config.json`: `f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7`.

O download fixado terminou com 10 arquivos em cerca de 3 min 51 s. Ambiente e
cache totalizaram aproximadamente **1,3 GiB**. Uma medição posterior apontou
**5,2 GiB livres**, com 98% de ocupação; o volume também é utilizado por outros
processos e pelo swap. Não foi feito download de dataset.

## 3. Procedimento reproduzível

Instalação executada: criação de venv sem `--system-site-packages`, atualização
do pip, instalação de `mlx==0.32.4` e `mlx-lm==0.32.2` com `--no-cache-dir`,
seguida de `pip freeze` e `pip check`. Para reproduzir com as versões resolvidas:

```sh
python3 -m venv .cache/mlx-diagnostic/venv
.cache/mlx-diagnostic/venv/bin/python -m pip install --no-cache-dir pip==26.3
.cache/mlx-diagnostic/venv/bin/python -m pip install --no-cache-dir -r diagnostics/mlx-requirements.txt
.cache/mlx-diagnostic/venv/bin/python -m pip check
```

O download foi realizado com `snapshot_download`, revisão fixa, no cache local:

```sh
HF_HOME=.cache/mlx-diagnostic/huggingface HF_HUB_DISABLE_TELEMETRY=1 \
.cache/mlx-diagnostic/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'mlx-community/Qwen2.5-0.5B-Instruct-bf16',
    revision='56d07e766edd7159fbe12ed12d9cf114bf38bf1e',
    allow_patterns=['*.json', '*.safetensors', '*.txt', 'README.md'],
    max_workers=2,
)
PY
```

Depois do download, a execução do script foi offline, usando o snapshot local
e `trust_remote_code=False`. Exemplo do comando executado na segunda rodada:

```sh
HF_HUB_OFFLINE=1 HF_HOME=.cache/mlx-diagnostic/huggingface PYTHONPATH=src \
.cache/mlx-diagnostic/venv/bin/python diagnostics/mlx_labels.py \
  --snapshot .cache/mlx-diagnostic/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
  --output outputs/mlx-diagnostic-2a/run-02
```

Escolher uma pasta de saída nova para cada repetição; o script rejeita pastas
existentes. O JSON final só é escrito após terminar as verificações; arquivos
`.npy` isolados em uma execução interrompida não indicam diagnóstico concluído.
Não se trata do protocolo de storage do runner.

O script carrega o modelo uma vez, usa `model.eval()`, limita o cache do alocador
MLX a 256 MiB e executa geração greedy (`temp=0`), sem penalidades ou filtros de
logits. Esse limite não é um teto para toda a memória do processo.
Registra hashes, versões, prompts completos, IDs, formas/dtypes, probabilidades,
controles de cache e memória. Os logits completos ficam em `.npy` localmente.

Foram duas rodadas bem-sucedidas, `run-01` e `run-02`. A segunda acrescentou
medições de precisão, mantendo os mesmos prompts e procedimento de pontuação.
Em ambas, o pedido `Reply with the single word OK.` retornou **`OK`**.

## 4. Chat template e tokenização contextual

O template do snapshot insere uma mensagem de sistema padrão quando não há
mensagem explícita de sistema. Com `add_generation_prompt=True`, o primeiro
prompt de múltipla escolha ficou exatamente:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2 + 2?
A. 4
B. 3
C. 5
D. 6
Reply with exactly one letter: A, B, C, or D. Do not add spaces or explanations.<|im_end|>
<|im_start|>assistant
```

Há uma quebra de linha após `assistant`. O prompt possui **79 tokens** e termina
em `[151644, 77091, 198]`. Os IDs obtidos com `apply_chat_template(tokenize=True)`
foram iguais aos do texto renderizado seguido de
`encode(add_special_tokens=False)`, sem duplicação de tokens especiais.
Esse procedimento é coerente com a [documentação de chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating).

Para cada candidato `c`, foi comparado `encode(prompt + c)` com `encode(prompt)`:
o prefixo deve permanecer idêntico e o sufixo deve conter exatamente um token
distinto, não especial. Não foi usado apenas `encode(c)` isoladamente.

| Candidato | IDs isolados para A / B / C / D | Prefixo preservado no template exato? | Resultado |
| --- | --- | --- | --- |
| `A`, `B`, `C`, `D` | `[32]` / `[33]` / `[34]` / `[35]` | Sim | Um token contextual para cada rótulo |
| ` A`, ` B`, ` C`, ` D` | `[362]` / `[425]` / `[356]` / `[422]` | Sim | Um token, mas outro conjunto de candidatos |
| `\nA` etc. | `[198,32]` / `[198,33]` / `[198,34]` / `[198,35]` | Não | A quebra final do prompt é retokenizada |
| `\n A` etc. | `[198,362]` / `[198,425]` / `[198,356]` / `[198,422]` | Não | Mesmo problema de fronteira |
| `\n\nA` etc. | `[271,32]` / `[271,33]` / `[271,34]` / `[271,35]` | Não | Mesmo problema de fronteira |

Exemplo: `prompt + '\nA'` troca o último token `198` por `271` (duas quebras)
antes de `32`. Cortar a sequência pelo comprimento anterior produziria uma falsa
evidência de token único. O script detecta e rejeita essa interpretação.

Também foram executados os controles:

- **Espaço ao final do prompt:** `prompt + ' '` termina em `220`; ao acrescentar
  `A`, esse token é substituído por `362`. O prefixo não é preservado e o script
  não pontua os rótulos sem espaço nesse contexto.
- **Quebra adicional no prompt:** `assistant\n\n` termina em `271`. Os rótulos
  sem espaço passam na verificação contextual desse novo prefixo, mas os logits
  mudam; é um prompt diferente, não uma transformação neutra.
- **Sem cabeçalho de geração:** termina após a mensagem do usuário, com 76 tokens.
  Embora os rótulos sejam tokens únicos, o modelo prevê `<|im_start|>`; a posição
  ainda não é o início do conteúdo da resposta do assistente.
- **Texto bruto do usuário:** a terminação `explanations.` funde o ponto e `A`
  em outro token; o prefixo falha. Esse controle também não foi pontuado.

Foram inspecionados **8 contextos × 20 candidatos = 160 combinações** na rodada
completa. Seis contextos admitiram pontuação dos quatro rótulos sem espaço; dois
foram recusados explicitamente. Quatro dos seis são os prompts de chat regulares,
com as alternativas rotacionadas; os outros dois são controles.

## 5. Posição e evidências dos logits

Para a entrada `[1, T]`, `model(input_ids)` devolveu `[1, T, 151936]`, em
`bfloat16`. O vetor usado foi **`output[0, T-1, :]`**, promovido a float32 antes de
normalizar. Na primeira questão, `T=79`, portanto o índice é **78**, prevendo o
token seguinte à quebra de linha do cabeçalho do assistente. Nenhum candidato ou
gabarito foi inserido na entrada da inferência para obter esse vetor.

O [código oficial de geração na versão 0.32.2](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/generate.py)
seleciona igualmente `logits[:, -1, :]`. O diagnóstico capturou os logits brutos
da API `stream_generate` com um callback identidade e comparou os resultados
com uma execução manual com KV cache. Essa execução preenche o cache até `T-2`
e passa o último token de entrada para obter a previsão seguinte.

| Verificação | Resultado |
| --- | --- |
| Primeiro token greedy da API versus argmax direto | Iguais nos seis contextos pontuados |
| Logits com cache manual versus logits brutos da API | Diferença máxima absoluta **0**, nos seis contextos |
| Logits sem cache versus com cache | Diferença máxima absoluta de **0,21875 a 0,34375** no vocabulário |
| Repetição sem cache, entre as rodadas | Escores dos quatro rótulos idênticos nos seis contextos |
| Controle no índice `T-2` do prompt regular | Argmax `198` (quebra de linha), em vez de `32` (`A`) |

A posição está sustentada pelo código da biblioteca e pelos controles. A
diferença entre caminhos numéricos **não deve ser ocultada como igualdade exata**.
Os caminhos usam shapes/cache distintos com pesos BF16; não foi feito um controle
com todos os pesos float32 para atribuir cada diferença exclusivamente a um kernel.
Para `chat-0`, os escores com cache/API foram `[29.5, 23.875, 23.625, 24.25]`,
enquanto os escores sem cache estão na tabela seguinte.

Na conferência final, os seis arquivos do vocabulário completo sem cache também
foram idênticos byte a byte entre as rodadas. Uma referência adicional com
`math.exp`/`math.fsum`, lendo os vetores sem NumPy, confirmou as probabilidades
originais com diferença absoluta inferior a `1e-14`.

## 6. Probabilidades e massa original

Os resultados abaixo usam a passagem completa **sem cache**, com os rótulos
`A`–`D`, nesta ordem. A distribuição condicional usa o softmax da fundação.
Para conferência independente, NumPy/float64 normalizou o vetor inteiro do
vocabulário, depois dividiu as quatro probabilidades pela soma correspondente.
Os dois caminhos concordaram com tolerância absoluta `1e-12`.

| Contexto | Escores A, B, C, D | Probabilidades condicionais A, B, C, D | Massa original dos quatro rótulos |
| --- | --- | --- | ---: |
| `chat-0`: opções 4, 3, 5, 6 | 29.25; 23.75; 23.5; 24.125 | 0.986956609; 0.004033466; 0.003141267; 0.005868659 | 0.999300616166 |
| `chat-1`: opções 6, 4, 3, 5 | 29.375; 24.625; 24.0; 25.25 | 0.971396163; 0.008404224; 0.004498457; 0.015701157 | 0.999028067080 |
| `chat-2`: opções 5, 6, 4, 3 | 29.0; 24.625; 24.0; 25.0 | 0.963723772; 0.012131492; 0.006493520; 0.017651217 | 0.998757938311 |
| `chat-3`: opções 3, 5, 6, 4 | 29.125; 24.25; 24.125; 24.75 | 0.973746638; 0.007434647; 0.006561053; 0.012257661 | 0.999012939653 |

Em `chat-0`, as probabilidades originais foram
`[0.986266347363332, 0.004030645139122745, 0.0031390695906317453, 0.00586455407279618]`.
Os valores da tabela estão arredondados apenas para leitura; JSON e `.npy`
locais preservam a precisão usada no cálculo.

Os candidatos **com espaço** no mesmo contexto tiveram massa conjunta de apenas
`0.00000139496835826558`, embora sua distribuição renormalizada fosse
`[0.969291384, 0.010767855, 0.006952249, 0.012988512]`. Não foram somados aos
candidatos sem espaço nem tratados como equivalentes.

Sem o cabeçalho de geração, a massa dos rótulos cai para
`3.3582163059991036e-15`, mas a distribuição condicional ainda atribui cerca de
67,6% a `A`. Isso demonstra por que renormalização, sozinha, pode esconder uma
posição de resposta inadequada. Com uma quebra extra no cabeçalho, a massa é
`0.9983564962105387` e a probabilidade condicional de A é `0.9765230791695951`.

### Precisão: não usar `exp(logprobs)` BF16 como probabilidade exata

A API retorna `logprobs` em bfloat16 nessa configuração. A soma de suas
exponenciais, recalculadas em float64, variou de **1,000583272 a 1,036642747**
nos seis contextos. Comparando esses logprobs à normalização float64 dos
**mesmos logits brutos capturados da API**, o erro absoluto máximo variou de
**0,125582126 a 0,160987324**. Essa verificação separa o arredondamento da
normalização da diferença entre os caminhos com/sem cache.

Promover os logits antes de normalizar evita essa fonte adicional de erro,
sem recuperar a precisão já perdida na inferência BF16. O softmax MLX float32
dos quatro escores concordou com a referência com erro máximo de aproximadamente
`1.07e-7`. A massa calculada com softmax do vocabulário em MLX float32 concordou
com a referência float64 dentro das tolerâncias verificadas pelo script
(`rel_tol=1e-5`, `abs_tol=1e-7`). Não se alterou nenhuma fórmula do benchmark.

## 7. Memória, tempo e problemas observados

| Medida | Rodada 1 | Rodada 2 |
| --- | ---: | ---: |
| Tempo dentro de `load` | 1,019 s | 0,755 s |
| MLX ativo após carga | 988.065.544 bytes | 988.065.544 bytes |
| Pico de memória do alocador MLX | 1.128.820.664 bytes (~1,05 GiB) | 1.128.656.824 bytes (~1,05 GiB) |
| Máximo RSS do processo informado pelo macOS | 284.901.376 bytes | 371.736.576 bytes |
| MLX ativo após liberar modelo e cache | 8 bytes | 8 bytes |
| Swap global antes/depois do diagnóstico | 1.500,25 → 1.716,88 MiB | 1.660,88 → 1.840,00 MiB |

RSS e memória MLX **não são medidas intercambiáveis nem devem ser somadas**;
os contadores do processo não representam toda a memória Metal/unificada.
O swap é global e não pode ser atribuído integralmente a este script. Na segunda
rodada, a capacidade de swap passou de 2 para 3 GiB. O teste coube na máquina,
mas ocorreu com memória e armazenamento já pressionados por outras atividades.
Não garante capacidade para prompts longos, lotes ou uso prolongado.

Na segunda rodada, cada contexto pontuado levou cerca de 0,23–0,71 s, incluindo
passagem direta, referência via API, controle com cache e medições. Não são
tempos de throughput de geração. Os tempos de carga excluem download, importações
e hashing do snapshot.

Problemas de ambiente observados:

- A primeira consulta com `urllib` do Python do sistema falhou ao validar a
  cadeia TLS. Os metadados foram consultados com `curl` e verificação TLS ativa;
  instalação e download pelo ambiente isolado funcionaram. Não foi desabilitada
  a validação de certificados.
- O sandbox restringiu inicialmente consultas `sysctl`; medições de hardware e
  execução Metal foram realizadas com acesso local autorizado.
- Houve demoras em importações/leitura de arquivos locais. Uma amostra de pilha
  mostrou leitura de arquivo durante importação; um teste foi observado lendo
  um `.pyc`. Isso não demonstra falha de inferência nem permite atribuir a causa
  a um serviço específico de sincronização.
- Uma conferência auxiliar ficou bloqueada ao importar um `.pyc` do NumPy e foi
  encerrada; a conferência equivalente com a biblioteca padrão terminou com
  sucesso. Isso não interrompeu as duas inferências ou as suítes de testes,
  que já haviam terminado.
- Transformers avisou que PyTorch não estava instalado. Tokenização e inferência
  MLX funcionaram, sem necessidade de instalar PyTorch.

## 8. Validação, limitações e recomendação para 2B

O comando `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q`
terminou com **54 testes, OK, 17,873 s, saída 0**. A suíte inclui regressões F1/F2/F3
e integração simulada. A segunda execução, com
`PYTHONPYCACHEPREFIX=.cache/mlx-diagnostic/test-pycache PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`,
também passou: **54 testes, OK, 0,625 s, saída 0**; saída registrada em
`outputs/mlx-diagnostic-2a/foundation-tests.log`. São os mesmos 54 casos, não 108
testes distintos. Código, testes, configuração e dependências da fundação
permanecem sem alteração em relação a `f9c5352`.
O script de diagnóstico terminou com saída 0 nas duas rodadas e verificou
tokenização, IDs distintos/não especiais, valores finitos, posição/argmax,
soma e equivalência das normalizações. São verificações de diagnóstico com GPU,
não novos testes offline da suíte de 54 casos.

**Limitação comportamental importante:** nas quatro rotações da pergunta `2 + 2`,
o modelo produziu `A` como primeiro token, mesmo quando `4` estava em B, C ou D.
Isso impede interpretar a alta massa nos rótulos ou a baixa dispersão como prova
de raciocínio correto. Não foi ajustado o prompt para maximizar acertos e esses
quatro casos não constituem um benchmark científico nem um diagnóstico causal
de viés geral do modelo.

Recomendações para a Entrega 2B, sujeitas à aprovação:

1. Implementar um backend separado usando o snapshot/revisão identificados;
   manter o núcleo independente de MLX e a inferência em lote 1 inicialmente.
2. Fixar e versionar o template e a convenção **letra sem espaço imediatamente
   após o cabeçalho do assistente**; validar a fronteira contextual por questão.
   Falhar explicitamente em tokens múltiplos, prefixo alterado ou IDs duplicados.
3. Usar inicialmente a passagem completa sem cache para pontuação, reproduzindo
   as evidências desta tabela. Se o cache for adotado, identificá-lo e validá-lo
   explicitamente; não misturar escores dos dois caminhos como se fossem idênticos.
4. Promover logits a float32 antes de reduções; reutilizar o softmax da fundação
   para a distribuição condicional. Registrar também a massa original, calculada
   sobre o vocabulário completo com precisão adequada, e o dtype da inferência.
5. Acrescentar testes de fronteira, posição e precisão específicos do backend,
   além de controles comportamentais que exponham respostas constantes e mudanças
   na ordem das opções. Não excluir erros do modelo nem trocar a metodologia
   silenciosamente em função desses controles.
6. Medir memória para o comprimento de prompt efetivamente aprovado antes de
   ampliar o uso. Liberar recursos ao encerrar e preservar o tratamento de falhas
   da fundação.

**Decisão proposta:** manter `label_next_token` como caminho viável, com as
condições de contexto e precisão acima. Não há evidência que obrigue adotar
pontuação de sequências neste escopo; variantes com quebras de linha não podem
ser reduzidas silenciosamente a um único logit. O aceite metodológico definitivo
e a autorização para implementar 2B permanecem com o responsável.

Evidências completas permanecem locais em `outputs/mlx-diagnostic-2a/`: metadados
do modelo, diagnóstico preliminar de tokenização, `run-01/diagnostic.json`,
`run-02/diagnostic.json` e vetores de logits `.npy`. Pesos, ambientes, caches e
resultados são ignorados pelo Git. O relatório transcreve as evidências centrais
para permitir revisão sem publicar esses arquivos volumosos.
`final-checks.json` registra a conferência independente final, incluindo o hash
do script executado e a validade dos links locais da documentação.
