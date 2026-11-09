# Validação do piloto MMLU — Entrega 3B

Data: 2026-11-07. **Implementada e executada localmente, aguardando revisão técnica;
3B não concluída.** Branch `codex/feat-mmlu-pilot`. Entrega 3C não iniciada.

## Fechamento da Entrega 3A e autorização

O responsável aprovou tecnicamente a [3A](MMLU_DATASET_VALIDATION.md) e determinou
manter `high_school_biology/dev/2` intacta, com alternativas A/C repetidas, usando
`dev` exclusivamente para validação técnica. O diff de 3A foi revisado, hashes e
exclusões do Git conferidos e a suíte reexecutada: **86 testes, OK, 0,769 s**.

Antes de implementar 3B foi criado o commit local exclusivo de 3A:
`b74da20fef310739af540b312e7a42f9f8c4a557`.
O checkout ficou limpo e a implementação de 3B começou em uma nova feature branch.
Não houve push ou merge. **As alterações de 3B permanecem sem commit**, conforme solicitado.

A autorização de 3B permitiu executar 15 `dev` e, somente após sucesso técnico,
60 `test` congelados. Não há exigência de acurácia mínima nem autorização para
alterar prompt, índices, modelo, pontuação ou fórmulas a partir dos resultados.

## Implementação e preservação do protocolo

| Arquivo | Responsabilidade |
| --- | --- |
| `src/benchmark/mmlu_selection.py` | Consome somente o manifesto aprovado e Parquets locais. Confere hash do manifesto, revisão, tamanho/hash das fontes, populações, IDs, índices e hashes das linhas antes de retornar questões. Não sorteia, filtra ou acessa rede. |
| `src/benchmark/config.py` | Aceita `cais/mmlu`, split `dev`/`test`, disciplinas na ordem aprovada e conjunto `mmlu-3a-v1`. Exige revisão/seed/protocolo fixados e a quantidade integral aprovada por disciplina/split. Preserva configurações sintéticas anteriores. |
| `src/benchmark/runner.py` | Seleciona o adaptador configurado, carrega dados antes do backend e confere prompts/tokens contra 3A. Encaminha a procedência a cada registro e aos metadados. Mantém tratamento de falhas, scoring e persistência existentes. |
| `src/benchmark/schemas.py` | Acrescenta procedência opcional por registro, com validação de identidade, hashes, índices e coerência com metadados. Em simulações antigas ela é `null`; os contratos matemáticos permanecem iguais. |
| `configs/mmlu_dev.yaml`, `configs/mmlu_test.yaml` | Configurações separadas e reproduzíveis das duas fases. |
| `diagnostics/mmlu_pilot_validation.py` | Auditoria somente de leitura, sem carregar uma LLM. Reconstrói os contratos, confere fonte/prompt/tokens/protocolo/persistência e gera evidência compacta e resumos por disciplina usando `summarize` existente. |
| `tests/test_mmlu_runner.py` | 14 testes offline novos de integração, deriva de dados/configuração, procedência, anomalia, falhas e condição técnica de avanço. |
| README, arquitetura e roadmap | Capacidade integrada, aprovação de 3A e resultados de 3B aguardando revisão. |
| Este relatório e `MMLU_PILOT_EVIDENCE.json` | Evidências das duas fases, métricas, recursos e limitações para revisão. |

`mmlu.py`, o manifesto/plano de 3A, o backend MLX, `prompts.py`, `numerics.py`,
`metrics.py` e scoring **não foram alterados em 3B**. O adaptador continua sem
conhecer o modelo; o backend continua sem conhecer dataset ou gabaritos.

Protocolo mantido conforme [2B](VALIDATION_DELIVERY_2B.md):

- Qwen `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, snapshot
  `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`, pesos/configuração/tokenizer conferidos.
- Prompt **exato** `current-no-enumeration-v1`, chat template padrão do snapshot,
  zero-shot, uma questão por entrada, nenhuma demonstração de `dev`.
- A–D sem espaços, `label_next_token`, forward completo sem KV cache,
  batch 1, posição `T-1`, BF16 promovido a float32 e floats Python antes da normalização.
- Funções matemáticas existentes; probabilidades condicionadas às alternativas e
  massa original no vocabulário registradas separadamente; limite de 1024 tokens,
  sem truncamento.
- Nenhuma geração livre, rotação extra, retentativa por acerto, reamostragem,
  adaptação do prompt ou correção artificial das respostas.

### Seleção, configuração e procedência

Dataset `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`, seed 42,
conjunto `mmlu-3a-v1`, somente os índices da [seleção aprovada](MMLU_SELECTION_MANIFEST.json).
O SHA-256 do manifesto completo é fixado no carregador:
`2e030192cdba40f31ec35ab33b32da1cdfe0207501e09b0edbdd1107dfd7fae1`.
Hash da seleção de 75 itens:
`3d547d3a335c55d2cdbddff09d8987c55e6bb1dffba8114ae7851ec7b2316844`.

`dataset.subjects` permite uma ou mais das três disciplinas, sempre na ordem de
3A. `dataset.split` escolhe `dev` ou `test`; `limit` deve corresponder exatamente
a 5 ou 20 por disciplina. Não há opção de substituir índices, trocar seed ou
limitar parcialmente um estrato. `selection_manifest` e `snapshot_path` indicam
arquivos locais; mudar o caminho não dispensa o hash fixado. A CLI não baixa dados.

Cada `results.jsonl` contém a `Question` integral, prompt final, scoring e métricas,
além de `provenance`: ID derivado original, disciplina, split, revisão, posição no
Parquet (`original_index`), posição no manifesto de 75 (`selection_index`), posição
na execução da fase (`execution_index`), hashes de fonte/linha, conjunto e anomalias.
**Todos os índices são base zero.** Os Parquets não possuem ID próprio por linha;
a identidade estável combina revisão, disciplina, split e posição original.

O texto, as quatro alternativas em sua ordem e o gabarito foram comparados com
as fontes em **75/75 registros**. As três disciplinas foram executadas na ordem
psicologia, história mundial e biologia, sem misturar `dev` ao denominador de `test`.

## Fases e condição de avanço

### Fase 1 — Dev exclusivamente técnico

Uma execução de 15 questões, identificada por:
`20261107T135918604041Z_0ca97243beec40a98ed44a36e3c402cb`.
Status `completed`, `simulation=false`, 15 válidas, 0 falhas e `close_error=null`.
Os quatro arquivos finais foram lidos e auditados antes de iniciar `test`.

A auditoria de `dev` terminou às **2026-11-07 13:59:34,246475 UTC**;
`test` começou às **2026-11-07 13:59:48,062030 UTC**.
Foram conferidos: origem, conteúdo/ordem/gabarito, IDs, hashes de prompt e tokens,
validação contextual do backend, posição T-1, quatro rótulos, escores finitos,
probabilidades/massa, entropia, resumo, contagens, encerramento e persistência.
Não houve falha técnica que exigisse interromper ou corrigir a fase real.

A acurácia de `dev` foi 6/15 (40%); **não foi condição de aprovação técnica**.
O teste automatizado de acurácia zero também passa pela condição técnica quando
todos os registros são íntegros. `dev` não foi utilizado para ajustar qualquer parâmetro.

### Fase 2 — Test congelado

Após aprovação dos controles técnicos de `dev`, uma execução de 60 questões:
`20261107T135948062030Z_5a52c49279c246fc8bc3dfdaef2bc1bb`.
Status `completed`, `simulation=false`, 60 válidas, 0 falhas e `close_error=null`.
Modelo, configuração técnica e hashes dos fontes são idênticos entre fases.
Apenas identificação da fase, split, quantidade e pasta de saída diferem.

Foram **34 acertos e 26 erros legítimos do modelo**, acurácia **56,6667%** e
entropia normalizada média **0,4785157438890066**. Nenhum item foi repetido para
melhorar desempenho. O total de apresentações MMLU foi exatamente 15 + 60.

## Métricas gerais e por disciplina

Os agregados usam somente as funções já existentes, sem nova métrica ou fórmula.
As duas fases são apresentadas separadamente; o resultado do piloto corresponde
aos 60 registros de `test`, não à união com `dev`. Valores arredondados apenas nesta tabela:

| Fase | Disciplina | Válidas | Acertos | Acurácia | Entropia normalizada média | Falhas técnicas |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dev` | Geral da fase | 15 | 6 | 0.400000 | 0.529707 | 0 |
| `dev` | high_school_psychology | 5 | 1 | 0.200000 | 0.553113 | 0 |
| `dev` | high_school_world_history | 5 | 4 | 0.800000 | 0.288285 | 0 |
| `dev` | high_school_biology | 5 | 1 | 0.200000 | 0.747724 | 0 |
| `test` | Geral da fase | 60 | 34 | 0.566667 | 0.478516 | 0 |
| `test` | high_school_psychology | 20 | 12 | 0.600000 | 0.387762 | 0 |
| `test` | high_school_world_history | 20 | 13 | 0.650000 | 0.468024 | 0 |
| `test` | high_school_biology | 20 | 9 | 0.450000 | 0.579761 | 0 |

Distribuição de predições A/B/C/D: `dev` **4/7/1/3**, `test` **16/23/10/11**.
Massa original dos rótulos: `dev` de **0,9887633004609512** a **0,9997288440050897**;
`test` de **0,9539227430611102** a **0,9999329892110992**. Massa elevada não implica acerto.

### Anomalia aprovada em biologia

`high_school_biology/dev/2` foi avaliada exatamente como na origem, com A e C
idênticas e gabarito D. O registro tem `anomalies: ["repeated_choice_text"]`,
`original_index=2`, `selection_index=52` e `execution_index=12`.

Predição A, `correct=false`, escores `[27.25, 27.125, 26.75, 25.25]`, probabilidades
aproximadas `[0.381045, 0.336271, 0.231115, 0.051569]`, entropia
`0.8840599582006703` e massa `0.9996599487410702`. O prompt tem 155 tokens.
É um erro de resposta do modelo em um item com anomalia documentada, não uma falha
operacional. Não houve exclusão, fusão de alternativas ou alteração do gabarito.

## Testes e auditorias

| Verificação | Evidência |
| --- | --- |
| Fechamento exclusivo de 3A | 86 testes, `OK`, 0,769 s, saída 0, antes do commit `b74da20`. |
| Suíte de 3B | **100 testes, OK, 1,089 s**, saída 0: 86 anteriores + 14 novos. |
| Configuração e arquivos | Round-trip dev/test, seleção por disciplina, rejeição de revisão/seed/índices/opções não aprovadas; manifesto adulterado, fonte ausente/corrompida e linha divergente falham antes de carregar o backend. |
| Integração offline | Procedência completa, anomalia preservada, 15 forwards sem retentativa, tokenização divergente detectada; mudança de prompt bloqueia forward. |
| Condição técnica | Acurácia zero é aceita; falha de tokenização, alteração de procedência, ausência de metadados ou erro de encerramento bloqueiam a auditoria. Registros já calculados são preservados. |
| Regressões existentes | Simulação, contratos, métricas, MLX offline e falhas F1/F2/F3 continuam passando. |
| Simulação avulsa | 3 válidas/0 falhas, acurácia `0.6666666666666666`, entropia média `0.7603746960459724`. |
| Execuções reais | Uma dev + uma test, ambas com saída CLI 0, registros completos e auditoria posterior aprovada. |
| Prompt/tokens | 75/75 hashes iguais aos congelados em 3A; candidatos contextuais 32/33/34/35, posição T-1 e entrada dentro de 1024. |
| Auditoria matemática adicional | Recalculo independente com Decimal, precisão 70: maior erro absoluto de softmax e de entropia **1,1102230246251565e-16**, em 75/75 registros. Argmax, gabarito/acerto e agregados conferidos. |
| Código | Hashes dos fontes presentes nos dois metadados conferem com o checkout final; backend, prompt e fórmulas iguais ao commit de 3A. |

O recálculo independente é uma auditoria fora do núcleo, não uma fórmula alternativa
adotada pelo benchmark. A massa original é validada por limites, soma e coerência
com a distribuição condicionada, além dos testes de 2B; os logits completos do
vocabulário não são persistidos para um novo recálculo independente dessa massa.

Simulação: `20261107T140015662976Z_d28164ec97984442b46b8ad6572400db`.
Testes e medições brutas permanecem em `outputs/mmlu-pilot-resource/`.

## Recursos e ambiente

Apple M2, 8 GiB de memória unificada, macOS 15.7.4 arm64, Python 3.12.2.
MLX/MLX Metal 0.32.4, MLX-LM 0.32.2, PyArrow 26.0.1, Transformers 5.22.1,
Tokenizers 0.23.3, Hugging Face Hub 2.4.0. Demais versões constam dos metadados.
Foi reutilizado o ambiente de 2A/2B/3A e os seis Parquets existentes (377.382 bytes).
Nenhuma dependência, dataset ou peso foi baixado nesta tarefa.

Disco livre antes: **20.381.630.464 bytes** (~18,98 GiB); depois:
**19.138.162.688 bytes** (~17,82 GiB). Swap global usado passou de **2.243,75 MiB**
para **3.102,06 MiB**; o total alocado passou de 3 para 4 GiB. São medidas globais
do macOS, não uso exclusivo do benchmark nem volume de downloads/artefatos.

| Fase | Tempo total do processo | Pico do alocador MLX (bytes) | Peak memory footprint do processo (bytes) | RSS máximo (bytes) |
| --- | ---: | ---: | ---: | ---: |
| `dev` | 7.37 s | 1.423.667.128 | 1.630.015.552 | 261.472.256 |
| `test` | 14.42 s | 1.541.566.364 | 1.751.781.568 | 517.881.856 |

O contador ativo MLX após `close()` ficou em **8 bytes** nas duas fases. O limite
de cache do alocador permanece em 256 MiB e é restaurado no encerramento; isso
não é KV cache. Os tempos de `/usr/bin/time -l` incluem inicialização, hashing,
carga, processamento e persistência; não representam throughput ou latência pura
de inferência. RSS, footprint e pico MLX são medidas diferentes e não devem ser
somados. O swap global preexistente limita conclusões sobre folga de memória.

## Artefatos e reprodução

O [JSON compacto de evidências](MMLU_PILOT_EVIDENCE.json) contém configurações,
metadados completos, resumos por fase/disciplina e os **75 resultados individuais**,
incluindo prompts exatos, gabaritos/predições, escores, probabilidades, entropia,
massa, IDs/índices, hashes e anomalias. Para reduzir repetição, os vetores de IDs
de entrada estão representados por hash/contagem na evidência compacta; os IDs
completos permanecem nos `results.jsonl` originais.

As pastas originais, ignoradas pelo Git, preservam `config.yaml`, `results.jsonl`,
`summary.json` e `metadata.json` completos. Os quatro hashes por fase estão no JSON.
O manifesto de 3A não foi reescrito. Os resultados de 3B registram o commit-base
`b74da20...` e `dirty=true`; hashes dos fontes identificam a implementação local,
mas não substituem um futuro commit revisado.

Comandos usados na raiz, no ambiente local existente:

```sh
PYTHONPYCACHEPREFIX=/tmp/mmlu-3b-pycache PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests -v

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
PYTHONPYCACHEPREFIX=/tmp/mmlu-3b-pycache PYTHONPATH=src \
  /usr/bin/time -l /tmp/llm-prompt-comparison-venv/bin/python \
  -m benchmark.cli --config configs/mmlu_dev.yaml

HF_HUB_OFFLINE=1 PYTHONPYCACHEPREFIX=/tmp/mmlu-3b-pycache PYTHONPATH=src \
  /tmp/llm-prompt-comparison-venv/bin/python diagnostics/mmlu_pilot_validation.py \
  --run outputs/mmlu-pilot-dev/20261107T135918604041Z_0ca97243beec40a98ed44a36e3c402cb \
  --output outputs/mmlu-pilot-resource/dev-validation.json
```

**Somente após a auditoria dev retornar sucesso**, foi executado o comando de
avaliação com `configs/mmlu_test.yaml`, sem outras mudanças. A auditoria final
recebeu `--run` para cada uma das duas pastas e gerou o JSON em `docs/reports/`.
O JSON foi complementado com medições antes/depois, `/usr/bin/time`, resultados
dos testes e recálculo independente; isso não modifica os artefatos do Runner.
O script de auditoria nunca dispara avaliação ou faz retentativas.

Para reproduzir em outro ambiente, usar as dependências fixadas em
`diagnostics/mlx-requirements.txt` e `diagnostics/mmlu-requirements.txt`, com os
snapshots locais aprovados. Os caminhos de cache são configuráveis; identidades,
hashes e conteúdo não. Uma nova chamada ao Runner cria uma pasta nova, sem
sobrescrever os resultados desta validação.

## Limitações, falhas e pendências

- **Nenhuma falha técnica nas duas fases reais.** Erros de resposta são resultados
  válidos: 9 em dev e 26 em test. Não houve correção de protocolo entre fases.
- A amostra cobre três disciplinas e 60 itens de test; não é um score MMLU completo,
  uma comparação estatística entre disciplinas ou conclusão científica de generalização.
  Não foi investigada contaminação dos dados de treinamento.
- Dev é exclusivamente técnico. A anomalia de biologia permanece visível e
  aprovada; não se infere sua influência causal sobre o desempenho do modelo.
- Entropia e probabilidades são condicionadas aos quatro rótulos. Baixa entropia
  ou alta massa original não demonstram acerto, calibração ou raciocínio.
- **Caminhos com e sem KV cache podem produzir logits diferentes em BF16.**
  Só o caminho completo aprovado foi usado. Não houve repetição de inferência real
  nesta entrega; estabilidade entre novas execuções e outras plataformas não foi medida.
- Dados/revisão/manifesto corrompidos ou ausentes falham antes da carga do modelo;
  essas falhas iniciais não produzem uma execução completa. A persistência existente
  trata exceções de I/O e publica metadados por último, sem garantir recuperação
  após queda de energia.
- A sequência dev → auditoria → test foi respeitada nesta tarefa e está evidenciada.
  A CLI individual continua permitindo invocar uma configuração explicitamente;
  não há um agendador que imponha automaticamente essa sequência a chamadas futuras.
- A seleção e o snapshot inicial são intencionalmente restritos aos aprovados.
  Outras seleções, revisões, políticas de exclusão ou protocolos exigem nova decisão.

**Próximo passo:** revisar o relatório, o JSON e o diff de 3B. Aguardar aceite técnico
e eventual autorização de commit/publicação ou 3C. **3B permanece aguardando revisão;
nenhuma análise gráfica foi implementada.**
