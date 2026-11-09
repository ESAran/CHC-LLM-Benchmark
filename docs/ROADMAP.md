# Roadmap de desenvolvimento — LLM Benchmark

> **Documento vivo:** este arquivo acompanha a implementação e a validação do benchmark. Deve ser atualizado à medida que o projeto evoluir, sem transformar planos ou resultados não verificados em fatos.
>
> **Última atualização:** 2026-11-09 — avaliação integral concluída e aprovada tecnicamente. Adendo documental sobre 33 empates, BF16, frequências de B e 1.081 conteúdos distintos. 128 testes novamente aprovados; artefatos preservados por hash. Fechamento em commit local autorizado, sem push/merge.

## 1. Objetivo

Construir, no repositório definitivo, um benchmark experimental em Python para avaliar LLMs em questões de múltipla escolha, relacionando **acurácia** e **entropia normalizada de Shannon entre alternativas**. O projeto deve ser pequeno, testável, reproduzível e extensível a outros modelos, datasets, estratégias de pontuação e métricas.

A entropia é calculada sobre probabilidades **renormalizadas no conjunto de alternativas**; não representa a incerteza sobre todo o vocabulário nem sobre respostas livres. Não assumir relação necessária entre baixa entropia e acerto, nem implementar índice próprio de eficiência informacional nesta fase.

Para objetivos, contratos e decisões técnicas, consultar [`README.md`](../README.md) e [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

## 2. Estado atual

**Fase atual:** avaliação integral das três disciplinas MMLU **concluída e
aprovada tecnicamente em 2026-11-09**, com fechamento local autorizado. Entregas 1, 2, 3A e 3B aprovadas.
A implementação 3C foi publicada em `e581438`, sem merge ou aceite automático;
seus artefatos foram preservados e reutilizados na expansão autorizada.

**Escopo definitivo:** 545 Psicologia + 237 História mundial + 310 Biologia =
1.092 questões test, substituindo a proposta de 1.000. Revisão de 3A, todos os
índices originais, sem amostragem/filtragem. Piloto de 60 e dev técnico intactos.

**Evidências:** [relatório integral](reports/MMLU_FULL_EVALUATION.md),
[JSON](reports/MMLU_FULL_EVIDENCE.json), [manifesto](reports/MMLU_FULL_SELECTION_MANIFEST.json)
e [run completo arquivado](reports/mmlu_full/run.tar.gz). Preflight de 1.092
entradas com 64–788 tokens, candidatos contextuais A–D válidos. Run
`20261109T120102573628Z_d6b1d88132a14f129a8a1002dd34cf5a`: 1.092 válidas, 0 falhas,
593 acertos, acurácia 0,543040293040293; macro 0,5309322201451923;
entropia média 0,4553146632905629. Todos os valores comparados nos 60 itens
sobrepostos repetiram exatamente (diferenças máximas zero).

**Validação:** 128 testes offline aprovados (117 anteriores + 11 novos), 1,120 s;
recálculo independente Decimal nos 1.092 registros e NumPy/Counter nos quatro
grupos. Anomalia: 11 pares duplicados em Psicologia preservados, sem conflito
de gabarito; nenhum item test com alternativas repetidas. Não houve mudança do
prompt, modelo, backend, método, limite ou fórmulas em função dos resultados.

**Próximo passo:** propor comparação com modelo maior nas mesmas 1.092 questões,
com preflight e protocolo preservado, e investigação separada de robustez A–D.
Aguardar autorização antes de iniciar outro modelo. Branch de fechamento local:
`codex/feat-mmlu-full-evaluation`; sem push ou merge.

### Legenda de status

- `[x]` Concluído **e verificado** (com evidência registrada).
- `[ ]` Pendente (inclui trabalho ainda não verificado).
- `Em andamento` ou `Bloqueado`: indicar em texto, com justificativa; não marcar tarefas incompletas como concluídas.

## 3. Entregas

### Entrega 1 — Fundação e validação offline

**Status:** Concluída — aprovação técnica explícita do responsável em 2026-11-06, correções F1/F2/F3 verificadas e suíte completa novamente aprovada.

- [x] Confirmar a estrutura existente e sua aderência à arquitetura no escopo da fundação; `datasets.py` simples é permitido.
- [x] Verificar os contratos e formatos compartilhados; invariantes de pontuação, registro e metadados reforçadas e testadas.
- [x] Verificar cálculo e testes de acurácia, normalização, entropia, empates e entradas inválidas; recálculo independente dos artefatos também conferido.
- [x] Verificar isolamento entre dataset, prompt, backend, scoring, métricas, runner e storage; não há dependência de inferência real.
- [x] Executar suíte atual pelo código-fonte e pelo pacote reinstalado: 54/54 testes em cada modo, sem falhas; a evidência de 31 testes abaixo é histórica.
- [x] Executar fluxo completo **com dados e backend simulados**, sem baixar uma LLM, pelos dois pontos de entrada documentados.
- [x] Conferir os quatro arquivos de saída, não sobrescrita, publicação final de metadados e preservação dos resultados em falha de encerramento.
- [x] Conferir comandos do README, corrigir links da arquitetura e registrar o estado real da revisão.

**Correções verificadas e aceite registrado:**

- [x] F1: publicar `metadata.json` somente após todos os demais arquivos; falhas parciais deixam a pasta sem metadados finais e propagam erro.
- [x] F2: preservar os registros após erro de `backend.close()` e registrar `close_error`; manter erros anteriores visíveis.
- [x] Adicionar regressões para abertura, escrita parcial, flush, fechamento, fsync, publicação, encerramento e erros combinados.
- [x] F3: validar probabilidades contra escores, entropia contra distribuição, status/contagens/horários/proveniência e coerência do conjunto antes da escrita, reutilizando funções puras.
- [x] Corrigir referências para `docs/ROADMAP.md`, manter `AGENTS.md` na raiz e alinhar links ao relatório em `docs/reports/`.
- [x] Reexecutar a validação após as correções e registrar evidências, conformidade e limites em `docs/VALIDATION_DELIVERY_1_FINAL.md`.
- [x] Obter aprovação explícita da fundação antes de iniciar a Entrega 2 — concedida pelo responsável em 2026-11-06.

**Critério de conclusão:** testes necessários passando, simulação completa verificável, resultados matematicamente consistentes e limitações registradas. Se testes falharem, registrar o bloqueio e não avançar silenciosamente.

**Evidências da revisão inicial em 2026-10-20 (antes das correções):**

| Verificação | Evidência |
| --- | --- |
| Suíte sobre o código atual | `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` → 31 testes, `OK`. |
| Suíte sobre a instalação existente | `.venv/bin/python -m unittest discover -s tests -v` → 31 testes, `OK`; arquivos Python instalados e fontes iguais em conteúdo. |
| Simulação pelo código-fonte | `PYTHONPATH=src .venv/bin/python -m benchmark.cli --config configs/synthetic.yaml` → saída 0; execução `20261020T175338332611Z_62f2cb4eeff4477296cac647fb5ecd75`. |
| Simulação pela CLI instalada | `.venv/bin/llm-benchmark --config configs/synthetic.yaml` → saída 0; execução `20261020T175340167584Z_78144fecf4a04c21ba24a460c4873d93`. |
| Resultado das duas simulações | 3 válidas, 0 falhas, acurácia sintética `0.6666666666666666`, entropia normalizada média `0.7603746960459724`. |
| Verificação matemática adicional | Probabilidades e entropias recalculadas com `decimal`, precisão 60, concordância absoluta menor que `1e-14`. |
| Ambiente | Python 3.12.2, PyYAML 6.0.2, pacote 0.1.0, Darwin 24.6.0, arm64. |
| Limite de proveniência | HEAD `090a6452db5ac8346753104e4c18585e336ea7e7`, `dirty=true` na execução por fonte; esse commit sozinho não contém a implementação não versionada. |
| Achados adicionais | F1/F2 reproduzidos com falhas controladas em diretórios temporários; detalhes e procedimentos no [relatório](reports/VALIDATION_DELIVERY_1.md). |

Os artefatos permanecem locais em `outputs/`, ignorados pelo Git. A evidência complementar está em `outputs/review-foundation-hgimvjo1/evidence.json`, incluindo hashes dos arquivos auditados. O relatório preserva as conclusões mesmo sem publicar esses artefatos. Não houve commit nem envio ao GitHub.

**Evidências após as correções em 2026-10-20:**

| Verificação | Evidência |
| --- | --- |
| Suíte atual sobre fonte | `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` → 54 testes, `OK`; 23 casos novos além dos 31 originais. |
| Suíte sobre pacote atualizado | `.venv/bin/python -m unittest discover -s tests -v` → 54 testes, `OK`, após reinstalação local sem downloads. |
| Simulação corrigida | `PYTHONPATH=src python3 -m benchmark.cli --config configs/synthetic.yaml` → saída 0; execução `20261020T181422276767Z_2322bdf97acc433fa8f8226078c0e00f`. |
| CLI instalada corrigida | `.venv/bin/llm-benchmark --config configs/synthetic.yaml` → saída 0; execução `20261020T181738149696Z_c926eeec1a7f486fb1edbc8e05d18e91`. |
| Resultado | 3 válidas, 0 falhas, acurácia `0.6666666666666666`, entropia média `0.7603746960459724`; protocolo e resultados nominais preservados. |
| Conferência dos artefatos | Quatro arquivos finais, sem temporário; probabilidades/entropias recalculadas com `decimal` (precisão 60, erro menor que `1e-14`). |
| F1 | Testes de falhas em todos os arquivos e fases de escrita/publicação; nenhum `metadata.json` final publicado em falha. |
| F2 | Registros preservados após erro de encerramento; testes de carga + encerramento, pontuação + encerramento e persistência + erros anteriores. |
| F3 | Construções inconsistentes e conjuntos divergentes rejeitados; nenhuma fórmula duplicada nos schemas ou no storage. |

**Nova validação técnica em 2026-10-20 — relatório final para revisão:**

| Verificação | Evidência |
| --- | --- |
| Suíte completa sobre fonte | `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` → 54 testes, `OK`, 0,600 s, saída 0. |
| Suíte completa sobre pacote instalado | `.venv/bin/python -m unittest discover -s tests -v`, sem `PYTHONPATH` → 54 testes, `OK`, 0,090 s, saída 0; fontes e arquivos Python instalados idênticos. |
| Nova simulação por fonte | `PYTHONPATH=src .venv/bin/python -m benchmark.cli --config configs/synthetic.yaml` → saída 0; execução `20261020T182459149278Z_96c8e2543a614756b239c60a873ebcc3`. |
| Nova simulação pela CLI instalada | `.venv/bin/llm-benchmark --config configs/synthetic.yaml` → saída 0; execução `20261020T182459256460Z_37882379b2044f868baf760c99c61179`. |
| Resultados e regressões nominais | Ambas com 3 válidas/0 falhas, acurácia `0.6666666666666666`, entropia média `0.7603746960459724`; configuração, registros e resumo idênticos entre modos e à execução corrigida anterior. |
| Conferência independente | Probabilidades e entropias recalculadas com `decimal`, precisão 60, diferença absoluta menor que `1e-14`; quatro arquivos finais e nenhum temporário. |
| Sondagens adicionais | F1: falha de abertura de resultados → CLI 2, sem metadados finais. F2: erro por item + encerramento → 3 registros preservados, `completed_with_errors`. F3: os três contraexemplos centrais do relatório anterior foram rejeitados. |
| Escopo e parecer | Nenhuma regressão ou novo bloqueio identificado nos cenários verificados. Código e testes preservados por conferência SHA-256. Somente relatório final e roadmap alterados nesta revisão; aprovação pendente. |

Detalhes no [relatório final](VALIDATION_DELIVERY_1_FINAL.md).
Logs, resultados das verificações e hashes permanecem locais em
`outputs/validation-final-fer8wo7c/`, ignorados pelo Git. Não houve commit ou envio
ao GitHub.

**Limitações remanescentes:** não há garantia de recuperação após perda de energia
ou interrupção abrupta, retomada de pastas parciais ou snapshot das alterações
locais do código. Falhas de inicialização continuam sem artefatos de avaliação,
mas erros secundários são reportados. Essas limitações estão documentadas e não
implicam validação de tokenização, inferência real ou piloto MMLU. A validação
atual usa Python 3.12.2 no ambiente existente; instalação limpa, outras versões
e plataformas não foram testadas. O relatório final detalha os limites de
reprodutibilidade e de simulação das falhas.

**Evidências de conclusão em 2026-11-06:**

- Aprovação técnica explícita do responsável após revisão do relatório final.
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` → **54 testes, `OK`, 2,656 s, saída 0**, incluindo as regressões F1/F2/F3 e os testes de integração simulada. Não foi executada uma nova simulação avulsa nesta tarefa; as execuções anteriores estão identificadas acima.
- Fundação versionada no commit `65e2dc7`; código, testes, configuração, relatórios e instruções presentes em `git ls-files`. Estado inicial limpo na `main`, alinhada à referência local `origin/main`; nenhuma consulta ou atualização remota nesta tarefa.
- `git status --ignored --short` e `git check-ignore -v` confirmaram exclusão de `.venv/`, `outputs/`, caches, build e arquivos do sistema. Acrescentadas regras para temporários `*.tmp`, `*.swp`, `*.swo` e `*~`. `git ls-files -ci --exclude-standard` não encontrou arquivos já versionados abrangidos pelas regras de exclusão.
- Registro de conclusão preparado na branch `codex/feat-conclusao-entrega-1`; README alinhado ao aceite. Código, testes e protocolo científico preservados. Publicação remota e implementação MLX-LM não autorizadas nesta tarefa.

### Entrega 2 — Integração e diagnóstico MLX-LM

**Status:** Concluída — 2A (diagnóstico) e 2B (integração ao Runner) aprovadas tecnicamente; aceite explícito de 2B em 2026-11-06. O protocolo aprovado e suas limitações permanecem documentados.

- [x] Confirmar versões do ambiente e compatibilidade com Apple M2 / 8 GB: Python 3.12.2 arm64, macOS 15.7.4, MLX 0.32.4 e MLX-LM 0.32.2 executados.
- [x] Carregar e executar `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, revisão `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`; inferência simples retornou `OK`.
- [x] Inspecionar prompt final, template de chat e variantes de rótulos: 160 combinações contextuais por rodada, com detecção de retokenização na fronteira.
- [x] Identificar a posição `T-1` dos logits e verificar correspondência; controle na posição anterior, comparação com geração e cache manual registrados.
- [x] Conferir softmax condicional e massa original contra NumPy float64; documentar diferenças BF16, efeitos do cache e limites do prompt testado.
- [x] Medir memória e registrar revisão/dependências: pico MLX observado de aproximadamente 1,05 GiB, com swap global preexistente e limitações descritas.
- [x] Aprovar provisoriamente `label_next_token` para a integração 2B: autorização explícita em 2026-11-06, com prompt sem enumeração e forward completo. O aceite técnico de 2B foi concedido; o piloto MMLU depende de autorização própria.

#### Entrega 2A — Diagnóstico técnico isolado

**Estado:** concluída; diagnóstico executado, comparação e protocolo inicial aprovados pelo
responsável em 2026-11-06 para iniciar 2B. Evidências
em [`docs/reports/MLX_DIAGNOSTIC.md`](reports/MLX_DIAGNOSTIC.md); script em
`diagnostics/mlx_labels.py`, dependências em `diagnostics/mlx-requirements.txt`.
Na etapa isolada, a fundação foi preservada sem integração ao Runner; a integração
autorizada está registrada em 2B abaixo. Não havia adaptador MMLU naquela etapa; a preparação atual está registrada em 3A.

- Duas rodadas offline concluídas com oito contextos cada; seis pontuados e dois
  recusados por falha de fronteira para os rótulos sem espaço. Os escores diretos
  dos quatro rótulos repetiram exatamente entre rodadas.
- Rótulos `A`–`D` têm IDs 32–35 no template de chat testado. Variantes com espaço
  têm outros IDs; quebras de linha podem alterar o prefixo e não são equivalentes.
- `chat-0`: escores `[29.25, 23.75, 23.5, 24.125]`, probabilidades condicionais
  `[0.986956609, 0.004033466, 0.003141267, 0.005868659]` e massa original
  `0.999300616166` (valores arredondados para exibição).
- Limites: `logprobs` BF16 da geração não normalizam exatamente; passagem com
  cache difere numericamente da passagem direta; o modelo respondeu `A` nas
  quatro rotações da pergunta, o que não valida qualidade de raciocínio.
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q` → **54 testes,
  OK, 17,873 s, saída 0**. Segunda execução com `-v` e cache de bytecode separado:
  **54 testes, OK, 0,625 s, saída 0**. `pip check` do ambiente isolado sem conflitos.
- Recomendação técnica: viável no contexto verificado, com precisão promovida,
  convenção de rótulos fixa e validação contextual obrigatória. Não constitui
  aprovação automática do método para o piloto.

**Investigação complementar de 2A (2026-11-06):** executada e incorporada ao diagnóstico aprovado;
ver [MLX_BEHAVIOR_VALIDATION.md](reports/MLX_BEHAVIOR_VALIDATION.md) e suas evidências.
Foram 64 condições (4 perguntas × 4 rotações × 4 instruções), mais 8 controles.
Com a instrução original, A venceu em 13/16 casos, e a remoção da enumeração final
produziu 13/16 vencedores corretos (contra 6/16 originais). Geração e argmax
concordaram no primeiro token em 64/64 casos; índices, tokenização e cache/API
foram conferidos. A aritmética continuou problemática; a variante semântica gerou
9/16 saídas fora do formato estrito. Não foi adotada nenhuma mudança definitiva
de prompt ou correção artificial. Os 54 testes da fundação passaram (3,891 s).
As hipóteses fundamentaram a comparação posterior e a aprovação explícita do
protocolo inicial de 2B, sem correções artificiais do comportamento do modelo.

**Comparação de formatos de prompt (2026-11-06):** executada e aprovada pelo responsável;
ver [relatório](reports/PROMPT_COMPARISON.md) e
[evidências por caso](reports/PROMPT_COMPARISON_EVIDENCE.json). Quatro perguntas
novas × quatro rotações × três formatos, com modelo/revisão e pontuação fixos.
Prompt atual sem enumeração: 11/16; referências Harness e LightEval adaptadas:
10/16 cada. Tokenização contextual validada nos 48 casos; estratégias com/sem
espaço separadas. Caminhos BF16 com e sem cache diferiram, sem mudar o vencedor
entre A–D, mas com duas diferenças de argmax global. 54 testes passaram (2,618 s).
Nenhum prompt foi escolhido automaticamente. Após revisão, o responsável aprovou
explicitamente `current-no-enumeration`, rótulos nus e forward completo para 2B.

#### Entrega 2B — Backend MLX

**Estado:** **concluída e aprovada tecnicamente** pelo responsável em 2026-11-06,
com base no relatório e no JSON de evidências. Implementação e resultados
validados preservados; suíte reexecutada neste fechamento.

- [x] Integrar backend MLX local ao Runner e seleção por configuração, preservando a simulação.
- [x] Preservar texto aprovado em `current-no-enumeration-v1`, chat template padrão, zero-shot e A–D sem espaços.
- [x] Validar prefixo contextual, token unitário/distinto/não especial, posição T-1 e forward completo sem KV cache.
- [x] Reutilizar softmax/métricas existentes com precisão promovida; persistir massa e probabilidades originais, condicionais e diagnósticos.
- [x] Registrar snapshot/revisão/hashes, template, versões, configuração e hashes dos fontes locais.
- [x] Executar 71 testes offline (54 existentes + 17 novos), todos aprovados; regressões de encerramento, configuração e registros inconsistentes incluídas.
- [x] Executar simulação (3 válidas, 0 falhas) e duas avaliações reais de 16 apresentações pelo Runner (11 acertos, 5 erros legítimos, 0 falhas operacionais).
- [x] Auditar igualdade exata de prompts/IDs/logits/probabilidades condicionais com 2A; os dois JSONL reais são idênticos. Pico MLX de 1.125.887.928 bytes; snapshot reutilizado, nenhum download.
- [x] Receber revisão e aprovação técnica explícita da Entrega 2B pelo responsável em 2026-11-06.
- [x] Reexecutar a suíte no fechamento: 71 testes, `OK`, 0,883 s, saída 0; revisar alterações de 2A/2B e exclusões do Git.

Evidências em [VALIDATION_DELIVERY_2B.md](reports/VALIDATION_DELIVERY_2B.md) e
[JSON auditável](reports/VALIDATION_DELIVERY_2B_EVIDENCE.json). A configuração
`configs/controlled_mlx.yaml` usa somente fixture sintética controlada. O adaptador
MMLU foi preparado separadamente em 3A e executado em 3B, conforme evidências abaixo. Limites de 2B: snapshot único aprovado,
entrada até 1024 tokens sem truncamento, batch 1 e caminho completo fixo;
cache pode alterar logits em BF16. Fechamento autorizado em commit local; sem push ou merge.

**Critério de conclusão:** procedimento de pontuação tecnicamente demonstrado e documentado, sem aproximações silenciosas na tokenização. **Não avançar ao piloto se a estratégia de pontuação permanecer inválida ou indeterminada.**

### Entrega 3 — Piloto MMLU com modelo real

**Status:** Em andamento — 3A e 3B concluídas e aprovadas; 3C implementada e verificada, aguardando revisão.

#### Entrega 3A — Integração e delimitação do dataset

**Estado:** concluída e aprovada tecnicamente pelo responsável em 2026-11-07.

- [x] Conferir estrutura/revisões oficiais e fixar `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`.
- [x] Implementar adaptador independente de MLX, preservando `Question` e a fundação.
- [x] Preservar texto, ordem das quatro alternativas, índice correto, disciplina, split e posição original; validar os 1.107 registros dos seis arquivos.
- [x] Selecionar por índices/seed 42: 5 `dev` + 20 `test` em cada uma das três disciplinas; registrar 75 IDs, hashes e índices base zero.
- [x] Baixar apenas seis Parquets, 377.382 bytes; conferir hashes contra o Hub, sem baixar outros splits/disciplinas.
- [x] Reportar duplicações sem filtrar: 11 pares nos arquivos de origem, nenhum par na seleção; alternativas A/C iguais em `high_school_biology/dev/2`, preservadas.
- [x] Medir entrada completa com prompt/tokenizer aprovados: 72–574 tokens, 0/75 acima de 1024; sem truncamento, pesos ou inferência.
- [x] Executar 86 testes offline, todos aprovados (0,769 s); simulação com 3 válidas/0 falhas e métricas preservadas.
- [x] Repetir preparação offline: JSONL e manifesto idênticos byte a byte; auditar conteúdo original e seleção independentemente.
- [x] Registrar procedimento, evidências, limitações e próximos passos em [MMLU_DATASET_VALIDATION.md](reports/MMLU_DATASET_VALIDATION.md).
- [x] Receber aprovação técnica explícita em 2026-11-07; manter A/C iguais em biologia `dev/2`, com anomalia documentada e uso apenas técnico de `dev`.
- [x] Revisar diff, hashes, exclusões do Git e reexecutar os 86 testes no fechamento: `OK`, 0,769 s, saída 0.

A seleção não depende de gabarito, conteúdo ou resultados. Hash do conjunto:
`3d547d3a335c55d2cdbddff09d8987c55e6bb1dffba8114ae7851ec7b2316844`.
O [manifesto](reports/MMLU_SELECTION_MANIFEST.json) contém o plano, procedência,
contagens de tokens e hashes. A preparação não gera acurácia, entropia ou logits.

#### Entrega 3B — Execução do piloto

**Estado:** concluída e aprovada tecnicamente pelo responsável em 2026-11-08.

- [x] Integrar a seleção congelada ao Runner/configuração, com disciplina/split/conjunto e procedência por registro.
- [x] Conferir manifesto fixado por SHA-256, revisão, arquivos, populações, índices, IDs e conteúdo antes de carregar o backend; sem rede ou reamostragem.
- [x] Preservar prompt/tokenizer, zero-shot, A–D nus, forward completo sem KV cache, promoção de logits e funções matemáticas de 2B.
- [x] Executar fase 1: 15 dev válidas/0 falhas, tokenização e persistência verificadas. Anomalia A/C de biologia mantida; 6 acertos não usados como critério de avanço.
- [x] Auditar dev antes de iniciar fase 2; auditoria às 13:59:34 UTC, test iniciado às 13:59:48 UTC em 2026-11-07.
- [x] Executar fase 2: 60 test válidas/0 falhas, 34 acertos; acurácia 0,5666666666666667 e entropia média 0,4785157438890066. Sem retentativas ou mudanças entre fases.
- [x] Persistir questões, prompts, procedência, escores, probabilidades, massa original, predições, gabaritos, entropias e metadados completos.
- [x] Gerar resumos por fase/disciplina e JSON compacto de evidências com 75 registros; validar cálculos e hashes dos artefatos/fontes.
- [x] Executar 100 testes offline (86 anteriores + 14 novos), `OK`, 1,089 s; simulação de 3 válidas/0 falhas, métricas preservadas.
- [x] Medir recursos reutilizando snapshots: pico MLX de 1.423.667.128 bytes em dev e 1.541.566.364 bytes em test; sem downloads. Limitações de swap global documentadas.
- [x] Receber aprovação técnica explícita em 2026-11-08, baseada no relatório e JSON do piloto.
- [x] Revisar diff completo, conferir hashes e exclusões; reexecutar 100 testes, `OK`, 1,033 s.
- [x] Receber autorização de commit local exclusivo de 3B e início de 3C; push e merge não autorizados.

Evidências em [MMLU_PILOT_VALIDATION.md](reports/MMLU_PILOT_VALIDATION.md) e
[MMLU_PILOT_EVIDENCE.json](reports/MMLU_PILOT_EVIDENCE.json). Os resultados do modelo
não foram corrigidos artificialmente; a simulação e as regressões F1/F2/F3 permanecem válidas.

#### Entrega 3C — Análise exploratória dos resultados

**Estado:** implementada e verificada localmente, **aguardando revisão técnica**.

- [x] Encerrar 3B em commit local exclusivo `5db73f9`, após aprovação e reexecução dos 100 testes.
- [x] Ler somente artefatos test persistidos e aprovados; conferir hashes, schemas, 60 IDs, ordem, conteúdo, prompt e tokens contra 3A/3B. Sem pesos, Parquets ou inferência.
- [x] Implementar `analysis.py`, leitor compartilhado `results.py` e gerador `analysis_report.py`, preservando fórmulas e módulos experimentais.
- [x] Calcular acurácia e denominadores gerais/por disciplina; entropia e sua distribuição; entropia por acerto/erro; frequências A–D; descritivas da massa original.
- [x] Documentar limiar descritivo inclusivo de 0,90 para concentração condicional: quatro erros entre 26; listar todos sem filtro dos agregados.
- [x] Gerar 12 exemplos pela regra simétrica de extremos de entropia por disciplina/acerto/erro, desempate por ID; preservar todos os 60 registros no JSON.
- [x] Produzir quatro PNGs científicos com fundo branco, denominadores e legendas; inspeção visual concluída, sem recortes ou valores ilustrativos.
- [x] Executar 117 testes offline, `OK`, 1,062 s: 100 anteriores e 17 de agregação, vazios, limites, denominadores, exemplos, leitura íntegra e separação dev/test.
- [x] Conferir descritivas dos quatro grupos (geral + disciplinas) independentemente com NumPy/Counter; repetir Markdown/JSON/PNGs com igualdade byte a byte.
- [x] Registrar observações, interpretações preliminares e limites, incluindo proibição de tratar a mesma amostra ajustada como validação independente.
- [ ] Receber revisão e aprovação técnica de 3C.
- [x] Receber autorização para commit e publicação de 3C na branch de feature em 2026-11-08.
- [ ] Integrar à main após revisão técnica de 3C.

Evidências: [relatório](reports/MMLU_EXPLORATORY_ANALYSIS.md),
[JSON](reports/MMLU_EXPLORATORY_EVIDENCE.json), logs e conferências locais em
`outputs/mmlu-analysis-check/` (ignorados). Dependências gráficas já disponíveis:
Matplotlib 3.9.1.post1, NumPy 1.26.4, Pillow 10.4.0; nenhum download.

**Critério de conclusão da Entrega 3:** piloto e análise rastreáveis e auditáveis,
com revisão técnica de cada parte. 3A e 3B aprovadas; 3C aguarda aceite. Resultados
não são conclusões científicas nem critério para ajustar o protocolo e reutilizar
a mesma amostra test como validação independente.

### Entrega 4 — Ampliação e análise experimental

**Status:** Em andamento — avaliação integral destas três disciplinas concluída
e aprovada em 2026-11-09. Comparação entre modelos e demais ampliações continuam
futuras; esta aprovação não conclui toda a Entrega 4.

#### Avaliação integral MMLU — três disciplinas

**Estado:** concluída e aprovada tecnicamente em 2026-11-09.

- [x] Fixar seleção `mmlu-full-test-v1` com os 1.092 índices test originais e hashes; preservar piloto/dev.
- [x] Reutilizar os três Parquets e snapshot do modelo da revisão aprovada, sem novos downloads desses artefatos.
- [x] Validar preflight contextual de todos os itens, sem truncamento: 64–788 tokens, limite 1024.
- [x] Executar um único run real integral pelo Runner: 1.092 válidas, 0 falhas técnicas, encerramento/persistência conferidos.
- [x] Comparar 60 itens com o piloto: igualdade de logits, probabilidades, entropia, massa, predições e conteúdo.
- [x] Preservar 11 pares duplicados em Psicologia; documentar IDs, cobertura e limitações.
- [x] Ampliar análise reutilizável com macro e seleção explícita; gerar tabelas, cinco PNGs, JSON e arquivo do run.
- [x] Executar 128 testes offline e auditorias independentes das métricas/agregados; manter regressões anteriores.
- [x] Receber aprovação técnica explícita da avaliação integral em 2026-11-09.
- [x] Documentar 33 empates exatos/`lowest_index`, limitação BF16, B=415 versus 270 e 1.081 conteúdos distintos.
- [x] Revalidar 128 testes (OK, 1,131 s), diff e preservação dos artefatos antes do fechamento.
- [x] Receber autorização para commit local de conclusão.
- [ ] Receber autorização para eventual publicação/merge; não realizada neste fechamento.

Resultados detalhados no [relatório integral](reports/MMLU_FULL_EVALUATION.md).
Não representa todo o MMLU ou confirmação independente do piloto; o protocolo
não foi ajustado pelos resultados. Acurácia macro é média simples entre disciplinas,
sem criação de métrica composta de eficiência informacional.


- [ ] Avaliar aumento da amostra e comparações entre modelos, mantendo protocolos compatíveis.
- [ ] Considerar outros modelos locais e, se necessário, backend Transformers para execução em servidor.
- [ ] Avaliar novas métricas ou análises sem alterar retroativamente os resultados brutos.
- [ ] Considerar WinoGrande e justificar associações exploratórias ao CHC, sem alegações psicométricas.
- [ ] Definir protocolo final, critérios de comparação, limitações e artefatos de reprodução.

**Critério de conclusão:** a definir com base nos resultados do piloto e nas decisões metodológicas posteriores.

## 4. Decisões e pendências importantes

| Tema | Estado atual | Observação |
| --- | --- | --- |
| Métricas iniciais | Definido | Acurácia e entropia normalizada entre alternativas; sem índice composto. |
| Arquitetura | Fundação e integração 2B aprovadas | Responsabilidades separadas; contratos de inferência real implementados e testados. |
| Primeiro dataset | 3A e 3B aprovadas | MMLU fixado, três disciplinas; 15 dev técnicos e 60 test executados sem falhas técnicas. |
| Primeiro backend | Integração 2B concluída e aprovada | MLX local pelo Runner, com 71 testes offline e duas avaliações controladas. |
| Primeiro modelo | Executado em diagnóstico e piloto 3B | `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, revisão fixa; piloto MMLU aprovado tecnicamente, sem conclusões científicas. |
| Pontuação por rótulos | Protocolo provisório aprovado para 2B | A–D sem espaços, template aprovado e forward completo; integração aceita tecnicamente. |
| Tamanho da amostra do piloto | Executado em 3B | 15 dev técnicos + 60 test, conjuntos separados nos agregados; não define a amostra final da pesquisa. |
| Resultados por execução | Fluxo nominal e regressões verificados | Publicação de metadados ao final; erro de encerramento não descarta registros calculados. |
| Aceite da fundação | Concluído em 2026-11-06 | Aprovação técnica explícita; 54 testes continuam passando durante 2A. |

## 5. Como manter este roadmap atualizado

Após **cada alteração relevante** de código, teste, integração ou decisão:

1. **Ler este arquivo antes do trabalho** e identificar a entrega correspondente.
2. Atualizar o **status real**, marcando `[x]` somente o que foi comprovadamente implementado e validado.
3. Registrar em `Histórico de progresso` uma entrada curta com data, mudança, evidência e próximo passo.
4. Anotar bloqueios, testes falhos e decisões técnicas abertas em vez de escondê-los.
5. Se uma decisão modificar o escopo, atualizar também `docs/ARCHITECTURE.md` e, quando afetar o uso público, `README.md`.
6. Não inventar testes, resultados, arquivos, commits ou execuções. Não inserir nomes pessoais, dados privados, tokens ou caminhos locais sensíveis.
7. Antes de encerrar a tarefa, informar no resumo do Codex o estado do roadmap, os testes executados e a próxima ação.

**Observação:** atualizar o roadmap não substitui versionamento Git nem validação. Uma tarefa com código escrito, mas sem testes suficientes, continua pendente ou em andamento.

## 6. Histórico de progresso

| Data | Mudança / situação | Evidência | Próxima ação |
| --- | --- | --- | --- |
| 2026-10-20 | Roadmap criado a partir do plano arquitetural. A fundação foi informada como gerada pelo Codex, sem relatório de validação disponível aqui. | Relato de implementação; testes e arquivos ainda não verificados neste documento. | Revisar a Entrega 1 e registrar os resultados reais dos testes. |
| 2026-10-20 | Revisão técnica da fundação; status alinhados ao código e relatório registrado. Somente documentação alterada. | 31 testes passando em cada modo; duas simulações com 3 válidas/0 falhas; recálculo independente; F1/F2 reproduzidos. Ver [relatório](reports/VALIDATION_DELIVERY_1.md). | Responsável revisar os achados e autorizar eventuais correções antes do aceite; manter Entrega 2 pendente. |
| 2026-10-20 | Correções autorizadas de F1/F2/F3 e alinhamento documental implementados. | 54 testes passando; simulação corrigida com 3 válidas/0 falhas; detalhes na seção 7 do [relatório](reports/VALIDATION_DELIVERY_1.md). | Responsável revisar as correções e aprovar a fundação; Entrega 1 aguardando aprovação, Entrega 2 não iniciada. |
| 2026-10-20 | Nova validação técnica após as correções; apenas documentação atualizada. | 54 testes passando em cada modo, duas novas simulações, recálculo independente e sondagens F1/F2/F3; ver [relatório final](VALIDATION_DELIVERY_1_FINAL.md). | Responsável revisar o relatório e decidir o aceite; Entrega 1 aguardando revisão e aprovação, Entrega 2 não iniciada. |
| 2026-11-06 | Entrega 1 aprovada pelo responsável e marcada como concluída; README alinhado e exclusões de temporários reforçadas. | 54 testes, `OK`, saída 0; fundação versionada em `65e2dc7`; auditoria do índice e regras de exclusão sem artefatos gerados versionados. | Criar commit local de conclusão na branch de feature, sem push; aguardar autorização para iniciar a Entrega 2. |

> Novas entradas devem acrescentar **evidência objetiva**: comando e saída de testes, arquivo alterado, identificador de commit, ID da execução ou referência a relatório. Não registrar sucesso sem evidência.

**2026-11-06 — Entrega 2A:** diagnóstico autorizado executado em ambiente isolado,
sem mudanças no núcleo. Duas execuções reais locais, tokenização contextual,
logits, probabilidades e memória documentados no [relatório MLX](reports/MLX_DIAGNOSTIC.md).
54 testes da fundação aprovados. Próxima ação: responsável revisar os achados e
autorizar ou ajustar 2B. Entrega 2 não concluída; nenhum push realizado.

**2026-11-06 — Investigação comportamental complementar:** modo `--behavior`
adicionado ao diagnóstico existente, com matriz isolada, geração livre e evidência
por caso. Relatório e JSON de evidências publicados apenas como arquivos locais;
sem commit ou push. Execução concluída em ambiente temporário com as mesmas
dependências fixadas após demoras de leitura no ambiente do projeto. Nenhuma
alteração no núcleo; 2B aguarda revisão e autorização.

### Política de versionamento após a fundação

Em 2026-10-20, o responsável autorizou publicar o estado atual diretamente na
`main`. Para as próximas alterações, o fluxo passa a usar branches de feature,
com revisão e validação antes da integração à `main`, conforme registrado em
`AGENTS.md`. A autorização de publicação não altera o status de aprovação técnica
da Entrega 1 nem inicia a Entrega 2. As evidências anteriores de ausência de commit
ou envio descrevem o momento de cada validação.

**2026-11-06 — Comparação controlada de prompts:** referências oficiais fixadas
por commit e Qwen2.5 consultadas; 48 casos concluídos em
`outputs/prompt-comparison-03/diagnostic.json`. [Relatório](reports/PROMPT_COMPARISON.md)
e JSON preservam prompts, probabilidades e controles. Ambiente temporário
restaurado com versões fixadas após bloqueio de arquivos `dataless`; snapshot
existente recuperado do iCloud e hash dos pesos conferido, sem novo modelo.
54 testes da fundação aprovados; auditoria dos 48 registros e 96 vetores passou,
assim como a comparação de 16 textos com a função oficial LightEval. Entrega 2
permanece em andamento; 2B não iniciada.
Próxima ação: revisão e aprovação explícita do protocolo candidato. Sem commit/push.

**2026-11-06 — Entrega 2B autorizada e implementada:** comparação de prompts
aprovada pelo responsável; protocolo provisório integrado ao Runner na branch
`codex/feat-mlx-backend`. 71 testes passaram em 0,730 s; simulação preservada.
Execução real final `20261106T171107979803Z_1c1baaac1d0c4cd6857cdd930d858b50`:
16 válidas, 0 falhas, acurácia 0,6875, entropia média 0,2938674223513097.
[Relatório](reports/VALIDATION_DELIVERY_2B.md) e evidências registram igualdade
com 2A, repetição dos resultados, hashes e memória. Fórmulas preservadas;
sem novos downloads, MMLU, commit ou push. **2B aguarda revisão; não concluída.**
Próximo passo: responsável revisar e decidir o aceite antes de qualquer Entrega 3.

**2026-11-06 — Conclusão e aceite da Entrega 2:** o responsável aprovou tecnicamente
2B com base em `VALIDATION_DELIVERY_2B.md` e `VALIDATION_DELIVERY_2B_EVIDENCE.json`.
Entrega 2 e sua etapa 2B marcadas como concluídas; README e arquitetura alinhados
somente quanto ao status. Resultados aprovados preservados: duas execuções reais
com 16 válidas/0 falhas, 11 acertos, acurácia 0,6875 e entropia média
0,2938674223513097; igualdade com o diagnóstico e repetição verificadas.

Comando de fechamento:
`PYTHONPYCACHEPREFIX=/tmp/benchmark-delivery2-final-pycache PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`
→ **71 testes, OK, 0,883 s, saída 0**. Revisão abrangeu código/configuração/testes,
scripts, requisitos, relatórios e JSONs de evidência de 2A e 2B ainda não
versionados; hashes dos fontes conferem com a execução aprovada. Ambientes,
pesos, caches e `outputs/` seguem ignorados; os JSONs de evidência em
`docs/reports/` são artefatos de validação deliberadamente incluídos. Nenhuma
mudança nas fórmulas, no protocolo ou na implementação durante o fechamento.
Commit local autorizado na branch `codex/feat-mlx-backend`, sem push ou merge.
**Próximo passo:** aguardar autorização específica para a Entrega 3, não iniciada.

**2026-11-06 — Autorização de integração e publicação:** após o commit local
`2f12368` (conclusão da Entrega 2), o responsável autorizou merge, push e
organização do Git. A integração inclui também `f9c5352`, aprovação da Entrega 1
que ainda estava apenas local. Preservar os commits aprovados, integrar sem
reescrever histórico e remover somente branches já incorporadas à `main`.
Validação disponível: 71 testes aprovados no fechamento, sem mudanças posteriores
nos fontes. Ambientes, pesos, caches e saídas temporárias permanecem ignorados.
As referências anteriores a “sem push ou merge” descrevem as autorizações da época.
Próximo trabalho de desenvolvimento: Entrega 3, somente após autorização específica.

### 2026-11-07 — Preparação MMLU em 3A, aguardando revisão

Autorizada preparação de `high_school_psychology`, `high_school_world_history` e
`high_school_biology`, com 5 `dev` e 20 `test` por disciplina, substituindo o escopo
antigo de `validation`. Adaptador independente, plano com seed 42/revisão/hashes,
script de seis downloads e contagem de tokens local implementados. Foram validados
1.107 registros; 75 selecionados, sem exclusão por gabarito, duplicação ou tamanho.
Nenhuma entrada selecionada excede 1024 tokens. Ressalva: alternativas A/C iguais
em biologia `dev/2`, intactas; 11 pares duplicados em psicologia fora da seleção.

**Evidências:** [relatório 3A](reports/MMLU_DATASET_VALIDATION.md),
[manifesto](reports/MMLU_SELECTION_MANIFEST.json), 86 testes `OK` (0,769 s),
duas preparações offline idênticas e simulação de 3 válidas/0 falhas, acurácia 2/3,
entropia média 0,7603746960459724. Nenhum modelo executado sobre MMLU; prompt,
backend, contratos centrais e fórmulas preservados. Branch `codex/feat-mmlu-dataset`,
sem commit/push. **Próximo passo:** revisão e aceite de 3A antes de autorizar 3B.

### 2026-11-07 — Aceite e fechamento local da Entrega 3A

Aprovação técnica explícita do responsável com base no relatório 3A. A anomalia
A/C idênticas em biologia `dev/2` deve permanecer intacta; `dev` é exclusivamente
técnico. Diff pendente revisado, manifesto e hashes dos fontes conferidos, ambientes,
pesos e caches ignorados. Suíte reexecutada: **86 testes, OK, 0,769 s**, saída 0.
Commit local exclusivo da 3A autorizado; sem push/merge. Relatório original preserva
o estado histórico anterior ao aceite. Próximo passo autorizado: integrar 3B e
executar dev antes de test, sem condicionar o avanço à acurácia e sem iniciar 3C.

### 2026-11-07 — Execução controlada 3B, aguardando revisão

Após o commit exclusivo de 3A `b74da20`, implementação em `codex/feat-mmlu-pilot`:
consumo local da seleção por manifesto/hash, configuração por disciplina/split,
procedência persistida e conferência de prompts/tokens contra 3A. Backend, scoring,
prompt e fórmulas preservados. 100 testes passaram (1,089 s) e a simulação manteve
3 válidas/0 falhas, acurácia 2/3 e entropia média 0,7603746960459724.

Fase 1 dev: 15 válidas, 6 acertos, 0 falhas técnicas; anomalia de biologia intacta.
A auditoria técnica passou antes de autorizar operacionalmente a fase 2 test:
60 válidas, 34 acertos, 0 falhas, entropia média 0,4785157438890066. Apenas uma
execução de cada fase; sem ajustes/retentativas por desempenho. Recálculo independente
com Decimal em 75 registros: erro máximo 1,1102230246251565e-16 em softmax/entropia.
Hashes de fonte, conteúdo, prompt, tokens e artefatos conferidos. Snapshots reutilizados,
sem downloads; recursos e limitações documentados no relatório e JSON de evidências.

**Próximo passo:** responsável revisar 3B. Sem commit de 3B, push, merge ou início de 3C.

### 2026-11-08 — Aceite e fechamento local exclusivo de 3B

Aprovação técnica explícita do responsável com base no relatório e JSON do piloto.
Diff completo revisado: integração/configuração/procedência, auditoria, testes,
relatórios e documentação dentro do escopo; backend, prompt, scoring e fórmulas
preservados. Hashes dos fontes e oito arquivos das duas execuções conferidos.
Ambientes, caches, pesos e saídas temporárias ignorados; nenhum arquivo já
versionado abrangido pelas regras de exclusão. Suíte reexecutada: **100 testes,
OK, 1,033 s**, saída 0. Commit local exclusivo autorizado, sem push ou merge.
Próximo passo: 3C, somente análise de 60 test persistidos, mantendo dev separado,
sem nova inferência, alteração de protocolo ou commit de 3C antes de revisão.

### 2026-11-08 — Análise exploratória 3C, aguardando revisão

Após fechamento local de 3B em `5db73f9`, implementação em `codex/feat-mmlu-analysis`.
Leitor JSON extraído do diagnóstico 3B sem mudança de lógica; agregações reutilizam
schemas/`summarize`, sem duplicar fórmulas de scoring. Análise exclusivamente dos
60 test congelados, com 15 dev separados e nenhuma LLM executada. Todos os artefatos
originais mantiveram seus hashes. 34 acertos/60; entropia média de acertos 0,346090
e de erros 0,651688; quatro erros com máx. p(A–D) ≥ 0,90. Critério descritivo e
regra de exemplos simétricos declarados antes da tabulação; nenhuma seleção de
casos favoráveis ou alegação de calibração, viés de posição ou diferenças cognitivas.

**Evidências:** relatório Markdown, JSON com 60 registros/valores e quatro PNGs
em `docs/reports/`. 117 testes `OK` (1,062 s); quatro grupos conferidos com referência
independente NumPy/Counter; segunda geração Markdown/JSON/PNGs idêntica byte a byte;
figuras inspecionadas. Imports `mlx.core`/`mlx_lm` ausentes na geração. Cache de fontes
gráficas direcionado a `/tmp`; dependências gráficas preexistentes, sem downloads.
**Próximo passo:** revisão técnica de 3C. Sem commit de 3C, push ou merge.


### 2026-11-08 — Publicação autorizada no GitHub

Usuário autorizou publicar todo o trabalho pendente. A branch
`codex/feat-mmlu-analysis` reúne os commits exclusivos aprovados de 3A e 3B
e a implementação, testes, relatórios, evidências e quatro PNGs de 3C.
A suíte foi reexecutada: **117 testes, OK, 1,074 s**. Diff sem erros de whitespace;
nenhum arquivo versionado abrangido pelo `.gitignore`. Ambientes, pesos, caches
e resultados temporários permanecem excluídos. Publicação não equivale ao aceite
técnico de 3C: revisão ainda pendente, sem marcar a entrega concluída ou integrar
à main. Próximo passo: revisar o relatório exploratório e decidir o aceite.

### 2026-11-09 — Expansão integral autorizada, executada e aguardando revisão

Escopo de 1.000 substituído pelos 1.092 test completos (545/237/310), sem alterar
revisão, alternativas, prompt, modelo, scoring, limite ou fórmulas. Branch local
`codex/feat-mmlu-full-evaluation`, base `e581438`. Preflight antes da inferência:
1.092 registros/IDs/hashes íntegros, 64–788 tokens, 4.368 continuações A–D válidas;
11 pares duplicados originais mantidos. Ambiente temporário anterior ausente;
reutilizado ambiente local e restaurado apenas PyArrow 26.0.1. Sem download de
pesos/dataset, sem execução de dev ou retentativas por desempenho.

126 testes aprovados antes do run. Execução única
`20261109T120102573628Z_d6b1d88132a14f129a8a1002dd34cf5a`: 1.092 válidas/0 falhas,
593 acertos; Psicologia 323/545, História 130/237, Biologia 140/310.
Global 0,543040293040293; macro 0,5309322201451923. Sobreposição de 60 repetiu
exatamente logits, probabilidades, entropia e massa. Auditoria Decimal (70 dígitos):
erros máximos 1,1102230246251565e-16 / 1,1709383462843448e-16 em softmax/entropia.
NumPy/Counter conferiram os quatro grupos. Suíte final: **128 testes, OK, 1,120 s**.

Relatório, JSON, manifesto integral e cinco PNGs em `docs/reports/`; run original
em `outputs/mmlu-full-test/` e cópia compactada auditável com os quatro arquivos
originais em `docs/reports/mmlu_full/run.tar.gz`. Evidências anteriores preservadas.
Próximo passo: revisão técnica. Sem commit, push, merge ou conclusão automática.

Conferência final: 128 testes novamente aprovados (1,120 s), cinco figuras
inspecionadas visualmente, dados dos gráficos conferidos e arquivo compactado
comparado byte a byte com os quatro artefatos originais. Hashes dos fontes,
links Markdown, preservação dos relatórios anteriores e ausência de caminhos
pessoais nos novos artefatos conferidos. Diff sem erros de whitespace.

### 2026-11-09 — Aprovação e fechamento da avaliação integral

Aprovação técnica explícita com base no relatório, evidências e manifesto.
Adendo documental conferido nos 1.092 registros persistidos: 33 empates exatos
no maior logit (17 Psicologia, 9 História, 7 Biologia), todos resolvidos pela regra
existente `lowest_index`; predições nos empates A=21/B=8/C=4/D=0. BF16 limita
precisão, sem evidência de que explique sozinho todos os empates. B aparece em
415 predições e 270 gabaritos; nenhuma atribuição causal a viés de posição.
Os 11 pares idênticos de Psicologia permanecem: 1.092 registros/1.081 conteúdos
distintos, sem mudar denominadores ou recalcular predições.

Diff da expansão revisado; alterações deste fechamento exclusivamente documentais.
Suíte `PYTHONPYCACHEPREFIX=/tmp/mmlu-closure-pycache PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q`:
**128 testes, OK, 1,131 s**. Manifesto, JSON, quatro arquivos do run, arquivo auditável
e cinco PNGs preservados por SHA-256; arquivo compactado conferido byte a byte.
Resultados, fontes experimentais, fórmulas e configurações permanecem os aprovados.
O JSON/gerador de relatório preservam o estado histórico pré-aceite; o adendo
editorial e este roadmap registram a aprovação sem reescrever essas evidências.
Ambientes, pesos, caches e temporários continuam ignorados. Commit local de
conclusão autorizado na branch `codex/feat-mmlu-full-evaluation`; sem push/merge.

Próxima etapa proposta, não iniciada: comparar modelo maior após fixar snapshot,
precisão e compatibilidade contextual/memória, usando o mesmo manifesto, prompt,
zero-shot, forward sem cache, scoring, desempate e métricas. Experimento separado
de robustez por rotações balanceadas A–D, com remapeamento explícito do gabarito,
resultados próprios e análise de estabilidade por conteúdo, sem substituir o run
principal nem otimizar o protocolo a partir destas questões.
