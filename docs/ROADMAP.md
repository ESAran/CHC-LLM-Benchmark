# Roadmap de desenvolvimento — LLM Benchmark

> **Documento vivo:** este arquivo acompanha a implementação e a validação do benchmark. Deve ser atualizado à medida que o projeto evoluir, sem transformar planos ou resultados não verificados em fatos.
>
> **Última atualização:** 2026-11-06 — Entrega 2 (2A e 2B) concluída após aprovação técnica explícita do responsável. Suíte reexecutada: 71 testes aprovados. Entrega 3 permanece pendente e não autorizada; fechamento somente em commit local, sem push ou merge.

## 1. Objetivo

Construir, no repositório definitivo, um benchmark experimental em Python para avaliar LLMs em questões de múltipla escolha, relacionando **acurácia** e **entropia normalizada de Shannon entre alternativas**. O projeto deve ser pequeno, testável, reproduzível e extensível a outros modelos, datasets, estratégias de pontuação e métricas.

A entropia é calculada sobre probabilidades **renormalizadas no conjunto de alternativas**; não representa a incerteza sobre todo o vocabulário nem sobre respostas livres. Não assumir relação necessária entre baixa entropia e acerto, nem implementar índice próprio de eficiência informacional nesta fase.

Para objetivos, contratos e decisões técnicas, consultar [`README.md`](../README.md) e [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

## 2. Estado atual

**Fase atual:** Entrega 2 — Integração e diagnóstico MLX-LM **concluída**,
com aprovação técnica explícita da Entrega 2B em 2026-11-06. Entrega 1 permanece
concluída e aprovada.

**Próxima etapa:** Entrega 3 — Piloto MMLU, ainda não iniciada. Aguardar autorização
específica antes de implementar o adaptador ou executar o piloto.

**Evidências do aceite:** [relatório 2B](reports/VALIDATION_DELIVERY_2B.md),
[registros auditáveis](reports/VALIDATION_DELIVERY_2B_EVIDENCE.json), 71 testes
novamente aprovados e duas execuções reais controladas já verificadas. Os relatórios
e o histórico preservam o estado no momento de cada validação; menções anteriores
a aprovação pendente foram superadas pelo aceite explícito registrado neste fechamento.

**Limite da autorização atual:** conclusão e commit local da Entrega 2, sem push,
merge ou implementação da Entrega 3. O protocolo técnico continua provisório para
as avaliações futuras, sem alterações metodológicas adicionais.

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
autorizada está registrada em 2B abaixo. Não há adaptador MMLU.

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
MMLU permanece na Entrega 3, não iniciada. Limites: snapshot único aprovado,
entrada até 1024 tokens sem truncamento, batch 1 e caminho completo fixo;
cache pode alterar logits em BF16. Fechamento autorizado em commit local; sem push ou merge.

**Critério de conclusão:** procedimento de pontuação tecnicamente demonstrado e documentado, sem aproximações silenciosas na tokenização. **Não avançar ao piloto se a estratégia de pontuação permanecer inválida ou indeterminada.**

### Entrega 3 — Piloto MMLU com modelo real

**Status:** Pendente.

- [ ] Implementar/verificar adaptador de MMLU com procedência, revisão, partição e IDs estáveis.
- [ ] Selecionar deterministicamente até 20 questões de `validation` e registrar exatamente quais foram usadas.
- [ ] Executar o modelo usando a estratégia de pontuação aprovada na Entrega 2.
- [ ] Salvar prompt, alternativas, escores, probabilidades, predição, gabarito, acerto/erro e entropia por questão.
- [ ] Produzir resumo agregado com acurácia, contagens e estatísticas identificadas.
- [ ] Revisar amostras manualmente, conferir integridade dos registros e documentar falhas.

**Critério de conclusão:** primeira execução real rastreável e auditável, sem tratar resultados do piloto como conclusões científicas. Preservar o conjunto `test` para avaliações posteriores.

### Entrega 4 — Ampliação e análise experimental

**Status:** Futuro; escopo a decidir após o piloto.

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
| Primeiro dataset | Planejado | MMLU, inicialmente `validation`. |
| Primeiro backend | Integração 2B concluída e aprovada | MLX local pelo Runner, com 71 testes offline e duas avaliações controladas. |
| Primeiro modelo | Executado em diagnóstico | `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, revisão registrada no relatório; sem piloto científico. |
| Pontuação por rótulos | Protocolo provisório aprovado para 2B | A–D sem espaços, template aprovado e forward completo; integração aceita tecnicamente. |
| Tamanho da amostra do piloto | Proposto | Até 20 questões, não é tamanho definitivo do experimento científico. |
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
