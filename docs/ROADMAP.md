# Roadmap de desenvolvimento — LLM Benchmark

> **Documento vivo:** este arquivo acompanha a implementação e a validação do benchmark. Deve ser atualizado à medida que o projeto evoluir, sem transformar planos ou resultados não verificados em fatos.
>
> **Última atualização:** 2026-10-20 — nova validação técnica confirmou F1/F2/F3, 54 testes em dois modos e duas simulações; fundação aguardando revisão do relatório final e aprovação do responsável.

## 1. Objetivo

Construir, no repositório definitivo, um benchmark experimental em Python para avaliar LLMs em questões de múltipla escolha, relacionando **acurácia** e **entropia normalizada de Shannon entre alternativas**. O projeto deve ser pequeno, testável, reproduzível e extensível a outros modelos, datasets, estratégias de pontuação e métricas.

A entropia é calculada sobre probabilidades **renormalizadas no conjunto de alternativas**; não representa a incerteza sobre todo o vocabulário nem sobre respostas livres. Não assumir relação necessária entre baixa entropia e acerto, nem implementar índice próprio de eficiência informacional nesta fase.

Para objetivos, contratos e decisões técnicas, consultar [`README.md`](../README.md) e [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

## 2. Estado atual

**Fase atual:** Entrega 1 — Fundação, **revalidação técnica realizada; aguardando revisão e aprovação**. A entrega não está marcada como concluída.

**Próxima ação:** o responsável revisará o [relatório de validação final](VALIDATION_DELIVERY_1_FINAL.md), suas evidências e limitações para decidir sobre o aceite da fundação. O [relatório anterior](reports/VALIDATION_DELIVERY_1.md) permanece como histórico.

**Condição para avançar:** obter aprovação explícita da fundação e autorização para a próxima entrega. A Entrega 2 permanece não iniciada.

### Legenda de status

- `[x]` Concluído **e verificado** (com evidência registrada).
- `[ ]` Pendente (inclui trabalho ainda não verificado).
- `Em andamento` ou `Bloqueado`: indicar em texto, com justificativa; não marcar tarefas incompletas como concluídas.

## 3. Entregas

### Entrega 1 — Fundação e validação offline

**Status:** Aguardando revisão técnica e aprovação — correções F1/F2/F3 e regressões novamente verificadas. Os itens marcados abaixo descrevem verificações realizadas, não uma aprovação automática da entrega.

- [x] Confirmar a estrutura existente e sua aderência à arquitetura no escopo da fundação; `datasets.py` simples é permitido.
- [x] Verificar os contratos e formatos compartilhados; invariantes de pontuação, registro e metadados reforçadas e testadas.
- [x] Verificar cálculo e testes de acurácia, normalização, entropia, empates e entradas inválidas; recálculo independente dos artefatos também conferido.
- [x] Verificar isolamento entre dataset, prompt, backend, scoring, métricas, runner e storage; não há dependência de inferência real.
- [x] Executar suíte atual pelo código-fonte e pelo pacote reinstalado: 54/54 testes em cada modo, sem falhas; a evidência de 31 testes abaixo é histórica.
- [x] Executar fluxo completo **com dados e backend simulados**, sem baixar uma LLM, pelos dois pontos de entrada documentados.
- [x] Conferir os quatro arquivos de saída, não sobrescrita, publicação final de metadados e preservação dos resultados em falha de encerramento.
- [x] Conferir comandos do README, corrigir links da arquitetura e registrar o estado real da revisão.

**Correções verificadas e aceite pendente:**

- [x] F1: publicar `metadata.json` somente após todos os demais arquivos; falhas parciais deixam a pasta sem metadados finais e propagam erro.
- [x] F2: preservar os registros após erro de `backend.close()` e registrar `close_error`; manter erros anteriores visíveis.
- [x] Adicionar regressões para abertura, escrita parcial, flush, fechamento, fsync, publicação, encerramento e erros combinados.
- [x] F3: validar probabilidades contra escores, entropia contra distribuição, status/contagens/horários/proveniência e coerência do conjunto antes da escrita, reutilizando funções puras.
- [x] Corrigir referências para `docs/ROADMAP.md`, manter `AGENTS.md` na raiz e alinhar links ao relatório em `docs/reports/`.
- [x] Reexecutar a validação após as correções e registrar evidências, conformidade e limites em `docs/VALIDATION_DELIVERY_1_FINAL.md`.
- [ ] Obter aprovação explícita da fundação antes de iniciar a Entrega 2.

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

### Entrega 2 — Integração e diagnóstico MLX-LM

**Status:** Pendente.

- [ ] Confirmar versões do ambiente e compatibilidade com Apple M2 / 8 GB de memória unificada.
- [ ] Carregar e executar localmente um modelo pequeno candidato (Qwen2.5-0.5B-Instruct, variante MLX a confirmar).
- [ ] Inspecionar prompt final, template de chat, tokenização de `A`, `B`, `C`, `D` e variantes com espaço.
- [ ] Identificar a posição correta dos logits e verificar correspondência entre tokens e alternativas.
- [ ] Validar numericamente a pontuação por rótulos e o softmax condicionado às alternativas; registrar diagnósticos e eventuais limitações.
- [ ] Medir/observar consumo de memória e registrar modelo, revisão e dependências.
- [ ] Definir se `label_next_token` está aprovado para o piloto ou exige reformulação metodológica.

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
| Arquitetura | Aderência verificada na fundação | Responsabilidades separadas; contratos de inferência real continuam pendentes. |
| Primeiro dataset | Planejado | MMLU, inicialmente `validation`. |
| Primeiro backend | Planejado | MLX-LM local; Transformers/servidor ficam para evolução. |
| Primeiro modelo | Candidato | Qwen2.5-0.5B-Instruct compatível com MLX-LM; variante exata a validar. |
| Pontuação por rótulos | **Em aberto** | Requer diagnóstico de tokenização, prompt e logits. |
| Tamanho da amostra do piloto | Proposto | Até 20 questões, não é tamanho definitivo do experimento científico. |
| Resultados por execução | Fluxo nominal e regressões verificados | Publicação de metadados ao final; erro de encerramento não descarta registros calculados. |
| Aceite da fundação | Aguardando aprovação | Critérios técnicos atendidos e correções verificadas; nenhuma aprovação ou avanço à Entrega 2 foi presumido. |

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

> Novas entradas devem acrescentar **evidência objetiva**: comando e saída de testes, arquivo alterado, identificador de commit, ID da execução ou referência a relatório. Não registrar sucesso sem evidência.

### Política de versionamento após a fundação

Em 2026-10-20, o responsável autorizou publicar o estado atual diretamente na
`main`. Para as próximas alterações, o fluxo passa a usar branches de feature,
com revisão e validação antes da integração à `main`, conforme registrado em
`AGENTS.md`. A autorização de publicação não altera o status de aprovação técnica
da Entrega 1 nem inicia a Entrega 2. As evidências anteriores de ausência de commit
ou envio descrevem o momento de cada validação.
