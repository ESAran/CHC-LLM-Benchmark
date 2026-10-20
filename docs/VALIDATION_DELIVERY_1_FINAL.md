# Revalidação técnica — Entrega 1: Fundação

Data: 2026-10-20.

**Parecer:** F1, F2 e F3 estão resolvidos nos cenários verificados. Os critérios
técnicos da fundação estão atendidos, sem novos bloqueios ou regressões
identificados nesta revisão. **A Entrega 1 permanece aguardando revisão e
aprovação explícita do responsável; não está concluída.**

## 1. Escopo e referências

Esta revisão confrontou a implementação com o [relatório anterior](reports/VALIDATION_DELIVERY_1.md),
o [README](../README.md), a [arquitetura](ARCHITECTURE.md),
o [roadmap](ROADMAP.md) e as [instruções permanentes](../AGENTS.md).
O relatório anterior foi preservado, incluindo os achados originais e sua seção
de correções. Seu caminho atual é `docs/reports/VALIDATION_DELIVERY_1.md`.

Foram executadas a suíte completa, duas simulações e verificações adicionais de
artefatos, fórmulas e falhas controladas. Nesta revisão, somente este relatório e
`docs/ROADMAP.md` foram criados ou editados como documentação do projeto.
Código, testes, configuração, README, arquitetura e AGENTS permaneceram
inalterados, conforme comparação SHA-256 com o início da revisão.
Não houve integração de LLM, mudança metodológica, início da Entrega 2, commit
ou envio ao GitHub.

## 2. Ambiente e testes automatizados

Ambiente observado: Python 3.12.2, PyYAML 6.0.2, pacote `llm-benchmark` 0.1.0,
Darwin 24.6.0, arm64. A instalação local existente foi utilizada; esta revisão
não fez instalação limpa nem downloads. Os arquivos Python instalados foram
comparados com `src/benchmark` e têm conteúdo idêntico.

Comandos executados a partir da raiz:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
```

No segundo comando, `PYTHONPATH` foi removido do ambiente para verificar o pacote
instalado.

| Execução | Resultado | Tempo informado pelo unittest | Código de saída |
| --- | --- | --- | --- |
| Código-fonte | 54 testes, `OK`, sem falhas ou erros | 0,600 s | 0 |
| Pacote instalado | 54 testes, `OK`, sem falhas ou erros | 0,090 s | 0 |

São **54 testes distintos executados duas vezes**, incluindo os 23 acrescentados
às correções em relação aos 31 testes da revisão inicial. Os subcasos de falhas
de escrita não são contados como testes distintos nessa soma.

| Arquivo | Testes | Responsabilidade verificada |
| --- | ---: | --- |
| `tests/test_config.py` | 3 | YAML, campos obrigatórios, chaves duplicadas e limites da fundação |
| `tests/test_contract_invariants.py` | 10 | Invariantes semânticas, tipos, proveniência e coerência entre artefatos |
| `tests/test_metrics.py` | 7 | Acurácia, entropia, empates, zeros, referências independentes e entradas inválidas |
| `tests/test_runner_storage.py` | 14 | Fluxo completo, CLI, repetibilidade, falhas por item, encerramento e erros combinados |
| `tests/test_schemas_datasets.py` | 8 | Questões, adaptação, seleção determinística, prompts e contratos |
| `tests/test_scoring.py` | 6 | Softmax, estabilidade, empates e ciclo do backend simulado |
| `tests/test_storage_failures.py` | 6 | Falhas de I/O, publicação final e rejeição prévia de conjuntos inconsistentes |
| **Total** | **54** | **Todos passaram nos dois modos** |

## 3. Nova simulação e conferência matemática

```sh
PYTHONPATH=src .venv/bin/python -m benchmark.cli --config configs/synthetic.yaml
.venv/bin/llm-benchmark --config configs/synthetic.yaml
```

Ambos terminaram com código de saída **0**:

| Modo | ID da execução em `outputs/` |
| --- | --- |
| Código-fonte | `20261020T182459149278Z_96c8e2543a614756b239c60a873ebcc3` |
| CLI instalada | `20261020T182459256460Z_37882379b2044f868baf760c99c61179` |

Cada execução contém exatamente `config.yaml`, `results.jsonl`, `summary.json`
e `metadata.json`, sem arquivo temporário remanescente. O estado final é
`completed`, com `close_error=null`, 3 registros válidos e 0 falhas.
A configuração salva corresponde à configuração de entrada resolvida.

| Resultado sintético | Valor |
| --- | ---: |
| Acurácia | `0.6666666666666666` (2/3) |
| Entropia normalizada média | `0.7603746960459724` |

A conferência independente usou `decimal` com precisão de 60 dígitos, calculando
`p_i = exp(s_i) / sum(exp(s_j))` e
`H_norm = -sum(p_i * ln(p_i)) / ln(K)` a partir dos escores persistidos,
sem chamar as funções numéricas do benchmark.
Probabilidades, entropias individuais e média concordaram com os artefatos com
diferença absoluta inferior a `1e-14`. Predições e acertos também foram conferidos.

| Ordem persistida | Alternativas (K) | Acerto | Entropia de referência, arredondada |
| --- | ---: | --- | ---: |
| `synthetic-002` | 3 | Não | `0.7576791106615823` |
| `synthetic-001` | 2 | Sim | `0.8399415379831692` |
| `synthetic-003` | 4 | Sim | `0.6835034394931654` |

`config.yaml`, `results.jsonl` e `summary.json` são idênticos byte a byte entre
os dois modos e em relação à execução corrigida anterior
`20261020T181422276767Z_2322bdf97acc433fa8f8226078c0e00f`.
Metadados variam legitimamente em ID, horários e proveniência.
Isso confirma a preservação do comportamento nominal para esta fixture e seed.
Esses valores são evidências de funcionamento, não resultados de uma LLM real.

## 4. Verificação dos achados anteriores

### F1 — Persistência dos resultados: resolvido nos cenários testados

`storage.save_run` valida e serializa o conjunto antes de criar a pasta.
A criação exclusiva impede sobrescrita. Os arquivos são gravados, descarregados
com `flush` e `fsync` e fechados antes da publicação final dos metadados.
`metadata.json.tmp` é renomeado para `metadata.json` por último, no mesmo diretório.
O retorno de sucesso ocorre após essa publicação.

As regressões cobrem falhas de abertura nos quatro arquivos, escrita parcial,
flush, fechamento, fsync e renomeação final. Também verificam a ordem de
publicação e a rejeição de inconsistências antes de criar o diretório.
Em falhas de persistência injetadas, não há `metadata.json` final e o erro
propaga com indicação de execução incompleta.

Uma sondagem adicional pela CLI bloqueou a abertura exclusiva de `results.jsonl`:
a saída foi **2**, restou somente `config.yaml`, não surgiu `metadata.json` e o
diagnóstico informou a persistência incompleta. O erro foi intencional e não
representa falha da suíte.

**Limite:** a presença dos metadados finais é a convenção de conclusão da
persistência; consumidores devem respeitá-la e consultar o status da avaliação.
Ela não certifica ausência de erros por item, nem protege contra alterações
manuais posteriores ou perda de energia.

### F2 — Encerramento do backend: resolvido nos cenários testados

O runner captura exceções ordinárias de `backend.close()` e registra
`close_error`. Quando a avaliação produziu registros, segue para persistência;
uma falha de encerramento impede o status nominal `completed`.
Falhas primárias de inicialização ou interrupção continuam sendo propagadas,
com o erro secundário de encerramento anexado como nota.
Se a escrita falha depois, o erro de persistência também mantém notas dos erros
anteriores de encerramento e dos itens.

Passaram as regressões de encerramento isolado, item + encerramento, carga +
encerramento, interrupção + encerramento, escrita + erros anteriores e
diagnósticos da CLI. A CLI retorna **1** para a execução persistida com erro de
encerramento e apresenta o diagnóstico.

Uma sondagem adicional combinou falha na questão de três alternativas e erro
de encerramento: os três registros foram persistidos, com 2 válidos e 1 falho,
status `completed_with_errors`, erro original do item e `close_error` presentes.
Não houve perda silenciosa de resultados calculados.

### F3 — Validação dos contratos: resolvido nos cenários testados

| Contrato/limite | Invariantes verificadas |
| --- | --- |
| `ScoringResult` | Tipos e valores finitos, cardinalidades, rótulos únicos, probabilidades compatíveis com softmax dos escores, predição/empate coerentes e método/diagnósticos suportados |
| `EvaluationRecord` | Tipos aninhados, formato de sucesso ou erro, acerto compatível com o gabarito e entropia compatível com a distribuição |
| `RunMetadata` | Identificação, contagens, status e `close_error` coerentes, horários com fuso e ordem temporal, proveniência e campos de ambiente |
| Limite de persistência | Revalidação dos registros e dicionários mutáveis, IDs e ordem, resumo, configuração, identidades e template coerentes |

Os três contraexemplos centrais do relatório anterior foram novamente tentados:
escores iguais com probabilidades `(0.9, 0.1)`; distribuição uniforme com
entropia zero; e status `completed` com zero válidos e três falhas.
Todos foram rejeitados com `ValueError` explicativo.

As fórmulas permanecem compartilhadas em `numerics.py`; os schemas reutilizam
essas funções e a validação agregada reutiliza `metrics.summarize`.
Não foi encontrada duplicação de fórmulas nos schemas ou no storage.

## 5. Arquitetura, protocolo e ausência de regressões observadas

| Componente | Responsabilidade e conformidade observada |
| --- | --- |
| `config.py` e `cli.py` | Configuração declarativa validada, comando de execução e códigos de saída coerentes |
| `schemas.py` | Contratos comuns e invariantes entre etapas |
| `datasets.py` | Fixture sintética, IDs estáveis e seleção determinística; módulo simples permitido na fundação |
| `prompts.py` | Template identificado e ordem explícita das alternativas |
| `backends/` | Contrato de ciclo de vida/capacidades e backend exclusivamente simulado |
| `scoring/` e `numerics.py` | Pontuação e normalização separadas da inferência, funções numéricas puras compartilhadas |
| `metrics.py` | Acurácia e entropia, agregação somente de registros válidos e contagens explícitas |
| `runner.py` | Orquestração sequencial, coleta de registros e tratamento de falhas |
| `storage.py` | Validação do conjunto, persistência por execução e proveniência |

A suíte completa preserva as verificações de seleção, prompts, empates,
estabilidade do softmax, não sobrescrita, falhas parciais, denominadores e
métricas ausentes quando não há itens válidos. A comparação dos artefatos
nominais e do pacote instalado complementa esses testes; não foi observada
regressão nos cenários exercitados.

A separação permite adicionar adaptadores, backends e métricas em suas etapas,
sem introduzir agora um sistema de plugins ou abstrações adicionais.
Configuração e schemas continuam deliberadamente restritos à simulação;
extensibilidade não significa aceitar métodos reais ainda não validados.

O significado científico foi preservado: entropia de Shannon normalizada por
`ln(K)` sobre alternativas renormalizadas, acurácia sobre itens válidos,
sem índice composto ou alegação de incerteza sobre todo o vocabulário.
Não há inferência real, validação de tokenização ou execução MMLU nesta entrega.
`AGENTS.md` permanece na raiz e as referências ao roadmap apontam para
`docs/ROADMAP.md`.

## 6. Rastreabilidade, limitações e pendências

Configuração resolvida, seed, IDs selecionados, prompts, escores, probabilidades,
gabaritos, erros, resumo e versões permitem auditar as execuções observadas.
Há, contudo, limites explícitos:

- **Reprodução do código:** a execução por fonte registra HEAD
  `090a6452db5ac8346753104e4c18585e336ea7e7` e `dirty=true`.
  Esse commit sozinho não contém a implementação local ainda não versionada.
  Os metadados não armazenam um snapshot dessas alterações. A instalação fora
  do checkout admite commit e estado Git indisponíveis.
- **Persistência:** pastas parciais permanecem para inspeção; não há retomada,
  limpeza automática ou garantia de recuperação após queda de energia ou
  interrupção abrupta. `fsync` dos arquivos e publicação final não constituem
  garantia de durabilidade do diretório frente a falhas físicas.
- **Inicialização e interrupção:** falhas antes de completar a avaliação podem
  terminar sem artefatos. A preservação da causa primária foi testada; recuperação
  de registros após interrupção abrupta não foi implementada nem aprovada.
- **Abrangência dos testes:** falhas de I/O foram simuladas, não provocadas em
  hardware. Não foi medida cobertura percentual. Python 3.11, outros sistemas,
  instalação limpa e diferentes versões de dependências não foram exercitados.
  As dependências têm intervalos de versão; não há ambiente totalmente fixado.
- **Abrangência científica:** a fixture tem três questões e backend determinístico.
  Os resultados não validam logits, tokens, chat templates, MLX-LM, MMLU ou
  desempenho científico de qualquer modelo.

Não foi identificada correção adicional obrigatória de F1/F2/F3 no escopo
exercitado. A pendência da Entrega 1 é a **revisão deste relatório e aprovação
explícita do responsável**, incluindo o aceite dos limites descritos.
Antes de uma futura integração, recomenda-se preservar uma versão identificável
da fundação aprovada e seguir os diagnósticos já previstos na Entrega 2,
somente após autorização. Isso não exige publicação no GitHub nesta revisão.

## 7. Evidências locais e próximo passo

Os artefatos das duas simulações estão nas pastas de `outputs/` identificadas na
seção 3. Evidências complementares estão em
`outputs/validation-final-fer8wo7c/`:

- `commands.json` e quatro arquivos `.log`: comandos, saídas e códigos de retorno.
- `checks.json`: recálculo independente, comparação de artefatos e sondagens F1/F2/F3.
- `manifest.json`: hashes SHA-256 dos arquivos preservados durante a revisão.

Esses arquivos são locais e ignorados pelo Git; os resultados relevantes foram
transcritos neste relatório para que seu parecer não dependa da publicação deles.

**Próximo passo:** o responsável revisa este relatório e decide o aceite da
fundação. O roadmap permanece com **Entrega 1 aguardando revisão e aprovação**
e **Entrega 2 pendente, não iniciada**.
