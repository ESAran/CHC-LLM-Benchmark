# Revisão técnica — Entrega 1: Fundação

**Data:** 2026-10-20.  
**Parecer atual:** F1/F2/F3 corrigidos e revalidados; critérios técnicos da fundação atendidos, aguardando aprovação do responsável.  
**Histórico:** as seções 1–6 preservam a revisão inicial, anterior às correções. A seção 7 registra a implementação autorizada e sua revalidação. Não houve integração MLX nem envio ao GitHub.

Referências: [README](../../README.md), [arquitetura](../ARCHITECTURE.md),
[roadmap](../ROADMAP.md) e [instruções do projeto](../../AGENTS.md).

## 1. Aderência e responsabilidades

| Módulo | Responsabilidade implementada | Avaliação |
| --- | --- | --- |
| `schemas.py` | `Question`, `ScoringResult`, `EvaluationRecord`, `RunMetadata` e validação de probabilidades. | Contratos mínimos presentes; validação semântica parcial, detalhada em F3. |
| `config.py` | YAML seguro, rejeição de chaves duplicadas/desconhecidas, campos obrigatórios e configuração resolvida. | Adequado à fundação; aceita somente fixture e backend simulados. |
| `datasets.py` | Três questões originais, adaptação, IDs únicos, ordem das alternativas, seleção com seed. | Adequado; 2, 3 e 4 alternativas. O arquivo simples é permitido pela arquitetura. |
| `prompts.py` | Prompt exato e template `synthetic_multiple_choice_v1`. | Identificadores numéricos exclusivos da simulação; não valida rótulos de uma LLM. |
| `backends/base.py` e `simulated.py` | Protocolo com identidade, capacidades, carga, pontuação e encerramento; escore `float(i)`. | Backend não recebe gabarito nem calcula métricas. Interface deliberadamente limitada. |
| `scoring/` | Obtenção dos escores, validação de tamanho, softmax entre alternativas e predição. | Normalização independente do backend; método e unidade da simulação explícitos. |
| `metrics.py` | Argmax, desempate, entropia normalizada, acurácia e resumo. | Funções puras, sem bibliotecas de inferência ou acesso a arquivos. |
| `runner.py` | Coordenação sequencial, uma carga por execução, registros de falhas por item e metadados. | Fluxo nominal correto; encerramento pode impedir persistência, conforme F2. |
| `storage.py` | JSON/YAML/JSONL, diretório exclusivo, metadados de código e ambiente. | Sem sobrescrita pelo programa; gravação não atômica, conforme F1. |
| `cli.py` | Configuração, chamada ao runner, apresentação do resumo e códigos de saída. | Testados sucesso (0), falhas por item (1) e configuração inexistente (2). |

Não há MLX, Transformers, APIs, adaptador MMLU, pontuação de sequências ou índice
composto. A ausência é coerente com a Entrega 1.

Os módulos permitem evolução incremental, mas novos componentes não podem ser
ativados apenas por YAML: dataset, backend, método, template e lista de métricas
estão fixados explicitamente. Será necessário ampliar configuração e orquestração
quando cada etapa for autorizada. A separação atual permite conservar as funções
matemáticas e de armazenamento. Não é necessário antecipar um sistema de plugins.

O protocolo de backend ainda não prepara chat templates nem inspeciona tokens;
`ScoringResult` usa valores padrão específicos da simulação. Eles não devem ser
reutilizados em inferência real sem revisão explícita do contrato.

## 2. Testes executados

Ambiente observado: Python 3.12.2, PyYAML 6.0.2, pacote 0.1.0, Darwin 24.6.0,
arm64. A instalação já existente foi utilizada; não houve reinstalação nem
validação de instalação limpa nesta revisão.

Na raiz do repositório:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
```

**Resultado:** 31 testes, `OK`, em cada modo. Duração informada pelo unittest:
1,405 s sobre fonte e 2,063 s sobre instalação. São os mesmos 31 casos, não 62
testes distintos. Uma execução adicional com
`PYTHONPATH=src python3 -m unittest discover -s tests -v` também passou
(31 testes, 0,129 s). Os arquivos `.py` instalados foram comparados byte a byte
com `src/benchmark/` e coincidiram.

| Arquivo | Casos | Cobertura observada |
| --- | ---: | --- |
| `test_metrics.py` | 7 | Uniformes, determinísticas, referências intermediárias, entradas inválidas, argmax, acurácia e resumo vazio. |
| `test_scoring.py` | 6 | Softmax de referência, grandes magnitudes, entradas inválidas, ciclo de vida simulado, quantidade de escores e empate. |
| `test_schemas_datasets.py` | 8 | Validação dos contratos, adaptação, duplicatas, ordem, seleção determinística, prompt e metadados. |
| `test_config.py` | 3 | Exemplo/round-trip, campos inválidos, bloqueio de etapas posteriores e chaves YAML duplicadas. |
| `test_runner_storage.py` | 7 | Fluxo completo, repetibilidade, não sobrescrita, falhas parciais/totais, incompatibilidade, limpeza após falha de carga e códigos da CLI. |

Não foi medida cobertura percentual. A suíte não exercita falhas de escrita nem
de `close()`. Os diagnósticos adicionais da seção 5 não foram adicionados à
suíte permanente, respeitando o escopo de revisão.

## 3. Simulações e conferência matemática

```bash
PYTHONPATH=src .venv/bin/python -m benchmark.cli --config configs/synthetic.yaml
.venv/bin/llm-benchmark --config configs/synthetic.yaml
```

Ambos os comandos retornaram 0. Execuções locais:

- Fonte: `20261020T175338332611Z_62f2cb4eeff4477296cac647fb5ecd75`.
- Instalação: `20261020T175340167584Z_78144fecf4a04c21ba24a460c4873d93`.

O fluxo percorreu configuração → dataset → prompt → backend → scoring →
métricas → arquivos. As pastas em `outputs/` contêm `config.yaml`,
`metadata.json`, `results.jsonl` e `summary.json`. O resumo das duas execuções foi:

```json
{
  "total_count": 3,
  "valid_count": 3,
  "failed_count": 0,
  "accuracy": 0.6666666666666666,
  "mean_normalized_entropy": 0.7603746960459724
}
```

Os registros da execução por fonte foram recalculados independentemente com
`decimal`, precisão de 60 dígitos: `p_i = exp(s_i) / sum(exp(s_j))` e
`H_norm = -sum(p_i * ln(p_i)) / ln(K)`. Não foram chamadas as funções
de normalização ou entropia do benchmark nesse recálculo. Probabilidades,
entropias individuais e média coincidiram com erro absoluto menor que `1e-14`;
predições e indicadores de acerto também coincidiram.

| Ordem registrada | K | Acerto | Entropia de referência, arredondada apenas nesta tabela |
| --- | ---: | --- | ---: |
| `synthetic-002` | 3 | Não | 0.7576791106615823 |
| `synthetic-001` | 2 | Sim | 0.8399415379831692 |
| `synthetic-003` | 4 | Sim | 0.6835034394931654 |

Acurácia é a proporção de acertos entre itens válidos. A entropia usa a fórmula
de Shannon dividida por `log2(K)` e omite termos com probabilidade zero. O softmax
subtrai o máximo antes da exponenciação. Empates usam o menor índice. Esses
comportamentos correspondem à metodologia documentada.

Os testes incluem `H_norm(1/2, 1/4, 1/8, 1/8) = 0.875`, uniformes com diferentes
K, distribuições determinísticas, NaN/infinito, negativos e soma inválida.
Nenhuma evidência indica erro nas fórmulas ou no fluxo nominal. Os números acima
são resultados de software com escores controlados, sem interpretação científica
sobre uma LLM.

## 4. Rastreabilidade e limites de reprodução

Verificados: configuração completa, seed, revisão e partição sintéticas,
IDs na ordem selecionada, alternativas e gabaritos, prompt, método/unidade,
escores brutos, probabilidades, predições, métricas, erros, horários, ambiente
e contagens. A massa original de probabilidade dos rótulos fica nula, corretamente,
pois não há distribuição de vocabulário.

As repetições da suíte conferem igualdade dos registros e resumos para a mesma
configuração e impedimento de sobrescrita de diretórios existentes. JSON preserva
os floats sem arredondamento de apresentação. Isso comprova reprodução no ambiente
atual, não em todas as versões de Python e dependências.

A execução por fonte registrou HEAD
`090a6452db5ac8346753104e4c18585e336ea7e7` e `dirty=true`. A implementação ainda
contém arquivos não versionados; esse commit não representa sozinho o código
executado. `dirty=true` identifica a limitação, mas não guarda o conteúdo alterado.
Na instalação comum, commit e estado local ficam nulos, como descrito no README.

Foi guardado um manifesto SHA-256 dos fontes, testes, configuração,
`pyproject.toml` e `.gitignore` em
`outputs/review-foundation-hgimvjo1/evidence.json`. Ele identifica os conteúdos
auditados, mas não substitui seu arquivamento. O arquivo também contém o recálculo
e os resultados dos diagnósticos. Esses artefatos são locais e ignorados pelo Git.
Nenhum commit ou envio remoto foi feito.

Dependências têm intervalo de versões, sem lock; o ambiente virtual reutiliza
pacotes do sistema. A reprodução em instalação limpa e em Python 3.11 não foi
testada. A proteção contra sobrescrita é da aplicação, não um bloqueio de edição
manual dos arquivos.

## 5. Problemas e limitações encontrados

### F1 — Prioridade média: gravação incompleta pode parecer concluída

**Local:** `storage.py`, linhas 64–73.

`metadata.json` com status final e `summary.json` são gravados antes de
`results.jsonl`. Ao simular `OSError` na abertura deste último, a execução
propagou o erro, mas deixou `config.yaml`, `metadata.json` e `summary.json`,
com `status=completed` e `valid_count=3`, sem nenhum arquivo de resultados.
Uma auditoria posterior que leia somente os metadados pode confundir a pasta
incompleta com uma execução íntegra.

**Reprodução realizada:** usar `unittest.mock.patch.object(Path, "open", ...)`
para lançar `OSError` apenas quando `path.name == "results.jsonl"` e o modo
for `"x"`; chamar `run()` com saída em diretório temporário; capturar a exceção
e inspecionar os arquivos restantes. Não houve falha real do disco.

**Recomendação:** sinalizar conclusão somente depois de todos os artefatos
estarem persistidos; definir comportamento explícito para gravações parciais e
adicionar teste de regressão. Correção não realizada.

### F2 — Prioridade média: erro no encerramento perde resultados calculados

**Local:** `runner.py`, linhas 53–55 e 85–86.

`backend.close()` executa antes da persistência. Uma exceção nesse método
interrompe o fluxo mesmo que todas as questões já tenham sido avaliadas.
O diagnóstico pontuou os três itens e depois lançou `RuntimeError` em `close()`:
o diretório de saída sequer foi criado. O problema não ocorre com o backend
padrão atual, cujo encerramento só altera um booleano, mas afeta o contrato
de backend já exposto e a garantia de rastreabilidade em falhas.

**Reprodução realizada:** subclassificar `SimulatedBackend`, manter identidade,
capacidades e pontuação; contar chamadas a `score()` e lançar `RuntimeError`
depois de `super().close()`. Chamar `run()` com essa instância e saída temporária.
Resultado observado: três chamadas de pontuação, exceção e nenhuma pasta.

**Recomendação:** preservar resultados já calculados e registrar separadamente
o erro de encerramento, sem mascarar falhas anteriores; incluir regressão.
Correção não realizada.

### F3 — Validação parcial dos contratos

Construções diretas aceitaram:

- `ScoringResult` com escores `(0, 0)` e probabilidades `(0.9, 0.1)`, embora
  declare normalização por softmax.
- `EvaluationRecord` com essa distribuição e entropia `0`.
- `RunMetadata` com uma falha, zero itens válidos e `status="completed"`;
  horários como `"start"` e `"end"` e metadados de ambiente vazios também passam.

O runner nominal não produziu essas inconsistências. Os contratos verificam
algumas invariantes, mas não certificam integralmente a semântica dos registros.
É necessário decidir se essa garantia cabe aos produtores, a uma validação de
integração ou aos schemas. Recalcular métricas no storage não é recomendado.
Não é necessário duplicar fórmulas nos contratos para resolver essa responsabilidade.

### Limites documentais e de escopo

- Links do README para a antiga arquitetura na raiz e links relativos incorretos
  do roadmap foram corrigidos nesta revisão.
- `AGENTS.md` e a árvore direcional da arquitetura ainda citam `ROADMAP.md` na
  raiz; o arquivo real está em `docs/ROADMAP.md`. Registrado para alinhamento,
  sem alterar instruções permanentes ou decisões arquiteturais nesta tarefa.
- A seção geral de testes da arquitetura menciona exemplos MMLU e tokenização
  simulada; esses testes não existem. A seção específica da Entrega 1 permite
  adaptador sintético, e o protocolo real ainda está pendente. Não considerar
  tokenização ou adaptação MMLU validadas.
- Falhas de inicialização não geram artefatos, conforme o README. Não houve
  teste de queda do processo, interrupção abrupta ou recuperação de execução.
- A suíte passa, mas não esgota os contratos e os caminhos de falha. Seu sucesso
  não elimina F1/F2 nem constitui aprovação do protocolo com modelo real.

## 6. Pendências e recomendação antes do MLX-LM

1. O responsável deve revisar este parecer e decidir sobre F1/F2; recomenda-se
   corrigi-los na fundação antes do aceite e cobri-los com testes de regressão.
2. Definir o nível de garantia dos contratos observado em F3 e alinhar as
   referências documentais ao roadmap.
3. Validar eventuais correções e aprovar explicitamente a Entrega 1.
4. Antes de usar resultados reais, preservar uma versão exata do código e do
   ambiente; isso pode ser feito localmente, sem publicar no GitHub.
5. Somente após autorização da Entrega 2, revisar o contrato de entrada final,
   template de chat, tokenização contextual, posição dos logits, identidade e
   revisão do modelo. Diagnosticar tokens distintos e unitários e consumo de
   memória conforme a arquitetura. Não presumir que os identificadores e escores
   sintéticos demonstram viabilidade de `label_next_token`.

**Próximo passo:** revisão e decisão do responsável sobre este relatório.
A Entrega 1 permanece em revisão; a Entrega 2 não foi iniciada.

## 7. Correções e revalidação de F1/F2/F3

**Data:** 2026-10-20. Correções autorizadas pelo responsável, limitadas à Entrega 1.

### F1 — Persistência

`save_run()` valida e serializa o conjunto antes de criar a pasta, que continua
exclusiva. Grava `config.yaml`, `results.jsonl` e `summary.json`; cada arquivo é
fechado após `flush` e `fsync`. Em seguida grava e sincroniza `metadata.json.tmp`
e o renomeia atomicamente para `metadata.json`, no mesmo diretório.

A presença de **`metadata.json` final é o sinal de persistência terminada**.
Uma falha anterior deixa a pasta sem esse arquivo, propaga a exceção e informa
que a execução está incompleta. Os arquivos parciais são preservados para
inspeção; não são retomados ou sobrescritos. O temporário nunca é tratado como
metadado final. O campo `status` do metadado publicado descreve o resultado da
avaliação; uma avaliação com erros também pode ter persistência terminada.

As regressões cobrem falhas de abertura, escrita parcial, flush, fechamento,
sincronização dos quatro arquivos e publicação final. Também verificam que a
publicação acontece depois dos demais artefatos e que um conjunto inconsistente
é rejeitado antes de criar a pasta.

### F2 — Encerramento do backend

Uma falha em `close()` após a avaliação é registrada em `RunMetadata.close_error`.
Os registros já calculados continuam até a persistência. O status fica
`completed_with_errors` quando há itens válidos e `failed` quando todos falharam.
A CLI informa o erro em stderr e retorna 1; as métricas não são alteradas pelo
erro operacional.

Quando a inicialização ou uma interrupção já causou uma exceção, essa mesma
exceção é preservada. A falha secundária em `close()` é anexada como nota.
Se a escrita falhar depois de erros por item ou de encerramento, o diagnóstico
da escrita inclui esses erros anteriores, sem afirmar que houve persistência.
A CLI também exibe as notas das exceções tratadas.

As regressões verificam sucesso dos itens + falha de encerramento, falhas dos
itens + encerramento, carga + encerramento, interrupção + encerramento e
escrita + erros anteriores, além dos códigos e mensagens da CLI.

### F3 — Invariantes e limites entre componentes

- `ScoringResult`: tuplas, rótulos textuais únicos, números finitos sem booleanos,
  probabilidades coerentes com softmax dos escores, argmax e desempate corretos,
  método/unidade/diagnósticos correspondentes à simulação suportada.
- `EvaluationRecord`: tipos dos objetos, prompt/template, forma exclusiva de
  sucesso ou falha, acerto coerente com gabarito e entropia coerente com a
  distribuição.
- `RunMetadata`: IDs/contagens, status coerente com erros, horários ISO 8601
  com fuso e ordenados, identidade/revisão, campos explícitos de código,
  ambiente e dependências.
- Antes da escrita: revalidação dos registros e metadados, inclusive dicionários
  mutáveis; IDs em ordem, contagens, resumo e configuração devem concordar.

`numerics.py` reúne as funções puras compartilhadas. As interfaces existentes de
`metrics.py` e `scoring/normalization.py` são preservadas. Os schemas reutilizam
as funções, e o resumo é conferido pela mesma implementação de `metrics.py`.
O storage delega a validação sem conter fórmulas. As comparações numéricas usam
tolerância absoluta de `1e-12`, sem modificar os dados armazenados.

### Arquivos modificados

| Grupo | Arquivos |
| --- | --- |
| Implementação | `src/benchmark/storage.py`, `runner.py`, `cli.py`, `schemas.py`, `metrics.py`, `scoring/normalization.py` |
| Novo núcleo numérico puro | `src/benchmark/numerics.py` |
| Testes ampliados | `tests/test_runner_storage.py`, `tests/test_schemas_datasets.py` |
| Novos testes | `tests/test_storage_failures.py`, `tests/test_contract_invariants.py` |
| Documentação | `AGENTS.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md` e este relatório |

`AGENTS.md` permanece na raiz. Referências ao roadmap agora apontam para
`docs/ROADMAP.md`; links ao relatório respeitam sua localização em `docs/reports/`.

### Evidências após as correções

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src python3 -m benchmark.cli --config configs/synthetic.yaml
```

**Suíte:** 54 testes, `OK` (23 novos casos além dos 31 originais); 0,633 s no
unittest desta execução. Vários casos exercitam múltiplas falhas por subtestes.
**Simulação:** saída 0; execução
`20261020T181422276767Z_2322bdf97acc433fa8f8226078c0e00f`.
Resultado: 3 válidas, 0 falhas, acurácia `0.6666666666666666` e entropia média
`0.7603746960459724`. O protocolo científico e o resultado nominal foram preservados.

Também foram conferidos os quatro arquivos finais, ausência do temporário,
status/contagens/IDs e `close_error=null`. O recálculo independente com `decimal`
(precisão 60) confirmou probabilidades, entropias individuais e média com
diferença absoluta menor que `1e-14`.

A primeira tentativa de reinstalação pelo `pip` da `.venv` ficou parada durante
a importação de seus módulos e foi interrompida antes de instalar. A alternativa
abaixo concluiu a atualização no mesmo ambiente, sem downloads:

```bash
python3 -m pip --python .venv/bin/python install --no-index --no-deps --no-build-isolation .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/llm-benchmark --config configs/synthetic.yaml
```

**Pacote atualizado:** os mesmos 54 testes passaram (`OK`, 0,186 s). A CLI
retornou 0 e gerou a execução
`20261020T181738149696Z_c926eeec1a7f486fb1edbc8e05d18e91`, com os mesmos
valores do resumo. Isso valida a instalação atualizada; não é uma prova de
instalação limpa em outra máquina ou versão de Python.

### Limitações e estado da entrega

F1/F2/F3 estão resolvidos nos cenários testados. Permanecem os limites da
fundação: sem recuperação de queda de energia/interrupção abrupta, sem retomada
de pastas parciais, sem snapshot do código local e sem garantia de reprodução
entre versões distintas do ambiente. Se o dispositivo impedir gravar, a
persistência não pode ser garantida; a operação retorna erro e não publica
metadados finais. Falhas de inicialização continuam sem artefatos de avaliação,
mas a causa original e erros secundários são mantidos visíveis.

Os critérios técnicos de conclusão do roadmap foram atendidos. **A Entrega 1
permanece aguardando aprovação do responsável, não marcada como concluída.**
O próximo passo é a revisão deste resultado; a Entrega 2 não foi iniciada.
