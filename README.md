# LLM Benchmark — Desempenho e Incerteza

> **Estado:** Entregas 1, 2, 3A e 3B concluídas e aprovadas. **3C implementada e verificada, aguardando revisão técnica:** análise dos 60 test persistidos, quatro gráficos PNG e JSON auditável; 117 testes offline aprovados. Nenhuma nova inferência. Veja a [análise exploratória](docs/reports/MMLU_EXPLORATORY_ANALYSIS.md) e o [roadmap](docs/ROADMAP.md).

## Visão geral

Este repositório reúne o desenvolvimento de um **benchmark experimental em Python para avaliação de Large Language Models (LLMs)** em tarefas de múltipla escolha. A investigação analisa conjuntamente:

- **Acurácia:** proporção de questões em que a alternativa de maior pontuação corresponde ao gabarito.
- **Entropia normalizada de Shannon:** incerteza da distribuição de probabilidades **entre as alternativas da questão**, após normalização dentro desse conjunto.

A proposta é investigar relações entre desempenho, acertos, erros e incerteza, sem pressupor que menor entropia implica maior acurácia ou maior inteligência. O estudo é inspirado exploratoriamente em dimensões do modelo Cattell–Horn–Carroll (CHC), **sem reivindicar uma mensuração psicométrica de inteligência**.

O projeto não pretende, nesta etapa, formular um índice composto de “eficiência informacional”. A análise inicial mantém acurácia e entropia como medidas distintas.

## Escopo do MVP

A primeira versão será implementada **no repositório principal do projeto**, de forma incremental e extensível:

| Item | Escolha inicial proposta |
| --- | --- |
| Linguagem | Python |
| Dataset | `cais/mmlu`, revisão fixa, três disciplinas; `dev` técnico e `test` reservado ao piloto posterior |
| Questões | 5 `dev` + 20 `test` por disciplina: 15 + 60, sem execução experimental em 3A |
| Modelo de desenvolvimento | Qwen2.5-0.5B-Instruct em variante compatível com MLX-LM (candidata) |
| Inferência | Local, utilizando MLX-LM em Apple Silicon |
| Pontuação das alternativas | Rótulos de resposta (`A`, `B`, `C`, `D`), **condicionada à validação da tokenização e dos logits** |
| Métricas | Acurácia e entropia normalizada |
| Saída | Resultados por questão, resumo e metadados por execução |

O equipamento de referência para a fase inicial é um **Mac com Apple M2 e 8 GB de memória unificada**. O tamanho do modelo, o uso de memória e a forma exata de pontuação serão confirmados com testes antes de considerar o piloto validado.

O piloto de desenvolvimento **não será apresentado como avaliação científica conclusiva**. O subconjunto de `test` preparado em 3A fica reservado ao piloto posterior, sem ajustar o protocolo a partir dos seus resultados. O pedido de 3A substitui explicitamente a proposta anterior de até 20 questões de `validation`.

## Arquitetura

O sistema separa sete responsabilidades:

1. **Datasets:** carregar, selecionar e adaptar questões para um formato interno comum.
2. **Prompts:** produzir a entrada do modelo, preservando o formato e a versão utilizados.
3. **Backends:** adaptar bibliotecas de inferência (inicialmente MLX-LM; no futuro, Transformers ou outras tecnologias compatíveis).
4. **Scoring:** calcular pontuações das alternativas e transformá-las numa distribuição normalizada.
5. **Metrics:** calcular medidas por questão e agregadas, de maneira independente do modelo.
6. **Runner:** orquestrar uma execução configurada.
7. **Storage:** registrar resultados, configurações e metadados de reprodutibilidade.

O formato compartilhado das questões e resultados será definido por estruturas tipadas (`schemas`). As configurações dos experimentos ficarão fora do código, preferencialmente em YAML.

**Detalhamento técnico:** consulte [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Métricas iniciais

Para uma questão com `K` alternativas (`K >= 2`), `p_i` representa a probabilidade **renormalizada entre essas alternativas**, com `Σ p_i = 1`:

\[
H_{norm} = \frac{-\sum_{i=1}^{K} p_i \log_2(p_i)}{\log_2(K)}
\]

Por convenção, `0 · log₂(0) = 0`. A entropia normalizada pertence ao intervalo `[0, 1]` quando a distribuição é válida: próximo de `0` indica maior concentração em uma alternativa; próximo de `1`, maior uniformidade entre alternativas.

A predição é o índice da alternativa com maior probabilidade. A acurácia de uma execução é a média dos indicadores de acerto (`0` ou `1`) das questões avaliadas.

**Limitação de interpretação:** essa é a entropia da **distribuição condicionada ao conjunto de alternativas pontuadas**, não a entropia de todo o vocabulário da LLM nem uma entropia semântica de respostas abertas. Diferentes estratégias de pontuação podem produzir distribuições distintas e precisam ser identificadas nos resultados.

## Estrutura prevista

```text
.
├── configs/
│   └── pilot_mmlu.yaml
├── docs/
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
├── src/
│   └── benchmark/
│       ├── cli.py
│       ├── runner.py
│       ├── schemas.py
│       ├── datasets/
│       │   ├── base.py
│       │   └── mmlu.py
│       ├── backends/
│       │   ├── base.py
│       │   └── mlx.py
│       ├── scoring/
│       │   ├── labels.py
│       │   └── normalization.py
│       ├── prompts.py
│       ├── metrics.py
│       └── storage.py
├── tests/
├── outputs/              # gerados durante a execução; não versionar por padrão
├── pyproject.toml
├── README.md
└── .gitignore
```

Esta árvore representa a **arquitetura desejada**, não uma lista de arquivos já implementados. Os módulos devem ser introduzidos conforme necessários, mantendo o projeto pequeno e compreensível.

## Etapas de desenvolvimento

1. Criar a estrutura, os formatos de dados, a configuração e testes com um **backend simulado**.
2. Implementar e testar matematicamente a normalização, a entropia e a acurácia.
3. Integrar MLX-LM e realizar **diagnóstico de tokenização, formato de prompt e acesso aos logits**.
4. Preparar a seleção MMLU em 3A; após revisão e autorização de 3B, executar a validação técnica em `dev` e o piloto reservado em `test`.
5. Registrar resultados por questão, configurações e versões, analisar limitações e evoluir gradualmente.

O backend Transformers, a execução em servidores, novas métricas e novos datasets pertencem a fases futuras; **não fazem parte do MVP**.

## Execução e instalação

### Entrega 1 implementada

O pacote em `src/benchmark/` contém contratos em dataclasses, adaptador sintético,
prompt versionado, contrato de backend, backend simulado, softmax estável, métricas
puras, configuração YAML estrita, runner sequencial, armazenamento e CLI. As fórmulas
compartilhadas ficam em `numerics.py` e são reutilizadas na validação dos contratos. O adaptador
sintético fica em `datasets.py`; ainda não há necessidade de um pacote de adaptadores.
A configuração executável é [`configs/synthetic.yaml`](configs/synthetic.yaml).
A árvore da seção anterior continua sendo a estrutura prevista para as etapas futuras.

O exemplo contém três questões originais, com 2, 3 e 4 alternativas. O backend
`controlled_scores`, revisão `v1`, atribui o escore `float(i)` à alternativa de índice
`i`, sem consultar o gabarito. Isso testa o fluxo; **não representa respostas de uma
LLM nem resultados do MMLU**. Não há download de datasets ou pesos.

### Comandos verificados

Requer Python 3.11 ou superior. A validação local usou Python 3.12.2, PyYAML 6.0.2,
setuptools 69.2.0 e wheel 0.45.1, já disponíveis no ambiente. Os comandos abaixo
foram executados na raiz do repositório. O ambiente virtual reutiliza essas
dependências instaladas; `--no-deps --no-build-isolation` pressupõe sua disponibilidade.
`--no-index` impede consultar um índice de pacotes.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-index --no-deps --no-build-isolation .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/llm-benchmark --config configs/synthetic.yaml
```

Na revalidação das correções, o `pip` do ambiente virtual ficou parado ao importar
seus próprios módulos e foi interrompido antes da instalação. A atualização do
mesmo ambiente foi concluída sem downloads usando o `pip` do sistema:

```bash
python3 -m pip --python .venv/bin/python install --no-index --no-deps --no-build-isolation .
```

Também é possível executar o mesmo ponto de entrada como módulo:

```bash
.venv/bin/python -m benchmark.cli --config configs/synthetic.yaml
```

A instalação comum acima foi validada. A instalação editável (`-e .`) apresentou
um problema neste ambiente macOS: o arquivo `.pth` recebeu a marca de oculto e
foi ignorado pelo Python. Por isso, ela não é o caminho documentado de execução.
Após modificar o código, reinstale o pacote ou use diretamente o código-fonte:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m benchmark.cli --config configs/synthetic.yaml
```

O uso direto do código-fonte também permite identificar o checkout Git nos
metadados. Na instalação comum, o pacote em `site-packages` não contém o
checkout: commit e estado de alterações ficam nulos, sem atribuir ao pacote
instalado uma revisão não comprovada.

Cada execução cria uma nova pasta `outputs/<timestamp_UTC>_<uuid>/`, contendo:

- `config.yaml`: configuração completa utilizada.
- `metadata.json`: identificação da simulação, revisões, ordem dos IDs, semente,
  ambiente, versões, commit e estado de alterações locais, horários, contagens e
  eventual `close_error`.
- `results.jsonl`: questão e alternativas em ordem, gabarito, prompt exato,
  método, escores, probabilidades condicionais, predição, acerto, entropia e erro.
- `summary.json`: contagens, acurácia e média da entropia nas questões válidas.

`metadata.json` é publicado por último, após gravar, fechar e sincronizar os
demais arquivos. **Uma pasta sem esse arquivo final é incompleta**, mesmo que
contenha `summary.json` ou `metadata.json.tmp`. Em uma falha de escrita, a pasta
parcial é preservada para inspeção e a CLI retorna erro; execute novamente para
criar uma nova pasta. Não há retomada automática.

O exemplo completo produz 3 questões válidas, nenhuma falha e acurácia sintética
`2/3`. Esse valor decorre dos escores controlados e serve apenas como verificação
do software. Os testes cobrem referências matemáticas independentes, entradas
inválidas, empates, seleção reproduzível, falhas parciais/totais e persistência
sem perda de precisão nem sobrescrita.
As regressões de F1/F2/F3 também cobrem abertura, escrita parcial, flush,
fechamento, sincronização e publicação dos arquivos, encerramento do backend,
erros combinados e inconsistências entre contratos.

### Convenções e limites da simulação

- A configuração rejeita campos desconhecidos, ausentes, chaves YAML duplicadas
  e valores que ativariam funcionalidades posteriores. O limite da fixture é de
  1 a 3 questões. O diretório de saída relativo é resolvido a partir do diretório
  de trabalho do comando.
- A seleção embaralha a fixture com uma instância local de `random.Random(seed)`
  e aplica o limite. Os IDs e sua ordem são sempre registrados.
- O prompt `synthetic_multiple_choice_v1` usa identificadores numéricos para
  suportar diferentes quantidades de alternativas. O método `simulated_scores`
  registra a unidade `synthetic_logits`, sem tokenização ou template de chat.
  Isso não define nem valida o futuro protocolo de rótulos `A`–`D`.
- As probabilidades são condicionadas ao conjunto de alternativas. A validação
  exige valores finitos em `[0, 1]` e soma 1, com tolerância absoluta de `1e-12`.
  Empates de probabilidade usam o menor índice na ordem original.
- Falhas por questão são registradas com métricas nulas; os agregados usam
  somente as questões válidas e exibem o denominador em `valid_count`. Se nenhuma
  questão for válida, acurácia e média de entropia são nulas. Esse tratamento
  operacional não estabelece critérios de exclusão para o piloto científico.
- A CLI retorna `0` no sucesso, `1` quando há falhas por questão ou no encerramento
  do backend e `2` para erros
  de configuração, inicialização ou acesso aos arquivos. Erros de inicialização
  interrompem a execução antes da persistência de resultados; uma falha adicional
  de encerramento é informada sem substituir o erro original.
- Quando só o encerramento falha, os resultados calculados são preservados e
  `close_error` é registrado. O status é `completed_with_errors` se houver itens
  válidos, ou `failed` se todos os itens falharam. Esse erro operacional não muda
  as métricas calculadas por questão.
- A aplicação não reabre nem sobrescreve pastas de execuções anteriores. Sem
  um checkout Git identificável, commit e estado de alterações ficam nulos.
  Saídas, ambiente virtual, caches e pesos são ignorados pelo Git por padrão.
- O protocolo de gravação trata exceções de I/O, mas não garante recuperação
  após queda de energia ou interrupção abrupta do processo. Falhas que impeçam
  a própria escrita são reportadas como erros, sem prometer persistência.

Na Entrega 1 não havia backend real. A integração MLX da Entrega 2B está descrita
abaixo; backend Transformers, APIs e pontuação de sequências continuam fora do escopo. O adaptador MMLU preparado em 3A está conectado ao Runner em 3B, descrita abaixo. Na simulação, a massa original dos rótulos no vocabulário
permanece nula, pois o backend simulado não fornece essa evidência. A Entrega 1 valida
apenas contratos, matemática e fluxo; **o MVP com modelo real ainda não está validado**.

### Diagnóstico isolado da Entrega 2A

O script [`diagnostics/mlx_labels.py`](diagnostics/mlx_labels.py) executa a investigação
de MLX-LM em ambiente separado, com dependências fixadas em
[`diagnostics/mlx-requirements.txt`](diagnostics/mlx-requirements.txt). Não altera o
runner ou a configuração sintética. Procedimento, revisão do modelo, comandos,
resultados e limitações estão no [relatório de diagnóstico](docs/reports/MLX_DIAGNOSTIC.md).
Rótulos sem espaço foram viáveis no template testado; fronteiras de tokenização,
precisão BF16 e diferenças entre caminhos com/sem cache exigem tratamento explícito.
O protocolo inicial e o início de 2B foram aprovados pelo responsável em 2026-11-06;
a implementação descrita abaixo foi aprovada tecnicamente no fechamento da Entrega 2.

A [investigação comportamental complementar](docs/reports/MLX_BEHAVIOR_VALIDATION.md)
compara quatro perguntas, rotações de alternativas e instruções explícitas por
meio de `diagnostics/mlx_labels.py --behavior`. As variantes são diagnósticas;
naquele diagnóstico não houve adoção definitiva. A comparação posterior fundamentou
a aprovação explícita do protocolo provisório usado em 2B.

### Entrega 2B — execução real local, concluída e aprovada

O Runner seleciona o backend por `model.backend`. A configuração
[`configs/controlled_mlx.yaml`](configs/controlled_mlx.yaml) usa 16 apresentações
controladas (quatro questões × quatro rotações), sem MMLU. O template configurável
`current-no-enumeration-v1` preserva exatamente o texto aprovado em
[2A](docs/reports/PROMPT_COMPARISON.md); o backend aplica somente o chat template
do snapshot, sem conhecer a pergunta, o gabarito ou a instrução de múltipla escolha.

O protocolo provisório é zero-shot, candidatos `A`–`D` sem espaços, posição `T-1`,
batch 1 e forward completo **sem KV cache**. Logits BF16 são promovidos a float32
e convertidos a floats Python antes do softmax compartilhado. São persistidos
os escores, probabilidades condicionais, probabilidades originais, massa válida,
IDs contextuais, posição, precisão, revisão, hashes e template efetivo.
**Cache e forward completo podem produzir logits diferentes**; não são
intercambiáveis neste protocolo.

MLX é opcional: a instalação básica e os testes offline continuam exigindo apenas
as dependências da fundação. No Apple Silicon, use um ambiente separado com
as versões de [`diagnostics/mlx-requirements.txt`](diagnostics/mlx-requirements.txt).
A configuração exige um snapshot **já disponível localmente**; não baixa pesos.
O manifesto em `backends/mlx.py` verifica os arquivos do snapshot aprovado.
O limite configurável de entrada é de até 1024 tokens nesta etapa, com erro
explícito em vez de truncamento. Não foram validados prompts longos.

Com o ambiente MLX de 2A disponível, execute na raiz:

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=src \
.cache/mlx-diagnostic/venv/bin/python -m benchmark.cli --config configs/controlled_mlx.yaml
```

A validação desta entrega reutilizou o ambiente temporário de 2A e o mesmo
snapshot em `.cache/`; o comando exato está no relatório. `snapshot_path` e
`output.directory` são relativos ao diretório de trabalho quando não absolutos.
Use caminhos relativos em configurações compartilhadas, sem dados pessoais.

O backend inicial aceita apenas o snapshot Qwen aprovado. Novos modelos exigem
manifesto e validação próprios. Templates são selecionados pela configuração e
implementados em `prompts.py`, separados do backend; adicionar um template não
altera a inferência, mas uma nova convenção experimental requer aprovação.
Erros de resposta do modelo são registros válidos com `correct: false`; falhas de
execução/tokenização são erros explícitos com métricas ausentes. As regras de
persistência e encerramento da fundação continuam aplicadas.

Resultado controlado verificado: **16 válidas, 0 falhas operacionais, 11/16 acertos**;
não é um score MMLU nem evidência científica de generalização. Os 71 testes
incluem os 54 da fundação e 17 novos testes offline; veja
[VALIDATION_DELIVERY_2B.md](docs/reports/VALIDATION_DELIVERY_2B.md).

## Preparação MMLU — Entrega 3A

O adaptador independente `src/benchmark/mmlu.py` preserva texto, quatro alternativas,
gabarito, disciplina e posição original. `configs/mmlu_selection.json` fixa revisão,
seed 42, arquivos e hashes. O script `diagnostics/mmlu_dataset.py` prepara 15 questões
de `dev` e 60 de `test` e conta tokens com o tokenizer local aprovado em 2B.
O script de preparação não carrega pesos nem executa avaliação. O consumo pelo Runner foi acrescentado separadamente em 3B.

As dependências opcionais estão em `diagnostics/mmlu-requirements.txt`; use um
ambiente separado do núcleo. Com essas dependências instaladas e o snapshot local
já disponível, a preparação inicial pode ser reproduzida na raiz:

```sh
PYTHONPATH=src python diagnostics/mmlu_dataset.py --download \
  --tokenizer-snapshot .cache/mlx-diagnostic/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
  --output outputs/mmlu-preparation
```

`--download` autoriza apenas os seis Parquets fixados (377.382 bytes); o tokenizer
continua exclusivamente local. Para repetir, omita a opção, use `HF_HUB_OFFLINE=1`
e escolha outra pasta de saída. Nenhum arquivo existente será sobrescrito.
`selected_questions.jsonl` guarda conteúdo e procedência; `manifest.json`, publicado
por último, guarda seleção, hashes, contagem de tokens e duplicações. Pastas sem o
manifesto final não devem ser consumidas. Os artefatos de trabalho ficam ignorados.

A seleção verificada tem entradas de 72 a 574 tokens, nenhuma acima do limite de
1024. Há alternativas A/C idênticas em `high_school_biology/dev/2`; o conteúdo foi
preservado e sinalizado. Consulte o [relatório](docs/reports/MMLU_DATASET_VALIDATION.md)
e o [manifesto auditável](docs/reports/MMLU_SELECTION_MANIFEST.json). A 3A foi aprovada com essa anomalia preservada e uso exclusivamente técnico de dev.

## Piloto MMLU pelo Runner — Entrega 3B

`configs/mmlu_dev.yaml` e `configs/mmlu_test.yaml` usam `mmlu-3a-v1`, revisão e
índices congelados. `dataset.subjects` permite uma ou mais disciplinas na ordem
aprovada; `limit` deve ser exatamente 5 por disciplina em dev ou 20 em test.
O manifesto e todos os Parquets selecionados são verificados antes da carga do
modelo. Não há download automático nem seleção por resultado.

Use o ambiente MLX de 2B com as dependências MMLU de 3A e os snapshots locais:

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=src python -m benchmark.cli --config configs/mmlu_dev.yaml
```

Antes de executar test, audite a pasta retornada com
`diagnostics/mmlu_pilot_validation.py --run <pasta-dev> --output <evidencia-dev.json>`
usando o mesmo `PYTHONPATH=src`. A auditoria não exige acurácia mínima e retorna erro
se houver falha técnica ou persistência incompleta. Só após sucesso técnico executar
a CLI com `configs/mmlu_test.yaml`. A CLI individual não automatiza essa sequência.

Na validação de 3B, dev teve 15 válidas/0 falhas (6 acertos) e test teve 60 válidas/0
falhas (34 acertos, acurácia 56,6667%, entropia média 0,4785157438890066). Dev permaneceu
exclusivamente técnico; o item de biologia com A/C iguais foi preservado. Nenhum
parâmetro foi ajustado entre as fases. Os registros incluem procedência, prompt,
tokens, escores, probabilidades, massa original, predição, gabarito e entropia.
O [relatório](docs/reports/MMLU_PILOT_VALIDATION.md) e o
[JSON compacto](docs/reports/MMLU_PILOT_EVIDENCE.json) foram aprovados tecnicamente em 2026-11-08.

## Análise exploratória — Entrega 3C

O módulo `benchmark.analysis` calcula descritivas dos registros `test`, mantendo
os `dev` fora da análise experimental. `benchmark.results` reconstrói os schemas
e confere os artefatos; `benchmark.analysis_report` gera Markdown, JSON com todos
os valores e quatro PNGs. A entrada do relatório exige o piloto de 60 test e os
hashes aprovados; não depende de pesos, tokenizer, Parquets ou rede.

Com as dependências opcionais de `diagnostics/analysis-requirements.txt` disponíveis:

```sh
XDG_CACHE_HOME=/tmp/mmlu-analysis-cache MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mmlu-analysis-mpl \
PYTHONPYCACHEPREFIX=/tmp/mmlu-3c-pycache PYTHONPATH=src \
  .venv/bin/python -m benchmark.analysis_report \
  --date 2026-11-08 --output docs/reports
```

É necessário preservar as pastas de resultados identificadas no JSON de 3B.
O comando não executa a LLM nem modifica os artefatos originais. Agregações vazias
têm contagem zero e métricas nulas; dados dev/mistos e IDs duplicados são recusados.
A acurácia usa registros válidos, com falhas e denominadores explícitos.

O [relatório](docs/reports/MMLU_EXPLORATORY_ANALYSIS.md) mostra 34/60 acertos,
entropia média 0,346090 nos acertos e 0,651688 nos erros. Lista todos os quatro
erros com máx. p condicional ≥ 0,90 e 12 exemplos escolhidos por uma regra simétrica.
Isso não demonstra calibração, viés de posição ou diferenças cognitivas. Os valores
completos, regras e hashes estão no [JSON](docs/reports/MMLU_EXPLORATORY_EVIDENCE.json).
A 3C aguarda revisão técnica; commit e publicação na branch de feature autorizados em 2026-11-08.

## Reprodutibilidade e transparência

Cada execução deverá registrar, no mínimo:

- identificação e versão/revisão do dataset, partição e IDs das questões;
- identificação e revisão do modelo, backend e dependências;
- prompt efetivamente enviado (inclusive o template de chat, se usado);
- método de pontuação, rótulos, escores e probabilidades;
- alternativa prevista, gabarito, acerto/erro e entropia por questão;
- parâmetros de seleção, semente, configuração utilizada e versão do código;
- estatísticas agregadas e eventuais falhas ou questões descartadas.

Pesos dos modelos, caches, credenciais e resultados volumosos não devem ser enviados automaticamente ao repositório. A publicação de dados derivados deverá observar licenças e condições de uso dos datasets e modelos utilizados.

## Referências técnicas

- [MLX-LM](https://github.com/ml-explore/mlx-lm) — biblioteca de inferência para modelos de linguagem em MLX.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) — biblioteca alternativa para inferência, inicialmente fora do escopo.
- [LM Evaluation Harness (EleutherAI)](https://github.com/EleutherAI/lm-evaluation-harness) — referência arquitetural, sem intenção de reproduzir toda a sua complexidade.
- [Guia de tarefas do LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) — referência para configuração de tarefas de múltipla escolha.

## Status das decisões

- **Metodologia estabelecida:** múltipla escolha, acurácia por alternativa de maior probabilidade e entropia normalizada entre alternativas.
- **Plano técnico inicial:** Python, MLX-LM local, MMLU e um Qwen2.5 pequeno.
- **Protocolo provisório aprovado para 2B:** prompt sem enumeração, chat template do snapshot, zero-shot, rótulos nus e forward completo sem cache.
- **Concluído:** aceite técnico da integração 2B em 2026-11-06, com 71 testes novamente aprovados.
- **3A concluída e aprovada:** revisão, seed, índices, hashes e orçamento de tokens de 75 questões MMLU; anomalia A/C de biologia mantida por decisão explícita.
- **3B concluída e aprovada:** fases dev/test sem falhas técnicas, 100 testes novamente aprovados; fechamento em commit local exclusivo autorizado.
- **3C aguardando revisão:** análise dos 60 test, quatro PNGs, evidência completa e 117 testes aprovados; sem nova inferência; publicação autorizada na branch de feature, mantendo o aceite técnico pendente.

Consulte [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) antes de alterar contratos, pontuação ou fórmulas.
