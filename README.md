# LLM Benchmark — Desempenho e Incerteza

> **Estado:** Entrega 1 — Fundação com correções F1/F2/F3 implementadas e 54 testes passando sobre o código-fonte e o pacote instalado; aguardando aprovação do responsável. Consulte o [relatório de validação](docs/reports/VALIDATION_DELIVERY_1.md) e o [roadmap](docs/ROADMAP.md). A inferência real e o método de pontuação ainda exigem validação técnica.

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
| Dataset | MMLU, com pequena amostra identificável da partição `validation` |
| Questões | Até 20, inicialmente, para validação técnica |
| Modelo de desenvolvimento | Qwen2.5-0.5B-Instruct em variante compatível com MLX-LM (candidata) |
| Inferência | Local, utilizando MLX-LM em Apple Silicon |
| Pontuação das alternativas | Rótulos de resposta (`A`, `B`, `C`, `D`), **condicionada à validação da tokenização e dos logits** |
| Métricas | Acurácia e entropia normalizada |
| Saída | Resultados por questão, resumo e metadados por execução |

O equipamento de referência para a fase inicial é um **Mac com Apple M2 e 8 GB de memória unificada**. O tamanho do modelo, o uso de memória e a forma exata de pontuação serão confirmados com testes antes de considerar o piloto validado.

O piloto de desenvolvimento **não será apresentado como avaliação científica conclusiva**. A partição de teste deverá ser preservada para avaliações posteriores, evitando ajustar o protocolo a partir dos resultados finais.

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
4. Se o método de pontuação for validado, executar um piloto pequeno com MMLU `validation`.
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

### Convenções e limites desta entrega

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

Não foram implementados MLX, Transformers, APIs, adaptação do MMLU, diagnóstico
de tokens ou pontuação de sequências. A massa original dos rótulos no vocabulário
permanece nula, pois a simulação não fornece essa evidência. A Entrega 1 valida
apenas contratos, matemática e fluxo; **o MVP com modelo real ainda não está validado**.

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
- **Pendente de validação:** tokenização dos rótulos, posição de pontuação, uso de template de chat, extração e normalização dos escores, consumo de memória e critérios de aceitação.

Consulte [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) antes de alterar contratos, pontuação ou fórmulas.
