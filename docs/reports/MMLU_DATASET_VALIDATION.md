# Validação do dataset MMLU — Entrega 3A

Data: 2026-11-07. **Preparação implementada e validada localmente; aguardando revisão
técnica. Entrega 3B não iniciada.** Branch `codex/feat-mmlu-dataset`, base
`d5405721b895830229f9cb16190fd5776435efb4`. Nenhum commit ou push nesta tarefa.

## Resultado e delimitação autorizada

Foram preparados **75 itens reais**, preservados integralmente: 15 de `dev` para
validação técnica futura e 60 de `test` reservados ao piloto posterior. Duas
preparações offline produziram JSONL e manifesto **idênticos byte a byte**.
As entradas completas têm **72–574 tokens**, nenhuma acima do limite atual de 1024.
A suíte passou com **86 testes** (71 anteriores + 15 novos); a simulação permanece
com 3 válidas, 0 falhas e as mesmas métricas.

Há uma ressalva de conteúdo: `high_school_biology/dev/2` tem alternativas A e C
idênticas. O registro permanece intacto; o gabarito fornecido é D (índice 3), que
não pertence ao par repetido. A seleção não contém pares de questões duplicadas.
A revisão deve considerar essa ressalva antes de autorizar 3B.

O pedido atual substitui **explicitamente** a proposta anterior de até 20 questões
de `validation` por 5 `dev` + 20 `test` em cada disciplina autorizada. Isso foi
registrado no README, na arquitetura e no roadmap. `dev` não fornece exemplos
few-shot: o protocolo continua zero-shot. Não foram alterados prompt, chat
template, fórmulas, scoring, limite de tokens ou backend MLX.

**Nenhum item MMLU foi submetido ao modelo.** Não há logits, probabilidades,
acurácia ou entropia MMLU nesta entrega. Foram carregados somente arquivos de
Parquet e de tokenizer/configuração; não houve carga ou download de pesos.

## Origem, revisão e arquivos

Fonte: [dataset oficial `cais/mmlu`](https://huggingface.co/datasets/cais/mmlu),
com licença MIT declarada no card. O [histórico oficial](https://huggingface.co/datasets/cais/mmlu/commits/main)
e a API do Hub foram consultados em 2026-11-07. A revisão fixada é:

`c30699e8356da336a370243923dbaf21066bb9fe`

Na consulta, `main` apontava para esse commit (2024-03-08, “Upload dataset”).
O histórico retornou 20 commits, incluindo o anterior
`b4f97952bba29f55186343a7693089e4261f3e22` (conversão para Parquet).
Não havia tags. A referência de conversão `refs/convert/parquet` apontava para
`d183e18c31b6d5563d00fb87257819c64e76b985`; essa referência não foi usada.
O carregamento usa o SHA imutável do dataset, nunca `main` ou uma conversão móvel.

A [árvore fixada](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe)
separa disciplinas e splits. Baixamos somente
`{subject}/{split}-00000-of-00001.parquet` para os seis pares abaixo:

| Disciplina | Split | Linhas na origem | Selecionadas | Tokens selecionados mín.–máx. | Bytes do Parquet |
| --- | --- | ---: | ---: | ---: | ---: |
| `high_school_psychology` | `dev` | 5 | 5 | 88–164 | 5.185 |
| `high_school_psychology` | `test` | 545 | 20 | 72–165 | 92.752 |
| `high_school_world_history` | `dev` | 5 | 5 | 169–280 | 10.248 |
| `high_school_world_history` | `test` | 237 | 20 | 200–574 | 201.554 |
| `high_school_biology` | `dev` | 5 | 5 | 78–155 | 4.939 |
| `high_school_biology` | `test` | 310 | 20 | 76–189 | 62.704 |

Total: **1.107 linhas verificadas**, **seis Parquets / 377.382 bytes** (aproximadamente
369 KiB). O cache do snapshot contém exatamente esses seis Parquets. Não foram
baixados `validation`, `auxiliary_train`, as outras disciplinas ou o dataset inteiro.
Os SHA-256 de cada arquivo vieram dos metadados do Hub e foram conferidos contra
os bytes locais; estão em [configs/mmlu_selection.json](../../configs/mmlu_selection.json)
e no [manifesto auditável](MMLU_SELECTION_MANIFEST.json).

O download usa [hf_hub_download com filename e revision](https://huggingface.co/docs/huggingface_hub/guides/download),
sem `load_dataset('all')` nem execução de script remoto. A leitura usa
[ParquetFile](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html)
para abrir somente o arquivo indicado.

## Implementação e contratos

| Arquivo | Responsabilidade |
| --- | --- |
| `src/benchmark/mmlu.py` | Leitura local com hash/tamanho conferidos; adaptação para `Question`; procedência tipada; seleção por índices; diagnóstico de duplicações. Sem rede ou backend. |
| `configs/mmlu_selection.json` | Revisão, três disciplinas, splits/quantidades, seed, algoritmo, seis hashes e identidade do tokenizer/protocolo. |
| `diagnostics/mmlu_dataset.py` | Download opt-in dos seis arquivos, tokenizer exclusivamente local, contagem de tokens e publicação dos artefatos de preparação. Não importa Runner/MLX. |
| `diagnostics/mmlu-requirements.txt` | Dependências opcionais fixadas de Parquet, Hub e tokenizer, independentes de MLX. |
| `tests/test_mmlu.py` | 15 testes offline com fixtures autorais e doubles, sem Hub, Arrow obrigatório, pesos ou Metal. |
| `docs/reports/MMLU_SELECTION_MANIFEST.json` | Cópia exata do manifesto final, com todos os IDs, posições, hashes, tokens e duplicações; sem caminhos locais privados. |
| README, arquitetura e roadmap | Capacidade atual, alteração explícita de delimitação e revisão pendente. |

O schema Arrow real nos seis arquivos é `question: string`, `subject: string`,
`choices: list<string>` e `answer: int64`; a descrição do dataset identifica
`answer` como classe A–D. O adaptador exige exatamente esses quatro campos,
disciplina coerente com a origem, quatro alternativas textuais não vazias e
índice **inteiro** entre 0 e 3 (rejeita bool, float, strings e nulos).
Não há coerção do gabarito, `strip`, normalização Unicode, tradução ou reordenação.

`Question` continua sendo o contrato interno já validado. `MMLUItem` associa a
questão à origem e confere coerência do ID, disciplina, posição e hash. Os arquivos
não fornecem um ID por questão: a identidade derivada é
`cais/mmlu@{revision}/{subject}/{split}/{original_index}`, com índice **base zero**
na ordem física do Parquet. O SHA da revisão impede confundir posições de snapshots
diferentes. `datasets.py`, schemas centrais e Runner permanecem inalterados.
A configuração de execução atual ainda não aceita MMLU: a conexão pertence a 3B.

## Seleção determinística e questões selecionadas

Seed **42**, método versionado **`sha256_index_rank_v1`**, aplicado separadamente
em cada disciplina/split. Para cada índice original `i`, calcular SHA-256 dos bytes
UTF-8 do JSON compacto:

```text
["sha256_index_rank_v1","cais/mmlu",revision,subject,split,42,i]
```

A serialização usa `ensure_ascii=False`, `sort_keys=True`, `separators=(',', ':')`
e `allow_nan=False`. Ordenar por `(digest_hex, i)`, tomar os primeiros 5 ou 20 e
ordenar os índices escolhidos crescentemente para apresentação na ordem original.
Os grupos seguem a ordem de arquivos registrada no plano. Não se usa o gerador
pseudoaleatório do Python; o algoritmo pode ser reproduzido pelos bytes especificados.

A função recebe somente população, quantidade, seed e origem. **Gabarito, texto,
comprimento e resultados de modelo não participam do ranking.** O conteúdo é
validado, mas não filtrado; fonte inválida interrompe a preparação. Todos os índices
são definidos antes da contagem de tokens. `dev` tem exatamente cinco itens por
disciplina, portanto todos são preservados. Os índices selecionados são:

- **high_school_psychology / dev:** `0, 1, 2, 3, 4`.
- **high_school_psychology / test:** `1, 69, 74, 107, 119, 158, 184, 188, 211, 213, 230, 236, 281, 373, 453, 469, 483, 494, 520, 532`.
- **high_school_world_history / dev:** `0, 1, 2, 3, 4`.
- **high_school_world_history / test:** `10, 19, 21, 29, 48, 53, 60, 83, 89, 90, 97, 98, 118, 137, 138, 150, 168, 173, 227, 231`.
- **high_school_biology / dev:** `0, 1, 2, 3, 4`.
- **high_school_biology / test:** `6, 24, 25, 31, 40, 49, 52, 93, 94, 106, 113, 116, 118, 150, 174, 220, 225, 243, 253, 305`.

O manifesto lista os 75 IDs completos e seus hashes, sem depender de abreviações
deste relatório. O JSONL local contém os 75 textos, alternativas e gabaritos, com
procedência por registro. A auditoria comparou **todos os 75 itens**, campo a campo,
com suas linhas no Parquet e recalculou independentemente o ranking: concordância integral.
Não houve escolha de seed por desempenho nem tentativa de balancear os gabaritos.

## Duplicações e integridade

Nos 1.107 registros de origem, foram encontrados **11 pares exatos de enunciado e
alternativas**, todos em `high_school_psychology/test`, nas posições:

`(14,222), (24,423), (43,323), (51,68), (71,539), (111,425), (135,153),
(193,298), (198,466), (251,505), (380,396)`.

Os pares têm gabaritos iguais. Nenhuma dessas posições está na seleção congelada;
**não houve remoção ou substituição para obter esse resultado**. Não foram
observadas identidades repetidas ou duplicações exatas entre os splits/disciplinas
carregados. Na seleção, não há enunciados ou pares enunciado/alternativas repetidos.

Em `high_school_biology/dev/2`, A e C têm texto exatamente igual; ambas são
alternativas incorretas segundo o gabarito D do arquivo. O item atende ao schema
de quatro posições, mas tem apenas três textos distintos. Isso é uma característica
da origem, não um efeito do adaptador. Foi sinalizado no manifesto e preservado,
sem deduplicar alternativas ou mudar o gabarito.

A detecção é exata, preservando espaços e Unicode: não detecta paráfrases,
semelhanças semânticas ou duplicações com diferenças de formatação. Não audita
outras disciplinas/splits não baixados. Validar o intervalo do gabarito não
significa validar independentemente a correção científica de cada resposta.

## Orçamento de tokens e compatibilidade

Tokenizer local de `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, snapshot
`56d07e766edd7159fbe12ed12d9cf114bf38bf1e`, os mesmos sete hashes de configuração
e tokenização de 2B. Prompt `current-no-enumeration-v1` obtido da função existente;
chat template padrão fixado, uma mensagem de usuário, zero-shot e cabeçalho de
resposta via `add_generation_prompt=True`. A contagem inclui instrução, enunciado,
quatro alternativas, mensagem de sistema padrão e demais tokens do chat.

Para **75/75 itens**, `encode(rendered_prompt, add_special_tokens=False,
truncation=False)` coincide com `apply_chat_template(tokenize=True,
add_generation_prompt=True, return_dict=False)`. Não há resposta gerada na contagem.
Resultado: **mínimo 72, máximo 574; 0/75 acima de 1024**. O array `over_limit_ids`
está vazio. O diagnóstico sinaliza excessos sem cortar texto, remover itens ou
refazer a seleção; esse comportamento também foi exercitado com fixtures longas.

A primeira tentativa local detectou uma diferença de **tipo de retorno** da API:
Transformers 5.22.1 retorna `BatchEncoding` por padrão, enquanto o wrapper de
MLX-LM 0.32.2 fixa `return_dict=False`. O diagnóstico passou a explicitar essa mesma
opção. As listas de IDs então coincidiram integralmente. Foi um ajuste no diagnóstico,
não falha de tokenização do dataset ou alteração no backend. A tentativa interrompida
não publicou uma pasta de preparação válida; as duas execuções posteriores concluíram.

Esta verificação não demonstra inferência, memória no limite ou validade contextual
dos rótulos em uma avaliação real MMLU. O backend existente deverá repetir suas
validações contextuais obrigatórias em 3B. Os caminhos com e sem KV cache podem
produzir logits diferentes; a preparação não executa nenhum deles e mantém o
protocolo aprovado de forward completo para a futura avaliação.

## Ambiente, comandos e evidências de validação

Mac Apple M2 / 8 GiB, macOS 15.7.4 arm64, Python 3.12.2. Preparação no ambiente
isolado já utilizado em 2A/2B: PyArrow **26.0.1**, Hugging Face Hub **2.4.0**,
Transformers **5.22.1**, Tokenizers **0.23.3**. Apenas PyArrow precisou ser instalado
nesse ambiente (wheel de aproximadamente 37,3 MB); o ambiente básico não foi alterado.
Havia aproximadamente 18 GiB livres antes da preparação. O diagnóstico não mediu
pico de memória e não produz evidência sobre o custo da futura avaliação MLX.

Na raiz, suíte final e simulação executadas com saída 0:

```sh
PYTHONPYCACHEPREFIX=/tmp/mmlu-3a-pycache PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/mmlu-3a-pycache PYTHONPATH=src \
  .venv/bin/python -m benchmark.cli --config configs/synthetic.yaml
```

Preparação final, repetida mudando apenas `--output` de `...-01` para `...-02`:

```sh
HF_HUB_OFFLINE=1 USE_TORCH=0 USE_TF=0 \
PYTHONPYCACHEPREFIX=/tmp/mmlu-3a-pycache PYTHONPATH=src \
/tmp/llm-prompt-comparison-venv/bin/python diagnostics/mmlu_dataset.py \
  --tokenizer-snapshot .cache/mlx-diagnostic/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
  --output outputs/mmlu-dataset-3a-02
```

O ambiente em `/tmp` é reutilizável nesta máquina, não um requisito de caminho.
Para outro ambiente, instalar `diagnostics/mmlu-requirements.txt` e usar seu Python;
o snapshot do tokenizer deve estar disponível localmente. Na primeira preparação
sem cache de dataset, remover `HF_HUB_OFFLINE=1` e acrescentar `--download`.
O modo padrão é exclusivamente local; ausências/falhas de integridade são erros,
não gatilhos para baixar o dataset inteiro. Use uma pasta de saída nova a cada execução.

| Verificação | Resultado observado |
| --- | --- |
| Suíte completa | **86 testes, OK, 0,769 s**, saída 0; 71 preexistentes + 15 de MMLU. |
| Contratos e seleção offline | Texto/ordem/gabarito preservados, IDs por origem, entradas inválidas recusadas, ranking conferido por referência independente e alteração de gabaritos sem alterar índices. |
| Integridade e duplicações offline | Hash conferido antes da leitura, identidades duplicadas recusadas, conteúdo repetido e respostas conflitantes sinalizados sem remoção. |
| Preparação e limites offline | Seis pedidos de arquivo fixados, cache local por padrão, excessos preservados, limite inclusivo, retorno de IDs explícito e divergência de tokenização recusada. |
| Parquets reais | Seis hashes/tamanhos conferidos; 1.107 registros válidos; schema Arrow e correspondência exata dos 75 selecionados auditados. |
| Repetição real da preparação | Manifestos e JSONL de `outputs/mmlu-dataset-3a-01/` e `...-02/` idênticos byte a byte, sem rede. |
| Simulação preservada | 3 válidas, 0 falhas; acurácia `0.6666666666666666`, entropia média `0.7603746960459724`. |
| Inferência MMLU | Não executada. O script verifica ausência de imports `mlx.core` e `mlx_lm`. |

Simulação identificada por `20261107T134054737555Z_a318334823f04aa283bcc7077fa6a01e`.
Os testes usam doubles; a leitura real de Parquet e a tokenização são verificadas
adicionalmente pelas preparações locais, não exigidas para a suíte offline.
O log da suíte está em `outputs/mmlu-reference/tests.log`.

### Hashes de reprodução

- Plano canônico: `5013e0184dc1a013da8f23469654d291a2a2ca03d48dfae58eae83fa92bd06b3`.
- Seleção (lista ordenada de procedência): `3d547d3a335c55d2cdbddff09d8987c55e6bb1dffba8114ae7851ec7b2316844`.
- JSONL completo, bytes UTF-8: `8df15f82250b63acdcce84a9f45471a9b0282aa1d27b06017a9eba8e055a88a9`.

Os hashes canônicos usam a serialização JSON descrita acima. O hash de linha cobre
os quatro campos originais, **incluindo** o gabarito, para auditoria de conteúdo;
isso não faz o gabarito participar do ranking. `prompt_sha256` é o hash canônico
da string de prompt, `input_ids_sha256` da lista de tokens, e
`chat_template_sha256_utf8` usa diretamente os bytes UTF-8 do template.
O manifesto registra ainda os hashes dos fontes efetivamente executados e versões.

`selected_questions.jsonl` é gravado antes do manifesto, que é publicado por rename
após a escrita completa. Pastas incompletas não devem ser consumidas; esta preparação
não oferece retomada ou garantia contra perda de energia. Os arquivos de trabalho
em `outputs/`, cache, ambientes e pesos continuam ignorados pelo Git. O manifesto
copiado para `docs/reports/` é uma evidência intencionalmente revisável, não resultados
de modelo. O checkout ainda tem mudanças locais; hashes não substituem um commit revisado.

## Pendências e recomendação para 3B

1. Revisar e aprovar esta preparação e a ressalva de alternativas A/C iguais em
   biologia `dev/2`. Recomenda-se manter a fonte intacta e a sinalização visível;
   qualquer política de exclusão/substituição requer decisão explícita antes de executar.
2. Após autorização, conectar a seleção congelada ao Runner com procedência completa
   e separar claramente execução técnica `dev` e piloto `test`. Conferir novamente
   hashes e índices antes de avaliar; não selecionar itens a partir de resultados.
3. Preservar o protocolo de 2B e validar tokens contextuais, recursos e persistência
   no fluxo real. Compatibilidade de comprimento não prova viabilidade operacional.
4. Manter `test` reservado ao piloto posterior, sem escolher prompt, seed ou método
   por seus acertos. A amostra de três disciplinas não representa o MMLU completo
   nem define a amostra final da pesquisa; contaminação de treinamento não foi investigada.

Não há bloqueio de schema, integridade ou comprimento entre os 75 itens. A ressalva
de conteúdo e o aceite técnico permanecem pendentes. **A Entrega 3 não está concluída;
aguardar revisão antes de iniciar 3B.**
