# Comparação controlada de formatos de prompt — complemento da Entrega 2A

Data: 2026-11-06. Estado: comparação executada; protocolo candidato aguarda revisão e aprovação. Entrega 2B não autorizada.

## Escopo e referências

Esta investigação compara textos de prompts em condições comuns. Não reproduz uma
execução oficial de MMLU, não carrega esse dataset e não instala os frameworks.
Foram consultados os relatórios [do diagnóstico](MLX_DIAGNOSTIC.md) e
[da investigação comportamental](MLX_BEHAVIOR_VALIDATION.md).

Referências oficiais consultadas em 2026-11-06, com revisões fixadas:

- EleutherAI LM Evaluation Harness, commit `d6de81643928d653435c431bae19945d41d32520`:
  [template MMLU default](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/mmlu/default/_default_template_yaml),
  [descrição por assunto](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/mmlu/default/mmlu_abstract_algebra.yaml),
  [construção de requisições](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/api/task.py) e
  [configuração padrão](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/config/task.py).
- Hugging Face LightEval, commit `b97a6533bec22c4184b88526366fdf5d73d502c9`:
  [implementação MMLU](https://github.com/huggingface/lighteval/blob/b97a6533bec22c4184b88526366fdf5d73d502c9/src/lighteval/tasks/tasks/mmlu.py).
- Qwen: [quickstart oficial da versão 2.5](https://qwen.readthedocs.io/en/v2.5/getting_started/quickstart.html),
  com `apply_chat_template(..., add_generation_prompt=True)`;
  o template efetivamente usado pertence ao [snapshot MLX fixado](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-bf16/blob/56d07e766edd7159fbe12ed12d9cf114bf38bf1e/tokenizer_config.json).

O Harness configura `multiple_choice`, candidatos `A`–`D` e acurácia por
log-verossimilhança. Em completion, o delimitador padrão é espaço; na revisão
consultada, o caminho com chat template remove esse delimitador quando não há
`gen_prefix`. Portanto não é correto afirmar que o Harness sempre pontua ` A`.
A configuração usa `test` para avaliação e `dev` para few-shot; a quantidade de
exemplos depende da execução. Aqui usamos zero-shot e nenhuma dessas partições.

O LightEval acrescenta `Question: ` antes da pergunta, declara escolhas com espaço
inicial e, **nesta revisão**, configura `Metrics.exact_match`, geração de até cinco
tokens e parada em quebra de linha. Não se deve atribuir automaticamente ao seu
MMLU atual o mesmo método de pontuação do Harness. Nossa comparação importa seu
formato textual, não sua avaliação por geração.

## Desenho fixado antes da leitura dos resultados

Quatro questões autorais simples, independentes das quatro usadas anteriormente:

| ID | Pergunta exata | Assunto | Alternativas canônicas (correta primeiro) |
| --- | --- | --- | --- |
| triangle | How many sides does a triangle have? | elementary_mathematics | 3; 2; 4; 5 |
| planet | Which planet is known as the Red Planet? | astronomy | Mars; Venus; Jupiter; Saturn |
| organ | Which organ pumps blood through the human body? | anatomy | Heart; Liver; Lung; Stomach |
| meter | How many centimeters are in one meter? | elementary_mathematics | 100; 10; 1000; 1 |

Rotação cíclica à direita de 0, 1, 2 e 3 posições: o gabarito percorre A, B, C, D.
Os distratores também rodam, sempre de modo idêntico entre formatos. Não são as
24 permutações possíveis. São 4 questões × 4 rotações × 3 formatos = 48 casos;
as 16 apresentações por formato não são 16 questões independentes.

### Textos exatos parametrizados

Atual sem enumeração final (`current-no-enumeration`):

```text
{question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
Reply with exactly one letter. Do not add spaces or explanations.
```

Harness adaptado (`harness-adapted`):

```text
The following are multiple choice questions (with answers) about {subject}.

{question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
Answer:
```

LightEval adaptado (`lighteval-adapted`): mesmo texto do Harness acima, com
`Question: ` imediatamente antes de `{question}`. Sublinhados do assunto são
substituídos por espaços. Não há newline nem espaço acrescentado ao final de
nenhum texto de usuário. As perguntas selecionadas não exigem limpeza de espaços.

### Adaptação comum para Qwen-Instruct

O texto inteiro acima entra em uma única mensagem `user`. O template do snapshot
adiciona o sistema padrão e abre o turno do assistente:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
```

A quebra de linha depois de `assistant` faz parte do prompt. `Answer:` permanece
no turno do usuário; não é um preenchimento prévio do assistente. Essa escolha é
uma adaptação explícita e comum, não uma alegação de equivalência integral aos
runners dos frameworks. Não adicionamos a instrução do prompt atual aos outros
formatos. Harness × LightEval isola `Question: `; atual × referências modifica
um pacote de convenções (introdução, assunto, instrução final e `Answer:`), cuja
causalidade individual não pode ser separada por esta matriz.

### Pontuação e condições comuns

Modelo `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, revisão
`56d07e766edd7159fbe12ed12d9cf114bf38bf1e`; Apple M2, 8 GiB, execução MLX-LM
isolada do Runner. Dependências fixadas em `diagnostics/mlx-requirements.txt`.
Um forward completo por prompt, batch 1, parâmetros BF16, sem quantização,
`model.eval()`, sem cache KV no caminho primário. A posição `[0, T-1, :]` prevê
o primeiro token do assistente. Logits são promovidos para float32 e então
float64 na análise. Pontuam-se exclusivamente os candidatos `A`, `B`, `C`, `D`,
com a função `softmax` existente, e `argmax` com desempate pelo primeiro índice.

Acurácia e entropia usam as funções existentes em `benchmark.metrics` e
`benchmark.numerics`. A entropia de cada caso é a Shannon normalizada entre as
quatro alternativas; o agregado é sua média, não a entropia das contagens de
rótulos selecionados. A massa original usa softmax sobre o vocabulário completo,
antes da renormalização. Não há novos índices ou correções de viés.

Como sensibilidade separada, o mesmo vetor de logits permite pontuar ` A`, ` B`,
` C`, ` D`. Essa mudança altera o evento condicionado e é apresentada em outra
tabela, sem misturá-la à comparação primária. Não soma massas das duas famílias.
A sonda de geração produz apenas um token greedy (`temp=0`, sem penalidades),
para verificar o alinhamento dos logits; não mede exact-match de resposta completa.
O cache da sonda e o forward manual com cache são controles numéricos, não
substitutos silenciosos do caminho primário.

## Execução, ambiente e reprodutibilidade

Execução completa: `outputs/prompt-comparison-03/diagnostic.json`, com estado
`diagnostic_completed_not_backend_approved` e 48 casos. Nenhum dataset ou peso
novo foi solicitado ao Hugging Face. O ambiente temporário da investigação
anterior não existia mais; o ambiente do projeto e o snapshot estavam marcados
`dataless` pelo macOS/iCloud, bloqueando leituras. O ambiente Python foi restaurado
em `/tmp/llm-prompt-comparison-venv` a partir das versões fixadas. O snapshot foi
recuperado pelo iCloud; o responsável também selecionou “Manter Baixado”.
A tentativa de usar o venv temporário antigo falhou por inexistência; a tentativa
no venv descarregado foi interrompida antes da matriz. Nenhuma tentativa parcial
foi usada para preencher resultados. `outputs/prompt-comparison-01` já estava vazio
no início desta retomada; a execução válida é exclusivamente `03`.

| Componente | Versão/condição |
| --- | --- |
| Python / macOS | 3.12.2 / 15.7.4 arm64 |
| MLX / MLX Metal / MLX-LM | 0.32.4 / 0.32.4 / 0.32.2 |
| Transformers / Tokenizers | 5.22.1 / 0.23.3 |
| NumPy / Hugging Face Hub / Safetensors | 2.5.4 / 2.4.0 / 0.8.0 |
| Cache de alocação MLX | limite de 256 MiB |
| Memória de parâmetros | 988.065.536 bytes, BF16 |
| Pico de memória MLX | 1.133.162.424 bytes (aproximadamente 1,055 GiB) |
| RSS máximo informado pelo processo | 425.017.344 bytes; não equivale à memória unificada total |
| Swap global ao fim | 1.647,38 MiB usados; não atribuível exclusivamente ao diagnóstico |

SHA-256 dos pesos reutilizados:
`39da1e4174f12953ad7ace17239da672aa2b4b9ba723a09835963484c5a05b41`.
Os hashes do tokenizer, configuração, scripts e vetores brutos constam do
[JSON de evidências](PROMPT_COMPARISON_EVIDENCE.json), junto dos 48 prompts exatos,
mensagens, IDs de entrada, gabaritos, logits, probabilidades originais e
condicionadas, entropias, posições e controles. Os vetores completos `.npy`
permanecem em `outputs/`, ignorados pelo Git. O JSON é a evidência compacta para
revisão, sem pesos ou caminhos pessoais.

Comando executado (a pasta de saída deve ser nova para repetir):

```sh
HF_HUB_OFFLINE=1 HF_HOME=.cache/mlx-diagnostic/huggingface PYTHONPATH=src \
/tmp/llm-prompt-comparison-venv/bin/python diagnostics/mlx_labels.py \
  --snapshot .cache/mlx-diagnostic/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
  --output outputs/prompt-comparison-03 --prompt-comparison
```

Ordem fixa: questão, rotação, formato. Não houve amostragem aleatória nem seed
necessária ao greedy. Não foram executadas réplicas independentes ou uma análise
de estabilidade entre versões do MLX. Os hashes e as versões permitem repetir o
procedimento, sem prometer identidade binária entre caminhos de kernels.

## Resultados primários: rótulos sem espaço

| Formato adaptado | Acertos | Acurácia | Entropia normalizada média | Seleções A / B / C / D | Massa original: mínimo / média / máximo |
| --- | --- | --- | --- | --- | --- |
| Atual sem enumeração | 11/16 | 68,75% | 0,293867 | 7 / 2 / 3 / 4 | 0,993578 / 0,998755 / 0,999876 |
| Harness | 10/16 | 62,50% | 0,239550 | 10 / 2 / 2 / 2 | 0,284571 / 0,547754 / 0,735494 |
| LightEval | 10/16 | 62,50% | 0,247161 | 10 / 2 / 2 / 2 | 0,464292 / 0,619974 / 0,767015 |

Cada sequência abaixo corresponde aos gabaritos A, B, C, D, nessa ordem:

| Questão | Atual | Harness | LightEval |
| --- | --- | --- | --- |
| triangle | ADAD | AAAA | AAAA |
| planet | ABCD | ABCD | ABCD |
| organ | ABCD | ABCD | ABCD |
| meter | AAAC | AAAA | AAAA |

As referências tiveram a mesma predição em todas as 16 apresentações, mas
probabilidades e entropias diferentes. `Question: ` mudou a distribuição e a
massa válida sem mudar os rótulos selecionados nesta amostra. Menor entropia
não implicou maior acurácia. O comportamento de selecionar A não é universal:
as questões de planeta e órgão acompanharam todas as rotações nos três formatos.
As questões numéricas permaneceram sensíveis ao conteúdo e à posição.

Exemplo auditável: `triangle`, gabarito A, formato atual, posição 69 de 70 tokens,
saída `[1, 70, 151936]`, IDs candidatos `[32, 33, 34, 35]`:

| Rótulo | Logit | Probabilidade condicionada | Probabilidade original |
| --- | --- | --- | --- |
| A | 27,000 | 0,821223721713 | 0,819877587719 |
| B | 24,625 | 0,076385705003 | 0,076260495037 |
| C | 24,875 | 0,098081186695 | 0,097920413917 |
| D | 21,750 | 0,004309386590 | 0,004302322727 |

O JSON contém a precisão completa para todos os casos; números desta tabela
foram arredondados para apresentação.

## Sensibilidade separada: rótulos com espaço

Mesmos prompts, mesmos forwards e logits, outros IDs de candidato. Não são
resultados do protocolo primário, nem reprodução da avaliação oficial LightEval.

| Formato | Acertos | Acurácia | Entropia média | Seleções A / B / C / D | Intervalo de massa original |
| --- | --- | --- | --- | --- | --- |
| Atual | 12/16 | 75,00% | 0,338079 | 6 / 2 / 4 / 4 | 2,88e-7 a 2,17e-5 |
| Harness | 10/16 | 62,50% | 0,247942 | 10 / 2 / 2 / 2 | 4,43e-8 a 1,86e-6 |
| LightEval | 10/16 | 62,50% | 0,253339 | 10 / 2 / 2 / 2 | 3,91e-8 a 1,59e-6 |

O aumento para 12/16 no prompt atual não justifica trocar candidatos. Esses tokens
representam eventos muito raros no início deste turno do assistente. A
renormalização esconde a pequena massa original; escolher a família de tokens
pela acurácia deste conjunto introduziria ajuste ao diagnóstico. Espaços podem
ser apropriados em outros contextos, como completion após `Answer:`, mas isso
precisa de um protocolo próprio validado.

## Tokenização, posição e controles numéricos

- Nos 48 prompts com chat, anexar cada letra preservou integralmente os IDs do
  prefixo e acrescentou exatamente um token: A=32, B=33, C=34, D=35. O resultado
  de `apply_chat_template(tokenize=True)` coincidiu com a codificação do texto
  renderizado sem tokens especiais adicionais.
- As variantes com espaço também foram continuações unitárias estáveis:
  ` A`=362, ` B`=425, ` C`=356, ` D`=422. São eventos distintos, não aliases.
- O controle sem chat mostrou que anexar letras nuas ao texto de usuário altera
  o prefixo tokenizado, tanto após o ponto final do prompt atual quanto após
  `Answer:` nas referências. As variantes com espaço preservam o prefixo nesses
  controles. Logo, validar as letras isoladamente não basta, e os IDs do chat
  não podem ser transferidos cegamente para completion.
- A posição usada foi sempre `T-1`, entre 66 e 81, no vocabulário de 151.936
  tokens. O script registra também o controle `T-2`, que não prevê a mesma
  posição. O preenchimento manual de cache até o penúltimo token seguido do
  último token produziu logits exatamente iguais aos capturados pela geração
  da biblioteca em 48/48 casos.
- O softmax das quatro alternativas coincidiu com a razão entre probabilidades
  originais e massa válida (tolerância absoluta 1e-12). A diferença máxima para
  o softmax MLX float32 foi 1,33e-7, dentro de 1e-6.
- Forward completo e caminho com cache **não são numericamente idênticos** em
  BF16: maior diferença absoluta no vocabulário de 0,423828125 nos logits e de
  0,030672103 nas probabilidades condicionadas. Os vencedores entre A–D
  coincidiram em 48/48, mas o argmax global coincidiu em apenas 46/48.
- Em `organ-2-lighteval-adapted`, o argmax global sem cache foi `The`, enquanto
  cache/geração escolheram C. Em `meter-3-lighteval-adapted`, ocorreu `The` versus A.
  Isso reforça a necessidade de fixar o caminho de inferência. Não indica troca
  de índices de alternativas nem foi corrigido artificialmente.
- Somar `exp(logprobs)` retornados em BF16 pela API gerou totais entre
  0,940257 e 1,051130. Por isso as probabilidades deste relatório vêm de logits
  brutos com precisão promovida, não desses valores exponenciados diretamente.

## Validação e limites

A suíte da fundação foi executada com
`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q`: **54 testes,
OK, 2,618 s**, saída 0. O ambiente temporário passou em `pip check`.
As asserções do diagnóstico concluíram nos 48 casos. A auditoria dos 48 registros
e 96 vetores persistidos passou: scores, posições, normalização, métricas,
balanceamento de gabaritos e cache/geração conferidos. `git diff --check` passou.
Uma asserção inicial da auditoria comparou uma tupla retornada por `softmax`
com a lista serializada no JSON; a comparação de tipos foi corrigida para lista,
sem alterar números, resultados ou código da fundação. As 48 renderizações e
os controles sem chat foram reconferidos; os 16 textos LightEval coincidiram
com a função oficial extraída do código consultado. Os hashes dos dois scripts
coincidiram com os registrados na execução.

Esta comparação não prova desempenho em MMLU ou robustez geral: quatro perguntas
fáceis, escolhidas para diagnóstico, rotações correlacionadas, um único modelo e
uma execução. Não mede exact-match de respostas completas, não executa os
frameworks e não testa few-shot. Os resultados das adaptações não são scores
oficiais de Harness ou LightEval. Tampouco isolam todos os componentes textuais
da diferença entre prompt atual e referências. O limite do modelo é uma
hipótese compatível com as falhas numéricas; o diagnóstico não separa capacidade
interna, treinamento e seguimento de instruções causalmente.

## Recomendação para revisão antes de 2B

Recomenda-se considerar o prompt atual sem enumeração como **candidato de
continuidade**, com chat template fixado, rótulos nus e validação contextual
obrigatória. A justificativa é sua instrução explícita de formato e a maior massa
válida observada, além de exigir menos mudança em relação à investigação
anterior. A diferença de apenas um acerto não é critério de seleção nem evidência
estatística de superioridade. Harness adaptado continua sendo uma alternativa
justificável se a prioridade for proximidade textual com uma referência externa;
isso deve ser nomeado como adaptação e não reprodução integral do protocolo.

Antes de 2B, o responsável deve aprovar explicitamente: texto/versionamento do
prompt, uso do sistema padrão Qwen, zero-shot, rótulos sem espaço no início do
assistente e caminho de logits. O forward completo é o candidato já medido;
qualquer escolha por cache exige uma decisão explícita e testes correspondentes,
pois os caminhos diferiram numericamente. Não recomendamos trocar para rótulos
com espaço, misturar famílias ou selecionar o prompt automaticamente por acurácia.

Nenhuma dessas recomendações foi aplicada ao Runner. Não há backend definitivo,
adaptador MMLU, nova métrica ou alteração de fórmula. Entrega 2 permanece em
andamento; Entrega 2B aguarda revisão e autorização. Nenhum commit ou push foi
realizado nesta comparação.
