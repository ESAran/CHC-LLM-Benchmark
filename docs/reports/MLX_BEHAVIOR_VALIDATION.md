# Investigação complementar de comportamento — Entrega 2A

Data: 2026-11-06. Complementa [MLX_DIAGNOSTIC.md](MLX_DIAGNOSTIC.md).
Entrega 2 permanece em andamento; Entrega 2B não iniciada.

**Conclusão:** a preferência por A é frequente, mas depende da pergunta e da
instrução. Não foi encontrada troca de índices, erro na posição pontuada ou
divergência entre previsão e primeiro token gerado nos 64 casos. Remover a
enumeração final mudou substancialmente o comportamento de três perguntas,
mas não resolveu `2 + 2`. Há evidências de efeitos de prompt e de limitações de
seguimento de instruções/mapeamento de resposta. Não há base para atribuir toda
a ocorrência a uma única causa ou aprovar automaticamente um novo protocolo.

## Protocolo controlado

A investigação usa o script existente `diagnostics/mlx_labels.py`, com o modo
explícito `--behavior` e a matriz em `diagnostics/mlx_behavior.py`. Não houve
mudança no runner, MMLU, fórmulas ou dependências da fundação. As variantes abaixo
são condições diagnósticas, não alterações aprovadas do protocolo definitivo.

Modelo: `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, revisão
`56d07e766edd7159fbe12ed12d9cf114bf38bf1e`; pesos BF16, MLX 0.32.4, MLX-LM 0.32.2,
Python 3.12.2 arm64 no M2/8 GiB, conforme ambiente isolado de 2A.

| ID | Pergunta | Opções na ordem inicial (A correta) |
| --- | --- | --- |
| sum | What is 2 + 2? | 4; 3; 5; 6 |
| capital | What is the capital of France? | Paris; Rome; Berlin; Madrid |
| week | How many days are in one week? | 7; 5; 6; 8 |
| opposite | What is the opposite of hot? | cold; warm; large; fast |

Cada lista sofre quatro rotações cíclicas à direita: a resposta correta ocupa
A, B, C e D exatamente uma vez. As opções continuam identificadas em ordem A–D.
Não se trata de todas as 24 permutações possíveis. O gabarito é usado apenas
para conferir o resultado e não é passado ao modelo.

Quatro instruções finais, definidas antes de observar os resultados:

| Nome | Texto exato |
| --- | --- |
| original | Reply with exactly one letter: A, B, C, or D. Do not add spaces or explanations. |
| reverse-enumeration | Reply with exactly one letter: D, C, B, or A. Do not add spaces or explanations. |
| no-enumeration | Reply with exactly one letter. Do not add spaces or explanations. |
| semantic | Choose the correct answer to the question. Reply only with the letter of that option. |

São 4 perguntas × 4 rotações × 4 instruções = **64 condições**, em ordem de
pergunta, rotação e instrução. A variante invertida altera somente a enumeração
na instrução, sem inverter a identificação das opções. A variante sem enumeração
remove apenas a lista final; a semântica é uma mudança mais ampla e não permite
atribuir seu efeito a uma palavra isolada.

Cada condição usa `apply_chat_template(add_generation_prompt=True)`, a mensagem
de sistema padrão do snapshot e nenhuma conversa anterior. O prompt renderizado
e seus IDs são registrados. A pontuação usa `output[0, T-1, :]` sem cache e os
IDs contextuais dos rótulos sem espaço. Normalização reutiliza o softmax existente.

A geração livre é greedy (`temp=0`), limitada a 24 tokens, sem máscara de
rótulos, penalidades ou correção de resposta. São registrados texto integral,
tokens, primeiro token e motivo de término. A classificação estrita aceita
apenas A/B/C/D após remover espaços externos; prosa não é convertida em acerto.
Compara-se também a geração com os logits capturados da API e com cache manual.
A antiga assertiva de igualdade entre argmax direto e geração virou um campo
de evidência, para que uma diferença numérica não interrompa nem selecione casos.

Oito controles adicionais: cada pergunta sem opções seguida de
`Reply with the answer only.` e quatro pedidos `Reply with exactly the letter X.`
(X em A–D). Os controles ajudam a separar conhecimento da resposta e capacidade
de emitir cada letra da tarefa de mapear a resposta para uma alternativa.

## Reprodução e preservação

```sh
HF_HUB_OFFLINE=1 HF_HOME=.cache/mlx-diagnostic/huggingface PYTHONPATH=src \
/tmp/llm-mlx-behavior-venv/bin/python diagnostics/mlx_labels.py --behavior \
  --snapshot .cache/mlx-diagnostic/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
  --output outputs/mlx-behavior-03
```

Use uma pasta de saída nova. O ambiente e as dependências são os mesmos do
diagnóstico anterior. A primeira tentativa foi encerrada durante uma leitura de
cache `.pyc` de uma dependência, antes dos casos. A segunda tentativa, com cache de bytecode separado, também demorou lendo
arquivos das dependências e foi encerrada antes dos casos. A execução concluída
usou um novo venv em `/tmp/llm-mlx-behavior-venv`, instalado com
`python3 -m venv /tmp/llm-mlx-behavior-venv` e
`/tmp/llm-mlx-behavior-venv/bin/python -m pip install --no-cache-dir -r diagnostics/mlx-requirements.txt`.
`pip check` passou; as versões de runtime são as mesmas. Pip de instalação foi
24.0 nesse venv. Não houve novo download de pesos nem mudança de configuração
experimental. As demoras de leitura não são tempos de inferência.

Os resultados completos ficam em `diagnostic.json`; `behavior-progress.jsonl`
é um registro incremental e não indica conclusão por si só. Vetores completos
de logits ficam em `.npy`, em `outputs/`, ignorado pelo Git.

## Resultados completos da matriz

Cada sequência abaixo segue a ordem de gabaritos **A, B, C, D**. A coluna
“escores” indica os rótulos vencedores; a geração preserva pontuação e texto.

| Pergunta | Instrução | Vencedores pelos escores | Textos efetivamente gerados, nas quatro rotações |
| --- | --- | --- | --- |
| sum | original | AAAA | `A`; `A`; `A`; `A` |
| sum | reverse-enumeration | AAAA | `A`; `A`; `A`; `A` |
| sum | no-enumeration | AAAA | `A`; `A`; `A`; `A` |
| sum | semantic | ABBA | `A.`; `B`; `B`; `A. 3` |
| capital | original | AACA | `A`; `A`; `C`; `A` |
| capital | reverse-enumeration | AAAA | `A`; `A`; `A`; `A` |
| capital | no-enumeration | ABCD | `A`; `B`; `C`; `D` |
| capital | semantic | ABCD | `A. Paris`; `B`; `C`; `D` |
| week | original | AAAD | `A`; `A`; `A`; `D` |
| week | reverse-enumeration | AAAA | `A`; `A`; `A`; `A` |
| week | no-enumeration | ABCD | `A`; `B`; `C`; `D` |
| week | semantic | ABCD | `A. 7`; `B. 7`; `C. 7`; `D. 7` |
| opposite | original | AADA | `A`; `A`; `D`; `A` |
| opposite | reverse-enumeration | AAAA | `A`; `A`; `A`; `A` |
| opposite | no-enumeration | ABCD | `A`; `B`; `C`; `D` |
| opposite | semantic | ABCD | `A. cold`; `B`; `C`; `D.` |

| Instrução | A como vencedor | Vencedor correto | Geração correta no formato estrito | Geração fora do formato |
| --- | ---: | ---: | ---: | ---: |
| original | 13/16 | 6/16 | 6/16 | 0/16 |
| reverse-enumeration | 16/16 | 4/16 | 4/16 | 0/16 |
| no-enumeration | 7/16 | 13/16 | 13/16 | 0/16 |
| semantic | 5/16 | 14/16 | 6/16 | 9/16 |

Os denominadores incluem todos os casos. Essas contagens são descritivas deste
diagnóstico; não estimam desempenho em um dataset científico. A comparação
separa o acerto do rótulo pontuado da conformidade da geração completa. Não foi
usada extração permissiva para transformar `A. Paris` em uma resposta estrita A.
Todas as 64 gerações terminaram com `finish_reason=stop`, sem atingir o limite.

### Controles sem opções e pedidos de reprodução

| Controle | Texto gerado |
| --- | --- |
| What is 2 + 2? Reply with the answer only. | `4` |
| What is the capital of France? Reply with the answer only. | `Paris` |
| How many days are in one week? Reply with the answer only. | `7 days` |
| What is the opposite of hot? Reply with the answer only. | `cold` |
| Reply with exactly the letter A. | `Okay, Qwen.` |
| Reply with exactly the letter B. | `B` |
| Reply with exactly the letter C. | `okay` |
| Reply with exactly the letter D. | `disagree` |

Os controles abertos produziram o conteúdo correto, mas os de reprodução literal
falharam em três pedidos. Isso evidencia seguimento imperfeito de instruções.
Não significa incapacidade de emitir C ou D: ambas aparecem corretamente em
outras condições da matriz.

## Verificações de implementação e pontuação

- Os quatro prompts e vetores de quatro escores originais de `2 + 2` foram
  reproduzidos **exatamente** em comparação com `run-02` de 2A.
- Nos 64 casos, os quatro rótulos preservaram o prefixo e acrescentaram um único
  token, com IDs `[32,33,34,35]`, distintos, não especiais e decodificados como
  A/B/C/D. Foram verificados 256 candidatos centrais; o helper também registrou
  as 20 variantes contextuais por caso (1.280 combinações).
- O template tokenizado e o texto renderizado com `add_special_tokens=False`
  produziram os mesmos IDs. Comprimentos variaram de 65 a 79 tokens. A posição
  usada foi sempre `T-1`, com saída de dimensão de vocabulário 151.936.
- O gabarito de cada rotação foi conferido contra o texto da opção original
  correta. Os candidatos permanecem na ordem A/B/C/D mesmo na instrução que
  enumera D/C/B/A. Não houve inversão dos índices de logits.
- **64/64:** argmax do vocabulário da passagem direta = primeiro token da sonda
  da API = primeiro token da geração completa. O vencedor entre A–D também
  correspondeu a esse primeiro token em todos os casos.
- **64/64:** logits brutos da API = logits obtidos com cache manual, com diferença
  máxima absoluta zero. Persistem diferenças numéricas BF16 entre os caminhos
  com e sem cache, já descritas em 2A, sem troca de vencedor nesta matriz.
- Probabilidades condicionais somaram 1 dentro de `1e-12` e foram conferidas
  contra a normalização do vocabulário completo. A massa original de A–D ficou
  entre **0,984677639835 e 0,999783902807**. Massa alta não implicou acerto.

Exemplo reproduzido: na pergunta `2 + 2` com resposta correta em B e instrução
original, escores A–D foram `[29.375,24.625,24.0,25.25]`; o modelo realmente gerou
`A`. Portanto, esse erro não foi criado apenas pela restrição do argmax a quatro
candidatos ou pela interpretação de uma resposta livre diferente.

Os 54 testes automatizados da fundação passaram novamente:
`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q` → **54 testes,
OK, 3,891 s, saída 0**. Compilação dos dois scripts e `git diff --check` passaram.
Comparação com `f9c5352` confirmou núcleo, testes, configuração e `pyproject.toml`
inalterados. O diagnóstico terminou com saída 0 e 64 casos, além dos oito controles.

## Interpretação das hipóteses

| Hipótese | Evidência e limite |
| --- | --- |
| A ocorre por erro de índice ou tokenização no script | Não sustentada nos casos verificados: IDs, fronteiras, rotações, posição e API concordam. Isso não prova ausência de qualquer defeito possível na biblioteca ou em todos os prompts. |
| O método força A artificialmente | Não sustentada: não há máscara, a geração livre coincide e em outras condições gera B/C/D. |
| O modelo sempre responde A | Refutada nesta matriz: a instrução original já produziu C/D e as outras produziram B/C/D. Há preferência frequente, não invariância. |
| O modelo copia simplesmente a primeira letra enumerada na instrução | Não sustentada: inverter a enumeração para D/C/B/A produziu A em 16/16 casos, não D. |
| A formulação do prompt influencia fortemente o resultado | Sustentada por comparações pareadas: retirar só a enumeração final elevou vencedores corretos de 6/16 para 13/16. As perguntas/opções/template ficaram iguais. |
| O modelo desconhece as quatro respostas | Não sustentada pelos controles abertos, que retornaram o conteúdo correto. Isso não garante mapeamento consistente para rótulos. |
| Há limitações de seguimento de instruções e de mapeamento para opções | Sustentada pelos controles literais, pela pontuação adicional em nove respostas semânticas e pela persistência de erros em aritmética. Não identifica a causa interna dos pesos. |

O resultado não é restrito à pergunta original: a preferência por A reaparece
nas quatro perguntas, mas sua intensidade depende do prompt. Não se pode separar
completamente “efeito de prompt” e “limitação do modelo”: o efeito observado é a
resposta desta revisão do modelo à manipulação controlada da instrução.

Não foram comparados outro modelo, outra conversão, pesos FP32, outra biblioteca
de inferência ou outro template de sistema. Não há evidência para atribuir
causalmente o comportamento ao tamanho 0,5B, à quantização (não foi aplicada),
ao BF16 ou a um defeito de conversão. Uma rodada determinística, quatro perguntas
escolhidas manualmente e rotações correlacionadas não permitem generalização
estatística. Não houve ajuste iterativo para escolher os melhores exemplos.

## Recomendação para a Entrega 2B — sujeita a aprovação

1. Manter a conclusão técnica de que os escores podem ser obtidos corretamente;
   não tratar isso como validação do modelo como solucionador de questões.
2. Submeter para aprovação a adoção de uma instrução sem enumeração final como
   candidata ao próximo protocolo. A variante `no-enumeration` preservou o
   formato estrito e melhorou três perguntas, mas falhou na aritmética: não é
   uma solução universal. Não trocar automaticamente o prompt definitivo.
3. Não escolher a variante semântica somente por seus 14/16 vencedores corretos:
   ela produziu nove respostas fora do formato. Fixar explicitamente se o
   protocolo mede o primeiro token ou exige resposta completa restrita a letra;
   essas avaliações não são equivalentes.
4. Se 2B for autorizada, manter controles de fronteira, identificação do prompt,
   índices e precisão de 2A. Preservar os erros do modelo nos resultados, sem
   reponderar A, tentar novamente até acertar, selecionar prompts por questão ou
   usar o gabarito durante a inferência.
5. Antes do piloto científico, validar em exemplos independentes do conjunto
   exploratório qualquer prompt que venha a ser aprovado. Comparações de modelo
   ou precisão seriam novos diagnósticos a autorizar, não causas já demonstradas.

## Evidências e estado do trabalho

O arquivo [MLX_BEHAVIOR_EVIDENCE.json](MLX_BEHAVIOR_EVIDENCE.json) acompanha este
relatório com todos os 64 prompts, IDs de entrada, opções, gabaritos, escores,
probabilidades, massa original, geração integral e verificações principais,
além dos oito controles, versões, desenho e hashes. É uma extração sem caminhos
pessoais do JSON completo; não contém pesos. As tabelas acima usam valores
arredondados somente para apresentação.

O JSON completo, as variantes detalhadas de tokenização e os vetores de logits
permanecem em `outputs/mlx-behavior-03/`. O pico observado do alocador MLX foi
**1.128.656.824 bytes (~1,05 GiB)**; RSS máximo reportado pelo processo foi
421.658.624 bytes. O swap global ao final estava em 2.954 MiB, com capacidade
4 GiB: não representa consumo atribuível exclusivamente ao modelo. As medidas
não são garantias para lotes ou prompts maiores.

Não houve integração ao runner, uso de MMLU, mudança de fórmulas, commit ou push
nesta investigação. **A Entrega 2 não está concluída. A revisão do relatório e
a aprovação explícita continuam necessárias antes da Entrega 2B.**
