# Validação da Entrega 2B — Integração MLX-LM ao benchmark

Data: 2026-11-06. **Estado: implementada e validada localmente; aguardando revisão
técnica do responsável. Não concluída.** Branch `codex/feat-mlx-backend`.
Nenhum commit, merge ou envio ao remoto foi realizado nesta tarefa.

## Autorização e escopo

O responsável aprovou a [comparação de prompts](PROMPT_COMPARISON.md) e autorizou
explicitamente o protocolo provisório: texto `current-no-enumeration`, chat
template padrão do snapshot Qwen, zero-shot, candidatos nus A–D,
`label_next_token`, forward completo sem KV cache, logits promovidos antes de
normalizar e registro da massa original. Os relatórios
[MLX_DIAGNOSTIC.md](MLX_DIAGNOSTIC.md) e
[MLX_BEHAVIOR_VALIDATION.md](MLX_BEHAVIOR_VALIDATION.md) fundamentam os controles.

Não há adaptador MMLU, backend Transformers, API remota, nova métrica, ajuste de
prompt por questão, geração usada como pontuação ou correção dos erros do modelo.
As fórmulas em `numerics.py`, `metrics.py` e `scoring/normalization.py` não mudaram.

## Implementação e arquivos

| Arquivo | Responsabilidade / alteração |
| --- | --- |
| `src/benchmark/backends/mlx.py` (novo) | Carga exclusivamente local, manifesto SHA-256 do snapshot aprovado, template original, validação contextual, forward sem cache, extração e promoção dos logits, liberação e metadados. Imports MLX tardios. |
| `src/benchmark/backends/base.py` | Ciclo de vida comum e contratos separados para escores simulados e evidência do próximo token. |
| `src/benchmark/backends/simulated.py`, `backends/__init__.py` | Metadados do backend simulado e documentação do pacote; regra de escores preservada. |
| `src/benchmark/schemas.py` | `TokenizationEvidence`, `NextTokenOutput`, probabilidades originais e massa no `ScoringResult`; invariantes para inferência real e identificação `simulation: false`. |
| `src/benchmark/scoring/__init__.py` | Seleção da estratégia; softmax existente no vocabulário e entre candidatos. Backend não calcula acurácia, entropia ou probabilidades. |
| `src/benchmark/prompts.py` | Template `current-no-enumeration-v1` com texto exato aprovado; template numérico da simulação preservado. |
| `src/benchmark/config.py` | Seleção estrita de backend, template registrado, fixture/revisão, snapshot local e limite de tokens; rejeita campos e protocolos não suportados. |
| `src/benchmark/datasets.py` | Fixture original `controlled-v1`, quatro perguntas da comparação aprovada com quatro rotações; fixture v1 preservada. |
| `src/benchmark/runner.py`, `cli.py` | Integração pelo fluxo oficial, aplicação do template final, persistência de resultados reais e identificação adequada na CLI. |
| `src/benchmark/storage.py` | Acrescenta hashes dos fontes Python à proveniência. Publicação de `metadata.json` por último e regressões F1/F2 preservadas. |
| `configs/controlled_mlx.yaml` (novo) | Execução reproduzível de 16 apresentações, seed 42, revisão e caminho de snapshot fixados. |
| `tests/test_mlx_backend.py` (novo) | 17 testes offline com doubles de modelo, tokenizer e runtime; sem importar MLX ou NumPy. |
| `pyproject.toml` | Descrição do pacote alinhada à capacidade local; dependências básicas preservadas. |
| `README.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md` | Capacidade pública, decisões aprovadas, evidências e aprovação pendente. |
| Este relatório e `VALIDATION_DELIVERY_2B_EVIDENCE.json` (novos) | Resultado auditável da integração, com configuração, metadados e 16 registros completos. |

Os arquivos e alterações de 2A já presentes no checkout foram preservados.
`AGENTS.md` permanece na raiz. O backend não importa `prompts.py`: ele recebe
texto pronto e aplica somente o template do tokenizer. A configuração escolhe
um nome de template registrado; nesta etapa, MLX aceita a convenção aprovada de
quatro rótulos. Acrescentar outro texto é uma mudança explícita em `prompts.py`
e na validação da configuração, sem exigir lógica de prompt dentro do backend.

### Contratos e invariantes verificadas

1. Cada continuação de rótulo deve preservar **todos** os IDs do prompt, acrescentar
   um único ID, ser distinta, não especial e decodificar para o rótulo esperado.
   Tokenizar apenas a letra isolada não é considerado evidência suficiente.
2. `apply_chat_template(tokenize=True)` deve coincidir com a codificação do texto
   renderizado usando `add_special_tokens=False`. O Runner salva esse texto final.
3. O modelo recebe batch 1 completo, sem argumento de cache. A saída deve ter
   shape `[1, T, V]`; pontuam-se os logits `[0, T-1, :]`. Todos os logits do
   vocabulário são validados como finitos.
4. A promoção ocorre antes das reduções: BF16 → float32 → floats Python (double).
   O mesmo softmax estável já validado calcula ambas as distribuições. A massa
   original é a soma das probabilidades A–D no vocabulário completo; não é a soma
   trivial da distribuição condicionada, que vale 1.
5. Os schemas conferem índices, dimensões, argmax, probabilidades, massa e
   coerência entre as distribuições. O scoring registra `logits` como unidade,
   método `label_next_token`, caminho `full_no_kv_cache` e precisão utilizada.
6. Os erros do modelo permanecem registros válidos com `correct: false`. Falhas
   de tokenização/inferência têm erro explícito e métricas nulas, sem retentativas
   para obter acertos. O denominador válido continua explícito.
7. O manifesto confere os arquivos consumidos pelo carregador, incluindo pesos,
   configuração e tokenizer; arquivos inesperados no snapshot são recusados.
   A ausência do snapshot falha antes de chamar MLX-LM, sem download automático.
8. O limite operacional de até 1024 tokens é explícito na configuração; excessos
   são recusados, sem truncamento. Esse limite não foi ajustado pelos acertos.

## Ambiente e procedimento

Apple M2, 8.589.934.592 bytes de memória unificada (8 GiB), macOS 15.7.4 arm64,
Python 3.12.2. MLX/MLX Metal 0.32.4, MLX-LM 0.32.2, Transformers 5.22.1
(utilizado pelo tokenizer, não como backend), Tokenizers 0.23.3,
Hugging Face Hub 2.4.0, NumPy 2.5.4, Safetensors 0.8.0.

Reutilizados o ambiente `/tmp/llm-prompt-comparison-venv` de 2A e o snapshot
`mlx-community/Qwen2.5-0.5B-Instruct-bf16`, revisão
`56d07e766edd7159fbe12ed12d9cf114bf38bf1e`. Nenhum pacote, dataset ou peso foi
baixado nesta tarefa. As dependências continuam fixadas em
`diagnostics/mlx-requirements.txt`; não foram adicionadas ao ambiente da fundação.

Os relatórios foram confrontados com o código instalado de `mlx_lm.utils.load`
e com a documentação oficial de [carga MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py)
e [avaliação explícita de arrays MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html).
A API instalada recebe caminho local e opções de tokenizer; o backend verifica o
snapshot antes da carga, desabilita código remoto e exige arquivos locais.

### Comandos executados

Na raiz, suíte offline final (71 testes, exit 0):

```sh
PYTHONPYCACHEPREFIX=/tmp/benchmark-2b-pycache PYTHONPATH=src \
.venv/bin/python -m unittest discover -s tests -v
```

Simulação preservada (exit 0):

```sh
PYTHONPYCACHEPREFIX=/tmp/benchmark-2b-pycache PYTHONPATH=src \
.venv/bin/python -m benchmark.cli --config configs/synthetic.yaml
```

Avaliação real pelo **mesmo ponto de entrada oficial** (exit 0, executada duas vezes):

```sh
HF_HUB_OFFLINE=1 HF_HOME=.cache/mlx-diagnostic/huggingface \
PYTHONPYCACHEPREFIX=/tmp/benchmark-2b-pycache PYTHONPATH=src \
/tmp/llm-prompt-comparison-venv/bin/python -m benchmark.cli \
  --config configs/controlled_mlx.yaml
```

O cache Python temporário evita a lentidão de leitura de bytecode em arquivos
sincronizados, observada nas tarefas anteriores; não muda o protocolo numérico.
Não é necessário usar esse caminho específico em outra máquina.

## Testes e resultados

| Verificação | Resultado |
| --- | --- |
| 54 testes existentes da fundação, sem alterações | Passaram; regressões matemáticas, F1/F2/F3, CLI e simulação preservadas. |
| Suíte final completa | **71 testes, OK, 0,730 s**, exit 0. São 54 existentes + 17 novos, não somas de reexecuções. |
| Novos testes offline | Carga local/ciclo de vida, template exato, última posição, promoção antes da normalização, ausência de cache, massa com referência independente, validação contextual e rejeição de registros inconsistentes. |
| Regressões novas de falhas | Fronteira alterada, múltiplos tokens, token especial/duplicado/incompatível, prompt vazio/longo, shape inválido, NaN fora dos candidatos, template divergente, falha no load acompanhada de erro no close, falha no close após resultados, erro por item persistido. |
| Configuração e dependências opcionais | Round-trip YAML; seleção MLX pelo Runner; rejeição de revisão, modelo, cache, método e template não aprovados; importar o núcleo não importa MLX/Metal. |
| CLI simulada | 3 válidas, 0 falhas; acurácia 2/3; entropia média 0,7603746960459724. |
| CLI real | Duas execuções, ambas com 16 válidas e 0 falhas operacionais. |
| Auditoria de 16 resultados reais | Prompts, IDs, logits dos candidatos e probabilidades condicionais **exatamente iguais** ao diagnóstico aprovado para `current-no-enumeration`. |
| Auditoria das probabilidades originais | Maior diferença contra a referência NumPy de 2A: 4,440892098500626e-16. |
| Persistência e proveniência | Quatro arquivos obrigatórios presentes; sem arquivo `.tmp`; configuração resolvida, fontes, pesos, tokenizer, template, versões, IDs/ordem e método conferidos. |
| Repetição real | Os dois `results.jsonl` são idênticos byte a byte; os hashes da execução final correspondem ao código Python final. |

A suíte usa doubles para que possa executar sem Apple Silicon; isso não substitui
as execuções reais. Os testes não codificam a acurácia do Qwen como requisito de
qualidade. A referência congelada do prompt protege o texto aprovado.

### Execuções reais identificadas

- Primeira: `20261106T170913518884Z_2ef0f21af50b4966b632138fe75d707f`.
- Final: `20261106T171107979803Z_1c1baaac1d0c4cd6857cdd930d858b50`.
- Simulada: `20261106T171013600664Z_6f10f7c6a27c4925bf6c8cc2664c4cd9`.

A execução real final ocorreu das 17:11:07,979803 às 17:11:13,035610 UTC
(14:11 no horário de Brasília). O intervalo do Runner inclui hashing/carga e
processamento, mas não equivale a throughput de geração. Não houve geração livre
na avaliação real: as respostas previstas são argmax dos quatro escores.

| Resultado real final | Valor |
| --- | --- |
| Questões autorais / apresentações | 4 / 16, rotações correlacionadas |
| Registros válidos / falhas operacionais | 16 / 0 |
| Acertos / erros do modelo | 11 / 5 |
| Acurácia | 0,6875 |
| Entropia normalizada média | 0,2938674223513097 |
| Seleções A / B / C / D | 7 / 2 / 3 / 4 |
| Massa original mínima / média / máxima | 0,9935778729402164 / 0,998754582060833 / 0,9998760829055098 |
| Status / simulation / close_error | `completed` / `false` / `null` |

`completed` é o status operacional de uma execução persistida, **não** o aceite
da entrega ou a afirmação de que o modelo acertou tudo. Na ordem de gabaritos
A, B, C, D, as predições foram `ADAD` (triangle), `ABCD` (planet), `ABCD` (organ)
e `AAAC` (meter), como no diagnóstico. Não houve tratamento artificial dos erros.

O [arquivo de evidências](VALIDATION_DELIVERY_2B_EVIDENCE.json) acompanha este
relatório com os quatro conteúdos necessários à revisão (configuração, metadados,
resumo e registros) e hashes dos artefatos originais. Os diretórios originais
permanecem em `outputs/`, ignorados. Commit-base registrado:
`f9c53528ca04e6456f20bbfa7119fb29d40a5175`, `dirty: true`; os hashes dos arquivos
Python identificam o código local que efetivamente executou.

## Recursos observados

Antes da avaliação: aproximadamente 6,1 GiB de disco disponíveis, 8 GiB de memória
unificada e swap global de 1.439,38 MiB usados em 2 GiB. Após a validação:
aproximadamente 6,1 GiB de disco e swap de 1.611,31 MiB usados. Os pesos existentes
ocupam 988.097.734 bytes; SHA-256:
`39da1e4174f12953ad7ace17239da672aa2b4b9ba723a09835963484c5a05b41`.

O pico do alocador MLX nas duas avaliações foi **1.125.887.928 bytes**, cerca de
1,049 GiB. Após fechar o backend, o contador ativo ficou em **8 bytes**. O limite
de cache do alocador foi 256 MiB, restaurado no encerramento; esse cache de memória
não é KV cache. Swap é global e não pode ser atribuído exclusivamente ao benchmark.
Essas medições não garantem folga para prompts maiores, lotes ou execução prolongada.

## Limitações e pendências para revisão

- O protocolo é provisório e restrito ao snapshot aprovado; novos modelos exigem
  outro manifesto e validação contextual/numérica. O limite de entrada é até
  1024 tokens; não houve teste de capacidade no limite.
- A fixture reutiliza deliberadamente os casos de 2A para verificar a integração,
  não para estimar generalização. Não há MMLU nem avaliação científica independente.
- **Forward completo e KV cache podem gerar logits diferentes em BF16.** Em 2A,
  houve diferença máxima de 0,423828125 no vocabulário e dois argmax globais
  diferentes. 2B fixa o caminho completo e não mistura distribuições de ambos.
- A entropia é condicionada aos rótulos. Alta massa, baixa entropia ou repetição
  exata não garantem acerto, calibração ou raciocínio correto.
- O manifesto e os hashes dos fontes ajudam a auditar o checkout dirty, mas não
  substituem um commit revisado e um ambiente reproduzível. Não foi criado commit.
- Os vetores completos do vocabulário são transitórios no fluxo oficial; os
  resultados preservam escores dos candidatos, probabilidades e massa, não todos
  os logits. Os vetores de diagnóstico 2A permanecem como referência local.
- Persistência conserva o contrato da fundação para exceções de I/O; não promete
  recuperação após queda de energia. Falhas anteriores à carga não geram execução
  completa; erros de encerramento posteriores aos resultados são registrados.
- Testes offline e duas rodadas locais não demonstram estabilidade em outros
  sistemas, versões ou GPUs. Nada foi instalado ou baixado para ampliar o escopo.

**Próximo passo:** responsável revisar este relatório, contratos, configuração e
limitações; decidir o aceite da Entrega 2B. O roadmap mantém a revisão pendente.
Somente depois desse aceite e de autorização específica deve-se iniciar a
Entrega 3. Não há pendência conhecida que tenha impedido a validação controlada
solicitada, mas 2B não foi marcada como concluída automaticamente.
