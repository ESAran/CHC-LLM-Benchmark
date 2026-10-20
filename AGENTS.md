# Instruções de desenvolvimento para agentes de código

Estas instruções são públicas e se aplicam ao trabalho neste repositório. Respeite também quaisquer instruções mais específicas que já existam no projeto.

## Antes de começar uma tarefa

1. Consulte `README.md`, `docs/ARCHITECTURE.md` e **`docs/ROADMAP.md`** conforme a área modificada.
2. Identifique o estágio atual no roadmap e mantenha a implementação no escopo pedido.
3. Verifique o código existente antes de propor novos módulos ou substituir implementações.

## Durante o desenvolvimento

- Após a publicação inicial da fundação na `main`, desenvolver alterações em branches de feature e integrá-las à `main` após revisão e validação. Não trabalhar diretamente na `main` sem autorização explícita.
- Faça mudanças pequenas, testáveis e proporcionais à tarefa.
- Preserve o desacoplamento entre datasets, prompts, backends, scoring, métricas, runner e storage.
- Não altere silenciosamente fórmulas, protocolos experimentais ou o significado das probabilidades. Proponha mudanças metodológicas para revisão antes de implementá-las.
- Não presuma que tokenização, logits ou métodos de pontuação estejam validados sem evidência.
- Não invente resultados, testes executados, dados experimentais ou arquivos existentes.
- Não inclua informações pessoais, credenciais, tokens, caminhos locais sensíveis ou grandes arquivos de modelos no repositório público.

## Ao concluir uma tarefa

1. Execute os testes apropriados; informe claramente quais testes foram executados e seus resultados. Caso não seja possível, explique por quê.
2. **Atualize `docs/ROADMAP.md` quando houver progresso, decisão, bloqueio ou mudança relevante**: status da entrega, critérios atendidos e entrada curta no histórico com evidência e próximo passo.
3. Não marque `[x]` sem verificação. Código gerado não equivale automaticamente a etapa concluída.
4. Mantenha `README.md` e `docs/ARCHITECTURE.md` coerentes quando mudar uma capacidade pública ou uma decisão arquitetural aprovada.
5. No resumo final, informe: alterações, testes, estado do roadmap e próximo passo recomendado.

**Não transforme a atualização do roadmap em um commit separado obrigatório.** Ela deve acompanhar as mudanças relevantes do trabalho, preservando a rastreabilidade do Git.
