"""Relatório/figuras a partir de artefatos persistidos; nenhuma inferência."""

import argparse
from datetime import date
from importlib.metadata import version
import json
from pathlib import Path

from .analysis import analyze_test, load_approved_test
from .mmlu import file_hash
from .mmlu_selection import SUBJECTS
from .storage import code_metadata

NAMES = dict(zip(SUBJECTS, ('Psicologia', 'História mundial', 'Biologia')))
FIGURES = ('accuracy_by_subject.png', 'entropy_by_outcome.png',
           'entropy_by_subject.png', 'label_frequencies.png')


def build_plot_data(analysis):
    rows = [r for r in analysis['records'] if r['error'] is None]
    return {
        'accuracy_by_subject': [
            {'subject': s, 'accuracy': g['accuracy'], 'correct_count': g['correct_count'],
             'denominator': g['accuracy_denominator']} for s, g in analysis['by_subject'].items()],
        'entropy_by_outcome': {name: [r['normalized_entropy'] for r in rows if r['correct'] == outcome]
                               for name, outcome in (('correct', True), ('incorrect', False))},
        'entropy_by_subject': {s: [r['normalized_entropy'] for r in rows if r['subject'] == s]
                               for s in analysis['by_subject']},
        'label_frequencies': {name: {k: group[k] for k in ('gold_frequencies', 'prediction_frequencies')}
                              for name, group in [('overall', analysis['overall']), *analysis['by_subject'].items()]},
    }


def render_figures(data, directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    directory.mkdir(parents=True, exist_ok=True)
    blue, orange = '#29677c', '#b45b35'
    style = {'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
             'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
             'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True}
    with plt.rc_context(style):
        def save(fig, name):
            fig.savefig(directory/name, dpi=200, bbox_inches='tight')
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.0))
        points = data['accuracy_by_subject']
        ax.barh(range(len(points)), [p['accuracy'] for p in points], color=blue, height=.55)
        ax.set_yticks(range(len(points)), [NAMES[p['subject']] for p in points])
        ax.invert_yaxis();ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel('Acurácia observada no test');ax.set_title('Acurácia por disciplina')
        ax.grid(axis='x', alpha=.18)
        for i, point in enumerate(points):
            ax.text(point['accuracy']+.02, i,
                    f"{point['correct_count']}/{point['denominator']} · {point['accuracy']:.0%}", va='center')
        fig.tight_layout();save(fig, FIGURES[0])

        def boxes(values, labels, colors, title, name, width):
            fig, ax = plt.subplots(figsize=(width, 4.5))
            artists = ax.boxplot(values, positions=range(len(values)), widths=.36,
                                patch_artist=True, showfliers=False, whis=(0, 100),
                                medianprops={'color': '#202020', 'linewidth': 1.5})
            for patch, color in zip(artists['boxes'], colors):
                patch.set_facecolor(color);patch.set_alpha(.18)
            for i, (group, color) in enumerate(zip(values, colors)):
                # Deslocamento horizontal determinístico apenas para legibilidade.
                xs = [i + ((j % 7)-3)*.028 for j in range(len(group))]
                ax.scatter(xs, group, s=23, color=color, alpha=.8, zorder=3, linewidths=0)
            ax.set_xticks(range(len(values)), [f'{label}\n(n={len(group)})' for label, group in zip(labels, values)])
            ax.set_ylim(-.025, 1.025);ax.set_yticks([0, .2, .4, .6, .8, 1])
            ax.set_ylabel('Entropia normalizada entre A–D');ax.set_title(title)
            ax.grid(axis='y', alpha=.18)
            fig.tight_layout();save(fig, name)
        boxes(list(data['entropy_by_outcome'].values()), ['Acertos', 'Erros'], [blue, orange],
              'Entropia por resultado — test', FIGURES[1], 7)
        boxes(list(data['entropy_by_subject'].values()), [NAMES[s] for s in data['entropy_by_subject']],
              [blue]*3, 'Entropia por disciplina — test', FIGURES[2], 9)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (name, values) in zip(axes.flat, data['label_frequencies'].items()):
            gold, pred = values['gold_frequencies'], values['prediction_frequencies']
            ax.bar([i-.18 for i in range(4)], list(gold['counts'].values()), width=.34, color=blue, label='Gabarito')
            ax.bar([i+.18 for i in range(4)], list(pred['counts'].values()), width=.34, color=orange, label='Predição')
            ax.set_xticks(range(4), list('ABCD'));ax.set_ylabel('Contagem')
            ax.set_title(f"{'Geral' if name == 'overall' else NAMES[name]} (n={gold['denominator']})")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y', alpha=.18)
            upper = max(*gold['counts'].values(), *pred['counts'].values())
            ax.set_ylim(0, upper+max(2, upper*.18))
            for bars in ax.containers:
                ax.bar_label(bars, padding=2, fontsize=9)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.suptitle('Frequências de alternativas — test', fontsize=14, y=.99)
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.95), ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .89));save(fig, FIGURES[3])
    return {name: file_hash(directory/name) for name in FIGURES}


def _number(value, digits=6):
    return '—' if value is None else f'{value:.{digits}f}'


def markdown_report(evidence):
    a, source = evidence['analysis'], evidence['source']
    groups = [('Geral', a['overall'])] + [(NAMES[s], g) for s, g in a['by_subject'].items()]
    lines = [
        '# Análise exploratória do piloto MMLU — Entrega 3C', '',
        f"Data: {evidence['date']}. **Implementada e verificada localmente; aguardando revisão técnica. 3C não concluída.**", '',
        '## Escopo e procedência', '',
        '3B foi aprovada tecnicamente em 2026-11-08 e encerrada em commit local exclusivo '
        '`5db73f95bcc5f14e1c245ecf06bbf94e553cb434`, após revisão do diff e **100 testes aprovados (1,033 s)**. '
        'Esta etapa usa somente os **60 registros test**, 20 por disciplina, de `mmlu-3a-v1`. '
        'Os 15 dev permanecem separados e exclusivamente técnicos; não entram em tabelas, exemplos ou gráficos experimentais.', '',
        f"Execução test: `{source['run_id']}`. Dataset `cais/mmlu@{source['dataset_revision']}`; "
        f"modelo `{source['config']['model']['name']}@{source['config']['model']['revision']}`. "
        'Prompt, scoring e resultados são os persistidos em 3B. **Nenhuma LLM foi executada novamente.**', '',
        'O leitor confere os quatro hashes de artefatos contra a evidência aprovada, reconstrói os schemas, '
        'valida resumos/configuração/procedência e confronta os 60 IDs, ordem, conteúdo, prompts e tokens com 3A. '
        'Não é necessário carregar pesos, tokenizer, Parquets ou acessar rede. Entradas dev/mistas, duplicações '
        'de IDs ou artefatos adulterados são recusados. A carga do piloto exige exatamente 60 test válidos.', '',
        f"SHA-256 da evidência 3B: `{source['pilot_evidence_sha256']}`. "
        f"SHA-256 do manifesto 3A: `{source['selection_manifest_sha256']}`. "
        'Hashes de cada artefato, código, versões, valores dos gráficos e todos os registros analisados estão no '
        '[JSON de evidências](MMLU_EXPLORATORY_EVIDENCE.json). Os arquivos originais não foram modificados.', '',
        '## Método de análise', '',
        '- Acurácia e médias gerais reutilizam `summarize`; os schemas conferem a entropia e o scoring existentes. '
        'Não há fórmula nova de eficiência informacional ou nova normalização.',
        '- Denominador da acurácia: registros válidos; falhas técnicas são contadas separadamente. Neste piloto: '
        '60/60 válidos, 0 falhas. Gabaritos usam todos os itens; predições, apenas válidos. As contagens são sempre explícitas.',
        '- Descritivas: média, mínimo, máximo, mediana, quartis inclusivos por interpolação linear em `(n−1)p` '
        'e desvio padrão populacional descritivo. Grupos vazios têm `n=0` e valores nulos, não zero. '
        'O JSON contém histogramas de entropia em dez intervalos de largura 0,1, fechados à esquerda; o último inclui 1.',
        f"- Alta concentração: **máx. p(A–D) ≥ {a['policy']['high_concentration_threshold_inclusive']:.2f}**, "
        'limiar operacional descritivo declarado antes da tabulação de 3C. Não é limiar calibrado, probabilidade de acerto '
        'ou filtro de questões. Todos os erros que satisfazem o critério são listados; os demais continuam nos agregados.',
        '- Exemplos: menor e maior entropia por disciplina × acerto/erro; empate por ID, sem duplicar um mesmo exemplo. '
        'A regra mostra extremos de ambos os resultados, não casos representativos, e não foi ajustada para sustentar uma hipótese.',
        '- Somente descrições desta amostra. Não foram realizados testes de significância, inferência causal, '
        'avaliação de calibração ou comparação cognitiva entre disciplinas.', '',
        '## Resultados observados', '',
        '### Acurácia e denominadores', '',
        '| Grupo | Total | Válidas | Falhas | Acertos | Erros | Acurácia (acertos/válidas) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, g in groups:
        lines.append(f"| {name} | {g['total_count']} | {g['valid_count']} | {g['failed_count']} | {g['correct_count']} | {g['incorrect_count']} | {g['correct_count']}/{g['accuracy_denominator']} = {_number(100*g['accuracy'], 2)}% |")
    lines += ['', '![Acurácia por disciplina](mmlu_exploratory/accuracy_by_subject.png)', '',
              'Figura 1. Proporções observadas, com denominadores explícitos, sem intervalos inferenciais. '
              'A ordem das disciplinas é a da seleção, não um ranking de capacidade.', '',
              '### Entropia geral e por disciplina', '',
              '| Grupo | n | Média | DP populacional | Mínimo | Q1 | Mediana | Q3 | Máximo |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, g in groups:
        d = g['entropy']
        lines.append(f"| {name} | {d['count']} | " + ' | '.join(_number(d[k]) for k in ('mean','std_population','min','q1','median','q3','max'))+' |')
    lines += ['', '![Entropia por disciplina](mmlu_exploratory/entropy_by_subject.png)', '',
              'Figura 2. Cada ponto é um registro test. Caixas: Q1–Q3; linha: mediana; hastes: mínimo–máximo. '
              'O deslocamento horizontal determinístico apenas reduz sobreposição. Nenhuma observação foi removida.', '',
              '### Entropia de acertos e erros', '',
              '| Grupo | Resultado | n | Média | Mediana | Q1 | Q3 | Mínimo | Máximo |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, g in groups:
        for outcome, label in (('correct', 'Acerto'), ('incorrect', 'Erro')):
            d = g['entropy_by_outcome'][outcome]
            lines.append(f"| {name} | {label} | {d['count']} | "+' | '.join(_number(d[k]) for k in ('mean','median','q1','q3','min','max'))+' |')
    lines += ['', '![Entropia em acertos e erros](mmlu_exploratory/entropy_by_outcome.png)', '',
              'Figura 3. Todos os acertos e erros, com n próprios. Convenção das caixas idêntica à Figura 2; '
              'os grupos têm tamanhos diferentes. A entropia não expressa probabilidade de acerto.', '',
              '### Erros com alta concentração condicional', '',
              '| Grupo | Erros com máx. p ≥ 0,90 | Total de erros válidos | Fração dos erros |',
              '| --- | ---: | ---: | ---: |']
    for name, g in groups:
        h = g['high_concentration_errors']
        lines.append(f"| {name} | {h['count']} | {h['incorrect_denominator']} | {_number(h['fraction_of_incorrect'])} |")
    by_id = {r['id']: r for r in a['records']}
    high_ids = a['overall']['high_concentration_errors']['ids']
    lines += ['', 'Todos os casos que satisfazem o critério, em ordem decrescente de máx. p (desempate por ID):', '',
              '| ID (sufixo) | Gabarito | Predição | máx. p condicional | Entropia | Massa original |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    for identity in high_ids:
        r = by_id[identity]
        lines.append(f"| `{identity.split('/', 2)[2]}` | {r['gold']} | {r['prediction']} | {_number(r['max_conditional_probability'])} | {_number(r['normalized_entropy'])} | {_number(r['original_label_probability_mass'])} |")
    lines += ['', 'O prefixo comum dos IDs abreviados nas tabelas é '
              f"`cais/mmlu@{source['dataset_revision']}/`. IDs completos, enunciados, alternativas e distribuições "
              'estão no JSON. Não há ID nativo por linha: a identidade deriva de revisão/disciplina/split/índice base zero.', '',
              '### Frequências de gabaritos e predições', '',
              '| Grupo | Série | Denominador | A | B | C | D |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, g in groups:
        for field, label in (('gold_frequencies','Gabarito'), ('prediction_frequencies','Predição')):
            f = g[field]
            lines.append(f"| {name} | {label} | {f['denominator']} | "+' | '.join(str(f['counts'][letter]) for letter in 'ABCD')+' |')
    lines += ['', '![Frequências de alternativas](mmlu_exploratory/label_frequencies.png)', '',
              'Figura 4. Contagens dos mesmos registros. Os painéis indicam seus tamanhos e escalas. '
              'Diferenças de frequência, isoladamente, não demonstram existência nem ausência de viés de posição.', '',
              '### Massa original dos quatro rótulos', '',
              '| Grupo | n | Média | DP populacional | Mínimo | Q1 | Mediana | Q3 | Máximo |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, g in groups:
        d = g['original_label_mass']
        lines.append(f"| {name} | {d['count']} | "+' | '.join(_number(d[k]) for k in ('mean','std_population','min','q1','median','q3','max'))+' |')
    lines += ['', 'A massa é a soma original das probabilidades A–D no vocabulário, antes da renormalização. '
              'As distribuições usadas na entropia somam 1 **entre os rótulos**. Massa elevada não implica correção ou calibração.', '',
              '## Exemplos individuais auditáveis', '',
              'São incluídos extremos simétricos de acertos e erros nas três disciplinas. Não são uma amostra '
              'representativa nem uma seleção de casos favoráveis; os 60 registros permanecem disponíveis no JSON.', '',
              '| ID (sufixo) | Resultado / regra | Gabarito → predição | p(A), p(B), p(C), p(D) | Entropia |',
              '| --- | --- | --- | --- | ---: |']
    for ex in a['examples']:
        r = by_id[ex['id']]
        roles = ', '.join('menor H' if role == 'min_entropy' else 'maior H' for role in ex['roles'])
        lines.append(f"| `{r['id'].split('/',2)[2]}` | {'Acerto' if r['correct'] else 'Erro'} / {roles} | {r['gold']} → {r['prediction']} | "+', '.join(_number(p) for p in r['probabilities'])+f" | {_number(r['normalized_entropy'])} |")
    correct = a['overall']['entropy_by_outcome']['correct'];wrong = a['overall']['entropy_by_outcome']['incorrect']
    lines += ['', '## Interpretações preliminares e limitações', '',
              f"**Observado:** acertos (n={correct['count']}) têm entropia média {_number(correct['mean'])}; "
              f"erros (n={wrong['count']}) têm média {_number(wrong['mean'])}. "
              f"Há {len(high_ids)} erros com máx. p condicional ≥ 0,90. Esses valores descrevem esta amostra e não fornecem "
              'uma regra de decisão sobre a correção de uma resposta individual. '
              f"As faixas se sobrepõem: acertos {_number(correct['min'])}–{_number(correct['max'])}; "
              f"erros {_number(wrong['min'])}–{_number(wrong['max'])}.", '',
              '**Interpretação preliminar:** concentração e correção devem ser consideradas separadamente. '
              'Os erros concentrados mostram que uma distribuição restrita aos rótulos pode favorecer uma alternativa incorreta. '
              'As estatísticas agrupadas não estabelecem calibração, mecanismo causal, relação universal ou validade fora destes itens.', '',
              '- Apenas 60 itens, três disciplinas e um único modelo/snapshot/prompt. Os 20 itens por disciplina '
              'não permitem interpretar diferenças como características cognitivas ou rankings gerais de capacidade.',
              '- Não há rotações controladas neste piloto. Frequências dependem também do conteúdo, gabaritos '
              'e amostragem; não permitem concluir presença ou ausência de viés de posição.',
              '- Não foram estimadas incertezas populacionais, correlações, significância estatística, calibração '
              'ou contaminação de treinamento. Estatísticas de subgrupos são exploratórias.',
              '- Dev não participa da análise experimental. A anomalia A/C em biologia dev/2 permanece '
              'documentada em 3B e não foi removida nem corrigida para esta análise.',
              '- Os vetores completos do vocabulário não foram persistidos. Esta análise usa a massa e '
              'os escores registrados e validados em 3B, sem nova inferência.',
              '- Qualquer protocolo futuramente otimizado com base neste test deve ser avaliado em **nova amostra '
              'independente**. Não reutilizar estes mesmos 60 itens como validação independente após ajustes.', '',
              '## Reprodução e arquivos', '',
              'Agregações: `src/benchmark/analysis.py`; leitura compartilhada: `src/benchmark/results.py`; '
              'relatório e gráficos: `src/benchmark/analysis_report.py`. O leitor JSON antes presente no diagnóstico '
              '3B foi extraído sem alterar sua lógica, para reaproveitar os schemas. Scoring, métricas, prompt, '
              'seleção, Runner, backend e resultados originais não foram alterados.', '',
              'Dependências opcionais dos gráficos: `diagnostics/analysis-requirements.txt`. '
              'Matplotlib, NumPy e Pillow já estavam disponíveis no ambiente usado; não houve downloads. '
              'Versões e hashes dos fontes/PNGs estão no JSON. Estatísticas usam a biblioteca padrão.', '',
              '```sh', 'XDG_CACHE_HOME=/tmp/mmlu-analysis-cache MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mmlu-analysis-mpl \\',
              'PYTHONPYCACHEPREFIX=/tmp/mmlu-3c-pycache PYTHONPATH=src \\',
              '  .venv/bin/python -m benchmark.analysis_report \\',
              f"  --date {evidence['date']} --output docs/reports", '```', '',
              'O comando lê os resultados test indicados na evidência de 3B e grava apenas o relatório, '
              'JSON e quatro PNGs. Os caminhos das fontes e seus hashes constam do JSON. '
              'Valores não arredondados e todos os dados utilizados nas figuras são preservados; '
              'somente a apresentação nas tabelas é arredondada.', '',
              'Verificação automatizada e inspeção visual desta entrega estão registradas no roadmap. '
              'Este relatório é gerado pelo módulo; sua regeneração não executa a LLM.', '',
              '**Próximo passo:** revisão técnica da 3C. Sem commit de 3C, push ou merge; entrega não marcada como concluída.', '']
    return '\n'.join(lines)


def generate(evidence_path, manifest_path, output, report_date):
    records, source = load_approved_test(evidence_path, manifest_path)
    analysis = analyze_test(records)
    plot_data = build_plot_data(analysis)
    output.mkdir(parents=True, exist_ok=True)
    figure_hashes = render_figures(plot_data, output/'mmlu_exploratory')
    evidence = {'analysis_version': 'mmlu-exploratory-v1', 'date': report_date,
                'status': 'pending_technical_review', 'source': source, 'analysis': analysis,
                'plot_data': plot_data, 'figure_sha256': figure_hashes, 'code': code_metadata(),
                'plotting_versions': {p: version(p) for p in ('matplotlib','numpy','pillow')}}
    (output/'MMLU_EXPLORATORY_EVIDENCE.json').write_text(
        json.dumps(evidence, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    (output/'MMLU_EXPLORATORY_ANALYSIS.md').write_text(markdown_report(evidence), encoding='utf-8')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/reports/MMLU_PILOT_EVIDENCE.json'))
    parser.add_argument('--manifest', type=Path, default=Path('docs/reports/MMLU_SELECTION_MANIFEST.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--date', type=date.fromisoformat, required=True)
    args = parser.parse_args()
    result = generate(args.evidence, args.manifest, args.output, args.date.isoformat())
    print(json.dumps({'records': result['analysis']['overall']['total_count'],
                      'summary': result['analysis']['overall'], 'figures': list(result['figure_sha256'])}))


if __name__ == '__main__':
    main()
