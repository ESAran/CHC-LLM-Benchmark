"""Relatório integral a partir da auditoria persistida, sem executar a LLM."""
import argparse
from datetime import date
import gzip
import io
import json
from pathlib import Path
import re
import tarfile
from importlib.metadata import version

from benchmark.analysis import macro_accuracy
from benchmark.analysis_report import NAMES, build_plot_data, render_figures
from benchmark.mmlu import file_hash, canonical_hash
from benchmark.storage import code_metadata


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def number(value):
    return '—' if value is None else f'{value:.6f}'


def archive_run(directory, target):
    """Arquiva bytes originais, sem caminhos pessoais, com ordem/data determinísticas."""
    with target.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as archive:
            for name in ('config.yaml','metadata.json','results.jsonl','summary.json'):
                content=(directory/name).read_bytes()
                info=tarfile.TarInfo(f'{directory.name}/{name}')
                info.size=len(content);info.mode=0o644
                archive.addfile(info,io.BytesIO(content))
    return {'path':str(target),'sha256':file_hash(target),'size_bytes':target.stat().st_size,
            'contents':'four_original_runner_artifacts_unmodified'}


def histogram_figure(analysis, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    groups=[('Geral',analysis['overall'])]+[(NAMES[s],g) for s,g in analysis['by_subject'].items()]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,'figure.facecolor':'white',
                         'axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,(name,g) in zip(axes.flat,groups):
            h=g['entropy_histogram']
            ax.bar(h['edges'][:-1],h['counts'],width=.095,align='edge',color='#29677c')
            ax.set_xlim(0,1);ax.set_xlabel('Entropia normalizada entre A–D');ax.set_ylabel('Contagem')
            ax.set_title(f'{name} (n={h["denominator"]})')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',alpha=.18)
            ax.set_axisbelow(True)
        fig.tight_layout();fig.savefig(path,dpi=200,bbox_inches='tight',facecolor='white');plt.close(fig)
    return file_hash(path)


def report(e):
    a=e['analysis'];pilot=e['pilot_analysis'];groups=[('Geral',a['overall'])]+[(NAMES[s],g) for s,g in a['by_subject'].items()]
    pre=e['preflight'];overlap=e['overlap_comparison'];meta=e['metadata'];g=a['overall']
    text=['# Avaliação integral das três disciplinas MMLU','',
          f'Data: {e["date"]}. **Execução e análise aguardando revisão técnica; sem commit ou push.**','',
          '## Delimitação e protocolo','',
          'Esta execução substitui a proposta de 1.000 questões pela totalidade dos test de Psicologia (545), História mundial (237) e Biologia (310): **1.092 índices originais**. Não é uma avaliação de todas as disciplinas do MMLU. Dev fica excluído; o piloto congelado de 60 questões e todos os relatórios anteriores foram preservados.','',
          f'Dataset `cais/mmlu@{meta["dataset"]["revision"]}`; seleção `mmlu-full-test-v1`, sem amostragem, seed nula/não aplicável. Ordem: Psicologia, História mundial e Biologia, índices originais crescentes base zero. [Manifesto integral](MMLU_FULL_SELECTION_MANIFEST.json), SHA-256 `{e["selection_manifest_sha256"]}`.','',
          f'Modelo `{meta["model"]["name"]}@{meta["model"]["revision"]}`. Backend MLX; prompt exato `current-no-enumeration-v1`, chat template padrão fixado, zero-shot, A–D sem espaços, `label_next_token`, batch 1, forward completo sem KV cache, logits em T−1 promovidos antes das funções matemáticas existentes. Limite 1.024 tokens, sem truncamento, exclusão, retentativa por desempenho ou ajuste do protocolo.','',
          '**Caminhos com e sem KV cache podem produzir logits diferentes em BF16.** Foi usado exclusivamente o caminho completo já aprovado.','',
          '## Preflight e integridade','',
          table(['Disciplina','Registros test','Índices','Tokens mín.–máx.'],
                [[NAMES[s],d['count'],f'0–{d["count"]-1}',f'{d["min_tokens"]}–{d["max_tokens"]}'] for s,d in pre['by_subject'].items()]),'',
          'Hashes/tamanhos dos três Parquets, 1.092 IDs únicos, schema, quatro alternativas e gabaritos foram validados. Tokens contextuais A/B/C/D = 32/33/34/35 em todos os itens. Nenhuma entrada incompatível: máximo de 788 tokens. Os hashes de conteúdo, prompt e tokens dos 60 itens de sobreposição coincidem com 3A. Nenhum peso foi carregado no preflight.','',
          'Há **11 pares de questões com texto e alternativas duplicados em Psicologia**, sem conflito de gabarito. Foram mantidos os 22 índices originais; nenhuma deduplicação. Não há alternativas repetidas dentro das questões test. A anomalia A/C de Biologia dev/2 permanece somente na evidência técnica anterior.','',
          table(['Pares duplicados (Psicologia/test, índices originais)'],
                [[', '.join(i.rsplit('/',1)[1] for i in d['ids'])] for d in pre['duplicates']['same_question_and_choices']]),'',
          '## Execução, cobertura e reprodução do piloto','',
          f'Run `{meta["run_id"]}`, de `{meta["started_at"]}` a `{meta["finished_at"]}`. '
          f'Status `{meta["status"]}`; **{g["valid_count"]}/{g["total_count"]} registros válidos**, {g["failed_count"]} falhas técnicas; erro de encerramento: `{meta["close_error"]}`. '
          'A persistência final e os quatro arquivos foram conferidos. Uma única execução integral; nenhum novo dev ou rotação.','',
          'Tolerâncias fixadas **antes da inferência**: igualdade dos logits BF16 promovidos e das predições; diferença absoluta ≤ `1e-12`, relativa zero, para probabilidades condicionais/originais, massa e entropia. Esta é a tolerância dos contratos numéricos existentes, não um ajuste posterior aos resultados.','',
          f'Comparação por ID dos **{overlap["count"]} itens**: `{overlap["passed"]}`. '
          'Conteúdo, prompt, tokenização, posição dos logits, método, predição e acerto conferidos. Diferenças máximas:', '',
          table(['Grandeza','Diferença absoluta máxima','Tolerância'],
                [[k,number(v),str(overlap['absolute_tolerances'][k])] for k,v in overlap['max_absolute_differences'].items()]),'',
          'A auditoria interrompe a consolidação se encontrar divergência; todos os controles individuais estão no JSON. '
          'Esta repetição mede reprodutibilidade técnica dos mesmos itens, **não validação independente**.','',
          '## Método de análise','',
          'Agregações reutilizam `benchmark.analysis`, `metrics.summarize` e os schemas. Acurácia global = acertos/itens válidos; '
          'macro = média simples das três acurácias disciplinares, com pesos iguais. Falhas têm contagem e motivo explícitos, não entram no denominador válido. '
          'Macro fica indefinida se alguma disciplina não tiver itens válidos. Não há fórmula de eficiência informacional.','',
          'Entropia de Shannon normalizada é calculada sobre probabilidades **condicionadas a A–D**. '
          'Quartis usam interpolação inclusiva; desvio padrão é populacional descritivo. Histogramas mantêm todos os registros válidos, em dez intervalos de largura 0,1: [esquerda, direita), último incluindo 1. '
          'Erros concentrados usam o limiar descritivo anterior máx. p ≥ 0,90, sem alegação de calibração. Exemplos seguem a mesma regra simétrica de extremos por disciplina/acerto, com desempate por ID.','',
          '## Acurácia global, por disciplina e macro','',
          table(['Grupo','Total','Válidas','Falhas','Acertos','Erros','Acurácia'],
                [[n,d['total_count'],d['valid_count'],d['failed_count'],d['correct_count'],d['incorrect_count'],
                  f'{d["correct_count"]}/{d["accuracy_denominator"]} = {number(d["accuracy"])}'] for n,d in groups]),'',
          f'**Acurácia macro:** {number(e["macro_accuracy"]["value"])}; denominador: {e["macro_accuracy"]["subject_count"]} disciplinas. '
          'A global pondera as disciplinas por seus números de questões válidas; a macro lhes dá peso igual.','',
          '![Acurácia por disciplina](mmlu_full/accuracy_by_subject.png)','',
          '## Entropia e distribuições','']
    keys=('mean','std_population','min','q1','median','q3','max')
    text += [table(['Grupo','n','Média','DP','Mín.','Q1','Mediana','Q3','Máx.'],
                   [[n,d['entropy']['count']]+[number(d['entropy'][k]) for k in keys] for n,d in groups]),'',
             '![Entropia por disciplina](mmlu_full/entropy_by_subject.png)','',
             'Caixas: Q1–Q3, mediana e hastes mínimo–máximo; todos os pontos são registros reais, com deslocamento horizontal determinístico. Nenhum outlier foi excluído.','',
             '![Histogramas de entropia](mmlu_full/entropy_histograms.png)','',
             'Histogramas em contagens, com escalas por painel e todos os denominadores explícitos.','',
             '### Acertos e erros','',
             table(['Grupo','Resultado','n','Média','Mediana','Mín.','Máx.'],
                   [[n,label,d['entropy_by_outcome'][key]['count']]+
                    [number(d['entropy_by_outcome'][key][k]) for k in ('mean','median','min','max')]
                    for n,d in groups for key,label in (('correct','Acertos'),('incorrect','Erros'))]),'',
             '![Entropia por resultado](mmlu_full/entropy_by_outcome.png)','',
             '## Frequências A–D','',
             table(['Grupo','Série','Denominador','A','B','C','D'],
                   [[n,label,d[key]['denominator']]+[d[key]['counts'][c] for c in 'ABCD']
                    for n,d in groups for key,label in (('gold_frequencies','Gabarito'),('prediction_frequencies','Predição'))]),'',
             '![Frequências de alternativas](mmlu_full/label_frequencies.png)','',
             'As frequências não demonstram existência ou ausência de viés de posição. Não foram executadas rotações controladas.','',
             '## Erros com alta concentração condicional','',
             table(['Grupo','Erros com máx. p ≥ 0,90','Total de erros','Fração'],
                   [[n,d['high_concentration_errors']['count'],d['high_concentration_errors']['incorrect_denominator'],
                     number(d['high_concentration_errors']['fraction_of_incorrect'])] for n,d in groups]),'',
             'Todos os IDs que atendem ao critério estão listados no JSON, sem seleção de casos favoráveis ou exclusão dos demais erros. '
             'A concentração entre rótulos não implica probabilidade de acerto ou calibração.','',
             '## Massa original dos quatro rótulos','',
             table(['Grupo','n','Média','DP','Mín.','Q1','Mediana','Q3','Máx.'],
                   [[n,d['original_label_mass']['count']]+[number(d['original_label_mass'][k]) for k in keys] for n,d in groups]),'',
             'Massa original = soma das probabilidades dos quatro rótulos no vocabulário antes de renormalizar. '
             'Alta massa e baixa entropia não garantem resposta correta. O vocabulário completo de logits não foi persistido.','',
             '## Comparação descritiva com o piloto de 60','',
             table(['Grupo','Piloto: acertos/n','Piloto: acurácia','Integral: acertos/n','Integral: acurácia','H piloto','H integral'],
                   [[n,f'{p["correct_count"]}/{p["accuracy_denominator"]}',number(p['accuracy']),
                     f'{d["correct_count"]}/{d["accuracy_denominator"]}',number(d['accuracy']),
                     number(p['entropy']['mean']),number(d['entropy']['mean'])]
                    for (n,d),p in zip(groups,[pilot['overall'],*pilot['by_subject'].values()],strict=True)]),'',
             f'Macro piloto: {number(macro_accuracy(pilot)["value"])}; macro integral: {number(e["macro_accuracy"]["value"])}.','',
             'O piloto está inteiramente contido na população integral; seus 60 itens não são somados novamente aos 1.092. '
             'O piloto tinha 20 itens por disciplina; a expansão tem tamanhos diferentes. Comparações são descritivas, '
             'sem teste de independência ou alegação de confirmação das hipóteses do piloto.','',
             '## Exemplos auditáveis','']
    rows={r['id']:r for r in a['records']}
    text += [table(['ID (disciplina/test/índice)','Regra','Gabarito → predição','p(A), p(B), p(C), p(D)','H'],
                   [[ex['id'].split('/',2)[2],','.join(ex['roles']),f'{rows[ex["id"]]["gold"]} → {rows[ex["id"]]["prediction"]}',
                     ', '.join(number(v) for v in rows[ex['id']]['probabilities']),number(rows[ex['id']]['normalized_entropy'])]
                    for ex in a['examples']]),'',
             'Os exemplos são extremos simétricos de acertos e erros, não uma amostra representativa. '
             'Todos os 1.092 registros, incluindo enunciados, distribuições e escores, estão preservados nas evidências/arquivo do run.','',
             '## Recursos, ambiente e testes','',
             f'Python {meta["environment"]["python"]}, macOS 15.7.4, Apple M2/8 GiB. '
             'Versões MLX/Metal 0.32.4, MLX-LM 0.32.2, Transformers 5.22.1, Tokenizers 0.23.3, PyArrow 26.0.1. '
             'O ambiente temporário anterior havia sido removido; reutilizado o ambiente local, restaurando apenas PyArrow 26.0.1 (wheel de 37,3 MB). '
             'Dataset/tokenizer/pesos locais reutilizados e conferidos por hash; nenhum novo download desses artefatos.','',
             f'Pico alocador MLX: **{meta["model"].get("mlx_peak_bytes")} bytes**; ativos após close: {meta["model"].get("mlx_active_bytes_after_close")} bytes. '
             'Cache do alocador de 256 MiB, distinto de KV cache. Medições antes/depois, swap global e `/usr/bin/time -l` estão no JSON; '
             'não somar RSS, footprint e pico MLX nem atribuir todo o swap ao benchmark.','',
             table(['Medida','Valor'],[
                 ['Processo completo (s)',e['resources']['process_seconds']],
                 ['RSS máximo (bytes)',e['resources']['max_rss_bytes']],
                 ['Peak memory footprint (bytes)',e['resources']['peak_footprint_bytes']],
                 ['Disco livre antes (bytes)',e['resources']['before']['disk_free_bytes']],
                 ['Disco livre depois (bytes)',e['resources']['after']['disk_free_bytes']],
             ]),'',
             'O tempo inclui inicialização, hashing, carga, processamento e persistência; não é latência pura do modelo.','',
             f'Testes antes da inferência: **126, OK, 1,166 s**. Verificação final: **{e["verification"]["tests"]["count"]} testes, '
             f'{e["verification"]["tests"]["status"]}, {e["verification"]["tests"]["seconds"]} s**. '
             'Novas regressões cobrem contagens integrais, seed nula, índices completos, preflight bloqueado, separação de seleções, '
             'macro/denominadores, integração offline e comparação com o piloto; regressões anteriores preservadas.','',
             f'Auditoria independente Decimal (70 dígitos) nos 1.092 registros: erro absoluto máximo '
             f'{e["verification"]["softmax_max_absolute_error"]} no softmax e '
             f'{e["verification"]["entropy_max_absolute_error"]} na entropia. '
             'Descritivas, quartis, histogramas, frequências e macro conferidos independentemente com NumPy/Counter nos quatro grupos. '
             'Hashes dos artefatos anteriores do piloto e dos fontes executados conferidos.','',
             '## Observações, interpretações preliminares e limitações','',
             f'**Observado:** {g["correct_count"]}/{g["valid_count"]} acertos, entropia média {number(g["entropy"]["mean"])}. '
             f'Acertos: {number(g["entropy_by_outcome"]["correct"]["mean"])}; erros: {number(g["entropy_by_outcome"]["incorrect"]["mean"])}. '
             f'{g["high_concentration_errors"]["count"]} erros têm máx. p ≥ 0,90.','',
             '**Interpretação preliminar:** acerto e concentração devem ser tratados separadamente. As descritivas não '
             'estabelecem relação causal, calibração ou uma regra universal de correção de respostas.','',
             '- Três disciplinas, um modelo/snapshot e um único protocolo: não representam todo o MMLU nem permitem inferir capacidades cognitivas do CHC.',
             '- As 1.092 linhas cobrem integralmente os test dessas disciplinas na revisão fixada; duplicatas originais permanecem e limitam interpretações de independência entre itens.',
             '- Não foram examinados contaminação de treinamento, significância estatística, calibração ou robustez a reformulações/rotações.',
             '- A repetição dos 60 itens é controle técnico; a expansão sobreposta não é confirmação independente de hipóteses exploratórias.',
             '- Resultados test não foram usados para otimizar o protocolo. Qualquer ajuste futuro inspirado nestes resultados exige nova validação independente.',
             '- O forward completo pode divergir do caminho com cache em BF16; a repetição observada não garante invariância em outras plataformas/versões.',
             '- Persistência com publicação final de metadados não garante recuperação após queda de energia. O arquivo auditável preserva os bytes produzidos, não elimina essa limitação operacional.','',
             '## Artefatos e reprodução','',
             f'Run original: `{e["run_directory"]}` (ignorado pelo Git, preservado localmente). '
             f'[Arquivo completo auditável](mmlu_full/run.tar.gz), SHA-256 `{e["archive"]["sha256"]}`. '
             'Contém exatamente os quatro arquivos originais, com hashes individuais no [JSON de evidências](MMLU_FULL_EVIDENCE.json). '
             'A extração permite restaurar a pasta esperada de resultados sem executar a LLM.','',
             f'Código-base `{meta["code"]["commit"]}`, dirty={meta["code"]["dirty"]}; hashes dos fontes da execução e da análise constam nos metadados/JSON. '
             'Nenhum commit desta expansão foi criado antes da revisão.','',
             '```sh',
             'HF_HUB_OFFLINE=1 PYTHONPATH=src:. .cache/mlx-diagnostic/venv/bin/python -m diagnostics.mmlu_full_preflight \\',
             '  --snapshot <snapshot-local-do-dataset-fixado> --tokenizer-snapshot <snapshot-local-do-modelo-fixado> \\',
             '  --output outputs/<nova-pasta-de-preflight>',
             '# Apenas após conferir o preflight; isto cria NOVO run, não regenera a análise:',
             'HF_HUB_OFFLINE=1 PYTHONPATH=src .cache/mlx-diagnostic/venv/bin/python -m benchmark.cli --config configs/mmlu_full_test.yaml',
             '# Auditoria/análise sem inferência:',
             'HF_HUB_OFFLINE=1 PYTHONPATH=src:. .cache/mlx-diagnostic/venv/bin/python -m diagnostics.mmlu_full_validation \\',
             f'  --run {e["run_directory"]} --output outputs/<nova-auditoria>.json',
             'PYTHONPATH=src:. .venv/bin/python -m diagnostics.mmlu_full_report \\',
             '  --audit outputs/<nova-auditoria>.json --resources outputs/mmlu-full-resource \\',
             '  --output outputs/<novo-relatorio> --date 2026-11-09',
             '```','',
             'O preflight inclui versões/espaço livre; a seleção semântica é identificada por seu hash e seus índices, '
             'e o Runner exige o manifesto original fixado. A auditoria depende dos Parquets e artefatos do piloto locais; '
             'o relatório utiliza somente o JSON auditado e os recursos persistidos.','',
             '**Próximo passo:** revisão técnica dos resultados, evidências e diff. Sem commit, push ou aceite automático da expansão.','']
    return '\n'.join(text)


def generate(audit_path,resources,output,report_date):
    e=json.loads(audit_path.read_text())
    if not e['overlap_comparison']['passed'] or 'analysis' not in e:
        raise ValueError('Divergência no piloto: investigar antes de consolidar.')
    if (output/'MMLU_FULL_EVIDENCE.json').exists() or (output/'mmlu_full').exists():
        raise ValueError('Usar saída nova; não sobrescrever evidências anteriores.')
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=Path('docs/reports/MMLU_FULL_SELECTION_MANIFEST.json')
    m=json.loads(manifest_path.read_text())
    e.update(date=report_date,selection_manifest_sha256=file_hash(manifest_path),
             analysis_code=code_metadata(),audit_sha256=file_hash(audit_path),
             plotting_versions={p:version(p) for p in ('matplotlib','numpy','pillow')})
    e['preflight']={'status':m['status'],'duplicates':m['duplicates'],'selection_sha256':m['selection_sha256'],
                    'by_subject':{s:{'count':len(rows),'min_tokens':min(r['input_tokens'] for r in rows),
                        'max_tokens':max(r['input_tokens'] for r in rows)}
                        for s in NAMES for rows in [[r for r in m['selected_records'] if r['subject']==s]]}}
    e['resources']={name:json.loads((resources/f'{name}.json').read_text()) for name in ('before','after')}
    e['resources']['process_time_log']=(resources/'run.stderr.log').read_text()
    log=e['resources']['process_time_log']
    e['resources'].update(
        process_seconds=float(re.search(r'([\d.]+) real',log)[1]),
        max_rss_bytes=int(re.search(r'(\d+)\s+maximum resident set size',log)[1]),
        peak_footprint_bytes=int(re.search(r'(\d+)\s+peak memory footprint',log)[1]))
    e['verification']=json.loads((resources/'verification.json').read_text())
    e['plot_data']=build_plot_data(e['analysis'])
    directory=output/'mmlu_full'
    e['figure_sha256']=render_figures(e['plot_data'],directory)
    e['figure_sha256']['entropy_histograms.png']=histogram_figure(e['analysis'],directory/'entropy_histograms.png')
    e['archive']=archive_run(Path(e['run_directory']),directory/'run.tar.gz')
    # Compactação da evidência JSON: IDs completos ficam no arquivo byte-exato do run.
    for r in e['records']:
        if r['scoring'] and r['scoring']['tokenization']:
            token=r['scoring']['tokenization'];ids=token.pop('input_ids')
            token.update(input_tokens=len(ids),input_ids_sha256=canonical_hash(ids))
    e['diagnostic_source_sha256']={str(p):file_hash(p) for p in
        [Path('diagnostics')/name for name in ('mmlu_full_preflight.py','mmlu_full_validation.py','mmlu_full_report.py')]}
    (output/'MMLU_FULL_EVIDENCE.json').write_text(json.dumps(e,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    (output/'MMLU_FULL_EVALUATION.md').write_text(report(e))
    return e


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--resources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--date',type=date.fromisoformat,required=True)
    args=p.parse_args();e=generate(args.audit,args.resources,args.output,args.date.isoformat())
    print(json.dumps({'summary':e['summary'],'macro_accuracy':e['macro_accuracy'],'figures':list(e['figure_sha256'])}))


if __name__=='__main__':main()
