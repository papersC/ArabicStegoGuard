"""Render results/results.json into LaTeX tables (results/tables.tex) included by the paper."""
import json
R = json.load(open('results/results.json')); M = ['ZW1', 'KHN', 'KSH', 'PSP']
f3 = lambda x: '--' if x is None else '%.3f' % x; f1 = lambda x: '--' if x is None else '%.1f' % (100 * x)
L = []
L += [r'\begin{table*}[!t]', r'\caption{E1 (source-disjoint, all families seen) on the test partition. Threshold fixed on validation at FPR$\le$1\%. TPR/FPR in \%. AUROC 95\% CI: source-cluster bootstrap (300 resamples).}', r'\label{tab:e1}', r'\centering\footnotesize',
      r'\begin{tabular}{@{}lccccccccc@{}}', r'\toprule', r'Detector & AUROC [95\% CI] & AUPRC & TPR & FPR & FPR$_{\text{hard-neg}}$ & FPR$_{\text{clean}}$ & ECE & Brier & ms/doc \\', r'\midrule']
for k, v in R['E1'].items():
    lo, hi = v['AUROC_CI']
    L.append(r'%s & %.4f [%.4f, %.4f] & %.4f & %s & %s & %s & %s & %s & %s & %.2f \\' % (k, v['AUROC'], lo, hi, v['AUPRC'], f1(v['TPR']), f1(v['FPR']), f1(v['FPR_hardneg']), f1(v['FPR_clean']), f3(v.get('ECE')), f3(v.get('Brier')), v['ms_per_doc']))
L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
L += [r'\begin{table}[!t]', r'\caption{E2 leave-one-embedding-family-out: AUROC / TPR (\%) on the unseen family at the validation threshold (FPR$\le$1\%).}', r'\label{tab:e2}', r'\centering\footnotesize', r'\setlength{\tabcolsep}{3pt}',
      r'\begin{tabular}{@{}lcccc@{}}', r'\toprule', r'Detector & ZW1 & KHN & KSH & PSP \\', r'\midrule']
for k, v in R['E2'].items():
    L.append(k + ' & ' + ' & '.join('%.3f / %s' % (v[a]['AUROC'], f1(v[a]['TPR'])) for a in M) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
L += [r'\begin{table}[!t]', r'\caption{Payload survival rate PSR (\%, Eq.~(\ref{eq:psr})) per transformation, and TPR (\%) on surviving payloads / FPR (\%) of detectors trained on untransformed data.}', r'\label{tab:e4}', r'\centering\footnotesize', r'\setlength{\tabcolsep}{2.5pt}',
      r'\begin{tabular}{@{}lccccccc@{}}', r'\toprule', r'Transform & ZW1 & KHN & KSH & PSP & B1 & B2 & B6 \\', r'\midrule']
for t, p in R['PSR'].items():
    e = R['E4'][t]
    L.append(t.replace('_', r'\_') + ' & ' + ' & '.join(f1(p[a]) for a in M) + ' & ' + ' & '.join('%s/%s' % (f1(e[n]['TPR']), f1(e[n]['FPR'])) for n in ['B1-Rules', 'B2-CodepointNgramLR', 'B6-ArabicStegoGuard']) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('results/tables.tex', 'w').write('\n'.join(L)); print('\n'.join(L))
