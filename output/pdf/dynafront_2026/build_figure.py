"""Render recorded measurements only; never run or change experiments."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / 'results' / 'raw'
e9 = json.loads((RAW / 'e009_divergence_2026-07-20.json').read_text())
e11 = json.loads((RAW / 'e011a_direction_2026-07-22.json').read_text())
steps = [4, 16, 64, 256, 512, 1000, 4000, 16000, 64000, 128000, 143000]
p = [e9['cells'][f'P1@{t}']['sym_kl'] for t in steps]
j = [e9['cells'][f'J1@{t}']['sym_kl'] for t in steps]
cos = {c['t']: c['cos'] for c in e11['pairs']['P1']['curve']}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), layout='constrained')
axes[0].plot(steps, p, 'o-', color='#176b91', ms=3, lw=1.3, label='Order only (P1)')
axes[0].plot(steps, j, 's-', color='#bc5b2a', ms=3, lw=1.3, label='Init + order (J)')
for a, b, label, xy in [(1000, 2000, '1k–2k', (0, -13)),
                        (16000, 17000, '16k–17k', (-24, -14)),
                        (142000, 143000, '142k–143k', (-51, -15))]:
    val = e9['cells'][f'floor:{a}-{b}']['sym_kl']
    axes[0].scatter(b, val, marker='x', c='#333333', s=27, zorder=4)
    axes[0].annotate(label, (b, val), xytext=xy, textcoords='offset points', fontsize=6.5)
axes[0].set_ylabel('Symmetric KL (nats)')
axes[0].set_ylim(-.015, .74)
axes[0].legend(frameon=False, fontsize=7, loc='upper left')
axes[1].plot(steps, [a/b for a,b in zip(p,j)], 'o-', color='#176b91', ms=3, lw=1.3,
             label='D(P1) / D(J)')
axes[1].plot(steps, [cos[t] for t in steps], 's-', color='#8060a5', ms=3, lw=1.3,
             label='Raw direction cosine')
axes[1].set_ylabel('Ratio / cosine')
axes[1].set_ylim(-.025, 1.07)
axes[1].legend(frameon=False, fontsize=7, loc='upper left')
for ax in axes:
    ax.set_xscale('log')
    ax.set_xlabel('Training step (log scale)')
    ax.set_xticks([10, 100, 1000, 10000, 100000], ['10','100','1k','10k','100k'])
    ax.grid(alpha=.15)
fig.savefig(HERE / 'dynamics.pdf', bbox_inches='tight')
