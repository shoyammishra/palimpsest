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
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
fig = plt.figure(figsize=(5.5, 2.55), layout='constrained')
grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
left = fig.add_subplot(grid[:, 0])
ratio_ax = fig.add_subplot(grid[0, 1])
cos_ax = fig.add_subplot(grid[1, 1], sharex=ratio_ax)
left.plot(steps, p, 'o-', color='#176b91', ms=2.7, lw=1.3, label='Order only (P1)')
left.plot(steps, j, 's-', color='#bc5b2a', ms=2.7, lw=1.3, label='Init + order (J)')
for a, b, label, xy in [(1000, 2000, '1k-2k', (4, 3)),
                        (16000, 17000, '16k-17k', (-38, -13)),
                        (142000, 143000, '142k-143k', (-44, -14))]:
    val = e9['cells'][f'floor:{a}-{b}']['sym_kl']
    left.scatter(b, val, marker='x', c='#333333', s=25, zorder=4)
    left.annotate(label, (b, val), xytext=xy, textcoords='offset points', fontsize=7)
left.set_title('A  Functional divergence', loc='left', fontsize=8, weight='bold')
left.set_ylabel('Symmetric KL (nats)')
left.set_ylim(-.015, .76)
left.legend(frameon=False, fontsize=7, loc='upper left')
ratios = [a/b for a,b in zip(p,j)]
ratio_ax.plot(steps, ratios, 'o-', color='#176b91', ms=2.7, lw=1.3)
cos_ax.plot(steps, [cos[t] for t in steps], 's-', color='#8060a5', ms=2.7, lw=1.3)
ratio_ax.set_title('B  Output-divergence ratio', loc='left', fontsize=8, weight='bold')
cos_ax.set_title('C  Raw endpoint-direction alignment', loc='left', fontsize=8, weight='bold')
ratio_ax.set_ylabel('$R_t$')
cos_ax.set_ylabel('$C_t$')
for ax in [ratio_ax, cos_ax]:
    ax.set_ylim(-.06, 1.12)
    ax.set_yticks([0, .5, 1])
    ax.axvline(16000, color='#707070', lw=.8, ls=':', zorder=0)
ratio_ax.annotate(f'16k: {ratios[steps.index(16000)]:.3f}',
                  (16000, ratios[steps.index(16000)]), xytext=(-88, -3),
                  textcoords='offset points', fontsize=8, color='#176b91',
                  arrowprops={'arrowstyle': '-', 'color': '#176b91', 'lw': .7})
cos_ax.annotate(f'16k: {cos[16000]:.3f}', (16000, cos[16000]), xytext=(-66, 12),
                textcoords='offset points', fontsize=8, color='#8060a5',
                arrowprops={'arrowstyle': '-', 'color': '#8060a5', 'lw': .7})
for ax in [left, ratio_ax, cos_ax]:
    ax.set_xscale('log')
    ax.set_xticks([10, 100, 1000, 10000, 100000], ['10','100','1k','10k','100k'])
    ax.grid(alpha=.15)
ratio_ax.tick_params(labelbottom=False)
for ax in [left, cos_ax]:
    ax.set_xlabel('Training step (log scale)')
fig.savefig(HERE / 'dynamics.pdf')
