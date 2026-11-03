"""Recompute the tables and figures of Experiments A, B and C from the saved per-image predictions.

No GPU and no images are needed: the script reads results/A_seed*.csv, results/B_seed*.csv and results/C_seed*.csv
and rewrites results/tables/*.csv and results/figures/fig_A_*, fig_B_*, fig_C_* (Figure 1 is produced by the notebook).

Usage:  python reproduce_tables.py results
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results'
os.makedirs(f'{OUT}/tables', exist_ok=True)
os.makedirs(f'{OUT}/figures', exist_ok=True)
JITTER = [0.1, 0.2, 0.3, 0.5]
A_VARIANTS = (['gt_m0.00', 'gt_m0.25', 'gt_m0.50', 'gt_m1.00'] + [f'jit_{L:.1f}' for L in JITTER]
              + ['mp_m0.25', 'mp_m0.35', 'whole_squash', 'whole_pad'])


# ---- A ----
plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
                     'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})

VARIANT_NAMES = {
    'gt_m0.00': 'Annotated box, margin 0%', 'gt_m0.25': 'Annotated box, margin 25%',
    'gt_m0.50': 'Annotated box, margin 50%', 'gt_m1.00': 'Annotated box, margin 100%',
    'jit_0.1': 'Noisy box, error 10%', 'jit_0.2': 'Noisy box, error 20%',
    'jit_0.3': 'Noisy box, error 30%', 'jit_0.5': 'Noisy box, error 50%',
    'mp_m0.25': 'MediaPipe, margin 25%', 'mp_m0.35': 'MediaPipe, margin 35%',
    'whole_squash': 'Whole photo, squeezed', 'whole_pad': 'Whole photo, padded'}

def load_all(prefix):
    files = sorted(glob.glob(f'{OUT}/{prefix}_seed*.csv'))
    assert files, f'no {prefix}_seed*.csv yet: run section 6 first'
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def cboot_ci(values, clusters, n_boot=2000, seed=0):
    # 95% bootstrap interval of the mean, resampling whole clusters (people), not single photos
    values = np.asarray(values, float)
    codes, uniq = pd.factorize(pd.Series(np.asarray(clusters)))
    sums = np.bincount(codes, weights=values, minlength=len(uniq))
    cnts = np.bincount(codes, minlength=len(uniq))
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    stat = sums[idx].sum(1) / cnts[idx].sum(1)
    return np.percentile(stat, [2.5, 97.5])

def save_fig(fig, name):
    fig.savefig(f'{OUT}/figures/{name}.png', dpi=600, bbox_inches='tight')
    fig.savefig(f'{OUT}/figures/{name}.jpg', dpi=600, bbox_inches='tight', pil_kwargs={'quality': 95})

def save_table(t, name):
    t.to_csv(f'{OUT}/tables/{name}.csv', index=False)
    print(f'--- {name} ---')
    print(t.to_string(index=False))

A = load_all('A')
A['ok'] = (A['pred'] == A['y']).astype(float)
A['found'] = A['found'].astype(bool)
photo = A.groupby(['variant', 'id', 'person', 'label', 'found'], as_index=False).agg(ok=('ok', 'mean'), iou=('crop_iou', 'mean'))
seed_acc = A.groupby(['variant', 'seed'])['ok'].mean().groupby('variant').agg(['mean', 'std'])
print('seeds used:', [int(x) for x in sorted(A['seed'].unique())])

rows = []
for v in A_VARIANTS:
    g = photo[photo['variant'] == v]
    lo, hi = cboot_ci(g['ok'], g['person'])
    rows.append({'variant': v, 'accuracy_%': 100 * g['ok'].mean(), 'ci95_low': 100 * lo, 'ci95_high': 100 * hi,
                 'seed_std_pp': 100 * seed_acc.loc[v, 'std'] if not np.isnan(seed_acc.loc[v, 'std']) else np.nan,
                 'mean_crop_iou': g['iou'].mean(), 'n_photos': len(g)})
tabA1 = pd.DataFrame(rows).round(2)
save_table(tabA1, 'A1_accuracy_by_localization_variant')

wide = photo.pivot_table(index=['id', 'person', 'found'], columns='variant', values='ok').reset_index()
rows = []
for name, sub in (('MediaPipe found the hand', wide[wide['found']]), ('MediaPipe did not find the hand', wide[~wide['found']]),
                  ('all photos', wide)):
    for a, b in (('gt_m0.25', 'mp_m0.25'), ('gt_m0.25', 'mp_m0.35'), ('gt_m0.25', 'whole_squash'), ('gt_m0.25', 'whole_pad')):
        if len(sub) < 5:
            continue
        d = (sub[a] - sub[b]).values
        lo, hi = cboot_ci(d, sub['person'])
        rows.append({'photos': name, 'n': len(sub), 'reference': a, 'alternative': b, 'reference_%': 100 * sub[a].mean(),
                     'alternative_%': 100 * sub[b].mean(), 'diff_pp': 100 * d.mean(), 'ci95_low_pp': 100 * lo, 'ci95_high_pp': 100 * hi})
tabA2 = pd.DataFrame(rows).round(2)
save_table(tabA2, 'A2_same_photos_paired_differences')

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={'width_ratios': [1, 1.05]})
crops = A[A['variant'].str.startswith(('gt_', 'jit_')) | (A['variant'].str.startswith('mp_') & A['found'])]
edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0001])
xs, ys, los, his = [], [], [], []
for lo_e, hi_e in zip(edges[:-1], edges[1:]):
    part = crops[(crops['crop_iou'] >= lo_e) & (crops['crop_iou'] < hi_e)]
    if len(part) < 20:
        continue
    lo, hi = cboot_ci(part['ok'], part['person'])
    xs.append((lo_e + min(hi_e, 1.0)) / 2); ys.append(100 * part['ok'].mean()); los.append(100 * lo); his.append(100 * hi)
axes[0].errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)], marker='o', ms=3, lw=1, capsize=2, color='black')
axes[0].set_xlabel('Overlap with the annotated crop (intersection over union)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('(a)', loc='left', fontsize=8)
axes[0].set_ylim(0, 100)
t = tabA1.iloc[::-1]
axes[1].barh(t['variant'].map(VARIANT_NAMES), t['accuracy_%'], xerr=[t['accuracy_%'] - t['ci95_low'], t['ci95_high'] - t['accuracy_%']],
             color='0.7', ecolor='black', capsize=2, height=0.7)
axes[1].set_xlabel('Accuracy (%)')
axes[1].set_xlim(0, 100)
axes[1].set_title('(b)', loc='left', fontsize=8)
axes[1].tick_params(axis='y', labelsize=7)
plt.tight_layout()
save_fig(fig, 'fig_A_localization')
plt.close(fig)

# ---- B ----
B = load_all('B')
B['svm_ok'] = (B['svm'] == B['y']).astype(float)
B['cnn_ok'] = (B['cnn'] == B['y']).astype(float)
B['group_in_train'] = B['group_in_train'].astype(bool)
print('seeds used:', [int(x) for x in sorted(B['seed'].unique())])

cv = B[B['protocol'].isin(['random', 'group'])]
per_seed = cv.groupby(['protocol', 'seed']).agg(svm=('svm_ok', 'mean'), cnn=('cnn_ok', 'mean')).reset_index()
rows = []
for proto, g in per_seed.groupby('protocol'):
    rows.append({'protocol': f'{proto} 5-fold CV (all 500 images)', 'n_runs': len(g),
                 'HOG+SVM_%': 100 * g['svm'].mean(), 'HOG+SVM_std_pp': 100 * g['svm'].std(),
                 'MobileNetV2_%': 100 * g['cnn'].mean(), 'MobileNetV2_std_pp': 100 * g['cnn'].std()})
ho = B[B['protocol'] == 'holdout'].groupby(['seed', 'fold']).agg(svm=('svm_ok', 'mean'), cnn=('cnn_ok', 'mean'), n=('y', 'size')).reset_index()
rows.append({'protocol': f'single holdout split (~{int(ho["n"].mean())} test images)', 'n_runs': len(ho),
             'HOG+SVM_%': 100 * ho['svm'].mean(), 'HOG+SVM_std_pp': 100 * ho['svm'].std(),
             'MobileNetV2_%': 100 * ho['cnn'].mean(), 'MobileNetV2_std_pp': 100 * ho['cnn'].std()})
tabB1 = pd.DataFrame(rows).round(2)
save_table(tabB1, 'B1_accuracy_by_validation_protocol')

hmm = pd.DataFrame({'min_%': [100 * ho['cnn'].min(), 100 * ho['svm'].min()], 'max_%': [100 * ho['cnn'].max(), 100 * ho['svm'].max()]},
                   index=['MobileNetV2', 'HOG+SVM']).round(1).reset_index().rename(columns={'index': 'model'})
save_table(hmm, 'B2_holdout_range')

rows = []
for (proto, src), g in cv.groupby(['protocol', 'source']):
    ps = g.groupby('seed').agg(svm=('svm_ok', 'mean'), cnn=('cnn_ok', 'mean'))
    rows.append({'protocol': proto, 'source': src, 'n_images': g['idx'].nunique(),
                 'share_with_same_group_in_train_%': 100 * g['group_in_train'].mean(),
                 'HOG+SVM_%': 100 * ps['svm'].mean(), 'MobileNetV2_%': 100 * ps['cnn'].mean()})
tabB3 = pd.DataFrame(rows).round(2)
save_table(tabB3, 'B3_by_source_and_leakage_diagnostic')

pivot = per_seed.pivot(index='seed', columns='protocol', values='cnn')
if {'random', 'group'} <= set(pivot.columns):
    diff = 100 * (pivot['random'] - pivot['group'])
    print(f'\nMobileNetV2: random minus group 5-fold = {diff.mean():+.2f} pp (per-seed: {diff.round(2).tolist()})')

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={'width_ratios': [1, 1.2]})
for j, (col, lab, mk) in enumerate((('cnn', 'MobileNetV2', 'o'), ('svm', 'HOG + SVM', 's'))):
    for i, proto in enumerate(('random', 'group')):
        v = 100 * per_seed[per_seed['protocol'] == proto][col].values
        axes[0].scatter(np.full(len(v), i + (j - 0.5) * 0.25), v, marker=mk, s=14, facecolor='black' if j == 0 else 'white',
                        edgecolor='black', lw=0.7, label=lab if i == 0 else None)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Random split', 'Group split'])
axes[0].set_xlim(-0.6, 1.6); axes[0].set_ylabel('Accuracy over 500 images (%)'); axes[0].legend(frameon=False, fontsize=7)
axes[0].set_title('(a) 5-fold cross-validation, one point per seed', loc='left', fontsize=8)
rng = np.random.default_rng(0)
for j, (col, lab, mk) in enumerate((('cnn', 'MobileNetV2', 'o'), ('svm', 'HOG + SVM', 's'))):
    v = 100 * ho[col].values
    axes[1].scatter(j + rng.uniform(-0.12, 0.12, len(v)), v, marker=mk, s=14, facecolor='black' if j == 0 else 'white', edgecolor='black', lw=0.7)
    gm = 100 * per_seed[per_seed['protocol'] == 'group'][col].mean()
    axes[1].hlines(gm, j - 0.3, j + 0.3, color='0.4', lw=1, ls='--')
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['MobileNetV2', 'HOG + SVM']); axes[1].set_xlim(-0.6, 1.6)
axes[1].set_ylabel('Accuracy on one small test split (%)')
axes[1].set_title('(b) single holdout splits; dashed: group 5-fold mean', loc='left', fontsize=8)
plt.tight_layout()
save_fig(fig, 'fig_B_validation_protocol')
plt.close(fig)

# ---- C ----
C = load_all('C')
C['svm_ok'] = (C['svm'] == C['y']).astype(float)
C['cnn_ok'] = (C['cnn'] == C['y']).astype(float)
print('seeds used:', [int(x) for x in sorted(C['seed'].unique())])

order = ['hagrid in-domain CV', 'hagrid->own', 'hagrid->own, no low-res aug', 'own in-domain CV', 'own->hagrid']
ps = C.groupby(['setting', 'seed']).agg(svm=('svm_ok', 'mean'), cnn=('cnn_ok', 'mean'), n=('y', 'size')).reset_index()
rows = []
for s in order:
    g = ps[ps['setting'] == s]
    rows.append({'setting': s, 'test_images': int(g['n'].iloc[0]) if len(g) else 0, 'n_seeds': len(g),
                 'HOG+SVM_%': 100 * g['svm'].mean(), 'HOG+SVM_std_pp': 100 * g['svm'].std(),
                 'MobileNetV2_%': 100 * g['cnn'].mean(), 'MobileNetV2_std_pp': 100 * g['cnn'].std()})
tabC = pd.DataFrame(rows).round(2)
save_table(tabC, 'C1_domain_transfer')

fig, ax = plt.subplots(figsize=(6.5, 2.6))
x = np.arange(len(order))
for j, (col, lab, c) in enumerate((('cnn', 'MobileNetV2', '0.25'), ('svm', 'HOG + SVM', '0.7'))):
    m = [100 * ps[ps['setting'] == s][col].mean() for s in order]
    e = [100 * ps[ps['setting'] == s][col].std() if (ps['setting'] == s).sum() > 1 else 0 for s in order]
    ax.bar(x + (j - 0.5) * 0.38, m, 0.38, yerr=e, capsize=2, color=c, ecolor='black', label=lab)
ax.set_xticks(x); ax.set_xticklabels(['HaGRID\nin-domain', 'HaGRID to\nown photos', 'HaGRID to own,\nno low-res aug.',
                                      'Own photos\nin-domain', 'Own photos\nto HaGRID'], fontsize=7)
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=7)
plt.tight_layout()
save_fig(fig, 'fig_C_domain_transfer')
plt.close(fig)
