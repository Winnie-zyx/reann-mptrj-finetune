# 用于任意数量数据集的t-SNE可视化 + 多维度量化相似度指标
import sys
import os
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ase import Atom
import argparse
import matplotlib.colors as mcolors

# ===== 量化指标相关 imports =====
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import silhouette_score, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 原子类型列表（与训练时保持一致）
atomtype = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Co', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
            'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'I',
            'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

# 加载模型
pes = torch.jit.load("REANN_PES_FLOAT.pt")
pes.to(device).to(torch.float32)
pes.eval()
pes = torch.jit.optimize_for_inference(pes)

# 周期性边界条件设置
period_table = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)


def process_config_file(file_path):
    """处理单个构型文件，返回特征列表"""
    config_features = []
    config_index = 0
    line_number = 0

    with open(file_path, 'r') as f1:
        while True:
            config_index += 1
            config_start_line = line_number

            id_line = f1.readline()
            line_number += 1
            if not id_line:
                print(f"文件 {file_path} 读取完成，共处理 {config_index-1} 个构型")
                break
            id_line_stripped = id_line.strip()
            if not id_line_stripped:
                continue

            cell = np.zeros((3, 3), dtype=np.float32)
            cell_valid = True
            for i in range(3):
                cell_line = f1.readline()
                line_number += 1
                if not cell_line:
                    cell_valid = False
                    break
                cell_line_stripped = cell_line.strip()
                if not cell_line_stripped:
                    cell_valid = False
                    break
                try:
                    cell_values = list(map(float, cell_line_stripped.split()))
                    if len(cell_values) != 3:
                        cell_valid = False
                        break
                    cell[i] = cell_values
                except ValueError:
                    cell_valid = False
                    break
            if not cell_valid:
                continue

            separator_line = f1.readline()
            line_number += 1
            if not separator_line:
                continue
            separator_stripped = separator_line.strip()
            if separator_stripped and not any(key in separator_stripped.lower() for key in ['atom', 'species', 'element']):
                pass

            species = []
            cart = []
            mass = []
            while True:
                atom_line = f1.readline()
                line_number += 1
                if not atom_line:
                    break
                atom_line_stripped = atom_line.strip()
                if "abprop" in atom_line_stripped:
                    break
                if not atom_line_stripped:
                    continue
                tmp = atom_line_stripped.split()
                if len(tmp) < 8:
                    continue
                try:
                    element = tmp[0]
                    mass_val = float(tmp[1])
                    coords = list(map(float, tmp[2:5]))
                    if element not in atomtype:
                        pass
                    species.append(Atom(element).number)
                    mass.append(mass_val)
                    cart.append(coords)
                except (ValueError, IndexError):
                    continue

            if species and cart and mass:
                try:
                    species_tensor = torch.from_numpy(np.array(species)).to(device)
                    cart_tensor = torch.from_numpy(np.array(cart)).to(device).to(torch.float32)
                    mass_tensor = torch.from_numpy(np.array(mass)).to(device).to(torch.float32)
                    tcell_tensor = torch.from_numpy(cell).to(device).to(torch.float32)

                    with torch.set_grad_enabled(True):
                        _, _, _, density = pes(period_table, cart_tensor, tcell_tensor, species_tensor, mass_tensor)

                    density_np = density.detach().cpu().numpy()

                    mean_feat = np.mean(density_np, axis=0)
                    max_feat = np.max(density_np, axis=0)
                    min_feat = np.min(density_np, axis=0)
                    std_feat = np.std(density_np, axis=0)

                    config_feat = np.concatenate([mean_feat, max_feat, min_feat, std_feat], axis=0)
                    config_feat = config_feat.reshape(1, -1)

                    config_features.append(config_feat)
                except Exception:
                    pass

    return config_features


def get_distinct_colors(n):
    """生成n个不同的鲜明颜色"""
    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']
    if n <= len(base_colors):
        return base_colors[:n]
    additional_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    all_colors = base_colors + additional_colors
    return all_colors[:n]


# =====================================================================
#  量化指标计算
# =====================================================================

def stack_dataset_features(all_dataset_features):
    """将 list-of-list 特征转换为 list-of-2darray"""
    stacked = []
    for feats in all_dataset_features:
        stacked.append(np.concatenate(feats, axis=0))
    return stacked


def compute_knn_mixing_ratio(X_combined, n_first, k=10):
    """
    k-NN 混合比例。返回平均跨域邻居比例，范围 [0, 1]。
    0.5 = 完美混合，0.0 或 1.0 = 完全分离。
    """
    n_total = X_combined.shape[0]
    labels = np.zeros(n_total, dtype=int)
    labels[n_first:] = 1

    dist_matrix = cdist(X_combined, X_combined, metric='euclidean')
    np.fill_diagonal(dist_matrix, np.inf)
    knn_indices = np.argsort(dist_matrix, axis=1)[:, :k]

    neighbor_labels = labels[knn_indices]
    own_label = labels.reshape(-1, 1)
    cross_count = np.sum(neighbor_labels != own_label, axis=1)
    cross_ratio_per_sample = cross_count / k

    return np.mean(cross_ratio_per_sample)


def compute_self_baseline(X, n_splits=2, cv_folds=5, k_neighbors=None):
    """
    AIMD 自对比基线：将数据集随机分成两半，计算所有指标。
    """
    n = X.shape[0]
    idx = np.random.RandomState(42).permutation(n)
    half = n // n_splits
    X_i = X[idx[:half]]
    X_j = X[idx[half:2 * half]]
    n_i, n_j = X_i.shape[0], X_j.shape[0]

    X_pair = np.vstack([X_i, X_j])
    y_pair = np.array([0] * n_i + [1] * n_j)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pair)

    # ---- Silhouette ----
    try:
        sil = silhouette_score(X_scaled, y_pair, random_state=42)
    except Exception:
        sil = np.nan

    # ---- Balanced Accuracy + AUC + Proxy-A ----
    try:
        clf = LogisticRegression(max_iter=2000, random_state=42)
        cv = StratifiedKFold(n_splits=min(cv_folds, min(n_i, n_j)), shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_scaled, y_pair, cv=cv, method='predict')
        ba = balanced_accuracy_score(y_pair, y_pred)
        y_proba = cross_val_predict(clf, X_scaled, y_pair, cv=cv, method='predict_proba')[:, 1]
        auc = roc_auc_score(y_pair, y_proba)
        epsilon = 1.0 - ba
        proxy_a = max(0.0, min(1.0, 2.0 * (1.0 - 2.0 * epsilon)))
    except Exception:
        ba, auc, proxy_a = np.nan, np.nan, np.nan

    # ---- kNN Mix ----
    try:
        n_total = X_scaled.shape[0]
        if k_neighbors is None:
            k_adaptive = min(50, max(5, int(np.sqrt(n_total)), n_total // 4))
        else:
            k_adaptive = k_neighbors
        mix = compute_knn_mixing_ratio(X_scaled, n_i, k=k_adaptive)
    except Exception:
        mix, k_adaptive = np.nan, np.nan

    return {
        'silhouette': sil,
        'balanced_acc': ba,
        'auc': auc,
        'proxy_a_distance': proxy_a,
        'knn_mixing': mix,
        'k_used': k_adaptive,
        'n_i': n_i,
        'n_j': n_j,
    }


def compute_pairwise_metrics_v2(stacked_features, dataset_names, cv_folds=5, k_neighbors=None):
    """
    改进版 pairwise 指标：
    - Silhouette（主指标）
    - 平衡准确率 BalAcc
    - AUC（保留参考）
    - 代理 A-距离 Proxy-A
    - 自适应 k 的 kNN Mix
    """
    n_datasets = len(stacked_features)
    results = []

    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            X_i = stacked_features[i]
            X_j = stacked_features[j]
            n_i, n_j = X_i.shape[0], X_j.shape[0]
            name_pair = f"{dataset_names[i]}  vs  {dataset_names[j]}"

            X_pair = np.vstack([X_i, X_j])
            y_pair = np.array([0] * n_i + [1] * n_j)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_pair)

            # ---- Silhouette ----
            try:
                sil = silhouette_score(X_scaled, y_pair, random_state=42)
            except Exception:
                sil = np.nan

            # ---- Balanced Accuracy + AUC + Proxy-A ----
            try:
                clf = LogisticRegression(max_iter=2000, random_state=42)
                cv = StratifiedKFold(n_splits=min(cv_folds, min(n_i, n_j)), shuffle=True, random_state=42)
                y_pred = cross_val_predict(clf, X_scaled, y_pair, cv=cv, method='predict')
                ba = balanced_accuracy_score(y_pair, y_pred)
                y_proba = cross_val_predict(clf, X_scaled, y_pair, cv=cv, method='predict_proba')[:, 1]
                auc = roc_auc_score(y_pair, y_proba)
                epsilon = 1.0 - ba
                proxy_a = max(0.0, min(1.0, 2.0 * (1.0 - 2.0 * epsilon)))
            except Exception:
                ba, auc, proxy_a = np.nan, np.nan, np.nan

            # ---- kNN Mix (自适应 k) ----
            try:
                n_total = X_scaled.shape[0]
                if k_neighbors is None:
                    k_adaptive = min(50, max(5, int(np.sqrt(n_total)), n_total // 4))
                else:
                    k_adaptive = k_neighbors
                mix = compute_knn_mixing_ratio(X_scaled, n_i, k=k_adaptive)
            except Exception:
                mix, k_adaptive = np.nan, np.nan

            results.append({
                'pair': name_pair,
                'n_i': n_i,
                'n_j': n_j,
                'silhouette': sil,
                'balanced_acc': ba,
                'auc': auc,
                'proxy_a_distance': proxy_a,
                'knn_mixing': mix,
                'k_used': k_adaptive,
            })

    return results


def compute_auc_learning_curve(X_aimd, X_mlmd, n_trials=5, test_ratio=0.3):
    """
    轻量版 AUC Learning Curve：使用 train-test split 替代完整交叉验证。
    大幅降低计算量，同时保留诊断能力。
    """
    scaler = StandardScaler()
    X_all = np.vstack([X_aimd, X_mlmd])
    scaler.fit(X_all)
    X_a = scaler.transform(X_aimd)
    X_m = scaler.transform(X_mlmd)

    n_a, n_m = X_a.shape[0], X_m.shape[0]
    max_n = min(n_a, n_m)

    # 关键子样本量（对数间隔但精简）
    sample_sizes = [20, 50, 100, 200, 500, 1000]
    sample_sizes = [s for s in sample_sizes if s <= max_n]
    if sample_sizes and sample_sizes[-1] < max_n:
        sample_sizes.append(max_n)

    auc_means = []
    auc_stds = []

    for size in sample_sizes:
        aucs = []
        for trial in range(n_trials):
            idx_a = np.random.choice(n_a, size=size, replace=False)
            idx_m = np.random.choice(n_m, size=size, replace=False)
            X_sub = np.vstack([X_a[idx_a], X_m[idx_m]])
            y_sub = np.array([0] * size + [1] * size)

            # train-test split
            n_test = max(1, int(len(y_sub) * test_ratio))
            perm = np.random.permutation(len(y_sub))
            X_train = X_sub[perm[:len(y_sub) - n_test]]
            y_train = y_sub[perm[:len(y_sub) - n_test]]
            X_test = X_sub[perm[len(y_sub) - n_test:]]
            y_test = y_sub[perm[len(y_sub) - n_test:]]

            try:
                clf = LogisticRegression(max_iter=1000, random_state=trial)
                clf.fit(X_train, y_train)
                y_proba = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                aucs.append(auc)
            except Exception:
                pass

        if aucs:
            auc_means.append(np.mean(aucs))
            auc_stds.append(np.std(aucs))
        else:
            auc_means.append(np.nan)
            auc_stds.append(np.nan)

    return np.array(sample_sizes), np.array(auc_means), np.array(auc_stds)


def plot_auc_learning_curves(learning_results, save_path='auc_learning_curve.png'):
    """
    为每一对数据集画 AUC learning curve。
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(learning_results)))

    for idx, lr in enumerate(learning_results):
        sizes = lr['sample_sizes']
        means = lr['auc_means']
        stds = lr['auc_stds']
        label = lr['label']
        color = colors[idx % len(colors)]

        valid = ~np.isnan(means)
        ax.plot(sizes[valid], means[valid], 'o-', color=color, label=label, linewidth=2, markersize=6)
        ax.fill_between(sizes[valid],
                        (means - stds)[valid],
                        (means + stds)[valid],
                        color=color, alpha=0.15)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Random (0.5)')
    ax.set_xlabel('Sub-sample size (per dataset)', fontsize=14)
    ax.set_ylabel('AUC', fontsize=14)
    ax.set_title('AUC Learning Curve\n(lower curve at small N = better agreement)', fontsize=16)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim([0.45, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAUC Learning Curve 已保存至: {save_path}")
    plt.close()


# =====================================================================
#  输出函数
# =====================================================================

def print_metrics_table_v2(pairwise_results, baseline_info=None):
    """
    改进版输出：以 Silhouette 为第一指标，BalAcc 为核心辅助指标。
    """
    sep = "=" * 105
    header = (f"{'Pair':<35s} {'Silhouette★':>12s}  {'BalAcc':>8s}  "
              f"{'Proxy-A':>10s}  {'AUC(ref)':>10s}  {'kNN Mix':>12s}")
    sub =     (f"{'(dataset_i vs dataset_j)':<35s} {'(<0.10=OK)':>12s}  "
              f"{'(<0.65=OK)':>8s}  {'(<0.3=OK)':>10s}  {'(ref)':>10s}  {'(>0.20=OK)':>12s}")

    lines = [sep,
             "Quantitative Similarity Metrics (Improved - Effect Size Oriented)",
             sep, header, sub, "-" * 105]

    for r in pairwise_results:
        sil_s = f"{r['silhouette']:.4f}" if not np.isnan(r['silhouette']) else "N/A"
        ba_s  = f"{r['balanced_acc']:.4f}" if not np.isnan(r['balanced_acc']) else "N/A"
        pa_s  = f"{r['proxy_a_distance']:.4f}" if not np.isnan(r['proxy_a_distance']) else "N/A"
        auc_s = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "N/A"
        mix_s = f"{r['knn_mixing']:.4f} (k={r.get('k_used','?')})" if not np.isnan(r['knn_mixing']) else "N/A"

        lines.append(f"{r['pair']:<35s} {sil_s:>12s}  {ba_s:>8s}  {pa_s:>10s}  {auc_s:>10s}  {mix_s:>12s}")

    # 自对比基线
    if baseline_info is not None:
        lines.append("-" * 105)
        lines.append("AIMD Self-Baseline (AIMD split-half → finite-sampling noise floor):")
        b = baseline_info
        sil_s = f"{b['silhouette']:.4f}" if not np.isnan(b['silhouette']) else "N/A"
        ba_s  = f"{b['balanced_acc']:.4f}" if not np.isnan(b['balanced_acc']) else "N/A"
        pa_s  = f"{b['proxy_a_distance']:.4f}" if not np.isnan(b['proxy_a_distance']) else "N/A"
        auc_s = f"{b['auc']:.4f}" if not np.isnan(b['auc']) else "N/A"
        mix_s = f"{b['knn_mixing']:.4f} (k={b.get('k_used','?')})" if not np.isnan(b['knn_mixing']) else "N/A"
        lines.append(f"{'AIMD-A vs AIMD-B (split-half)':<35s} {sil_s:>12s}  {ba_s:>8s}  {pa_s:>10s}  {auc_s:>10s}  {mix_s:>12s}")

    lines.append(sep)
    lines.append("★ Silhouette is the PRIMARY metric (effect size, robust to sample size)")
    lines.append("  < 0.05 : Excellent    → MLMD can safely replace AIMD")
    lines.append("  0.05-0.10 : Very good → MLMD can likely replace AIMD")
    lines.append("  0.10-0.15 : Good      → minor deviations, check observables")
    lines.append("  0.15-0.25 : Moderate  → borderline, physical validation required")
    lines.append("  > 0.25    : Poor      → MLMD likely not suitable as AIMD replacement")
    lines.append("")
    lines.append("★ BalAcc (Balanced Accuracy) — key auxiliary metric")
    lines.append("  < 0.55 : Excellent    → classifier barely beats random")
    lines.append("  0.55-0.60 : Very good → very weak separability")
    lines.append("  0.60-0.70 : Good      → detectable but mild difference")
    lines.append("  0.70-0.85 : Fair      → clear separation, MLMD has systematic deviations")
    lines.append("  > 0.85    : Poor      → near-perfect separability")
    lines.append("")
    lines.append("  Proxy-A < 0.2 → excellent   |   kNN Mix > 0.20 → meaningful overlap")
    lines.append("  Note: AUC near 1.0 with Silhouette < 0.25 = narrow gap, large N effect")
    lines.append(sep)

    # 自动判断
    for r in pairwise_results:
        if not np.isnan(r['silhouette']):
            sil = r['silhouette']
            if sil < 0.05:
                verdict = "EXCELLENT — MLMD can safely replace AIMD"
            elif sil < 0.10:
                verdict = "VERY GOOD — MLMD can likely replace AIMD"
            elif sil < 0.15:
                verdict = "GOOD — minor deviations, recommend checking physical observables"
            elif sil < 0.25:
                verdict = "MODERATE — borderline, physical validation required"
            else:
                verdict = "POOR — MLMD likely not suitable as AIMD replacement"
            lines.append(f"  → {r['pair']}: Silhouette={sil:.4f} → {verdict}")

    if baseline_info is not None and not np.isnan(baseline_info['silhouette']):
        lines.append(f"  → Baseline Silhouette = {baseline_info['silhouette']:.4f} "
                     f"(AIMD vs AIMD; finite-sampling noise floor)")

    lines.append(sep)
    return "\n".join(lines)


def compute_knn_mixing_2d(dataset_2d_list, dataset_names, k=10):
    """在 t-SNE 2D 坐标上计算 k-NN 混合比"""
    n_datasets = len(dataset_2d_list)
    results = []
    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            X_i = dataset_2d_list[i]
            X_j = dataset_2d_list[j]
            X_pair = np.vstack([X_i, X_j])
            n_i = X_i.shape[0]
            name_pair = f"{dataset_names[i]}  vs  {dataset_names[j]}"
            try:
                mix = compute_knn_mixing_ratio(X_pair, n_i, k=k)
            except Exception:
                mix = np.nan
            results.append({'pair': name_pair, 'knn_mixing_2d': mix})
    return results


# =====================================================================
#  主函数
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='t-SNE可视化 + 多维度量化相似度指标')
    parser.add_argument('config_files', nargs='+', help='构型文件路径。第1个应为AIMD，后续为MLMD')
    parser.add_argument('--perplexity', type=int, default=20)
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--xylim', type=int, default=150)
    parser.add_argument('--tick_interval', type=int, default=50)
    parser.add_argument('--knn_k', type=int, default=None, help='kNN混合比的k，默认自适应')
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--no_learning_curve', action='store_true', help='跳过AUC learning curve（加速）')
    parser.add_argument('--n_trials_lc', type=int, default=5, help='learning curve重复次数')
    args = parser.parse_args()

    config_files = args.config_files
    if len(config_files) < 2:
        print("错误：至少需要2个构型文件（第1个=AIMD，后续=MLMD）")
        sys.exit(1)

    print("=" * 105)
    print("输入约定：第1个文件 = AIMD（用于自对比基线），后续 = MLMD")
    print("=" * 105)

    # ---- 处理所有构型文件 ----
    all_dataset_features = []
    dataset_names = []
    for idx, file_path in enumerate(config_files):
        print(f"\n开始处理数据集 {idx+1}/{len(config_files)}: {file_path}")
        features = process_config_file(file_path)
        all_dataset_features.append(features)
        dataset_name = os.path.basename(file_path)
        dataset_names.append(dataset_name)
        print(f"数据集 {file_path} 处理完成，共 {len(features)} 个构型")

    valid_datasets = [(name, feats) for name, feats in zip(dataset_names, all_dataset_features) if feats]
    if len(valid_datasets) < 2:
        print("错误：有效数据集不足")
        sys.exit(1)
    dataset_names, all_dataset_features = zip(*valid_datasets)
    print(f"\n有效数据集数量：{len(dataset_names)}")

    stacked_features = stack_dataset_features(all_dataset_features)

    # ================ 1. AIMD 自对比基线 ================
    print("\n" + "=" * 105)
    print(f"1. AIMD 自对比基线（文件1: {dataset_names[0]} 随机分半 → 有限采样噪声下限）")
    print("=" * 105)
    X_aimd = stacked_features[0]
    baseline = compute_self_baseline(X_aimd, cv_folds=args.cv_folds, k_neighbors=args.knn_k)

    print(f"  Silhouette  = {baseline['silhouette']:.4f}")
    print(f"  BalAcc      = {baseline['balanced_acc']:.4f}")
    print(f"  Proxy-A     = {baseline['proxy_a_distance']:.4f}")
    print(f"  AUC         = {baseline['auc']:.4f}")
    print(f"  kNN Mix     = {baseline['knn_mixing']:.4f} (k={baseline['k_used']})")

    if baseline['silhouette'] is not None and not np.isnan(baseline['silhouette']):
        if abs(baseline['silhouette']) < 0.03 and abs(baseline['balanced_acc'] - 0.5) < 0.05:
            print(f"  ✅ 基线干净：AIMD轨迹无时间漂移，10ps采样自洽")
        elif abs(baseline['silhouette']) < 0.08:
            print(f"  ⚠ 基线有轻微噪声：可能存在微弱的时间漂移或采样不足")
        else:
            print(f"  ❌ 基线异常：AIMD前后半段有明显差异，检查是否充分平衡")
    print(f"  解释：这是10ps有限采样下AIMD自身波动的上限。MLMD vs AIMD应与此对比。")

    # ================ 2. Pairwise 指标 ================
    print("\n" + "=" * 105)
    print("2. 量化相似度指标（原始描述符空间）")
    print("=" * 105)

    pairwise_results = compute_pairwise_metrics_v2(
        stacked_features, dataset_names,
        cv_folds=args.cv_folds, k_neighbors=args.knn_k
    )

    metrics_output = print_metrics_table_v2(pairwise_results, baseline_info=baseline)
    print(metrics_output)

    metrics_file = "similarity_metrics.txt"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(metrics_output)
    print(f"指标已保存至: {metrics_file}")

    # ================ 3. AUC Learning Curve ================
    if not args.no_learning_curve:
        print("\n" + "=" * 105)
        print("3. AUC Learning Curve（子样本量扫描）")
        print("=" * 105)

        learning_results = []

        # AIMD 自对比 learning curve
        print("  计算 AIMD 自对比 learning curve...")
        half = len(X_aimd) // 2
        ss_bl, am_bl, as_bl = compute_auc_learning_curve(
            X_aimd[:half], X_aimd[half:],
            n_trials=args.n_trials_lc
        )
        learning_results.append({
            'sample_sizes': ss_bl, 'auc_means': am_bl, 'auc_stds': as_bl,
            'label': 'AIMD-A vs AIMD-B (baseline)'
        })

        # 每个 MLMD 与 AIMD 的 learning curve
        for idx in range(1, len(stacked_features)):
            print(f"  计算 AIMD vs {dataset_names[idx]} learning curve...")
            ss, am, as_ = compute_auc_learning_curve(
                X_aimd, stacked_features[idx],
                n_trials=args.n_trials_lc
            )
            learning_results.append({
                'sample_sizes': ss, 'auc_means': am, 'auc_stds': as_,
                'label': f'AIMD vs {dataset_names[idx]}'
            })

        plot_auc_learning_curves(learning_results, 'auc_learning_curve.png')

        # 关键诊断：小样本下的 AUC
        print("\n  ★ 关键诊断（N=20 时的 AUC）：")
        for lr in learning_results:
            if len(lr['sample_sizes']) > 0 and lr['sample_sizes'][0] <= 20:
                idx_n = np.argmin(np.abs(lr['sample_sizes'] - 20))
                auc_n20 = lr['auc_means'][idx_n]
                print(f"     {lr['label']:<40s} AUC(N≈20) = {auc_n20:.4f}")
                if auc_n20 < 0.60:
                    print(f"       → 小样本下几乎不可分（效应量很小）✅")
                elif auc_n20 < 0.75:
                    print(f"       → 小样本下弱可分（效应量中等）⚠")
                else:
                    print(f"       → 小样本下明显可分（效应量大）❌")

    # ================ 4. t-SNE 可视化 ================
    print("\n" + "=" * 105)
    print("4. t-SNE 可视化")
    print("=" * 105)

    total_features = np.concatenate([np.concatenate(feats, axis=0) for feats in all_dataset_features], axis=0)
    print(f"总有效构型数: {total_features.shape[0]}, 特征维度: {total_features.shape[1]}")

    print(f"\n开始t-SNE降维（perplexity={args.perplexity}, max_iter={args.max_iter}）")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42,
                max_iter=args.max_iter, learning_rate=200.0)
    features_2d = tsne.fit_transform(total_features)
    print("t-SNE降维完成")

    dataset_2d_list = []
    start_idx = 0
    for feats in all_dataset_features:
        end_idx = start_idx + len(feats)
        dataset_2d_list.append(features_2d[start_idx:end_idx])
        start_idx = end_idx

    # ============ 导出 t-SNE 数据供 Origin 绘图 ============
    print("\n--- 导出 t-SNE 坐标数据 ---")

    # --- 方式一：合并文件（推荐用于 Origin 分组散点图）---
    # 格式：tSNE_1, tSNE_2, Dataset
    combined_lines = ["tSNE_1,tSNE_2,Dataset"]
    for idx, (data_2d, name) in enumerate(zip(dataset_2d_list, dataset_names)):
        for row in data_2d:
            combined_lines.append(f"{row[0]:.6f},{row[1]:.6f},{name}")
    combined_path = "tsne_all_datasets.csv"
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(combined_lines))
    print(f"  合并文件已保存: {combined_path}")
    print(f"    → 在 Origin 中导入后，用 'Dataset' 列作为分组列即可绘制不同颜色的散点图")

    # --- 方式二：每个数据集单独文件（用于分别导入后手动合并图层）---
    for idx, (data_2d, name) in enumerate(zip(dataset_2d_list, dataset_names)):
        safe_name = name.replace("/", "_").replace("\\", "_").replace(".", "_")
        single_path = f"tsne_{idx+1}_{safe_name}.csv"
        lines = ["tSNE_1,tSNE_2"]
        for row in data_2d:
            lines.append(f"{row[0]:.6f},{row[1]:.6f}")
        with open(single_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"  单独文件已保存: {single_path}  ({len(data_2d)} 点)")

    # --- 方式三：Origin 可以直接拖入的多列格式 ---
    # 每两列为一个数据集 (X1, Y1, X2, Y2, ...)
    # 各数据集点数不同时用空行补齐
    max_len = max(len(d) for d in dataset_2d_list)
    col_headers = []
    for name in dataset_names:
        safe_name = name.replace(",", "_")
        col_headers.append(f"{safe_name}_X,{safe_name}_Y")
    multicol_header = ",".join(col_headers)

    multicol_lines = [multicol_header]
    for i in range(max_len):
        row_vals = []
        for data_2d in dataset_2d_list:
            if i < len(data_2d):
                row_vals.append(f"{data_2d[i, 0]:.6f}")
                row_vals.append(f"{data_2d[i, 1]:.6f}")
            else:
                row_vals.append("")
                row_vals.append("")
        multicol_lines.append(",".join(row_vals))
    multicol_path = "tsne_multicol.csv"
    with open(multicol_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(multicol_lines))
    print(f"  多列格式已保存: {multicol_path}")
    print(f"    → 在 Origin 中直接 Ctrl+V 或拖入，每两列为一个数据集")

    # t-SNE 2D kNN Mix
    print("\nt-SNE 2D 空间的 k-NN 混合比例（补充参考）:")
    knn_k_2d = args.knn_k if args.knn_k else 10
    knn_2d_results = compute_knn_mixing_2d(dataset_2d_list, dataset_names, k=knn_k_2d)
    for r in knn_2d_results:
        print(f"  {r['pair']:<40s} kNN Mix(2D) = {r['knn_mixing_2d']:.4f}")

    # ---- 绘图 ----
    colors = get_distinct_colors(len(dataset_names))
    fixed_min, fixed_max = -args.xylim, args.xylim
    ticks = np.arange(fixed_min, fixed_max + args.tick_interval, args.tick_interval)

    # 组合图
    fig, ax = plt.subplots(figsize=(10, 10))
    for idx, (data, name, color) in enumerate(zip(dataset_2d_list, dataset_names, colors)):
        ax.scatter(data[:, 0], data[:, 1], c=color, s=40, alpha=0.8, label=name)
    ax.legend(loc='upper right', fontsize=10, title='Datasets', title_fontsize=12)
    ax.set_title(f't-SNE Visualization of {len(dataset_names)} Datasets', fontsize=16)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim([fixed_min, fixed_max])
    ax.set_ylim([fixed_min, fixed_max])
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale(enable=False)
    plt.savefig('combined_tsne.png', dpi=300, bbox_inches='tight')
    print(f"\n组合图已保存为 combined_tsne.png")

    # 单独图
    for idx, (data, name, color) in enumerate(zip(dataset_2d_list, dataset_names, colors)):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(data[:, 0], data[:, 1], c=color, s=40, alpha=0.8)
        ax.set_title(f't-SNE: {name}', fontsize=16)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim([fixed_min, fixed_max])
        ax.set_ylim([fixed_min, fixed_max])
        ax.set_aspect('equal', adjustable='box')
        ax.autoscale(enable=False)
        save_path = f'dataset_{idx+1}_{name.replace("/", "_")}_tsne.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"单数据集图 {name} 已保存为 {save_path}")

    plt.show()
    print("\n所有分析完成。")


if __name__ == "__main__":
    main()

