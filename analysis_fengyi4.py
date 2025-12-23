import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.io import savemat
from scipy.stats import ttest_rel, ttest_ind, permutation_test
from statsmodels.stats.multitest import fdrcorrection, multipletests
from tqdm import tqdm
from scipy import stats
import multiprocessing
from functools import partial
import xlsxwriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

#nohup python /home/fyu7/COMBINe/analysis_fengyi4.py > analysis_log.txt 2>&1 &

# ==========================================
# 核心量化函数 (仅处理数值)
# ==========================================

def quantify(sample_dir_path, save_dir, classes, header, template_path, analysis_depth=None, max_id_limit=5000):
    """
    量化函数：统计各脑区体积及细胞数量，计算密度。
    """
    dataset_name = os.path.basename(os.path.normpath(sample_dir_path))
    print(f"--- Quantifying Data: {dataset_name} ---")

    # 1. 加载图谱模板
    df_template = pd.read_csv(template_path)
    df_template = df_template[(df_template['id'] >= 0) & (df_template['id'] <= max_id_limit)].copy()
    ids = df_template['id'].values
    path = df_template['structure_id_path']
    current_max_id = int(np.max(ids))
    
    # 2. 计算各区域体积 (基于25um分辨率)
    # 依然需要读取 result.mhd 来获取每个 ID 占据的体素数量
    import SimpleITK as sitk # 仅用于读取 ID 阵列
    mhd_path = os.path.join(sample_dir_path, 'volume', 'result.mhd')
    try:
        img = sitk.ReadImage(mhd_path)
        mask_arr = sitk.GetArrayFromImage(img).astype(np.int64)
        voxel_size_mm3 = (25 * 25 * 25) / (1e9) # 25um 转换为 mm^3
        
        counts_per_id = np.bincount(mask_arr.ravel().clip(0, current_max_id), minlength=current_max_id + 1)
        volumes_raw = counts_per_id.astype(float) * voxel_size_mm3
    except Exception as e:
        print(f"Error reading volume mask for {dataset_name}: {e}")
        return None, None, None, None

    # 3. 脑区层级体积汇总 (父节点包含子节点)
    volumes_summarized = volumes_raw.copy()
    for i in range(1, len(ids)):
        parent_id = ids[i]
        child_mask = path.str.contains(f"/{parent_id}/", na=False)
        volumes_summarized[parent_id] = np.sum(volumes_raw[ids[child_mask]])
    
    # 4. 细胞计数与密度计算
    num_list = []
    den_list = []
    df_results = df_template.copy()
    df_results[f"{dataset_name}_Volume"] = volumes_summarized[ids]

    for i, class_name in enumerate(classes):
        csv_path = os.path.join(sample_dir_path, 'cell_registration', str(i), 'cell_registration.csv')
        cell_counts_raw = np.zeros(current_max_id + 1, dtype=int)
        
        if os.path.exists(csv_path):
            df_cells = pd.read_csv(
                csv_path, 
                header=None, 
                names=header, 
                usecols=range(len(header)), 
                index_col=False,
                on_bad_lines='skip' 
            )
            cell_ids = df_cells['id'].values
            valid_ids = cell_ids[(cell_ids >= 0) & (cell_ids <= current_max_id)].astype(int)
            cell_counts_raw = np.bincount(valid_ids, minlength=current_max_id + 1)

        # 层级汇总细胞数
        counts_summarized = cell_counts_raw.copy().astype(float)
        for j in range(1, len(ids)):
            p_id = ids[j]
            c_mask = path.str.contains(f"/{p_id}/", na=False)
            counts_summarized[p_id] = np.sum(cell_counts_raw[ids[c_mask]])

        # 计算密度
        densities = np.divide(counts_summarized, volumes_summarized, 
                              out=np.zeros_like(counts_summarized), where=volumes_summarized > 0)
        
        num_list.append(counts_summarized[ids])
        den_list.append(densities[ids])
        
        df_results[class_name] = counts_summarized[ids]
        df_results[f"{class_name}_density"] = densities[ids]

    # 5. 深度筛选与保存
    if analysis_depth is not None:
        depth_mask = df_results['depth'] == analysis_depth
        df_results = df_results[depth_mask].copy()
    
    output_csv = os.path.join(save_dir, f"quantification_depth_{analysis_depth}.csv")
    df_results.to_csv(output_csv, index=False)

    return volumes_summarized[ids], np.stack(num_list, axis=1), np.stack(den_list, axis=1), df_results

# ==========================================
# 统计分析逻辑 (纯计算)
# ==========================================

def statistic_diff_means(x, y, axis=0):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def worker_permutation(args):
    g1, g2, n = args
    if np.all(g1 == 0) and np.all(g2 == 0): return 1.0
    res = permutation_test((g1, g2), statistic_diff_means, vectorized=False, n_resamples=n)
    return res.pvalue

def run_stats(df_template, group_names, classes, group_data, output_dir):
    """
    进行组间对比分析，生成 Excel 报表。
    """
    print(f"--- Running Stats: {group_names[0]} vs {group_names[1]} ---")
    g1_label, g2_label = group_names
    # group_data 结构: [ [vol_arr, counts_list, dens_list], [...] ]
    
    info_cols = ['id', 'name', 'acronym', 'depth']
    df_info = df_template[info_cols].reset_index(drop=True)
    
    stats_output_path = os.path.join(output_dir, f"Stats_Report_{g1_label}_vs_{g2_label}.xlsx")
    writer = pd.ExcelWriter(stats_output_path, engine='xlsxwriter')

    with multiprocessing.Pool() as pool:
        # 定义计算子流程
        def process_metric(m_name, data1, data2):
            mean1, mean2 = np.mean(data1, axis=1), np.mean(data2, axis=1)
            pct_change = ((mean2 - mean1) / (mean1 + 1e-9)) * 100
            
            args = [(data1[i, :], data2[i, :], 1000) for i in range(data1.shape[0])]
            p_raw = np.array(pool.map(worker_permutation, args))
            _, p_fdr, _, _ = multipletests(p_raw, alpha=0.05, method='fdr_bh')
            
            return pd.DataFrame({
                f"{m_name}_Mean_{g1_label}": mean1,
                f"{m_name}_Mean_{g2_label}": mean2,
                f"%_Change": pct_change,
                "P_raw": p_raw,
                "P_FDR": p_fdr
            })

        # 1. 体积统计
        df_vol_stats = process_metric("Vol", group_data[0][0], group_data[1][0])
        pd.concat([df_info, df_vol_stats], axis=1).to_excel(writer, sheet_name="Volume", index=False)

        # 2. 各类细胞统计
        for i, c_name in enumerate(classes):
            df_cnt = process_metric(f"{c_name}_Count", group_data[0][1][i], group_data[1][1][i])
            df_den = process_metric(f"{c_name}_Den", group_data[0][2][i], group_data[1][2][i])
            pd.concat([df_info, df_cnt, df_den], axis=1).to_excel(writer, sheet_name=c_name[:30], index=False)

    writer.close()
    print(f"Stats report saved to: {stats_output_path}")

# ==========================================
# 主执行流
# ==========================================

if __name__ == '__main__':
    # 配置
    BASE_DIR = "/data/hdd12tb-1/fengyi/COMBINe/clearmap"
    TEMPLATE = "/home/fyu7/COMBINe/annotations/structure_template.csv"
    ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
    
    CLASSES = ['red glia','green glia','yellow glia','red neuron','green neuron','yellow neuron']
    HEADER = ['x', 'y', 'z', 'xt', 'yt', 'zt', 'id', 'name']
    GROUP_MAP = {"Control": "ff", "Experimental": "fw"}
    DEPTH = 5

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # 1. 扫描并处理每个样本
    samples = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and d != 'analysis']
    samples.sort()

    all_vols, all_counts, all_dens, names = [], [], [], []
    group_indices = {gn: [] for gn in GROUP_MAP}

    for s_name in samples:
        s_path = os.path.join(BASE_DIR, s_name)
        s_out = os.path.join(ANALYSIS_DIR, s_name)
        os.makedirs(s_out, exist_ok=True)

        res = quantify(s_path, s_out, CLASSES, HEADER, TEMPLATE, analysis_depth=DEPTH)
        
        if res[0] is not None:
            all_vols.append(res[0]); all_counts.append(res[1]); all_dens.append(res[2]); names.append(s_name)
            curr_idx = len(names) - 1
            for gn, kw in GROUP_MAP.items():
                if kw in s_name: group_indices[gn].append(curr_idx)

    # 2. 汇总保存
    savemat(os.path.join(ANALYSIS_DIR, "summary_data.mat"), {
        "vols": np.array(all_vols), "counts": np.array(all_counts), "dens": np.array(all_dens), "names": names
    })

    # 3. 运行组间统计
    active_g = [n for n in group_indices if len(group_indices[n]) >= 2]
    if len(active_g) >= 2:
        df_temp = pd.read_csv(TEMPLATE)
        df_temp = df_temp[df_temp['depth'] == DEPTH].reset_index(drop=True)
        
        # 构建统计输入
        g_data_input = []
        for gn in active_g[:2]:
            idxs = group_indices[gn]
            v = np.stack([all_vols[i] for i in idxs], axis=1)
            c = [np.stack([all_counts[i][:, k] for i in idxs], axis=1) for k in range(len(CLASSES))]
            d = [np.stack([all_dens[i][:, k] for i in idxs], axis=1) for k in range(len(CLASSES))]
            g_data_input.append([v, c, d])
        
        run_stats(df_temp, active_g[:2], CLASSES, g_data_input, ANALYSIS_DIR)
