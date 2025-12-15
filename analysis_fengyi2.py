import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.io import savemat
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

def quantify(sample_dir_path, classes, header, save_table, template_path):
    """
    对单个样本目录 (sample_dir_path) 进行量化。
    """
    
    # 从目录路径获取数据集名称
    dataset_name = os.path.basename(os.path.normpath(sample_dir_path))
    print(f"--- Quantifying: {dataset_name} ---")

    # --- 1. 测量结构体积 ---
    resolution = np.array([25, 25, 25])
    mm = np.prod(resolution) / (1E3**3)

    try:
        df_template = pd.read_csv(template_path)
    except FileNotFoundError:
        print(f"错误: 找不到结构模板文件: {template_path}")
        print("请检查 'template_path' 路径是否正确。")
        raise

    indexes = df_template['index'].values
    ids = df_template['id'].values
    path = df_template['structure_id_path']

    mhd_annotation_file = os.path.join(sample_dir_path, 'volume', 'result.mhd')

    try:
        img_original = sitk.ReadImage(mhd_annotation_file)
        img_casted = sitk.Cast(img_original, sitk.sitkUInt32)
        I_mask = sitk.GetArrayFromImage(img_casted)

    except Exception as e:
        print(f"错误: 无法读取MHD文件: {mhd_annotation_file}")
        print(f"详细信息: {e}")
        print("跳过此数据集。")
        return None, None, None, None

    I_mask_flat = I_mask.ravel()
    real_total_volume = np.sum(I_mask_flat > 0) * mm
    counted = np.bincount(I_mask_flat, minlength=len(indexes))
    df_volumes = counted.astype(float) * mm
    df_volumes[0] = 0

    df_new = df_volumes.copy()
    
    print(f"--- 1/2 汇总 {dataset_name} 体积 ---")
    for i in tqdm(range(1, len(ids)), desc=f"汇总体积", unit=" 结构"):
            parent_id = ids[i]
            current_id_path = f"/{parent_id}/"
            idx = path.str.contains(current_id_path, na=False)
            child_ids = ids[idx]
            parent_vol = np.sum(df_volumes[child_ids])
            df_new[parent_id] = parent_vol

    df_volumes = df_new
    print(f"{dataset_name}\t Total structure volume: {real_total_volume:.2f} mm^3")

    df_header = f"{dataset_name}_Volume"
    volumes_template = df_volumes[ids]
    table_volume = pd.DataFrame(volumes_template, columns=[df_header])
    
    

    df_vol = pd.concat([df_template.iloc[:, [0, -2, -1]], table_volume], axis=1)

    if save_table:
        save_vol = os.path.join(sample_dir_path, 'result_volume.csv')
        df_vol.to_csv(save_vol, index=False)

    # --- 2. 细胞计数/密度量化 ---
    num_list = []
    density_list = []
    df_cells = df_vol.copy()
    no_label_log = []

    print(f"--- 2/2 量化 {dataset_name} 细胞 ---")
    for i, class_name in tqdm(enumerate(classes), desc=f"量化细胞", total=len(classes), unit=" 类别"):
        df_new = np.zeros_like(df_volumes)
        
        csv_path = os.path.join(sample_dir_path, 'cell_registration', str(i), 'cell_registration.csv')

        try:
            df_csv = pd.read_csv(csv_path, header=None, names=header, dtype={'id': int})
            all_cell_ids = df_csv['id'].values
            
            no_label_count = np.sum(all_cell_ids == -1)
            no_label_log.append(f"Class '{class_name}': 发现 {no_label_count} 个 'no label' 细胞 (id = -1)。")
            valid_cell_ids = all_cell_ids[all_cell_ids >= 0]
            count = np.bincount(valid_cell_ids, minlength=len(df_volumes))

        except FileNotFoundError:
            print(f"警告: 文件未找到 {csv_path}. 将 {class_name} 计数设为 0。")
            no_label_log.append(f"Class '{class_name}': 文件未找到。")
            count = np.zeros(len(df_volumes), dtype=int)
            
        except Exception as e:
            print(f"警告: 无法读取 {csv_path}. 错误: {e}. 将 {class_name} 计数设为 0。")
            no_label_log.append(f"Class '{class_name}': 读取文件时出错 ({e})。")
            count = np.zeros(len(df_volumes), dtype=int)

        count[0] = 0

        count_summarized = count.copy()

        for j in range(1, len(ids)):
            parent_id = ids[j]
            
            current_id_path = f"/{parent_id}/"
            idx = path.str.contains(current_id_path, na=False)
            
            child_ids = ids[idx]
            
            total_count = np.sum(count[child_ids])
            
            count_summarized[parent_id] = total_count

        df_new = count_summarized.astype(float)
        num_list.append(df_new)

        density = np.divide(df_new, df_volumes, out=np.zeros_like(df_new), where=df_volumes!=0)
        density_list.append(density)

        count_in_order = df_new[ids]
        density_in_order = density[ids]

        table_count = pd.DataFrame(count_in_order, columns=[class_name])
        table_density = pd.DataFrame(density_in_order, columns=[f"{class_name}_density"])

        df_cells = pd.concat([df_cells, table_count, table_density], axis=1)

    num = np.stack(num_list, axis=1)
    density = np.stack(density_list, axis=1)

    if save_table:
        save_cells = os.path.join(sample_dir_path, 'result_density.csv')
        df_cells.to_csv(save_cells, index=False)

        log_file_path = os.path.join(sample_dir_path, 'no_label_cell_log.txt')
        try:
            with open(log_file_path, 'w') as f:
                f.write(f"--- 'No Label' (id = -1) 细胞日志 ---\n")
                f.write(f"样本: {dataset_name}\n")
                f.write("="*40 + "\n")
                for line in no_label_log:
                    f.write(line + "\n")
            print(f"--- 已保存 'no label' 日志到: {log_file_path} ---")
        except Exception as e:
            print(f"!!! 错误：无法保存 'no label' 日志文件到 {log_file_path}. 错误: {e}")
    
    return df_volumes, num, density, df_cells

def get_stats(group1, group2, paired, min_stat=0):
    # [此函数未更改]
    if group1.ndim == 1: group1 = group1.reshape(-1, 1)
    if group2.ndim == 1: group2 = group2.reshape(-1, 1)
    if group1.shape[1] == 0 or group2.shape[1] == 0:
        print("警告: get_stats 收到空组。")
        return np.zeros((group1.shape[0], 8))
        
    n_regions = group1.shape[0]
    stats = np.zeros((n_regions, 8))

    stats[:, 0] = np.mean(group1, axis=1)
    stats[:, 1] = np.mean(group2, axis=1)
    stats[:, 2] = np.std(group1, axis=1, ddof=1)
    stats[:, 3] = np.std(group2, axis=1, ddof=1)
    
    stats_0_safe = np.where(stats[:, 0] == 0, 1e-9, stats[:, 0])
    stats[:, 4] = 100 * (stats[:, 1] - stats[:, 0]) / stats_0_safe
    
    p_vals = np.ones(n_regions)
    if paired:
        if group1.shape[1] > 1 and group2.shape[1] > 1:
            p_vals = ttest_rel(group1, group2, axis=1)[1]
    else:
        if group1.shape[1] > 1 and group2.shape[1] > 1:
             p_vals = ttest_ind(group1, group2, axis=1, equal_var=False)[1]
        else:
             print("警告: 组中样本数不足 (N<2)，无法计算 t-test。p-value 将设为 1。")

    stats[:, 5] = np.nan_to_num(p_vals, nan=1.0)
    
    s_pos = ~np.isnan(p_vals) & (np.maximum(stats[:, 0], stats[:, 1]) > min_stat)
    
    stats[:, 6] = 1.0
    stats[:, 7] = 0.0

    if np.sum(s_pos) > 0:
        p_val_thresholded = p_vals[s_pos]
        
        reject, q_vals = fdrcorrection(p_val_thresholded, alpha=0.05, method='indep')
        stats[s_pos, 6] = q_vals
        
        sig_score = (q_vals < 0.01).astype(int) + \
                    (q_vals < 0.001).astype(int) + \
                    (q_vals < 0.0001).astype(int)
        stats[s_pos, 7] = sig_score

    stats[~s_pos, 0:5] = 0
    stats[~s_pos, 6] = 1
    stats[~s_pos, 7] = 0
    
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    
    return stats


def get_df_stats(df_template, group_names, classes, group_results):
    # [此函数未更改]
    all_stats_dfs = []
    header_prefix = ["Mean", "Mean", "StdDev", "StdDev", "PChange", "p", "p_adj", "sig"]

    stats_vol = get_stats(group_results[0][0], group_results[1][0], paired=False, min_stat=0)
    df_header = [f"{p}_{'Volume'}" for p in header_prefix]
    df_header[0] = f"{group_names[0]}_{df_header[0]}"
    df_header[1] = f"{group_names[1]}_{df_header[1]}"
    df_header[2] = f"{group_names[0]}_{df_header[2]}"
    df_header[3] = f"{group_names[1]}_{df_header[3]}"
    all_stats_dfs.append(pd.DataFrame(stats_vol, columns=df_header))

    stats_counts_list = []
    for c_idx, c_name in enumerate(classes):
        g1_counts = group_results[0][1][c_idx]
        g2_counts = group_results[1][1][c_idx]
        sub_stats = get_stats(g1_counts, g2_counts, paired=False, min_stat=0)
        df_header = [f"{p}_{c_name}_Counts" for p in header_prefix]
        df_header[0] = f"{group_names[0]}_{df_header[0]}"
        df_header[1] = f"{group_names[1]}_{df_header[1]}"
        df_header[2] = f"{group_names[0]}_{df_header[2]}"
        df_header[3] = f"{group_names[1]}_{df_header[3]}"
        stats_counts_list.append(pd.DataFrame(sub_stats, columns=df_header))
    all_stats_dfs.append(pd.concat(stats_counts_list, axis=1))

    stats_densities_list = []
    for c_idx, c_name in enumerate(classes):
        g1_densities = group_results[0][2][c_idx]
        g2_densities = group_results[1][2][c_idx]
        sub_stats = get_stats(g1_densities, g2_densities, paired=False, min_stat=0)
        df_header = [f"{p}_{c_name}_Densities" for p in header_prefix]
        df_header[0] = f"{group_names[0]}_{df_header[0]}"
        df_header[1] = f"{group_names[1]}_{df_header[1]}"
        df_header[2] = f"{group_names[0]}_{df_header[2]}"
        df_header[3] = f"{group_names[1]}_{df_header[3]}"
        stats_densities_list.append(pd.DataFrame(sub_stats, columns=df_header))
    all_stats_dfs.append(pd.concat(stats_densities_list, axis=1))
    
    df_final_stats = pd.concat([df_template] + all_stats_dfs, axis=1)
    
    return df_final_stats


# ##################################################################
# %% 第一部分：区域量化 (Regional quantitation)
# ##################################################################
print("--- 1. 开始区域量化 ---")

# --- 设置变量 ---
all_samples_dir = "/data/hdd12tb-1/fengyi/COMBINe/clearmap/fw2/results" #folder containing all Sample subfolders
template_path = "/home/fyu7/COMBINe/annotations/structure_template.csv" # annotation file, structure_template.csv
run_group_analysis = False 

classes = ['red glia','green glia','yellow glia','red neuron','green neuron','yellow neuron']
header = ['x', 'y', 'z', 'xt', 'yt', 'zt', 'id', 'name', 'sub1', 'sub2', 'sub3']
save_table = True

# --- 创建结果目录 ---
result_dir = os.path.join(all_samples_dir, 'analysis')
os.makedirs(result_dir, exist_ok=True)

# --- 查找数据集 (即查找所有 Sample 子文件夹) ---
data_subdirs_paths = []
for d in os.listdir(all_samples_dir):
    full_path = os.path.join(all_samples_dir, d)
    # 确保它是一个目录, 且不是 'analysis' 目录
    if os.path.isdir(full_path) and d != 'analysis':
        data_subdirs_paths.append(full_path)
data_subdirs_paths.sort() # 确保顺序一致

if not data_subdirs_paths:
    print(f"错误: 在 {all_samples_dir} 中未找到任何数据集子目录。")
    exit()
else:
    print(f"找到了 {len(data_subdirs_paths)} 个数据集。")

# --- 循环处理每个 Sample ---
volumes = []
counts = []
densities = []
df_results = []
valid_data_subdirs = [] 

# *** 变量名已更改 ***
for sample_dir_path in data_subdirs_paths:
    # 将 'sample_dir_path' (例如 '.../Sample_01') 传递给 quantify
    vol, num, den, df_c = quantify(sample_dir_path, classes, header, save_table, template_path)
    
    if vol is not None:
        volumes.append(vol)
        counts.append(num)
        densities.append(den)
        df_results.append(df_c)
        valid_data_subdirs.append(sample_dir_path)

if not volumes:
    print("错误：没有数据集被成功量化。退出脚本。")
    exit()
    
# --- 保存原始数据 ---
results_to_save = {
    'volumes': np.stack(volumes, axis=1), 
    'counts': np.stack(counts, axis=2),
    'densities': np.stack(densities, axis=2),
    'dataset_names': [os.path.basename(p) for p in valid_data_subdirs]
}
results_mat_path = os.path.join(result_dir, 'results.mat')
savemat(results_mat_path, results_to_save)
print(f"\n所有样本的量化结果已保存到 {results_mat_path}")


# ##################################################################
# %% 第二部分：分配组并进行统计 (可关闭)
# ##################################################################
# [此部分未更改]
if run_group_analysis:
    print("\n--- 2. 'run_group_analysis' 设置为 True，开始统计分析 ---")
    
    # --- 1. 定义组 ---
    # Python 是 0-索引! [1,4,6] 变为 [0,3,5]
    groups = [[0, 3, 5], [1, 2, 4]] 
    group_names = ["+/+", "F/+"]
    
    # --- 2. 安全检查 ---
    all_indices = [idx for group in groups for idx in group]
    max_required_index = max(all_indices) if all_indices else -1
    num_samples_found = len(volumes)

    if num_samples_found <= max_required_index:
        print(f"警告: 组分析失败。")
        print(f"    'groups' 变量需要索引高达 {max_required_index} (即至少 {max_required_index + 1} 个样本)。")
        print(f"    但只找到了 {num_samples_found} 个成功量化的样本。")
        print("    请检查 'groups' 变量或 'run_group_analysis' 设置。")
        print("    跳过统计分析。")
    
    else:
        # --- 3. 加载模板 ---
        df_template = pd.read_csv(template_path)
        
        # --- 4. 按组分离结果 ---
        group_results = []
        for j in range(len(groups)):
            group_indices = groups[j]
            group_vol = np.stack([volumes[i] for i in group_indices], axis=1)
            group_counts_by_class = []
            group_densities_by_class = []
            for k in range(len(classes)):
                c = np.stack([counts[i][:, k] for i in group_indices], axis=1)
                d = np.stack([densities[i][:, k] for i in group_indices], axis=1)
                group_counts_by_class.append(c)
                group_densities_by_class.append(d)
            group_results.append([group_vol, group_counts_by_class, group_densities_by_class])

        # --- 5. 执行统计 ---
        print("正在计算组间统计...")
        df_stats = get_df_stats(df_template, group_names, classes, group_results)

        # --- 6. 保存统计结果 ---
        stats_mat_path = os.path.join(result_dir, 'stats.mat')
        stats_to_save = {'df_stats_data': df_stats.values, 'df_stats_headers': list(df_stats.columns)}
        savemat(stats_mat_path, stats_to_save)

        if save_table:
            save_stats = os.path.join(result_dir, 'stats.csv')
            df_stats.to_csv(save_stats, index=False)
            print(f"统计表格已保存到 {save_stats}")
else:
    print("\n--- 2. 'run_group_analysis' 设置为 False，跳过统计分析 ---")

print("\n--- 分析完成 ---")