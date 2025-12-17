import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.io import savemat
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.multitest import multipletests
import multiprocessing
from functools import partial
import xlsxwriter

#id filtering
#analyze with depth information
#reading data from multiple sample folders
#statistical analysis between two groups

def quantify(sample_dir_path, classes, header, save_table, template_path, analysis_depth=None, max_id_limit=65535):
    """
    对单个样本进行量化分析。
    
    参数:
        sample_dir_path: 样本文件夹路径
        classes: 细胞类别列表
        header: 细胞CSV文件的表头
        save_table: 是否保存CSV结果
        template_path: 结构模板CSV路径
        analysis_depth (int, optional): 分析深度 (例如 5)。如果设置，最终结果只保留该深度的区域。
        max_id_limit (int): ID 最大值限制 (例如 5000)。大于此值的 ID 将被视为背景/无效，防止内存溢出。
    """
    
    # 从目录路径获取数据集名称
    dataset_name = os.path.basename(os.path.normpath(sample_dir_path))
    print(f"--- Quantifying: {dataset_name} ---")

    # --- 1. 基础设置 ---
    resolution = np.array([25, 25, 25])
    mm = np.prod(resolution) / (1E3**3)

    try:
        df_template_original = pd.read_csv(template_path)
    except FileNotFoundError:
        print(f"错误: 找不到结构模板文件: {template_path}")
        raise
        
    # --- [关键步骤] 过滤模板 ID ---
    # 这一步必须在最开始做，以确定我们只处理合理范围内的 ID
    df_template = df_template_original[
        (df_template_original['id'] >= 0) & 
        (df_template_original['id'] <= max_id_limit)
    ].copy()
    
    print(f"模板已加载: 保留 ID <= {max_id_limit} 的区域, 共 {len(df_template)} 个结构。")

    # --- 检查 depth 列是否存在 ---
    if 'depth' not in df_template.columns:
        if analysis_depth is not None:
            print("警告: 模板中没有 'depth' 列，无法进行深度过滤！将输出所有层级。")
        analysis_depth = None 

    indexes = df_template['index'].values
    ids = df_template['id'].values
    path = df_template['structure_id_path']
    
    # 确定计算用的数组大小
    current_max_id = np.max(ids) if len(ids) > 0 else 0
    bincount_minlength = int(current_max_id) + 1

    # --- 2. 测量结构体积 (Volume) ---
    mhd_annotation_file = os.path.join(sample_dir_path, 'volume', 'result.mhd')

    try:
        img_original = sitk.ReadImage(mhd_annotation_file)
        img_casted = sitk.Cast(img_original, sitk.sitkInt64) 
        I_mask = sitk.GetArrayFromImage(img_casted)
    except Exception as e:
        print(f"错误: 无法读取MHD文件: {mhd_annotation_file}\n详细信息: {e}")
        return None, None, None, None

    I_mask_flat = I_mask.ravel()
    
    # --- [关键步骤] 过滤 Mask 体素 ---
    # 只统计在 max_id_limit 范围内的体素
    valid_voxels_mask = (I_mask_flat >= 0) & (I_mask_flat <= current_max_id)
    valid_voxels = I_mask_flat[valid_voxels_mask]
    
    # 统计每个 ID 的体素数
    counted = np.bincount(valid_voxels, minlength=bincount_minlength)
    df_volumes = counted.astype(float) * mm
    df_volumes[0] = 0 # 背景体积设为 0

    # --- [关键步骤] 层级汇总 (Aggregation) ---
    # 即使我们要过滤 depth，也必须先在全集上做汇总，因为父级体积 = 子级体积之和
    df_new = df_volumes.copy()
    
    print(f"--- 1/2 汇总体积 {dataset_name} ---")
    # 注意：这里使用全量 ids 循环，确保父子关系计算正确
    for i in tqdm(range(1, len(ids)), desc=f"Volume Aggregation", unit=" struct"):
            parent_id = ids[i]
            current_id_path = f"/{parent_id}/"
            # 查找所有包含当前 ID 路径的子区域
            idx = path.str.contains(current_id_path, na=False)
            child_ids = ids[idx]
            
            # 累加
            parent_vol = np.sum(df_volumes[child_ids]) 
            df_new[parent_id] = parent_vol

    df_volumes = df_new
    
    # 构建包含体积的临时 DataFrame
    volumes_template = df_volumes[ids] 
    df_header_vol = f"{dataset_name}_Volume"
    table_volume = pd.DataFrame(volumes_template, columns=[df_header_vol])
    
    # 将模板信息与体积合并
    df_vol = pd.concat([df_template.reset_index(drop=True), table_volume], axis=1)

    # --- 3. 细胞计数 (Cell Counting) ---
    num_list = []
    density_list = []
    df_cells = df_vol.copy()
    no_label_log = []

    print(f"--- 2/2 量化细胞 {dataset_name} ---")
    for i, class_name in tqdm(enumerate(classes), desc=f"Cell Quantification", total=len(classes)):
        # ... [读取 CSV 和 bincount 的部分保持不变] ...
        csv_path = os.path.join(sample_dir_path, 'cell_registration', str(i), 'cell_registration.csv')

        try:
            df_csv = pd.read_csv(csv_path, header=None, names=header, dtype={'id': int})
            all_cell_ids = df_csv['id'].values
            
            # 记录异常
            no_label_count = np.sum(all_cell_ids == -1)
            if no_label_count > 0:
                no_label_log.append(f"Class '{class_name}': Found {no_label_count} cells with id=-1")

            # 过滤细胞 ID
            valid_cells_mask = (all_cell_ids >= 0) & (all_cell_ids <= current_max_id)
            valid_cell_ids = all_cell_ids[valid_cells_mask]
            
            # 计数 (这是原始的 lookup array，长度很大)
            count = np.bincount(valid_cell_ids, minlength=bincount_minlength)

        except FileNotFoundError:
            count = np.zeros(bincount_minlength, dtype=int)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            count = np.zeros(bincount_minlength, dtype=int)

        count[0] = 0
        count_summarized = count.copy()

        # 层级汇总 (Aggregation)
        for j in range(1, len(ids)):
            parent_id = ids[j]
            current_id_path = f"/{parent_id}/"
            idx = path.str.contains(current_id_path, na=False)
            child_ids = ids[idx]
            
            total_count = np.sum(count[child_ids])
            count_summarized[parent_id] = total_count

        df_new_counts = count_summarized.astype(float)
        
        # 计算密度 (Count / Volume) - 原始 lookup array
        density_raw = np.divide(df_new_counts, df_volumes, out=np.zeros_like(df_new_counts), where=df_volumes!=0)

        count_in_order = df_new_counts[ids]
        density_in_order = density_raw[ids]

        # 将对齐后的数据加入列表
        num_list.append(count_in_order)
        density_list.append(density_in_order)

        # 构建 DataFrame 列
        table_count = pd.DataFrame(count_in_order, columns=[class_name])
        table_density = pd.DataFrame(density_in_order, columns=[f"{class_name}_density"])

        # 拼接到总表中
        df_cells = pd.concat([df_cells, table_count, table_density], axis=1)

    num = np.stack(num_list, axis=1)
    density = np.stack(density_list, axis=1)

    # --- 4. [关键步骤] 最终结果筛选 (Filtering) ---
    # 默认返回所有数据（仅提取体积列的值）
    return_vol_values = df_vol.iloc[:, -1].values 

    if analysis_depth is not None:
        print(f"正在筛选结果: 仅保留 Depth == {analysis_depth} 的区域...")
        
        # 创建筛选掩码
        mask_depth = df_vol['depth'] == analysis_depth
        
        # 1. 筛选 DataFrame (用于保存 CSV)
        df_cells_filtered = df_cells[mask_depth].copy()
        df_vol_filtered = df_vol[mask_depth].copy() # 仅用于保存 volume csv
        
        # 2. 筛选返回给统计模块的矩阵/数组
        # 注意：这里必须保证维度缩减，否则 get_stats 会报错
        return_vol_values = df_vol.iloc[mask_depth.values, -1].values
        num_filtered = num[mask_depth.values, :]
        density_filtered = density[mask_depth.values, :]
        
        # 替换变量以便保存和返回
        df_cells = df_cells_filtered
        df_vol = df_vol_filtered
        num = num_filtered
        density = density_filtered
        
        print(f"筛选完成: 剩余 {len(df_cells)} 个区域。")

    # --- 5. 保存结果 ---
    if save_table:
        # 文件名加上后缀以示区别
        suffix = f"_depth{analysis_depth}" if analysis_depth is not None else ""
        
        save_vol_path = os.path.join(sample_dir_path, f'result_volume{suffix}.csv')
        df_vol.to_csv(save_vol_path, index=False)
        
        save_cells_path = os.path.join(sample_dir_path, f'result_density{suffix}.csv')
        df_cells.to_csv(save_cells_path, index=False)

        # 保存简单的 log (如果有异常)
        if no_label_log:
            log_path = os.path.join(sample_dir_path, 'no_label_log.txt')
            with open(log_path, 'w') as f:
                f.write('\n'.join(no_label_log))

    # 返回值说明:
    # 1. return_vol_values: 1D numpy array (长度 = 筛选后的区域数), 包含体积值
    # 2. num: 2D numpy array (筛选后的区域数 x 类别数), 包含计数
    # 3. density: 2D numpy array (筛选后的区域数 x 类别数), 包含密度
    # 4. df_cells: DataFrame, 包含完整信息的表格
    return return_vol_values, num, density, df_cells

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

def load_existing_results(csv_path, classes, dataset_name):
    """
    从已有的 CSV 结果文件中加载数据，模拟 quantify 的输出格式。
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 1. 提取体积 (Volume)
        # 根据之前的逻辑，体积列名通常是 "{dataset_name}_Volume"
        vol_col_name = f"{dataset_name}_Volume"
        if vol_col_name in df.columns:
            vol = df[vol_col_name].values
        else:
            # 如果找不到具体名字，尝试找最后一列包含 Volume 的（容错）
            vol_cols = [c for c in df.columns if 'Volume' in c]
            if vol_cols:
                vol = df[vol_cols[-1]].values
            else:
                raise ValueError(f"在 {csv_path} 中找不到体积列")

        # 2. 提取计数 (Counts) 和 密度 (Densities)
        num_list = []
        den_list = []
        
        for class_name in classes:
            # 计数列通常就是类别名
            if class_name in df.columns:
                num_list.append(df[class_name].values)
            else:
                raise ValueError(f"找不到类别列: {class_name}")
                
            # 密度列通常是 类别名_density
            den_col_name = f"{class_name}_density"
            if den_col_name in df.columns:
                den_list.append(df[den_col_name].values)
            else:
                raise ValueError(f"找不到密度列: {den_col_name}")

        # 堆叠成矩阵 (Rows x Classes)
        num = np.stack(num_list, axis=1)
        den = np.stack(den_list, axis=1)
        
        return vol, num, den, df
        
    except Exception as e:
        print(f"!!! 读取现有文件失败: {csv_path}")
        print(f"详细错误: {e}")
        return None, None, None, None

def statistic_diff_means(x, y, axis=0):
    """置换检验的统计量：均值之差"""
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def worker_permutation_task(args):
    """
    单个任务的工作函数：计算单个脑区、单个指标的 Permutation P-value
    args: (group1_data_array, group2_data_array, n_resamples)
    """
    g1_data, g2_data, n_resamples = args
    
    # 简单的方差检查，如果全是0或数据太少，直接返回 1.0
    if len(g1_data) == 0 or len(g2_data) == 0:
        return 1.0
    if np.all(g1_data == 0) and np.all(g2_data == 0):
        return 1.0
        
    # 执行置换检验
    res = stats.permutation_test(
        (g1_data, g2_data), 
        statistic_diff_means, 
        vectorized=False, 
        n_resamples=n_resamples, 
        alternative='two-sided'
    )
    return res.pvalue

def perform_advanced_stats(df_template, group_names, classes, group_results,result_dir, n_resamples=1000):
    """
    高级统计分析：Permutation Test + FDR Correction
    输出格式：多 Sheet 的 Excel 文件 (.xlsx)
    """
    print(f"--- 启动高级统计分析 (输出为多 Sheet Excel) ---")
    
    # --- 1. 数据准备 ---
    g1_vol = group_results[0][0]
    g2_vol = group_results[1][0]
    
    g1_counts = group_results[0][1] 
    g2_counts = group_results[1][1]
    
    g1_dens = group_results[0][2] 
    g2_dens = group_results[1][2]
    
    g1_label = group_names[0]
    g2_label = group_names[1]

    # 提取基础信息列 (出现在所有 Sheet 的左侧)
    # 根据你的截图，这些是关键列
    target_info_cols = ['id', 'atlas_id', 'parent_structure_id', 'depth', 'name', 'acronym']
    # 过滤出当前 df_template 中实际存在的列
    info_cols = [c for c in target_info_cols if c in df_template.columns]
    df_info = df_template[info_cols].copy()
    
    # 初始化显著性计数器
    sig_counters = np.zeros(len(df_template), dtype=int)
    
    # 用于临时存储计算好的 DataFrame，最后再一次性写入 Excel
    sheets_dict = {} 

    # --- 定义计算函数 (内部复用) ---
    def calculate_metric_stats(pool, metric_prefix, data1, data2):
        """计算均值、标准差、Permutation P值、FDR P值"""
        # 1. 描述性统计
        m1 = np.mean(data1, axis=1)
        s1 = np.std(data1, axis=1, ddof=1)
        m2 = np.mean(data2, axis=1)
        s2 = np.std(data2, axis=1, ddof=1)
        
        # 2. Permutation Test
        n_regions = data1.shape[0]
        perm_args = [(data1[r, :], data2[r, :], n_resamples) for r in range(n_regions)]
        raw_p = pool.map(worker_permutation_task, perm_args) # 调用外部定义的 worker
        raw_p = np.array(raw_p)
        
        # 3. FDR Correction
        reject, p_fdr, _, _ = multipletests(raw_p, alpha=0.05, method='fdr_bh')
        
        # 4. 组装成 DataFrame 部分
        df_res = pd.DataFrame({
            f"{metric_prefix} Mean ({g1_label})": m1,
            f"{metric_prefix} Std ({g1_label})": s1,
            f"{metric_prefix} Mean ({g2_label})": m2,
            f"{metric_prefix} Std ({g2_label})": s2,
            f"{metric_prefix} P-raw": raw_p,
            f"{metric_prefix} P-FDR": p_fdr
        })
        
        return df_res, p_fdr

    # --- 2. 开始并行计算 ---
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpu_count) as pool:
        
        # === A. 处理 Volume ===
        print("  -> 计算指标: Volume...")
        df_vol_stats, p_fdr_vol = calculate_metric_stats(pool, "Vol", g1_vol, g2_vol)
        # 累计显著性
        sig_counters += (p_fdr_vol < 0.05).astype(int)
        # 合并信息列和数据列
        sheets_dict["Volume"] = pd.concat([df_info, df_vol_stats], axis=1)

        # === B. 处理每种细胞 ===
        for i, class_name in enumerate(classes):
            print(f"  -> 计算细胞: {class_name}...")
            
            # 1. Count
            df_cnt_stats, p_fdr_cnt = calculate_metric_stats(pool, "Count", g1_counts[i], g2_counts[i])
            sig_counters += (p_fdr_cnt < 0.05).astype(int)
            
            # 2. Density
            df_den_stats, p_fdr_den = calculate_metric_stats(pool, "Density", g1_dens[i], g2_dens[i])
            sig_counters += (p_fdr_den < 0.05).astype(int)
            
            # 3. 合并该细胞的所有数据到一个 Sheet
            # 格式: Info + Count Stats + Density Stats
            df_cell_combined = pd.concat([df_info, df_cnt_stats, df_den_stats], axis=1)
            
            # Excel Sheet 名称长度有限制(31字符)，截断一下防止报错
            sheet_name = class_name[:30] 
            sheets_dict[sheet_name] = df_cell_combined

    # --- 3. 构建 Summary Sheet (放在最前面) ---
    df_summary = df_info.copy()
    df_summary['Total_Significant_Metrics'] = sig_counters
    # 按显著性降序排列
    df_summary = df_summary.sort_values(by='Total_Significant_Metrics', ascending=False)
    
    # --- 4. 写入 Excel 文件 ---
    filename_str = f"Stats_Report_{g1_label}_vs_{g2_label}.xlsx"
    output_path = os.path.join(result_dir, filename_str) # <--- 关键修改
    print(f"  -> 正在写入 Excel 文件: {output_path} ...")
    
    # 使用 xlsxwriter 引擎来支持格式化
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        
        # 定义一个辅助函数：写入 Sheet 并设置格式
        def write_sheet_with_format(df, sheet_name):
            df.to_excel(writer, sheet_name=sheet_name, index=False, freeze_panes=(1, 2)) # 冻结第1行和前2列
            
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # 设置高亮格式：如果 P-FDR < 0.05，标红
            red_format = workbook.add_format({'font_color': '#9C0006', 'bg_color': '#FFC7CE'})
            
            # 遍历列，找到包含 "P-FDR" 的列，应用条件格式
            for col_num, col_name in enumerate(df.columns):
                if "P-FDR" in col_name:
                    # Excel 列索引从 0 开始 (A=0, B=1...)
                    # 设置条件格式：Cell Value < 0.05
                    worksheet.conditional_format(1, col_num, len(df), col_num, # (first_row, first_col, last_row, last_col)
                                               {'type': 'cell',
                                                'criteria': '<',
                                                'value': 0.05,
                                                'format': red_format})
            
            # 自动调整列宽 (简单估算)
            for i, col in enumerate(df.columns):
                col_len = len(str(col)) + 2
                # 限制最大宽度，防止太宽
                if col_len > 30: col_len = 30
                if col_len < 10: col_len = 10
                worksheet.set_column(i, i, col_len)

        # A. 写入 Summary Sheet
        write_sheet_with_format(df_summary, "Summary")
        
        # B. 写入 Volume Sheet
        # 注意：这里需要按照 Summary 的顺序重新排序 Volume 表，保持一致性？
        # 或者保持原始顺序。为了查找方便，通常保持原始 ID 顺序，或者也按显著性排序。
        # 这里我们让其他 Sheet 保持原始 ID 顺序 (df_info 的顺序)，Summary 按显著性排序。
        write_sheet_with_format(sheets_dict["Volume"], "Volume")
        
        # C. 写入各个细胞的 Sheet
        for class_name in classes:
            sheet_name = class_name[:30]
            if sheet_name in sheets_dict:
                write_sheet_with_format(sheets_dict[sheet_name], sheet_name)

    print(f"完成！请查看生成的文件: {output_path}")
    return df_summary # 返回 summary 用于可能的打印查看

# ##################################################################
# %% 第一部分：区域量化 (Regional quantitation)
# ##################################################################

if __name__ == '__main__':

    print("--- 1. 开始区域量化 ---")

    all_samples_dir = "/data/hdd12tb-1/fengyi/COMBINe/clearmap" #folder containing all Sample subfolders
    template_path = "/home/fyu7/COMBINe/annotations/structure_template.csv" # annotation file, structure_template.csv
    run_group_analysis = True
    ANALYSIS_DEPTH = 5
    MAX_ID_LIMIT = 5000
    classes = ['red glia','green glia','yellow glia','red neuron','green neuron','yellow neuron']
    header = ['x', 'y', 'z', 'xt', 'yt', 'zt', 'id', 'name', 'sub1', 'sub2', 'sub3']
    save_table = True
    FORCE_RECALCULATE = False

    GROUP_CONFIG = {
        "Control": "ff",   # 文件夹名包含 'ff' 的归为 Control 组
        "Experimental": "fw"  # 文件夹名包含 'fw' 的归为 Experimental 组
    }

    result_dir = os.path.join(all_samples_dir, 'analysis')
    os.makedirs(result_dir, exist_ok=True)

    # --- 查找数据集 (即查找所有 Sample 子文件夹) ---
    data_subdirs_paths = []
    if os.path.exists(all_samples_dir):
        for d in os.listdir(all_samples_dir):
            full_path = os.path.join(all_samples_dir, d)
            # 排除 analysis 文件夹，只保留目录
            if os.path.isdir(full_path) and d != 'analysis' and d != 'analysis_grouped':
                data_subdirs_paths.append(full_path)
        data_subdirs_paths.sort() # 排序保证处理顺序一致
    else:
        print(f"错误: 找不到主目录 {all_samples_dir}")
        exit()

    print(f"在主目录中找到了 {len(data_subdirs_paths)} 个子文件夹。")

    # --- 循环处理每个 Sample ---
    volumes = []
    counts = []
    densities = []
    df_results = []
    valid_data_subdirs = [] 

    group_indices_map = {name: [] for name in GROUP_CONFIG.keys()}
    dataset_names_list = []

    current_data_index = 0 

    for sample_dir_path in data_subdirs_paths:
        dataset_name = os.path.basename(os.path.normpath(sample_dir_path))
        
        # --- 步骤 A: 判定分组 ---
        assigned_group = None
        for group_name, keyword in GROUP_CONFIG.items():
            if keyword in dataset_name:
                assigned_group = group_name
                break 
        
        # --- 步骤 B: 检查文件是否存在并决定操作 ---
        # 构建预期的输出文件名
        suffix = f"_depth{ANALYSIS_DEPTH}" if ANALYSIS_DEPTH is not None else ""
        expected_csv_name = f'result_density{suffix}.csv'
        expected_csv_path = os.path.join(sample_dir_path, expected_csv_name)
        
        vol, num, den, df_c = None, None, None, None
        
        # 逻辑：如果不强制重算 且 文件存在 -> 加载
        if not FORCE_RECALCULATE and os.path.exists(expected_csv_path):
            print(f"[{dataset_name}] 检测到已有结果 ({expected_csv_name}) -> 跳过计算，直接加载...")
            vol, num, den, df_c = load_existing_results(expected_csv_path, classes, dataset_name)
            # 如果加载失败（比如文件损坏），则回退到重新计算
            if vol is None:
                print(f"[{dataset_name}] 加载失败，尝试重新计算...")
                vol, num, den, df_c = quantify(
                    sample_dir_path, classes, header, save_table, template_path, 
                    analysis_depth=ANALYSIS_DEPTH, max_id_limit=MAX_ID_LIMIT
                )
                
        else:
            # 文件不存在 或 强制重算 -> 执行计算
            if FORCE_RECALCULATE:
                print(f"[{dataset_name}] 强制重新计算...")
            vol, num, den, df_c = quantify(
                sample_dir_path, classes, header, save_table, template_path, 
                analysis_depth=ANALYSIS_DEPTH, max_id_limit=MAX_ID_LIMIT
            )

        # --- 步骤 C: 收集数据 (无论是计算的还是加载的) ---
        if vol is not None:
            volumes.append(vol)
            counts.append(num)
            densities.append(den)
            df_results.append(df_c)
            dataset_names_list.append(dataset_name)
            
            if assigned_group:
                group_indices_map[assigned_group].append(current_data_index)
            
            current_data_index += 1
        else:
            print(f"警告: 样本 {dataset_name} 处理失败或无法加载。")

    if not volumes:
        print("错误：没有有效数据。退出脚本。")
        exit()

    # --- 4. 保存汇总结果 (results.mat) ---
    # 即使是加载的数据，我们也重新保存一份汇总的 .mat，保证它是最新的
    results_to_save = {
        'volumes': np.stack(volumes, axis=1), 
        'counts': np.stack(counts, axis=2),
        'densities': np.stack(densities, axis=2),
        'dataset_names': dataset_names_list
    }
    results_mat_path = os.path.join(result_dir, 'results_all.mat')
    savemat(results_mat_path, results_to_save)
    print(f"\n所有数据已汇总并保存到 {results_mat_path}")

# ##################################################################
# %% 第二部分：分配组并进行统计 (可关闭)
# ##################################################################
if run_group_analysis:
        print("\n--- 2. 开始统计分析 ---")
        
        # 1. 筛选有效组 (保持之前的逻辑)
        active_group_names = []
        active_groups_indices = []
        for g_name, indices in group_indices_map.items():
            if len(indices) > 0:
                active_group_names.append(g_name)
                active_groups_indices.append(indices)
        
        if len(active_groups_indices) < 2:
            print("错误: 有效分组少于 2 组，无法进行统计对比。")
        else:
            # 2. 准备数据
            df_template = pd.read_csv(template_path)
            # 确保 ID 过滤
            df_template = df_template[(df_template['id'] >= 0) & (df_template['id'] <= MAX_ID_LIMIT)]
            if ANALYSIS_DEPTH is not None and 'depth' in df_template.columns:
                 df_template = df_template[df_template['depth'] == ANALYSIS_DEPTH].reset_index(drop=True)

            group_names_to_compare = active_group_names[:2]
            groups_indices_to_compare = active_groups_indices[:2]
            
            print(f"正在对比: {group_names_to_compare[0]} vs {group_names_to_compare[1]}")

            # 3. 提取 Numpy 数组
            group_results_arrays = []
            for indices in groups_indices_to_compare:
                # 提取体积
                group_vol = np.stack([volumes[i] for i in indices], axis=1)
                # 提取计数和密度
                group_counts_by_class = []
                group_densities_by_class = []
                for k in range(len(classes)):
                    c = np.stack([counts[i][:, k] for i in indices], axis=1)
                    d = np.stack([densities[i][:, k] for i in indices], axis=1)
                    group_counts_by_class.append(c)
                    group_densities_by_class.append(d)
                
                group_results_arrays.append([group_vol, group_counts_by_class, group_densities_by_class])

            # 4. === 调用新的高级统计函数 ===
            # n_resamples=1000 是标准配置，如果非常慢可以暂时改为 500
            _ = perform_advanced_stats(
                df_template, 
                group_names_to_compare, 
                classes, 
                group_results_arrays, 
                result_dir=result_dir,  # <--- 传入结果目录
                n_resamples=1000        # 如果速度太慢，可改为 500
            )

            # 5. 保存
            print(f"\n统计分析完成，结果已保存到 {result_dir} 目录下的 Excel 文件中。") 
