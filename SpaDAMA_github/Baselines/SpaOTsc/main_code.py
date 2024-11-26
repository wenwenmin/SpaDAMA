# basic imports
import os
import time
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from spaotsc import SpaOTsc
from scipy import stats
import tangram as tg
from scipy.sparse import csr_matrix
import sys
# Example
# python ~/bio/SpatialBenchmarking/Codes/Deconvolution/SpaOTsc_pipeline.py \
# /home/share/xiaojs/spatial/sour_sep/mouce_brain_VISp/Ref_scRNA_VISp_qc2.h5ad \
# /home/share/xiaojs/spatial/sour_sep/tangram/merfish/MERFISH_mop.h5ad \
# cell_subclass \
# /home/share/xwanaf/sour_sep/simulation/SpaOTsc_test
import numpy as np
from scipy import stats


# def batch_spearmanr(a, b, batch_size=1000):
#     """
#     计算 a 的每一行与 b 的每一行的 Spearman 相关性，按批次处理。
#
#     参数:
#     a: 形状为 (m, n) 的 numpy 数组，m 是样本数，n 是特征数。
#     b: 形状为 (p, n) 的 numpy 数组，p 是样本数，n 是特征数。
#     batch_size: 每次处理的批次数量。
#
#     返回:
#     rho_final: (m, p) 相关性矩阵，m 行 x p 列。
#     pval_final: (m, p) p 值矩阵，m 行 x p 列。
#     """
#
#     # 获取 a 和 b 的行数
#     m, n_a = a.shape  # m: a 的行数，n_a: a 的列数
#     p, n_b = b.shape  # p: b 的行数，n_b: b 的列数
#
#     if n_a != n_b:
#         raise ValueError(f"The number of columns in a ({n_a}) must match the number of columns in b ({n_b}).")
#
#     # 初始化列表来存储每一批次的相关性结果
#     rho_list = []
#     pval_list = []
#     # 如果 a 或 b 是 pandas DataFrame，转换为 numpy 数组
#     if isinstance(a, pd.DataFrame):
#         a = a.values
#     if isinstance(b, pd.DataFrame):
#         b = b.values
#
#     # 将 a 按批次拆分
#     a_split = np.array_split(a, np.ceil(m / batch_size), axis=0)
#     # print(a_split.shape)
#     i=1
#     # 对每个批次进行处理
#     for batch in a_split:
#         print(batch.shape)
#         # 对每一行批次中的数据和 b 中的每一行计算 Spearman 相关性
#         batch_rho = []
#         batch_pval = []
#
#         for i in range(batch.shape[0]):  # batch 的行数
#             rho_row = []
#             pval_row = []
#             for j in range(p):  # b 的行数
#                 # 计算第 i 行的 batch 和第 j 行的 b 之间的 Spearman 相关性
#                 rho, pval = stats.spearmanr(batch[i, :], b[j, :])
#                 rho_row.append(rho)
#                 pval_row.append(pval)
#
#             # 将该行的结果加入到批次结果中
#             batch_rho.append(rho_row)
#             batch_pval.append(pval_row)
#
#         # 将批次的结果转为 numpy 数组并添加到最终结果列表中
#         rho_list.append(np.array(batch_rho))
#         pval_list.append(np.array(batch_pval))
#         print("第",i,"个批次结束")
#         i=i+1
#
#     # 将所有批次的结果按行拼接在一起
#     rho_final = np.concatenate(rho_list, axis=0)
#     pval_final = np.concatenate(pval_list, axis=0)
#
#     return rho_final, pval_final

# def batch_spearmanr(a, b, batch_size=5000):
#     m, n_a = a.shape
#     p, n_b = b.shape
#
#     if n_a != n_b:
#         raise ValueError(f"The number of columns in a ({n_a}) must match the number of columns in b ({n_b}).")
#
#     # Initialize lists to store the correlation results
#     rho_list = []
#     pval_list = []
#
#     # Convert to numpy arrays if pandas DataFrame
#     if isinstance(a, pd.DataFrame):
#         a = a.values
#     if isinstance(b, pd.DataFrame):
#         b = b.values
#
#     # Split 'a' by rows for batch processing
#     a_split = np.array_split(a, np.ceil(m / batch_size), axis=0)
#
#     # Process each batch
#     for i, batch in enumerate(a_split, start=1):
#         print(f"Batch {i} started.", batch.shape,b.shape)
#         # Compute Spearman correlation for each batch, use axis=1 to compare rows (row-wise correlation)
#         rho, pval = stats.spearmanr(batch, b, axis=1)  # Corrected to axis=1 for row-wise comparison
#         print(rho.shape, pval.shape)
#         rho_list.append(rho)
#         pval_list.append(pval)
#
#         # Optionally print progress
#         print(f"Batch {i} finished.")
#
#     # Concatenate the results from all batches
#     rho_final = np.concatenate(rho_list, axis=0)  # Ensure correct row-wise concatenation
#     pval_final = np.concatenate(pval_list, axis=0)  # Ensure correct row-wise concatenation
#
#     return rho_final, pval_final

def batch_spearmanr(a, b, batch_size=5000):
    m, n_a = a.shape
    p, n_b = b.shape

    if n_a != n_b:
        raise ValueError(f"The number of columns in a ({n_a}) must match the number of columns in b ({n_b}).")

    # Initialize lists to store the correlation results
    rho_list = []
    pval_list = []

    # Convert to numpy arrays if pandas DataFrame
    if isinstance(a, pd.DataFrame):
        a = a.values
    if isinstance(b, pd.DataFrame):
        b = b.values

    # Split 'a' by rows for batch processing
    a_split = np.array_split(a, np.ceil(m / batch_size), axis=0)
    print(f"Total number of batches: {len(a_split)}")

    # Process each batch
    for i, batch in enumerate(a_split, start=1):
        print(f"Batch {i} started. batch.shape={batch.shape}, b.shape={b.shape}")

        # Prepare lists for storing correlations for this batch
        batch_rho = []
        batch_pval = []
        j=1
        # Calculate correlation for each row in batch with each row in b
        for row_a in batch:
            batch_rho_row = []
            batch_pval_row = []
            for row_b in b:
                print(j)
                j=j+1
                rho, pval = stats.spearmanr(row_a, row_b)  # Calculate Spearman correlation for each pair of rows
                batch_rho_row.append(rho)
                batch_pval_row.append(pval)
            batch_rho.append(batch_rho_row)
            batch_pval.append(batch_pval_row)

        # Convert batch results to numpy arrays
        batch_rho = np.array(batch_rho)
        batch_pval = np.array(batch_pval)
        # Print the shape of the batch's rho and pval
        print(f"Batch {i} rho.shape={batch_rho.shape}, pval.shape={batch_pval.shape}")

        # Append to the final lists
        rho_list.append(batch_rho)
        pval_list.append(batch_pval)

        # Optionally print progress
        print(f"Batch {i} finished.")

    # Concatenate the results from all batches along axis 0 (rows)
    rho_final = np.concatenate(rho_list, axis=0)  # Ensure the results are concatenated along the rows
    pval_final = np.concatenate(pval_list, axis=0)

    # Check the final shape of rho_final and pval_final
    print(f"Final rho_final.shape={rho_final.shape}, pval_final.shape={pval_final.shape}")

    return rho_final, pval_final
def main(a, b, cell_key, x, y):
    for i in range(a, b):
        print("——————————第" + str(i) + "个数据——————————")
        start_time = time.time()
        sc_file = f'Datasets\preproced_data\dataset{i}\\Scdata_filter.h5ad'
        st_file = f'Datasets\preproced_data\dataset{i}\\Real_STdata_filter.h5ad'
        output_file_path = f'Baselines\SpaOTsc-master\Result\\dataset{i}'
        os.makedirs(output_file_path, exist_ok=True)

        ad_sc = sc.read(sc_file)
        ad_sp = sc.read(st_file)
        celltype_key = cell_key
        df_sc = ad_sc.to_df()
        df_IS = ad_sp.to_df()

        try:
            pts = ad_sp.obs[[x, y]].values
        except:
            pts = ad_sp.obs[[x, y]].values
        is_dmat = distance_matrix(pts, pts)

        df_is = df_IS

        gene_is = df_is.columns.tolist()
        gene_sc = df_sc.columns.tolist()
        gene_overloap = list(set(gene_is).intersection(gene_sc))
        a = df_is[gene_overloap]
        b = df_sc[gene_overloap]
        a = a.astype(np.float16)
        b = b.astype(np.float16)
        print(a.shape, b.shape)
        # rho, pval = batch_spearmanr(a, b, batch_size=1928)
        # print(rho.shape, pval.shape)
        rho, pval = stats.spearmanr(a, b, axis=1)
        rho[np.isnan(rho)] = 0
        mcc = rho[-(len(df_sc)):, 0:len(df_is)]
        C = np.exp(1 - mcc)

        issc = SpaOTsc.spatial_sc(sc_data=df_sc, is_data=df_is, is_dmat=is_dmat)

        issc.transport_plan(C ** 2, alpha=0, rho=1.0, epsilon=1.0, cor_matrix=mcc, scaling=False)
        gamma = issc.gamma_mapping
        for j in range(gamma.shape[1]):
            gamma[:, j] = gamma[:, j] / np.sum(gamma[:, j])
        ad_map = sc.AnnData(gamma, obs=ad_sc.obs, var=ad_sp.obs)
        tg.project_cell_annotations(ad_map, ad_sp, annotation=celltype_key)
        ad_sp.obsm['tangram_ct_pred'].to_csv(output_file_path + '/SpaOTsc_decon.csv')
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间
        print(f"SpaOTsc Total time taken: {total_time:.2f} seconds")  # 打印总时间
