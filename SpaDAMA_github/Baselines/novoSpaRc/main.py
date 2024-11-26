import os
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
import novosparc as nc
from scipy import stats
import tangram as tg
from scipy.spatial.distance import cdist
import sys
import sys
# Example
# python ~/bio/SpatialBenchmarking/Codes/Deconvolution/SpaOTsc_pipeline.py \
# /home/share/xiaojs/spatial/sour_sep/mouce_brain_VISp/Ref_scRNA_VISp_qc2.h5ad \
# /home/share/xiaojs/spatial/sour_sep/tangram/merfish/MERFISH_mop.h5ad \
# cell_subclass \
# /home/share/xwanaf/sour_sep/simulation/SpaOTsc_test
import time
import numpy as np


def batch_process_gromov_wasserstein(cost_marker_genes, cost_expression, cost_locations,
                                     alpha_linear, p_expression, p_location,batch_size=5000):
    # Total number of rows (or observations) in the dataset
    total_rows = cost_marker_genes.shape[0]

    # Initialize the final result variable, if needed
    result = []

    # Process in batches
    for start_row in range(0, total_rows, batch_size):
        end_row = min(start_row + batch_size, total_rows)

        # Slice the data for the current batch
        batch_marker_genes = cost_marker_genes[start_row:end_row, :]
        batch_expression = cost_expression[start_row:end_row, :]
        batch_locations = cost_locations[start_row:end_row, :]

        # gw = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,
        #                                                         alpha_linear, p_expression, p_location, 'square_loss',
        #                                                         epsilon=5e-3, verbose=True)
        # Call the gromov_wasserstein_adjusted_norm function on the current batch
        batch_result = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(
            batch_marker_genes, batch_expression, batch_locations,
            alpha_linear, p_expression, p_location, 'square_loss',epsilon=5e-3, verbose=True
        )

        # Store the result for this batch
        result.append(batch_result)

        # Optionally, clear variables to free memory after each batch
        del batch_marker_genes, batch_expression, batch_locations, batch_result

    # Concatenate the results from all batches if needed
    final_result = np.concatenate(result, axis=0)

    return final_result




"""https://github.com/YangLabHKUST/SpatialScope/blob/master/compared_methods/novoSpaRc_pipeline.py"""
def main(a, b, cell_key, x, y):
    for i in range(a, b):
        print("——————————第" + str(i) + "个数据——————————")
        start_time = time.time()
        sc_file = f'Datasets\preproced_data\dataset{i}\\Scdata_filter.h5ad'
        st_file = f'Datasets\preproced_data\dataset{i}\\Real_STdata_filter.h5ad'
        output_file_path = f'Baselines/novoSpaRc\Result\\dataset{i}'
        os.makedirs(output_file_path, exist_ok=True)

        ad_sc = sc.read(sc_file)
        ad_sp = sc.read(st_file)

        gene_names = np.array(ad_sc.var.index.values)
        dge = ad_sc.to_df().values
        num_cells = dge.shape[0]
        hvg = np.argsort(np.divide(np.var(dge, axis=0), np.mean(dge, axis=0) + 0.0001))
        dge_hvg = dge[:, hvg[-2000:]]

        locations = ad_sp.obs[[x, y]].values
        num_locations = locations.shape[0]

        p_location, p_expression = nc.rc.create_space_distributions(num_locations, num_cells)
        cost_expression, cost_locations = nc.rc.setup_for_OT_reconstruction(dge_hvg,locations,num_neighbors_source = 5,num_neighbors_target = 5)

        gene_is = ad_sp.var.index.tolist()
        gene_sc = ad_sc.var.index.tolist()
        insitu_genes = list(set(gene_is).intersection(gene_sc))

        markers_in_sc = np.array([], dtype='int')
        for marker in insitu_genes:
            marker_index = np.where(gene_names == marker)[0]
            if len(marker_index) > 0:
                markers_in_sc = np.append(markers_in_sc, marker_index[0])

        insitu_matrix = np.array(ad_sp.to_df()[insitu_genes])
        cost_marker_genes = cdist(dge[:, markers_in_sc] / np.amax(dge[:, markers_in_sc]),
                                  insitu_matrix / np.amax(insitu_matrix))

        alpha_linear = 0.5
        gw = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,
                                                                alpha_linear, p_expression, p_location, 'square_loss',
                                                                epsilon=5e-3, verbose=True)
        # Example usage:
        # gw = batch_process_gromov_wasserstein(cost_marker_genes, cost_expression, cost_locations,
        #                                                 alpha_linear, p_expression, p_location,
        #                                                 batch_size=5000)
        gamma = gw
        for j in range(gamma.shape[1]):
            gamma[:, j] = gamma[:, j] / np.sum(gamma[:, j])

        ad_map = sc.AnnData(gamma, obs=ad_sc.obs, var=ad_sp.obs)
        tg.project_cell_annotations(ad_map, ad_sp, annotation=cell_key)
        ad_sp.obsm['tangram_ct_pred'].to_csv(output_file_path + '/novoSpaRc_decon.csv')

        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间
        print(f"novoSpaRC Total time taken: {total_time:.2f} seconds")  # 打印总时间
