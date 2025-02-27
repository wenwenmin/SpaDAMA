import sys
import time

import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import subprocess

import os

import cell2location
import scvi

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
import seaborn as sns
from scipy.sparse import csr_matrix
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel
def main(a,b,cell_key):
    start_time = time.time()
    for i in range(a,b):
        sc_file_path = 'Datasets\preproced_data\dataset' + str(i)+ '\Scdata_filter.h5ad'
        #D:\pythonplaces\MACD_github\Datasets\preproced_data\dataset1\Scdata_filter.h5ad
        spatial_file_path = 'Datasets\preproced_data\dataset' + str(i)+ '\Real_STdata_filter.h5ad'
        celltype_key =cell_key
        output_file_path ='Baselines\Cell2location\Result\dataset' + str(i)
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        adata_snrna_raw = sc.read_h5ad(sc_file_path)
        adata_vis = sc.read_h5ad(spatial_file_path)
        adata_snrna_raw.X = csr_matrix(adata_snrna_raw.X)
        adata_vis.X = csr_matrix(adata_vis.X)
        print(adata_snrna_raw)

        adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs[celltype_key].isin(np.array(
            adata_snrna_raw.obs[celltype_key].value_counts()[
                adata_snrna_raw.obs[celltype_key].value_counts() <= 1].index))]

        # remove cells and genes with 0 counts everywhere
        sc.pp.filter_genes(adata_snrna_raw, min_cells=1)
        sc.pp.filter_cells(adata_snrna_raw, min_genes=1)

        adata_snrna_raw.obs[celltype_key] = pd.Categorical(adata_snrna_raw.obs[celltype_key])
        adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs[celltype_key].isna(), :]

        selected = filter_genes(adata_snrna_raw, cell_count_cutoff=5, cell_percentage_cutoff2=0.03,
                                nonz_mean_cutoff=1.12)

        # filter the object
        adata_snrna_raw = adata_snrna_raw[:, selected].copy()

        # scvi.data.setup_anndata(adata=adata_snrna_raw, labels_key=celltype_key)
        cell2location.models.RegressionModel.setup_anndata(adata=adata_snrna_raw, labels_key=celltype_key)

        # create and train the regression model
        if hasattr(adata_snrna_raw.X, 'toarray'):
            counts_dense1 = adata_snrna_raw.X.toarray()  # 转换为稠密矩阵
            counts_rounded1 = np.round(counts_dense1).astype(int)
            # 将结果转换回稀疏矩阵格式
            adata_snrna_raw.X = csr_matrix(counts_rounded1)
        else:
            adata_snrna_raw.X = np.round(adata_snrna_raw.X).astype(int)
        mod = RegressionModel(adata_snrna_raw)

        # mod.train(max_epochs=150, batch_size=2500, train_size=1, lr=0.002, use_gpu=True)
        mod.train(max_epochs=500, batch_size=2500, train_size=1, lr=0.002)

        # plot ELBO loss history during training, removing first 20 epochs from the plot
        # mod.plot_history(20)

        # In this section, we export the estimated cell abundance (summary of the posterior distribution).
        # adata_snrna_raw = mod.export_posterior(
        #     adata_snrna_raw, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
        # )
        adata_snrna_raw = mod.export_posterior(
            adata_snrna_raw, sample_kwargs={'num_samples': 1000, 'batch_size': 2500}
        )

        # export estimated expression in each cluster
        if 'means_per_cluster_mu_fg' in adata_snrna_raw.varm.keys():
            inf_aver = adata_snrna_raw.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                                        for i in adata_snrna_raw.uns['mod'][
                                                                            'factor_names']]].copy()
        else:
            inf_aver = adata_snrna_raw.var[[f'means_per_cluster_mu_fg_{i}'
                                            for i in adata_snrna_raw.uns['mod']['factor_names']]].copy()
        inf_aver.columns = adata_snrna_raw.uns['mod']['factor_names']
        inf_aver.iloc[0:5, 0:5]

        intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
        adata_vis = adata_vis[:, intersect].copy()
        inf_aver = inf_aver.loc[intersect, :].copy()
        if hasattr(adata_vis.X, 'toarray'):
            counts_dense1 = adata_vis.X.toarray()  # 转换为稠密矩阵
            counts_rounded1 = np.round(counts_dense1).astype(int)
            # 将结果转换回稀疏矩阵格式
            adata_vis.X = csr_matrix(counts_rounded1)
        else:
            adata_vis.X = np.round(adata_vis.X).astype(int)

        cell2location.models.Cell2location.setup_anndata(adata=adata_vis)
        # scvi.data.view_anndata_setup(adata_vis)
        # create and train the model
        mod = cell2location.models.Cell2location(
            adata_vis, cell_state_df=inf_aver,
            # the expected average cell abundance: tissue-dependent
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=30,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection (using default here):
            detection_alpha=200
        )

        mod.train(max_epochs=1500,
                  # train using full data (batch_size=None)
                  batch_size=None,
                  # use all data points in training because
                  # we need to estimate cell abundance at all locations
                  train_size=1,
                  # use_gpu=True
                  )

        # plot ELBO loss history during training, removing first 100 epochs from the plot
        # mod.plot_history(1000)
        # plt.legend(labels=['full data training'])

        adata_vis = mod.export_posterior(
            adata_vis, sample_kwargs={'num_samples': 200, 'batch_size': 200}
        )
        # print(adata_vis)
        adata_vis.obsm['q05_cell_abundance_w_sf'].to_csv(output_file_path + '/Cell2location_result.csv')

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    print(f"cell2location Total time taken: {total_time:.2f} seconds")  # 打印总时间
