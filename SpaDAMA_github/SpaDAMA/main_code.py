from collections import Counter

import anndata
import anndata as ad
import pandas as pd

from SpaDAMA_github.SpaDAMA.SpaDAMA_model import SpaDAMA
from MACD1.data_prepare import *
import copy
import time
import anndata as ad
import os
import copy
import pandas as pd
from collections import Counter


def main(a, b, cell_key):
    for i in range(a, b):
        st_file = 'SpaDAMA_github\Datasets\Real_datasets\dataset' + str(i) + '\Spatial.h5ad'
        sc_file = 'SpaDAMA_github\Datasets\Real_datasets\dataset' + str(i) + '\scRNA.h5ad'
        st_data1 = ad.read_h5ad(st_file)
        sc_data1 = ad.read_h5ad(sc_file)
        sc_data = copy.deepcopy(sc_data1)
        st_data = copy.deepcopy(st_data1)
        outfile = 'SpaDAMA_github\SpaDAMA\Result\Result\dataset' + str(i)
        datafile = 'SpaDAMA_github\Datasets\preproced_data\dataset' + str(i)
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        if not os.path.exists(datafile):
            os.makedirs(datafile)

        """数据处理"""
        data_prepare(sc_ad=sc_data, st_ad=st_data, celltype_key=cell_key,
                     h5ad_file_path=outfile, data_file_path=datafile,
                     n_layers=2, n_latent=2048)

        sc_adata = anndata.read_h5ad(datafile + '\sm_scvi_ad.h5ad')
        st_adata = anndata.read_h5ad(datafile + '\st_scvi_ad.h5ad')
        real_sc_adata = anndata.read_h5ad(datafile + '\Scdata_filter.h5ad')
        sm_labelad = anndata.read_h5ad(datafile + '\Sm_STdata_filter.h5ad')

        sm_data = pd.DataFrame(data=sc_adata.X, columns=sc_adata.var_names)
        sm_label = sm_labelad.obsm['label']
        st_data = pd.DataFrame(data=st_adata.X, columns=st_adata.var_names)
        #
        # print(st_data.shape, sm_data.shape, sm_label.shape)
        count_ct_dict = Counter(list(real_sc_adata.obs[cell_key]))
        celltypenum = len(count_ct_dict)
        start_time = time.time()  # 记录开始时间
        print("------Start Running Stage------")
        model_da = SpaDAMA(celltypenum, outdirfile=outfile,
                        used_features=list(sm_data.columns), num_epochs=300)
        model_da.double_train(sm_data=sm_data, st_data=st_data, sm_label=sm_label,mask_retio=0.2)
        final_preds_target = model_da.prediction()
        final_preds_target.to_csv(outfile + '/final_pro1.csv')
        final_preds_target.columns = sm_label.columns.tolist()
        pd.DataFrame(data=final_preds_target).to_csv(outfile + '/final_pro.csv')

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    print(f"MACD Total time taken: {total_time:.2f} seconds")  # 打印总时间

