"""读取数据"""
import copy
import multiprocessing as mp
from Baselines.Spoint.model import init_model
import anndata as ad
import torch
import os
import Baselines.Spoint.data_utils

import time
def main(a,b,cell_key):
    start_time = time.time()
    for i in range(a, b):
        print('第'+str(i)+'个切片')

        sc_file = 'Datasets\Real_datasets\dataset' + str(i) + '\scRNA.h5ad'
        st_file = 'Datasets\Real_datasets\dataset' + str(i) + '\Spatial.h5ad'
        st_ad1 = ad.read_h5ad(st_file)
        sc_data1 = ad.read_h5ad(sc_file)
        st_ad = copy.deepcopy(st_ad1)
        sc_data = copy.deepcopy(sc_data1)
        output_path = 'Baselines\Spoint\Result\dataset' + str(i)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        model = init_model(sc_ad=sc_data,
                                   st_ad=st_ad,
                                   celltype_key=cell_key,
                                   n_top_markers=500,
                                   n_top_hvg=2500)
        # model.train
        model.model_train(sm_lr=0.01,
                    st_lr=0.01)

        pre = model.deconv_spatial()

        pre.to_csv(output_path + "/proportion.csv")

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    print(f"spoint Total time taken: {total_time:.2f} seconds")  # 打印总时间
