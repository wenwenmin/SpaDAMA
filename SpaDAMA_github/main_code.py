import gc
import SpaDAMA.main_code as SpaDAMA
import Baselines.scpDeconv.scpDeconv_main.main as scp
import Baselines.Cell2location.main_code as cell2location
import Baselines.DestVi.main_code as DestVi
import Baselines.Stereoscope.main_code as Stereoscope
import Baselines.Tangram.main_code as Tangram
import Baselines.Spoint.main_code as Spoint
import Baselines.novoSpaRc.main as novoSpaRc
import Baselines.SpaOTsc.main_code as SpaOTsc
def train_Tangram_model(i,cell_key):
    print("-------------Tangram模型开始-----------------")
    Tangram.main(i, i + 1, cell_key)
    clear_memory()

def train_Spoint_model(i,cell_key):
    print("-------------Spoint模型开始-----------------")
    Spoint.main(i, i + 1, cell_key)
    clear_memory()
def train_DestVi_model(i,cell_key):
    print("-------------DestVi模型开始-----------------")
    DestVi.main(i, i + 1, cell_key)
    clear_memory()
def train_Stereoscope_model(i,cell_key):
    print("-------------Stereoscope模型开始-----------------")
    Stereoscope.main(i, i + 1, cell_key)
    clear_memory()
def train_cell2location_model(i,cell_key):
    print("-------------cell2location模型开始-----------------")
    cell2location.main(i, i + 1, cell_key)
    clear_memory()
def train_SpaDAMA_model(i,cell_key):
    print("-------------SpaDAMA模型开始-----------------")
    SpaDAMA.main(i, i + 1, cell_key)
    clear_memory()

def train_scp_model(i,cell_key):
    print("-------------scpDeconv模型开始-----------------")
    scp.main(i, i + 1, cell_key)
    clear_memory()

def train_novoSpaRc_model(i,cell_key,x, y):
    print("-------------novoSpaRc模型开始-----------------")
    novoSpaRc.main(i, i + 1, cell_key,x, y)
    clear_memory()

def train_SpaOTsc_model(i,cell_key,x, y):
    print("-------------novoSpaRc模型开始-----------------")
    SpaOTsc.main(i, i + 1, cell_key,x, y)
    clear_memory()
def clear_memory():
    # 删除所有不再需要的变量
    for name in dir():
        if not name.startswith('_') and name not in ['gc', 'train_scp_model', 'train_macm_model', 'clear_memory', 'i', 'number_of_datasets', 'scp', 'MACD']:
            del globals()[name]
    # 强制进行垃圾回收
    gc.collect()



if __name__ == '__main__':
        cell_key='cell_type'
        x='x'
        y='y'
        for i in range(1,2):
            print("——————————第" + str(i) + "个数据——————————")
            train_SpaDAMA_model(i, cell_key)
            train_scp_model(i, cell_key)
            train_Tangram_model(i, cell_key)
            train_DestVi_model(i, cell_key)
            train_cell2location_model(i, cell_key)
            train_Stereoscope_model(i, cell_key)
            train_Spoint_model(i,cell_key)
            train_novoSpaRc_model(i,cell_key,x,y)
            train_SpaOTsc_model(i,cell_key,x,y)
        




