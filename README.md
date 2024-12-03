# SpaDAMA
In this study, we introduce a Domain-Adversarial Masked Autoencoder (SpaDAMA) for cell type deconvolution  in spatial transcriptomics data.
SpaDAMA leverages Domain-Adversarial learning to align real ST data with simulated ST data generated from scRNA-seq data. By projecting both datasets into a shared latent space, SpaDAMA effectively minimizes the data modality gap. Furthermore, SpaDAMA incorporates masking techniques to enhance the model’s ability to learn robust features from real ST data, while mitigating noise and spatial confounding factors.
![(Variational)](fig1.png)


## System environment
To run `SpaDAMA`, you need to install [PyTorch](https://pytorch.org) with GPU support first. The environment supporting SpaDAMA and baseline models is specified in the `requirements.txt` file.

## Datasets
The publicly available  datasets were used in this study. You can download them from https://doi.org/10.5281/zenodo.14221635



## Run SpaDAMA and other Baselines models
After configuring the environment, download dataset4 from the Simulated_datasets in the data repository and place it into the Simulated_datasets folder. Then, Run `main.py`to start the process.If you want to run other data, simply modify the file path.

## Cite

## Contact
huanglin212@aliyun.com

minwenwen@ynu.edu.cn
