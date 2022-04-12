# Self-Supervised Contrastive Graph Clustering 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a PyTorch implementation of "SCGC : Self-Supervised Contrastive Graph Clustering".(https://arxiv.org/XXXXXXXXX)

The repo, inlcuding data sets and pretrained models are, has been forked initially from [SDCN](https://github.com/bdy9527/SDCN). We use also use the model code from AGCN [AGCN](https://github.com/ZhihaoPENG-CityU/MM21---AGCN) and portions of contrastive loss code from [Graph-MLP](https://github.com/yanghu819/Graph-MLP). 

## Dependencies
- CUDA 11.3.0
- python 3.6.9
- pytorch 1.3.1

Note : SCGC is able to run with no GPU if the GUP timing code is commented out, and then will not require CUDA. 

## Datasets

The dataset contains 2 folders, `data` and `graph`. Please obtain them from the [dataset Google drive links](https://github.com/bdy9527/SDCN/blob/master/README.md). You will need to set `--data_path` to the parent folder containing `data` and `graph`. Please note that the `data` folder contains the pre-trained `.pkl` models as well. We directly use the pre-trained models from SDCN.


## Usage
- All parameters are defined in train.py with comments and explanations. 

- To run SCGC on on the 6 datasets, for 10 iterations, use the following. This code has GPU time and memory profiling enabled, which can be turned off by commenting relevant code. Our published ACC,NMI,ARI and F1 was run with profiling commented. 
```
python train.py --name usps --iterations 10 --epochs 200 --model SCGC --verbosity 0   --alpha 1 --beta 0.1 --order 4 --tau 0.5 --lr 0.001 
python train.py --name hhar --iterations 10 --epochs 200 --model SCGC --verbosity 0   --alpha 1 --beta 10  --order 4 --tau 2.25 --lr 0.001 
python train.py --name reut --iterations 10 --epochs 200 --model SCGC --verbosity 0   --alpha 3 --beta 0.1 --order 3 --tau 1 --lr 0.001 
python train.py --name acm  --iterations 10 --epochs 200 --model SCGC --verbosity 0   --alpha 0.5 --beta 0.1 --order 2 --tau 0.25 --lr 0.001 
python train.py --name dblp --iterations 10 --epochs 200 --model SCGC --verbosity 0   --alpha 0.5 --beta 0.1 --order 1 --tau 0.25 --lr 0.001 
python train.py --name cite --iterations 10 --epochs 200 --model SCGC --verbosity 0   --alpha 1 --beta 0.1 --order 1 --tau 0.25 --lr 0.0001 
```

- To replicate the SCGC*, run the following.
```
python train.py --name usps --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 4 --beta 0.1 --order 4 --tau 0.25 --lr 0.001 --influence
python train.py --name hhar --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 1 --beta 10  --order 3 --tau 2.25 --lr 0.001 --influence
python train.py --name reut --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 0.5 --beta 0.1 --order 3 --tau 0.25 --lr 0.001 --influence
python train.py --name acm  --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 1 --beta 0.1 --order 1 --tau 0.25 --lr 0.001 --influence
python train.py --name dblp --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 1 --beta 0.1 --order 1 --tau 0.25 --lr 0.001 --influence
python train.py --name cite --iterations 10 --epochs 200 --model SCGC_TRIM --verbosity 0   --alpha 1 --beta 0.1 --order 1 --tau 0.25 --lr 0.0001 --influence
```

- To reun the prfiling for AGCN and SGCN we used the following. These use the SDCN and AGCN defaults.
```
python train.py --name usps --iterations 1 --epochs 200 --model SDCN --verbosity 0   --lr 0.001	--alpha 0.01 --beta 0.1
python train.py --name hhar --iterations 1 --epochs 200 --model SDCN --verbosity 0   --lr 0.001	--alpha 0.01 --beta 0.1
python train.py --name reut --iterations 1 --epochs 200 --model SDCN --verbosity 0   --lr 0.0001	--alpha 0.01 --beta 0.1
python train.py --name acm  --iterations 1 --epochs 200 --model SDCN --verbosity 0   --lr 0.001	--alpha 0.01 --beta 0.1
python train.py --name dblp --iterations 1 --epochs 200 --model SDCN --verbosity 0   --lr 0.001	--alpha 0.01 --beta 0.1
python train.py --name cite --iterations 1 --epochs 200 --model SDCN --verbosity 0   --lr 0.0001	--alpha 0.01 --beta 0.1

python train.py --name usps --iterations 1 --epochs 200 --model AGCN --verbosity 0   --lr 0.001	--alpha 1000 --beta 1000
python train.py --name hhar --iterations 1 --epochs 200 --model AGCN --verbosity 0   --lr 0.001	--alpha 0.1 --beta 1
python train.py --name reut --iterations 1 --epochs 200 --model AGCN --verbosity 0   --lr 0.0001	--alpha 10	--beta 10
python train.py --name acm  --iterations 1 --epochs 200 --model AGCN --verbosity 0   --lr 0.001	--alpha 0.01 --beta 0.1
python train.py --name dblp --iterations 1 --epochs 200 --model AGCN --verbosity 0   --lr 0.001	--alpha 0.01 --beta 0.1
python train.py --name cite --iterations 1 --epochs 200 --model AGCN --verbosity 0   --lr 0.0001	--alpha 0.01 --beta 0.1
```

## Data sources and code
Datasets and code is forked from [SDCN](https://github.com/bdy9527/SDCN). We use also use the model code from AGCN [AGCN](https://github.com/ZhihaoPENG-CityU/MM21---AGCN) and portions of contrastive loss code from [Graph-MLP](https://github.com/yanghu819/Graph-MLP). We acknowledge and thank the authors of these works for sharing their code.

## Citation
```
TODO
```
