VD-LAB 
==
This repository contains the source code for the paper:  
[VD-LAB: A view-decoupled network with local-global aggregation bridge for airborne laser scanning point cloud classification](https://www.sciencedirect.com/science/article/abs/pii/S0924271622000168).

## Environment
- Python>=3.5  
- CUDA>=10.0  
- PyTorch>=1.0
- numpy
- torchvision
- etw-pytorch-utils
- future
- h5py

## Installation  
``` 
pip install -r requirements.txt  
python setup.py build_txt --inplace
```

## Training  
```
python -m train.train --data_root
```

## Evaluation  
```
python -m train.eval --data_root --param_path
```

## Acknowledgement
- [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)  
- [KPConv](https://github.com/HuguesTHOMAS/KPConv)
- [ConvPoint](https://github.com/aboulch/ConvPoint)

## Citation
If you find this work useful in your research, please consider citing:
```
@article{li2022vd,  
    author = {Jihao Li and Martin Weinmann and Xian Sun and Wenhui Diao and Yingchao Feng and Stefan Hinz and Kun Fu},  
    title = {VD-LAB: A view-decoupled network with local-global aggregation bridge for airborne laser scanning point cloud classification},  
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},  
    volume = {186},  
    pages = {19-33},  
	year={2022},  
    publisher={Elsevier}  
}
```
