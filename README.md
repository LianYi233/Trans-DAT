# Trans-DAT
This repository contains the official implementation for the following paper:
[Domain Adaptation-aware Transformer for Hyperspectral Object Tracking](https://ieeexplore.ieee.org/document/10491347), 
which has been accepeted by IEEE TCSVT.

## UPDATE
:star: Our tracking results (txt files) on HOT2023 datasets are available [HERE](https://pan.baidu.com/s/1nwFSK0SDFRwTE3XSjFUpSQ). (Access Code: eugd)

:star: Our tracking results (txt files) on HOT2024 datasets are available [HERE](https://pan.baidu.com/s/1P9oTI2OYMTfCiqtGOHra7Q?pwd=2024). (Access Code: 2024)


### Environment
- Python 3.9
- Pytorch 1.13.0
- CUDA 11.7

### Dataset
In this paper, all the experimental results and comparisons are conducted on [HOT2023 Challenge](https://www.hsitracking.com/). The dataset could be downloaded from [CONTEST PAGE](https://www.hsitracking.com/contest/). Please remember to modify the error annotations according to "Problems and Updates" before using the dataset.

### Testing 
```
cd pysot_toolkit
python test_3_dataset.py
```

### Training
```
cd ltr
python run_training.py
```


### Acknowledgement
The code in this repository is based on [TransT](https://github.com/chenxin-dlut/TransT). We would like to thank the authors for providing the great frameworks and models.

### Citation

If you find our work useful in your research, please cite:

```
@ARTICLE{10491347,
  author={Wu, Yinan and Jiao, Licheng and Liu, Xu and Liu, Fang and Yang, Shuyuan and Li, Lingling},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Domain Adaptation-aware Transformer for Hyperspectral Object Tracking}, 
  year={2024},
  doi={10.1109/TCSVT.2024.3385273}}
```
