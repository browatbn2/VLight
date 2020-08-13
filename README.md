# VLight #

## Requirements

- Python 3.6 or newer
- PyTorch >= 1.1.0
- CUDA >= 10.0
- cuDNN >= 7.5

## Getting started

Download or clone the full repository.


### Install dependencies
```
pip install -r requirements.txt
```

### Datasets

You can train models or run evaluations on the following vessel segmenation datasets:

- [DRIVE](https://www.isi.uu.nl/Research/Databases/DRIVE/)
- [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/)
- [HRF](https://www5.cs.fau.de/research/data/fundus-images/)

Download the original datasets and place them in ./data. Alternatively, you can also update the paths in local_config.ini.


## Evaluation

To evaluate the pre-trained models:


```
python eval_vessels.py --model vlight --modelname models/pretrained/{drive,chase,hrf}_vlight --dataset {drive,chase,hrf}
```


## Training

To train VLight on DRIVE for 40 epochs (80k samples) with a learning rate of 0.001, for example, run:

```
python train_vessels.py --sessionname drive_vlight --lr 0.001 -e 40 --dataset-train drive --dataset-val drive 
```



