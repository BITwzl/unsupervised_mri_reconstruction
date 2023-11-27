# unsupervised_mri_reconstruction
The official code of paper *Adaptive Knowledge Distillation for High-Quality Unsupervised MRI Reconstruction with Model-Driven Priors*.

train teacher models
```sh
python main.py --cfg-path opts/bit_parcel.yaml
```
train student model
```sh
python main.py --cfg-path opts/bit_dccnn2_weighted_distill_parcel.yaml
```

train teacher models with fastmri dataset
```sh
python main.py --cfg-path opts/scalefastmri_parcel.yaml
```

train student models with fastmri dataset
```sh
python main.py --cfg-path opts/scalefastmri_dccnn2_weighted_distill_parcel.yaml
```
