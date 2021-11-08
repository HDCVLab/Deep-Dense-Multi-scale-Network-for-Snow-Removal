# **Deep Dense Multi-scale Network for Snow Removal Using Semantic and Depth Priors (TIP-2021)**
This project provides the source code for paper: ***Deep Dense Multi-scale Network for Snow Removal Using Semantic and Depth Priors (TIP-2021)***.


## **Pre-works**
1. Download pre-trained models of `VNL Monocular Depth Prediction Code` on KITTI and Cityscapes.
2. Download pre-trained models of `Semantic Segmentation Code` on KITTI and Cityscapes.
3. put pre-trained models of 1 and 2 into $STORE$/DDMSNet directory.
4. Download the Kitti-snow, cityscapes-snow and Snow100K datasets and put them into $STORE$/DDMSNet directory.

## **Train**
1. Change the train list file by modifying line 19 of train_data.py.
2. Change the validation set directory of the dataset you decide to run on by modifying the vatiable `val_data_dir` at line 79 of train.py.
3. Change the pre-trained model of semantic network by modifying line 122 of train.py (depend on which dataset you will train on, e.g. if you want to train on cityscapes dataset, then modify variable `ckpt_semantic_path` to `'cityscapes_best.pth'`).
4. run `python train.py`.

## **Test on Dataset**
1. Change the validation set directory of the dataset you decide to run on by modifying the variable `val_data_dir` at line 70 of test.py.
2. Change the pre-trained model of semantic network by modifying line 110 of test.py(depend on which dataset you will train on, e.g. if you want to train on cityscapes dataset, then modify variable `ckpt_semantic_path` to `'cityscapes_best.pth'`).
3. run `python test.py`.

## **Test on Single Image**
1. Step 1, 2 are same as `Test on Dataset`.

2. Change the location of raw image(tested image) and desnow image at line 157-160 in test_one.py.
3. run `python test_one.py`.

## **Download Links of Pretrained Models**
Links: https://drive.google.com/file/d/13ezCsznOm0C8qz1SRwgQY8vXHq-LNzCz/view?usp=sharing

Semantic Segmentation Network Pre-trained Models:
- Kitti: kitti_eigin.pth
- Cityscapes: cityscapes_best.pth

VNL Monocular Depth Prediction Network Pre-trained Models:
- Kitti and Cityscapes: kitti_eigen.path

DDMSNet Pre-trained Models:
- Kitti: kitti_DDMSNet
- Cityscapes: cityscapes_DDMSNet
- Snow100K: snow100k_DDMSNet

## **Download Links of Datasets**
- SnowKITTI2012: https://drive.google.com/file/d/1TB1WC60ZJvazepdvay18dCRr0yVLU6bH/view?usp=sharing
- SnowCityScapes: https://drive.google.com/file/d/1E6iXFV6K5UJ4Mrqer17v6KsHhQOFvjtO/view?usp=sharing
- Snow100K: https://sites.google.com/view/yunfuliu/desnownet

## **Download the Desnowed Results of Our Method** ##
- SnowKITTI2012 & SnowCityScapes: https://pan.baidu.com/s/1IuD1BMffXCS053D3dZ26dw, extraction code: 12ab
- Snow100K: https://pan.baidu.com/s/1iN4edXIFTeWJ0EJ4Vhossw, extraction code: 12ab


### Citation
```bibtex
  @article{zhang2021deep,
    title={Deep Dense Multi-scale Network for Snow Removal Using Semantic and Geometric Priors},
    author={Zhang, Kaihao and Li, Rongqing and Yu, Yanjiang and Luo, Wenhan and Li, Changsheng},
    journal={IEEE Transactions on Image Processing},
    year={2021}
  }
```

