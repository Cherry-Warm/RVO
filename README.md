>>>> HRED
# RVOnet: Automatic diagnostic network for retinal vein occlusion
![network]()
## Introduction
The implementation of: <br>
[**RVOnet: Automatic diagnostic network for retinal vein occlusion**](论文地址)
## Requirements
- python 3.9
- Pytorch 1.10.1
- torchvision 0.11.2
- opencv-python
- pandas
- scipy
## Setup
### Installation
Clone the repo and install required packages:
```
git clone https://github.com/yuhaomo/HoVerTrans.git
cd HoVerTrans
pip install -r requirements.txt
```
### Dataset
-  You can unpack your dataset into the ./data folder.
```
./data
└─GDPH&SYSUCC
      ├─label.csv
      └─img
          ├─benign(0).png
          ├─benign(1).png
          ├─benign(2).png
          ├─malignant(0).png
          ├─malignant(1).png
          ...
```
- The format of the label.csv is as follows:
```
+------------------+-------+
| name             | label |
+------------------+-------+
| benign(0).png    |   0   |
| benign(1).png    |   0   |
| benign(2).png    |   0   |
| malignant(0).png |   1   |
| malignant(1).png |   1   |
...
```
### Training
```
python train.py --data_path ./data/GDPH&SYSUCC/img --csv_path ./data/GDPH&SYSUCC/label.csv --batch_size 32 --class_num 2 --epochs 250 --lr 0.0001 
```
## Citation
If you find this repository useful or use our dataset, please consider citing our work:
```
@ARTICLE{}
```
=======
# RVO
Let's stop RVO！
>>>>>>> 7f6317d36f449c29b9d696118ae9cc7ae6ef6df5
