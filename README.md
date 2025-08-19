This project **heavily relies on the data loading and training framework of [HSIR](https://github.com/bit-isp/HSIR)**. We appreciate the excellent open-source work. 

📌 The model defined in `hybrid.py` is the one actually used during training.  
For better clarity and naming consistency, we also provide a cleaned-up and standardized version in `SMDe.py`.

📦 Pre-trained checkpoints and testing datasets can be downloaded from the following link:  
[Baidu Netdisk](https://pan.baidu.com/s/1ZQfjGeDEdHvA6ctDgWLeWA?pwd=1111)

📂 Datasets

[ICVL](https://doi.org/10.1007/978-3-319-46478-7_2)

[CAVE](https://doi.org/10.1109/TIP.2010.2046811)

[Urban](https://doi.org/10.1117/12.283843)

Please modify the dataset paths to your local storage location.
Use `train.py` to train the model, and `test.py` to test the model.

##  Usage

###  Installation
First, install the required dependencies:  
```bash
pip install -r requirements.txt
```
###  Train
Before training, modify the paths in the configuration file:
- `train_root`: set to your training dataset path
- `test_root`: set to your validation dataset path

Run the following command:
```bash
python train.py
```
###  Test
Before testing, modify the following parameters in the configuration file:
- `test_root`: set to the parent directory of your test dataset
- `resume_path`: set to the checkpoint file path to load the trained model
- `test_name`: set to the name of your test dataset (combined with test_root to form the complete path)

Run the following command:
```bash
python test.py
```
###  Using Our Model
```bash
import torch
from SMDe import smde

net = smde()
x = torch.randn(4, 1, 31, 64, 64)
y = net(x)
print(y.shape)
```
