This project **heavily relies on the data loading and training framework of [HSIR](https://github.com/bit-isp/HSIR)**. We appreciate the excellent open-source work. 

📌 The model defined in `hybrid.py` is the one actually used during training.  
For better clarity and naming consistency, we also provide a cleaned-up and standardized version in `SMDe.py`.

📦 Pre-trained checkpoints and testing datasets can be downloaded from the following link:  
[Baidu Netdisk](https://pan.baidu.com/s/1ZQfjGeDEdHvA6ctDgWLeWA?pwd=1111)

📂 Datasets
We use the following hyperspectral datasets in our experiments:

ICVL (DOI)

CAVE (DOI)

Urban (DOI)

🚀 Usage
Modify dataset paths
Update the dataset paths in the configuration files or scripts to match your local storage location.

Train the model
bash
python train.py
Test the model
bash
python test.py
