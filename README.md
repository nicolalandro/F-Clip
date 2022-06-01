[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolalandro/F-Clip/blob/master/F_Clip_Demo.ipynb)


# F-Clip — Fully Convolutional Line Parsing

* train
```
nohup python3.7 train.py -d 0 -i HG1_D3 config/fclip_HG1_D3.yaml > experiment_HG1_D3.log 2>&1 &
```
* show dataset
```
python3.7 read_dataset.py
```

# create dataset from europa

```
python3.7 dataset/europa2dataset.py
# check with 
show_dataset_debug.py

python3.7 dataset/europa.py /home/super/datasets-nas/line_detection/europa/128x128 /home/super/datasets-nas/line_detection/europa/FClip/valid

# check with 
show_dataset.py
```