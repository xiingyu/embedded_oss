import subprocess
from tqdm import tqdm
from glob import glob

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

train_img_list = glob('C:/Users/nrk52/anaconda3/envs/avala/Embedded/BB-13/train/images/*.jpg')
print(len(train_img_list)) 

valid_img_list = glob('C:/Users/nrk52/anaconda3/envs/avala/Embedded/BB-13/valid/images/*.jpg')
print(len(valid_img_list))

with open('C:/Users/nrk52/anaconda3/envs/avala/Embedded/BB-13/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('C:/Users/nrk52/anaconda3/envs/avala/Embedded/BB-13/val.txt', 'w') as f:
    f.write('\n'.join(valid_img_list) + '\n')

import yaml

with open('C:/Users/nrk52/anaconda3/envs/avala/Embedded/BB-13/data.yaml', 'r') as f:
    data = yaml.safe_load(f)

data['train'] = 'C:/Users/nrk52/anaconda3/envs/avala/Embedded/BB-13/train.txt'
data['val'] = 'C:/Users/nrk52/anaconda3/envs/avala/Embedded/BB-13/val.txt'

with open('C:/Users/nrk52/anaconda3/envs/avala/Embedded/BB-13/data.yaml', 'w') as f:
    yaml.dump(data, f)

# 명령어를 리스트 형태로 전달
command = [
    "python", 
    "C:/Users/nrk52/anaconda3/envs/avala/Embedded/yolov5/train.py",  # 정확한 경로로 수정
    "--img", "416",
    "--batch", "8",
    "--epochs", "90",
    "--data", "C:/Users/nrk52/anaconda3/envs/avala/Embedded/BB-13/data.yaml",
    "--cfg", "C:/Users/nrk52/anaconda3/envs/avala/Embedded/yolov5/models/yolov5s.yaml",
    "--weights", "yolov5s.pt",
    "--name", "blocks_nbb"
]

# subprocess.PIPE를 사용하여 실행 명령어의 출력을 받아올 수 있습니다.
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

# tqdm을 사용하여 진행률을 표시합니다.
for line in tqdm(iter(process.stdout.readline, ''), desc="Training YOLOv5"):
    print(line, end='')  # 출력된 로그를 화면에 표시합니다.

# 프로세스가 완료될 때까지 기다립니다.
process.wait()

from tensorflow.python.client import device_lib
device_lib.list_local_devices()