import os

path = '/home6/share_data/huangshan_16022/singlecoil_train/file1000323.h5'
print(os.path.exists(path))

import torch

# torch.ifft2()

print(len(os.listdir('/home6/share_data/huangshan_16022/dataset/singlecoil_train')))