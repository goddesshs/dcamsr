
# # import csv
# # import os

# # import logging
# # import pickle
# # import random
# # import xml.etree.ElementTree as etree
# # from pathlib import Path
# # from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
# # from warnings import warn
# # import pathlib

# # import h5py
# # import numpy as np
# # import torch
# # import yaml
# # from torch.utils.data import Dataset
# # import os

# # import sys
# # print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# # print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # from fastmri.data.mri_data import fetch_dir
# # def get_dataset(exp_name):
# #     # path_config = pathlib.Path.cwd() / "mriSR_dirs.yaml"
# #     path_config = pathlib.Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mriSR_dirs.yaml"))
# #     knee_path = fetch_dir("knee_path", path_config)
# #     knee_train_csv_path = fetch_dir("knee_train_csv_path", path_config)
# #     knee_val_csv_path = fetch_dir("knee_val_csv_path", path_config)
# #     knee_test_csv_path = fetch_dir("knee_val_csv_path", path_config)
# #     knee_challenge = 'singlecoil'
    
# #     # M4raw_path = fetch_dir("M4Raw_path", path_config)
# #     # M4RAW_train_csv_path = fetch_dir("M4Raw_train_csv_path", path_config)
# #     # M4RAW_val_csv_path = fetch_dir("M4Raw_val_csv_path", path_config)
# #     # M4RAW_test_csv_path = fetch_dir("M4Raw_test_csv_path", path_config)
# #     # M4RAW_challenge = 'multicoil'
    
# #     if exp_name == 'knee':
# #         return knee_path,knee_train_csv_path,knee_val_csv_path,knee_test_csv_path,knee_challenge
# #     # elif exp_name == 'M4Raw':
# #     #     return M4raw_path,M4RAW_train_csv_path,M4RAW_val_csv_path,M4RAW_test_csv_path,M4RAW_challenge

# # dataset_name = 'knee'
# # data_path,train_csv,val_csv,test_csv,challenge = get_dataset(dataset_name)

# # path_config = pathlib.Path.cwd() / "mriSR_dirs.yaml"
# # upscale = 4
# # net_name = 'DCAMSR'
# # # logdir = fetch_dir("log_path", path_config) / net_name / dataset_name / f"{upscale}x_SR"

# # with open('recode.txt', 'w') as t:
# #     with open(train_csv,'r') as f:
# #         reader=csv.reader(f)

# #         for row in reader:
# #             # pd_metadata, pd_num_slices = self._retrieve_metadata(os.path.join(self.cur_path,row[0]+'.h5'))

# #             # pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[1]+'.h5'))
# #             path_pd = os.path.join(data_path, row[0]+'.h5')
# #             if not os.path.exists(path_pd):
# #                 t.write(path_pd)
# #                 t.write('\n')
# #                 t.flush()
                
# #             path_pdfs = os.path.join(data_path, row[1]+'.h5')
# #             if not os.path.exists(path_pd):
# #                 t.write(path_pd)
# #                 t.write('\n')
# #                 t.flush()
            
# #         # for slice_id in range(min(pd_num_slices,pdfs_num_slices)):
# #         #     self.examples.append((os.path.join(self.cur_path, row[0]+'.h5'),os.path.join(self.cur_path, row[1]+'.h5')
# #         #                             ,slice_id,pd_metadata,pdfs_metadata))

# # # class SliceDataset():
# # #     def __init__(
# # #             self,
# # #             root,
# # #             transform,
# # #             csv_file,
# # #             challenge,
# # #             sample_rate=1,
# # #             dataset_cache_file=pathlib.Path("dataset_cache.pkl"),
# # #             num_cols=None,
# # #             mode='train',
# # #     ):
# # #         self.mode = mode

# # #         #challenge
# # #         if challenge not in ("singlecoil", "multicoil"):
# # #             raise ValueError('challenge should be either "singlecoil" or "multicoil"')
# # #         self.challenge = challenge
# # #         self.recons_key = (
# # #             "hf_" if challenge == "singlecoil" else "reconstruction_rss"
# # #         )
# # #         #transform
# # #         self.transform = transform

# # #         self.examples=[]

# # #         self.cur_path=root
# # #         # self.cur_path='/home/jupyter-huangshoujin/shoujinhuang/data/M4RawV1.1/motion/inter-scan_motion'
# # #         self.csv_file=os.path.join(csv_file)

# # #         #读取CSV
# # #         with open(self.csv_file,'r') as f:
# # #             reader=csv.reader(f)

# # #             for row in reader:
# # #                 pd_metadata, pd_num_slices = self._retrieve_metadata(os.path.join(self.cur_path,row[0]+'.h5'))

# # #                 pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[1]+'.h5'))

# # #                 for slice_id in range(min(pd_num_slices,pdfs_num_slices)):
# # #                     self.examples.append((os.path.join(self.cur_path, row[0]+'.h5'),os.path.join(self.cur_path, row[1]+'.h5')
# # #                                           ,slice_id,pd_metadata,pdfs_metadata))

# # #         if sample_rate < 1:
# # #             random.shuffle(self.examples)
# # #             num_examples = round(len(self.examples) * sample_rate)

# # #             self.examples=self.examples[0:num_examples]
            
            
# # #         def _retrieve_metadata(self, fname):
# # #             with h5py.File(fname, "r") as hf:
# # #                 et_root = etree.fromstring(hf["ismrmrd_header"][()])

# # #                 enc = ["encoding", "encodedSpace", "matrixSize"]
# # #                 enc_size = (
# # #                     int(et_query(et_root, enc + ["x"])),
# # #                     int(et_query(et_root, enc + ["y"])),
# # #                     int(et_query(et_root, enc + ["z"])),
# # #                 )
# # #                 rec = ["encoding", "reconSpace", "matrixSize"]
# # #                 recon_size = (
# # #                     int(et_query(et_root, rec + ["x"])),
# # #                     int(et_query(et_root, rec + ["y"])),
# # #                     int(et_query(et_root, rec + ["z"])),
# # #                 )

# # #                 lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
# # #                 enc_limits_center = int(et_query(et_root, lims + ["center"]))
# # #                 enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

# # #                 padding_left = enc_size[1] // 2 - enc_limits_center
# # #                 padding_right = padding_left + enc_limits_max

# # #                 num_slices = hf["kspace"].shape[0]

# # #             metadata = {
# # #                 "padding_left": padding_left,
# # #                 "padding_right": padding_right,
# # #                 "encoding_size": enc_size,
# # #                 "recon_size": recon_size,
# # #             }

# # #             return metadata, num_slices

# import importlib
# from argparse import ArgumentParser
# import os
# import sys


# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# import experimental


# selected_module = importlib.import_module('experimental.DCAMSR.DCAMSR')

# parent_parser = ArgumentParser(add_help=False)

# parser = selected_module.add_model_specific_args(parent_parser)

import torch

print(torch.cuda.is_available())