#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predictor_infer.py
- 使用 predictor.PREDICT 做推理
- 支持 CSV（含序列列）或 TXT（一行一条序列）
- 可选择输出到新 CSV/TXT
- 可选打印 one-hot 通道数以确认是 4 通道
"""


import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
from predictor_models import *

import re
import numpy as np
import torch
from torch.nn.functional import pad
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class PREDICT:
    def __init__(self, model_path):
        self.input_size = 4
        self.hidden_size = 256
        self.output_size = 1
        self.lambda_l2 = 0.001
        self.dropout_rate = 0.2
        self.model_path = model_path
        self.use_gpu = torch.cuda.is_available()
        self.model = self.load_model()
        print("[CHK] input_size passed to LSTMModel =", self.input_size)
        bn = self.model.cnn1[0]  # cnn1 = Sequential(BN, Conv, ReLU, Pool)
        print("[CHK] cnn1 BatchNorm1d num_features =", bn.num_features)



    def load_model(self):
        model = LSTMModel(self.input_size, self.hidden_size, self.output_size,
                          self.dropout_rate, self.lambda_l2)
        state = torch.load(
            self.model_path,
            map_location=("cuda" if self.use_gpu else "cpu")
        )
        model.load_state_dict(state)
        if self.use_gpu:
            model.cuda()
        model.eval()  # 关闭 dropout/bn
        return model

    @staticmethod
    def string_to_array(my_string: str):
        my_string = my_string.lower()
        my_string = re.sub('[^acgtACGT]', 'z', my_string)
        return np.array(list(my_string))

    @staticmethod
    def one_hot_encode(my_array: np.ndarray):
        # Channel order: A, C, G, T; unknown bases map to a discarded fifth slot.
        mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        idx = np.fromiter((mapping.get(ch, 4) for ch in my_array), dtype=np.int64, count=len(my_array))
        oh5 = np.eye(5, dtype=np.float32)[idx]  # 5 通道，其中最后一列是 z 占位
        return oh5[:, :4]  # 丢掉 z 列 → 4 通道

    def seq_onehot(self, seq_list):
        mats = [torch.tensor(self.one_hot_encode(self.string_to_array(s)),
                             dtype=torch.float32) for s in seq_list]
        max_len = max(m.shape[0] for m in mats) if mats else 0
        padded = [pad(m, (0, 0, 0, max_len - m.shape[0]), value=0) for m in mats]
        return torch.stack(padded, dim=0)  # (B, Lmax, 4)

    @torch.no_grad()
    def pre_seqs(self, population, batch_size: int = 512):
        """
        population: [{'sequence': <str>, 'expression': None}, ...]
        返回：原地写回，每个元素的 'expression' 为预测值(float)
        """
        seqs = [d['sequence'] for d in population]
        preds = []
        device = "cuda" if self.use_gpu else "cpu"

        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            x = self.seq_onehot(batch).to(device)
            y = self.model(x).detach().cpu().numpy().flatten().tolist()
            preds.extend([float(v) for v in y])

        for i, d in enumerate(population):
            d['expression'] = preds[i] if i < len(preds) else 0.0
        return population
