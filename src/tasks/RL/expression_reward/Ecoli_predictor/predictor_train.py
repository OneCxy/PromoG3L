#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import time
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from scipy.stats import pearsonr, spearmanr
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from pytorchtools import EarlyStopping
from predictor_models import *  

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DATA_PATH = PROJECT_ROOT / 'datasets' / 'ecoli.csv'
SAVE_ROOT = PROJECT_ROOT / 'outputs' / 'Predictor_Ecoli'


# Sample-weighted MSE
class SampleWeightedMSE(nn.Module):
    """
    每个样本一个权重 w_i:
    loss = mean( w_i * (y_pred - y_true)^2 )
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight):
        return ((pred - target) ** 2 * weight).mean()


class PREDICT():
    def __init__(self, run_name='predictor_', dataset='ECOLI', model_name='LSTMModel',
                 data_path=DATA_PATH,
                 val_ratio=0.1):
        self.model_name = model_name
        self.patience = 20
        self.val_acc_list = []
        self.dataset = dataset
        self.data_path = data_path
        self.val_ratio = val_ratio

        self.seq1, self.exp = self.data_load(self.data_path)   
        self.exp = self.exp.astype(np.float32)                 

        y_flat = self.exp.reshape(-1)
        abs_y = np.abs(y_flat)

        sample_w = np.ones_like(abs_y, dtype=np.float32)


        sample_w[abs_y >= 6.0] = 6.0
        # 5 <= |y| < 6
        mask_5_6 = (abs_y >= 5.0) & (abs_y < 6.0)
        sample_w[mask_5_6] = 3.0
        # 4 <= |y| < 5
        mask_4_5 = (abs_y >= 4.0) & (abs_y < 5.0)
        sample_w[mask_4_5] = 1.0

        self.sample_weights = sample_w.reshape(-1, 1)  # (N,1)
        print("[WEIGHT] |y|<4: {:.3f}, 4-5: {:.3f}, 5-6: {:.3f}, >=6: {:.3f}".format(
            (abs_y < 4).mean(),
            mask_4_5.mean(),
            mask_5_6.mean(),
            (abs_y >= 6.0).mean(),
        ))

        self.seq = self.seq_onehot(self.seq1)

        self._confirm_channels(expected=4)

        self.input_size = self.seq.shape[-1]   
        self.batch_size = 16                   
        self.hidden_size = 256
        self.conv_hidden = 128
        self.seq_len = 165
        self.output_size = 1
        self.lambda_l2 = 0.001
        self.dropout_rate = 0.2

        torch.cuda.set_device(0)
        self.use_gpu = True if torch.cuda.is_available() else False

        self.build_model()
        self.checkpoint_dir = os.path.join(SAVE_ROOT, run_name + dataset)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def _confirm_channels(self, expected=4):
        shape = tuple(self.seq.shape)  # [N, L, C]
        c = shape[-1] if len(shape) == 3 else None
        print(f"[CONFIRM] one-hot tensor shape = {shape}, channels = {c}, expected = {expected}")
        if c is None:
            print("[CONFIRM][WARN] 输入张量维度不是 (N, L, C)，请检查 one-hot 过程。")
        elif expected is not None and c != expected:
            print(f"[CONFIRM][WARN] 通道数为 {c}，与期望的 {expected} 不一致（仅提示，不中断训练）。")

    def data_load(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as data:
            seq = []
            exp = []
            for item in data:
                item = item.strip().split(",")
                if len(item) < 2:
                    continue
                seq.append(item[0])
                exp.append(item[1])
        expression = np.zeros((len(exp), 1), dtype=np.float32)
        for i in range(len(exp)):
            expression[i] = float(exp[i])
        return seq, expression

    def string_to_array(self, my_string):
        my_string = my_string.lower()
        my_string = re.sub('[^acgt]', 'z', my_string)
        my_array = np.array(list(my_string))
        return my_array

    def one_hot_encode(self, my_array):
        # Channel order: A, C, G, T; unknown bases map to a discarded fifth slot.
        mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        idx = np.fromiter((mapping.get(ch, 4) for ch in my_array),
                          dtype=np.int64, count=len(my_array))
        oh5 = np.eye(5, dtype=np.float32)[idx]  
        return oh5[:, :4] 

    def seq_onehot(self, seq):
        onehot_seq = [torch.tensor(self.one_hot_encode(self.string_to_array(s)),
                                   dtype=torch.float32)
                      for s in seq]
        max_length = max(matrix.shape[0] for matrix in onehot_seq)
        padded_tensor_list = []
        for matrix in onehot_seq:
            padding_length = max_length - matrix.shape[0]
            padded_tensor = pad(matrix, (0, 0, 0, padding_length), value=0)
            padded_tensor_list.append(padded_tensor)
        onehot_seq = torch.stack(padded_tensor_list, dim=0)  # [N, L, C]
        return onehot_seq

    def build_model(self):
        if self.model_name == 'OnlyCNNModel':
            self.model = OnlyCNNModel(self.input_size, self.hidden_size, self.output_size,
                                      self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'GRUModel':
            self.model = GRUModel(self.input_size, self.hidden_size, self.output_size,
                                  self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'LSTMModel':
            self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size,
                                   self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'BiLSTMModel':
            self.model = BiLSTMModel(self.input_size, self.hidden_size, self.output_size,
                                     self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'TransformerModel':
            self.model = TransformerModel(self.input_size, self.hidden_size, self.output_size,
                                          nhead=8, num_layers=2, dropout_rate=self.dropout_rate)
        elif self.model_name == 'LSTM_Transformer_Model':
            self.model = LSTM_Transformer_Model(self.input_size, self.hidden_size, self.output_size,
                                                nhead=8, num_layers=2, dropout_rate=self.dropout_rate)
        elif self.model_name == 'LSTM_Transformer_large':
            self.model = LSTM_Transformer_large(self.input_size, self.hidden_size, self.output_size,
                                                nhead=8, num_layers=2, dropout_rate=self.dropout_rate)
        elif self.model_name == 'Densenet':
            self.model = Densenet(input_nc=self.input_size, growth_rate=32,
                                  block_config=(2, 2, 4, 2),
                                  num_init_features=64, bn_size=4,
                                  drop_rate=self.dropout_rate)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

        if self.use_gpu:
            self.model.cuda()

        device = 'cuda' if self.use_gpu else 'cpu'
        self.loss_fn = SampleWeightedMSE().to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1e-3,
                                          weight_decay=self.lambda_l2)

    def train(self):
        seq = self.seq.numpy()                # (N, L, 4)
        expression = self.exp                 # (N,1)
        weights = self.sample_weights         # (N,1)

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            seq, expression, weights,
            test_size=self.val_ratio, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        w_train = torch.tensor(w_train, dtype=torch.float32)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        w_val = torch.tensor(w_val, dtype=torch.float32)

        train_data = TensorDataset(X_train, y_train, w_train)
        val_data = TensorDataset(X_val, y_val, w_val)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        num_epochs = 100

        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=True,
            path=os.path.join(self.checkpoint_dir, f'{self.model_name}.pth'),
            stop_order='max'  
        )

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_feature, batch_label, batch_w in train_loader:
                if self.use_gpu:
                    batch_feature = batch_feature.cuda()
                    batch_label = batch_label.cuda()
                    batch_w = batch_w.cuda()

                self.optimizer.zero_grad()
                output = self.model(batch_feature)          # [B,1]
                loss = self.loss_fn(output, batch_label, batch_w)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_preds, val_labels = [], []
                for val_feature, val_label, val_w in val_loader:
                    if self.use_gpu:
                        val_feature = val_feature.cuda()
                        val_label = val_label.cuda()
                    val_output = self.model(val_feature)
                    # Validation uses unweighted MSE for comparability.
                    loss_val = nn.functional.mse_loss(val_output, val_label)
                    val_loss += loss_val.item()

                    val_preds.extend(val_output.detach().cpu().numpy())
                    val_labels.extend(val_label.detach().cpu().numpy())

                avg_val_loss = val_loss / len(val_loader)
                val_labels_arr = np.array(val_labels).flatten()
                val_preds_arr = np.array(val_preds).flatten()

                val_rho = spearmanr(val_labels_arr, val_preds_arr)[0]
                val_cor = pearsonr(val_labels_arr, val_preds_arr)[0]

            print(
                f"Epoch:{epoch} "
                f"train_loss:{avg_loss:.4f} "
                f"val_loss:{avg_val_loss:.4f} "
                f"spearman:{val_rho:.4f} "
                f"pearson:{val_cor:.4f}"
            )

            # Early stopping maximizes Pearson correlation.
            early_stopping(val_loss=val_cor, model=self.model)
            if early_stopping.early_stop:
                print('Early Stopping......')
                break

    def load_model(self):
        path = os.path.join(self.checkpoint_dir, self.model_name + '.pth')
        state = torch.load(path, map_location='cuda' if self.use_gpu else 'cpu')
        self.model.load_state_dict(state)
        print(f'Loaded model from {path}')


if __name__ == '__main__':
    time_start = time.time()
    predict = PREDICT(
        run_name='predictor_',
        dataset='E_clio_final',
        model_name='LSTMModel',
        data_path=DATA_PATH,
        val_ratio=0.1
    )
    predict.train()
    time_end = time.time()
    print("Total training time: {:.2f}s".format(time_end - time_start))
