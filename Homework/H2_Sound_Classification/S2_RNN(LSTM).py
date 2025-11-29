# ==================================================================================
#                                   Import Model
# ==================================================================================

import os
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import gc
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# ==================================================================================
#                               Utility Functions
# ==================================================================================

def load_feat(path):
    """加载单个.pt，转换为张量"""
    feat = torch.load(path)   # [seq_len, 39]
    return feat

def preprocess_data(split, feat_dir, phone_path, train_ratio=0.8, train_val_seed=1337):
    """数据预处理，返回 utterance 列表"""
    class_num = 41
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()
        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError("Invalid split")

    usage_list = [line.strip('\n') for line in usage_list]
    print(f"[Dataset] - # phone classes: {class_num}, number of utterances for {split}: {len(usage_list)}")

    data = []
    for fname in tqdm(usage_list):
        feat = load_feat(os.path.join(feat_dir, mode, f"{fname}.pt"))  # [seq_len, 39]
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])                # [seq_len]
            data.append((feat, label))
        else:
            data.append(feat)

    return data

# ==================================================================================
#                                   Dataset
# ==================================================================================

class LibriSeqDataset(Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode != 'test':
            return self.data[idx][0], self.data[idx][1]   # (feat_seq, label_seq)
        else:
            return self.data[idx]                         # feat_seq only

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    if isinstance(batch[0], tuple):
        feats, labels = zip(*batch)
        feats = pad_sequence(feats, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return feats, labels
    else:
        lengths = [len(f) for f in batch]
        feats = pad_sequence(batch, batch_first=True)
        return feats, lengths



# ==================================================================================
#                                      Model
# ==================================================================================

class RNNClassifier(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=512, num_layers=4, output_dim=41, bidirectional=True, dropout=0.4):
        super(RNNClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.15
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # 用 LayerNorm 替代 BatchNorm，更适合序列
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)        # [B, T, H*dir]
        out = self.fc(out)           # 逐帧映射，不需要 reshape
        return out                   # [B, T, C]



# ==================================================================================
#                                      Training
# ==================================================================================

# parameters
train_ratio = 0.95
seed = 5201314
batch_size = 64
num_epoch = 200
learning_rate = 0.0001
model_path = './model_rnn.ckpt'

# fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(seed)

# load data
train_data = preprocess_data('train', './Data/libriphone/feat', './Data/libriphone', train_ratio=train_ratio)
val_data   = preprocess_data('val',   './Data/libriphone/feat', './Data/libriphone', train_ratio=train_ratio)
test_data  = preprocess_data('test',  './Data/libriphone/feat', './Data/libriphone')
print("Number of utterances:", len(test_data))
print("First utterance shape:", test_data[0].shape)


train_set = LibriSeqDataset(train_data, mode='train')
val_set   = LibriSeqDataset(val_data,   mode='val')
test_set  = LibriSeqDataset(test_data,  mode='test')

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE: {device}")

# model
model = RNNClassifier().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=8,T_mult=2,eta_min = learning_rate/2)

# training loop
best_acc = 0.0
for epoch in range(num_epoch):
    model.train()
    train_loss, train_acc, total = 0.0, 0.0, 0
    for feats, labels in tqdm(train_loader):
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(feats)   # [batch, seq_len, output_dim]
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, -1)
        train_acc += (preds == labels).masked_select(labels != -100).sum().item()
        total += (labels != -100).sum().item()
    print(f"[{epoch+1}/{num_epoch}] Train Acc: {train_acc/total:.6f} Loss: {train_loss/len(train_loader):.6f}")

    # validation
    model.eval()
    val_loss, val_acc, total = 0.0, 0.0, 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            val_loss += loss.item()
            _, preds = torch.max(outputs, -1)
            val_acc += (preds == labels).masked_select(labels != -100).sum().item()
            total += (labels != -100).sum().item()
    print(f"Val Acc: {val_acc/total:.6f} Loss: {val_loss/len(val_loader):.6f}")

    if val_acc/total > best_acc:
        best_acc = val_acc/total
        torch.save(model.state_dict(), model_path)
        print(f"Saving model with acc {best_acc:.3f}")

# test
model.load_state_dict(torch.load(model_path))
model.eval()
pred = []
with torch.no_grad():
    for feats, lengths in tqdm(test_loader):
        feats = feats.to(device)
        outputs = model(feats)              # [B, T, C]
        _, preds = torch.max(outputs, -1)   # [B, T]
        for i, p in enumerate(preds.cpu().numpy()):
            pred.extend(p[:lengths[i]])     # 用真实长度截断


print("len(pred):", len(pred))
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write(f"{i},{y}\n")
