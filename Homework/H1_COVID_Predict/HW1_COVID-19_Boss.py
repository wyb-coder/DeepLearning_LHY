# ==================================================================================
#                                   Import Model
# ==================================================================================

# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

import sklearn
from sklearn.feature_selection import SelectKBest, f_regression


# ==================================================================================
#                               Some Utility Functions
# ==================================================================================

def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training set and validation set.利用 seed 划分训练集、验证集"""
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    predicts = []
    # tqdm:为任意可迭代对象提供高性能进度条，显示已处理步数、速率、已用/剩余时间等，对性能影响很小
    for x in tqdm(test_loader):
        x = x.to(device)
        # 关闭梯度以用于预测
        with torch.no_grad():
            pred = model(x)
            predicts.append(pred.detach().cpu())
    predicts = torch.cat(predicts, dim=0).numpy()
    return predicts



# ==================================================================================
#                                   Dataset
# ==================================================================================

class COVID19Dataset(Dataset):
    """
    x: Features.
    y: Targets, if none, do prediction.
    """
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)



# ==================================================================================
#                              Neural Network Model
# ==================================================================================

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        # 采用Dropout
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.05),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.05),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.05),
            nn.Linear(8, 4),
            nn.LeakyReLU(0.05),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x



# ==================================================================================
#                               Feature Selection
# ==================================================================================

def select_feat(train_data, valid_data, test_data, select_all=True):
    """Selects useful features to perform regression"""
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        """
        feat_idx = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
            38, 39, 40, 41, 53,                 # Day 1
            54, 55, 56, 57, 69,                 # Day 2
            70, 71, 72, 73, 85,                 # Day 3
            86, 87, 88, 89, 101,                # Day 4
            102, 103, 104, 105
        ]
        """

        k = 24
        # 用 f_regression 计算每个特征和 y 的相关性
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(raw_x_train, y_train)

        # 得到分数最高的前 k 个特征索引（升序排列）
        # np.argsort(selector.scores_)：返回分数从小到大的特征索引
        idx = np.argsort(selector.scores_)[::-1]
        # np.sort非必要
        feat_idx = list(np.sort(idx[:k]))


    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid



# ==================================================================================
#                               Training Loop
# ==================================================================================

def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=0.000001)
    # 本文作者： Gality @藏器于身
    # 本文链接： https://gality.cn/ml/homework/HW-1/
    # 版权声明： 本站所有文章除特别声明外，均采用 (CC)BY-NC-SA 许可协议。转载请注明出处！

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'] * 75, weight_decay=1e-3)

    writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return



# ==================================================================================
#                              Configurations
# ==================================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.
    'batch_size': 128,
    'learning_rate': 1e-5,
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt',  # Your model will be saved here.
    'weight_decay': 1e-5
}



# ==================================================================================
#                                   Dataloader
# ==================================================================================

# Set seed for reproducibility
same_seed(config['seed'])

# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('./Data/covid.train.csv').values, pd.read_csv('./Data/covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)





model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)



# ==================================================================================
#                                   Predict
# ==================================================================================

def save_pred(preds, file):
    """ Save predictions to specified file """
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
predicts = predict(test_loader, model, device)
save_pred(predicts, 'pred.csv')