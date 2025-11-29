# ==================================================================================
#                               DataLoader And DataSet
# ==================================================================================
import numpy as np

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))

VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

import torch
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

BATCH_SIZE = 64

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)



# ==================================================================================
#                                   Clean
# ==================================================================================
import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()



# ==================================================================================
#                                   Model
# ==================================================================================
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 128),
            nn.Sigmoid(),
            nn.Linear(128, 39)
        )

    def forward(self, x):
        return self.layers(x)




# ==================================================================================
#                                   Utils
# ==================================================================================
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================================================================================
#                                   Configs
# ==================================================================================
# fix random seed for reproducibility
same_seeds(0)

# get device
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 20               # number of training epoch
learning_rate = 0.0001       # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# ==================================================================================
#                                   Training
# ==================================================================================
# start training
#
# best_acc = 0.0
# for epoch in range(num_epoch):
#     train_acc = 0.0
#     train_loss = 0.0
#     val_acc = 0.0
#     val_loss = 0.0
#
#     # training
#     model.train()  # set the model to training mode
#     for i, data in enumerate(train_loader):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         batch_loss = criterion(outputs, labels)
#         _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
#         batch_loss.backward()
#         optimizer.step()
#
#         train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
#         train_loss += batch_loss.item()
#
#     # validation
#     if len(val_set) > 0:
#         model.eval()  # set the model to evaluation mode
#         with torch.no_grad():
#             for i, data in enumerate(val_loader):
#                 inputs, labels = data
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 batch_loss = criterion(outputs, labels)
#                 _, val_pred = torch.max(outputs, 1)
#
#                 val_acc += (
#                             val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
#                 val_loss += batch_loss.item()
#
#             print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
#                 epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
#                 val_acc / len(val_set), val_loss / len(val_loader)
#             ))
#
#             # if the model improves, save a checkpoint at this epoch
#             if val_acc > best_acc:
#                 best_acc = val_acc
#                 torch.save(model.state_dict(), model_path)
#                 print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
#     else:
#         print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
#             epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
#         ))
#
# # if not validating, save the last epoch
# if len(val_set) == 0:
#     torch.save(model.state_dict(), model_path)
#     print('saving model at last epoch')



# ==================================================================================
#                                   Testing
# ==================================================================================
# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

# save predictions to CSV (existing behavior)
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))

# --- 新增：输出并记录预测信息 ---
from collections import Counter

num_pred = len(predict)
num_test = len(test_set) if 'test_set' in globals() else None

print('Prediction finished.')
print('Total predictions: {}'.format(num_pred))
if num_test is not None:
    print('Test set size   : {}'.format(num_test))
    if num_pred != num_test:
        print('Warning: prediction count and test set size differ!')

# 类别分布
class_counts = Counter(predict)
print('Class distribution (label:count):')
for label, cnt in sorted(class_counts.items()):
    print(f'  {label}: {cnt}')

# 打印前 20 个预测样本作为示例
print('First 20 predictions:', predict[:20])

# 将信息写入文件，便于后续查看
with open('prediction_info.txt', 'w') as info_f:
    info_f.write('Prediction summary\n')
    info_f.write('==================\n')
    info_f.write(f'Total predictions: {num_pred}\n')
    if num_test is not None:
        info_f.write(f'Test set size   : {num_test}\n')
        if num_pred != num_test:
            info_f.write('Warning: prediction count and test set size differ!\n')
    info_f.write('\nClass distribution (label:count):\n')
    for label, cnt in sorted(class_counts.items()):
        info_f.write(f'{label}: {cnt}\n')
    info_f.write('\nFirst 20 predictions:\n')
    info_f.write(','.join(map(str, predict[:20])) + '\n')

print('\nPrediction files written: prediction.csv, prediction_info.txt')
