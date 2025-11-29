"""
Baseline：基准参数结构模型
"""

# ==================================================================================
#                                   Import Model
# ==================================================================================
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchgen.api.types import layoutT
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold



# ==================================================================================
#                               Image Transforms
# ==================================================================================
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(180),
    transforms.RandomAffine(30),
    transforms.RandomGrayscale(0.2),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    transforms.RandomErasing(0.2)  # 随机擦除
])



# ==================================================================================
#                                   Dataset
# ==================================================================================
class FoodDataset(Dataset):

    def __init__(self, path=None, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        if path:
            self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        else:
            self.files = files
        self.transform = tfm
        print('Num of element: ', len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        # im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label



# ==================================================================================
#                               Model Structure
# ==================================================================================
class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            # 不激活，先残差连接，再激活。
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            # 对其特征，确保加法对齐
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class Classifier(nn.Module):
    def __init__(self, block, num_layers, num_classes=11):
        super(Classifier, self).__init__()
        self.preConv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer0 = self.makeResidualBlocks(block, 32, 64, num_layers[0], stride=2)
        self.layer1 = self.makeResidualBlocks(block, 64, 128, num_layers[1], stride=2)
        self.layer2 = self.makeResidualBlocks(block, 128, 256, num_layers[2], stride=2)
        self.layer3 = self.makeResidualBlocks(block, 256, 512, num_layers[3], stride=2)

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        out = self.preConv(x)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out.view(out.size(0), -1))
        return out

    def makeResidualBlocks(self, block, in_channels, out_channels, num_layer, stride=1):
        layers = [block(in_channels, out_channels, stride)]
        for i in range(1, num_layer):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
class MyCrossEntropy(nn.Module):
    def __init__(self, class_num):
        pass



# ==================================================================================
#                                   Config
# ==================================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
_exp_name = "Random_K-Fold"
myseed = 5201314  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

batch_size = 256
_dataset_dir = "../Data"
num_layers = [2, 3, 3, 1] # residual number layers
alpha = torch.Tensor([1, 2.3, 0.66, 1, 1.1, 0.75, 2.3, 3.5, 1.1, 0.66, 1.4]).view(-1,1)

k_fold = 4



# ==================================================================================
#                                       K-Fold
# ==================================================================================
# train_dir = "../Data/training"
# val_dir = "../Data/validation"
# train_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if x.endswith('.jpg')]
# val_files = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if x.endswith('.jpg')]
# total_files = train_files + val_files

# random.shuffle(total_files)
#
# num = len(total_files) // k_fold


# ==================================================================================
#                                   Training
# ==================================================================================
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"



# ==================================================================================
#                                   Testing
# ==================================================================================
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
import torch
import pandas as pd
from PIL import Image

# 定义温和的 TTA 增强
tta_transforms = [
    test_tfm,  # 原始图像
    transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomRotation(180),  # 小角度旋转
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.Resize((144,144)),   # 多尺度缩放
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])
]

# 构建测试集和 DataLoader
test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 加载每个 fold 的最佳模型
models = []
for fold in range(k_fold):
    model_best = Classifier(Residual_Block, num_layers, num_classes=11).to(device)
    model_best.load_state_dict(torch.load(f"Fold_{fold}_{_exp_name}_best.ckpt"))
    model_best.eval()
    models.append(model_best)

# 测试循环，带 TTA 和进度显示
prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader, desc="Testing with TTA"):
        test_preds = []
        for model_best in models:
            aug_preds = []
            for tfm in tta_transforms:
                # 对 batch 内每张图应用 tfm
                augmented = torch.stack([tfm(transforms.ToPILImage()(img.cpu())) for img in data])
                outputs = model_best(augmented.to(device))
                aug_preds.append(outputs.cpu().numpy())
            # 平均 TTA 结果
            aug_preds = np.mean(aug_preds, axis=0)
            test_preds.append(aug_preds)
        # 集成多个模型
        test_preds = np.sum(test_preds, axis=0)
        test_label = np.argmax(test_preds, axis=1)
        prediction += test_label.squeeze().tolist()

# 保存结果到 CSV
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
output_name = "submission_" + _exp_name + "_TTA.csv"
df.to_csv(output_name,index = False)
