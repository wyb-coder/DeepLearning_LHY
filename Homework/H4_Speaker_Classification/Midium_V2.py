"""
  Task description
- Classify the speakers of given features.
- Main goal: Learn how to use transformer.
- Baselines:
  - Easy:   Run sample code and know how to use transformer.
  - Medium: Know how to adjust parameters of transformer.
  - Strong: Construct [conformer](https://arxiv.org/abs/2005.08100) which is a variety of transformer.
  - Boss:   Implement [Self-Attention Pooling](https://arxiv.org/pdf/2008.01077v1.pdf) &
			[Additive Margin Softmax](https://arxiv.org/pdf/1801.05599.pdf) to further boost the performance.
"""

# ==================================================================================
#                                   Import Model
# ==================================================================================
import numpy as np
import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.optim import AdamW
import csv
import torchaudio.transforms as T
from torch.utils.data import Subset

# ==================================================================================
#                                    Utils
# ==================================================================================
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==================================================================================
#                                    Config
# ==================================================================================
set_seed(42)
_exp_name = "Midium_V2"
cuda_number = "cuda:6"


def parse_args():
    """arguments"""
    config = {
        "data_dir": "./Dataset",
        # "save_path": "model.ckpt",
        "save_path": f"model_{_exp_name}.ckpt",
        "batch_size": 128,
        "n_workers": 32,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 120000,
    }

    return config


# ==================================================================================
#                                    DataSet
# ==================================================================================
"""
Randomly select 600 speakers from Vox-celeb2.

- Args:
  - data_dir: The path to the data directory.
  - metadata_path: The path to the metadata.
  - segment_len: The length of audio segment for training.

For efficiency, we segment the mel-spectrograms into segments in the training step
"""


class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128, mode="train"):
        self.data_dir = data_dir
        self.segment_len = segment_len
        self.mode = mode

        # Load the mapping from speaker neme to their corresponding id.
        # 读取Speaker ID
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        # 每个Speaker对应的所有语句
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]

        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

        # SpecAugment:数据增强，随机抹去一段频率或一段时间，
        # 强迫模型不依赖特定的频段/时间点，提升泛化能力
        if self.mode == "train":
            self.freq_masking = T.FrequencyMasking(freq_mask_param=15)
            self.time_masking = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # data 中存储路径，数据存放于磁盘，因此存在I/O瓶颈
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segmemt mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)

        if self.mode == "train":
            # SpecAugment 需要输入维度为 (channel, freq, time) 或 (freq, time)
            # 现在的 mel 是 (128, 40) 即 (time, freq)
            # 所以需要转置: (128, 40) -> (40, 128)
            mel = mel.transpose(0, 1)

            # 应用遮挡
            mel = self.freq_masking(mel)
            mel = self.time_masking(mel)

            # 转置回来: (40, 128) -> (128, 40)
            mel = mel.transpose(0, 1)


        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


# ==================================================================================
#                                    DataLoader
# ==================================================================================
"""
- Split dataset into training dataset(90%) and validation dataset(10%).
- Create dataloader to iterate the data.
"""


def collate_batch(batch):
    """Collate a batch of data.
       对齐样本，Padding
    """
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)  # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
    """Generate dataloader"""

    # 1. 先创建一个临时的 dataset 只为了获取长度和索引
    tmp_dataset = myDataset(data_dir, mode="train")  # 这里的 mode 无所谓
    speaker_num = tmp_dataset.get_speaker_number()

    # 2. 计算切分长度 (保持原逻辑)
    trainlen = int(0.9 * len(tmp_dataset))
    lengths = [trainlen, len(tmp_dataset) - trainlen]

    # 3. 获取切分的【索引】，而不是直接获取 Subset
    # random_split 返回的是 Subset，我们需要取出里面的 indices
    train_subset_tmp, valid_subset_tmp = random_split(tmp_dataset, lengths)
    train_indices = train_subset_tmp.indices
    valid_indices = valid_subset_tmp.indices

    # 4. === 关键修改 ===
    # 创建两个独立的 Dataset 实例，分别设置不同的 mode
    train_dataset_obj = myDataset(data_dir, mode="train")  # 开启增强
    valid_dataset_obj = myDataset(data_dir, mode="valid")  # 关闭增强 (干净数据)

    # 5. 使用之前生成的索引，重新包装成 Subset
    trainset = Subset(train_dataset_obj, train_indices)
    validset = Subset(valid_dataset_obj, valid_indices)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num


# ==================================================================================
#                                        Model
# ==================================================================================
"""
- TransformerEncoderLayer:
  - Base transformer encoder layer in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - Parameters:
    - d_model: the number of expected features of the input (required).
      - 也就是经过Self-Attention后的Full-NN的层数
    - nhead: the number of heads of the multiheadattention models (required).
    - dim_feedforward: the dimension of the feedforward network model (default=2048).
    - dropout: the dropout value (default=0.1).
    - activation: the activation function of intermediate layer, relu or gelu (default=relu).

- TransformerEncoder:
  - TransformerEncoder is a stack of N transformer encoder layers
  - Parameters:
    - encoder_layer: an instance of the TransformerEncoderLayer() class (required).
    - num_layers: the number of sub-encoder-layers in the encoder (required).
    - norm: the layer normalization component (optional).
"""


class Classifier(nn.Module):
    def __init__(self, d_model=256, n_spks=600, dropout=0.3):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        # (512, 128, 40) 每次读取512个样本，每个样本128帧，每帧40个特征
        self.prenet = nn.Sequential(
            nn.Conv1d(40, d_model, kernel_size=1),
            nn.ReLU()
        )
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        # Project the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),  # BN 加速收敛
            nn.ReLU(),
            nn.Dropout(dropout),  # 防止过拟合
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        # 1. 维度适配 Conv1d
        # 原始 mels: (Batch, Length, 40)
        # Conv1d 需要: (Batch, 40, Length)
        out = mels.permute(0, 2, 1)
        # out: 40 -> d_model
        out = self.prenet(out)
        # out: (length, batch size, d_model)	更换Tensor三元组顺序，以适配transformer
        out = out.permute(0, 2, 1)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder(out)
        # out: (batch size, length, d_model)
        # out = out.transpose(0, 1)
        # mean pooling	进入 NN 必须是二维的Vector，类似(512, 80)，因此，在每一个特征列上，对全部行对应的值取平均
        # mean == Max Pooling 牺牲全部时序信息
        # 而判断说话者完全不需要任何时序信息，张三在第1秒的音色是张三。张三在第10秒的音色还是张三。

        mean_stats = out.mean(dim=1)
        std_stats = out.std(dim=1)

        # 拼接: (batch size, d_model * 2)
        stats = torch.cat((mean_stats, std_stats), dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
        The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""
    # 一次Train流程

    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)

    loss = criterion(outs, labels)

    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


def valid(dataloader, model, criterion, device):
    """Validate on validation set."""
    # 一次Test Validate

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            accuracy=f"{running_accuracy / (i + 1):.2f}",
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)


# ==================================================================================
#                                   Training
# ==================================================================================
def main(
        data_dir,
        save_path,
        batch_size,
        n_workers,
        valid_steps,
        warmup_steps,
        total_steps,
        save_steps,
):
    """Main function."""
    device = torch.device(cuda_number if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!", flush=True)

    # === 新增计算逻辑 ===
    dataset_size = len(train_loader.dataset)  # 获取训练集总样本数
    steps_per_epoch = len(train_loader)  # DataLoader 已经帮你算好了 (dataset_size / batch_size)

    print(f"训练集总样本数: {dataset_size}")
    print(f"1 个 Epoch 需要走: {steps_per_epoch} steps")
    print(f"总共将会训练: {total_steps / steps_per_epoch:.2f} 个 Epochs")

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!", flush=True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Updata model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()


if __name__ == "__main__":
    main(**parse_args())


# ==================================================================================
#                                Testing OR Inference
# ==================================================================================
class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)
    return feat_paths, torch.stack(mels)


def parse_args():
    """arguments"""
    config = {
        "data_dir": "./Dataset",
        "model_path": f"./model_{_exp_name}.ckpt",
        "output_path": f"./output_{_exp_name}.csv",
    }
    return config


def main(
        data_dir,
        model_path,
        output_path,
):
    """Main function."""
    device = torch.device(cuda_number if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=inference_collate_batch,
    )
    print(f"[Info]: Finish loading data!", flush=True)

    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[Info]: Finish creating model!", flush=True)

    results = [["Id", "Category"]]

    # 开始推理循环
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            # mels shape: (1, original_length, 40)
            mel = mels[0]  # 取出这个样本: (original_length, 40)

            # === TTA (Test Time Augmentation) 逻辑开始 ===
            segment_len = 128
            hop_len = 64  # 步长，意味着 50% 重叠

            segments = []
            length = mel.shape[0]

            # 1. 处理过短的音频 (Padding)
            if length <= segment_len:
                pad_len = segment_len - length
                # 使用 -20 填充，保持和训练时 collate_batch 一致的逻辑
                padding = torch.ones((pad_len, 40)) * -20
                segment = torch.cat([mel, padding], dim=0)
                segments.append(segment)

            # 2. 处理长音频 (Sliding Window)
            else:
                # 滑动窗口切片
                for start in range(0, length - segment_len + 1, hop_len):
                    end = start + segment_len
                    segments.append(mel[start:end])

                # 3. 补救措施：确保最后一段音频也被利用
                # 如果滑窗没有正好覆盖到结尾，我们强制取最后 128 帧
                if (length - segment_len) % hop_len != 0:
                    segments.append(mel[-segment_len:])

            # 4. 构造 TTA Batch
            # 将列表堆叠成 Tensor: (Num_Segments, 128, 40)
            # 这相当于把“一段”音频变成了“N个”样本的一个 Batch
            batch_mels = torch.stack(segments).to(device)

            # 5. 并行预测
            outs = model(batch_mels)  # 输出 shape: (Num_Segments, n_spks)

            # 6. 聚合结果 (取平均)
            # 对 N 个片段的预测结果取平均，消除噪音片段的影响
            final_out = outs.mean(dim=0)  # shape: (n_spks)

            # 7. 获取最终预测
            pred = final_out.argmax().cpu().item()
            # === TTA 逻辑结束 ===

            # feat_paths 是个 tuple，取第一个元素
            results.append([feat_paths[0], mapping["id2speaker"][str(pred)]])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if __name__ == "__main__":
    main(**parse_args())
