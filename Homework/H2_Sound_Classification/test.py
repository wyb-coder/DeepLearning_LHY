import os, torch
from collections import Counter

phone_path = './Data/libriphone'
feat_dir = './Data/libriphone/feat/test'

# 读取 usage_list 原始与去重
lines = open(os.path.join(phone_path, 'test_split.txt')).read().splitlines()
raw = [l.strip() for l in lines if l.strip()!='']
print('raw lines:', len(raw))
print('unique lines:', len(set(raw)))
dups = [k for k,v in Counter(raw).items() if v>1]
print('duplicates count:', len(dups))
if dups:
    print('example duplicate:', dups[:5])

# 汇总 .pt 帧数
total_frames = 0
missing = []
for fname in raw:
    path = os.path.join(feat_dir, f'{fname}.pt')
    if not os.path.exists(path):
        missing.append(path)
        continue
    feat = torch.load(path)
    total_frames += len(feat)
print('total frames from test_split files:', total_frames)
print('missing files:', len(missing))
if missing:
    print('example missing:', missing[:5])
