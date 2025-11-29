import os
from typing import List

import pandas as pd

feat_idx = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
            38, 39, 40, 41, 43, 44, 53,                 # Day 1
            54, 55, 56, 57, 59, 60, 69,                 # Day 2
            70, 71, 72, 73, 75, 76, 85,                 # Day 3
            86, 87, 88, 89, 91, 92, 101,                # Day 4
            102, 103, 104, 105, 107, 108
        ]

def read_header_columns(csv_path: str) -> List[str]:
    """Read only header (column names) from a CSV file.
    Prefer pandas if available; fallback to csv module otherwise.
    """
    # Try pandas first
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(csv_path, nrows=0)
        return list(df.columns)
    except Exception:
        # Fallback: use csv module to read the first row (header)
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            return header


def select_columns_by_index(columns: List[str], indices: List[int]) -> List[str]:
    selected: List[str] = []
    for i in indices:
        if 0 <= i < len(columns):
            selected.append(columns[i])
        else:
            selected.append(f"<Index {i} out of range: 0..{len(columns)-1}>")
    return selected


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'Data')  # 根据仓库结构，CSV 位于 Data 目录

    train_path = os.path.join(data_dir, 'covid.train.csv')
    test_path = os.path.join(data_dir, 'covid.test.csv')

    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到文件: {p}")

    train_cols = read_header_columns(train_path)
    test_cols = read_header_columns(test_path)

    train_selected = select_columns_by_index(train_cols, feat_idx)
    test_selected = select_columns_by_index(test_cols, feat_idx)

    print('Train CSV 选择的列名:')
    print(train_selected)
    print('\nTest CSV 选择的列名:')
    print(test_selected)

    # 额外信息：简单核对两个 CSV 的表头是否一致
    if train_cols != test_cols:
        print('\n警告：train 与 test 的表头不一致，可能导致索引错位。')
        print(f'train 列数: {len(train_cols)}, test 列数: {len(test_cols)}')


if __name__ == '__main__':
    main()
