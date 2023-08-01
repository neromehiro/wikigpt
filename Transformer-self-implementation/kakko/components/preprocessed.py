# ファイル名：/app/Transformer-self-implementation/kakko/components/preprocessed.py
import os
import json
from torch.utils.data import Dataset, DataLoader
import glob

# 括弧の種類
tokens = ["(", ")", "[", "]", "{", "}"]

# トークンとIDを対応付ける辞書
token2id = {token: i for i, token in enumerate(tokens)}

# IDとトークンを対応付ける辞書
id2token = {i: token for token, i in token2id.items()}

# データの保存先ディレクトリ
dirs = {
    "original": "/app/Transformer-self-implementation/kakko/kakko_dataset/original",
    "tokenize": "/app/Transformer-self-implementation/kakko/kakko_dataset/tokenize",
    "preprocessed": "/app/Transformer-self-implementation/kakko/kakko_dataset/preprocessed",
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_and_save_dataset(dataset, filepath):
    # 保存先ディレクトリの存在を確認し、存在しなければ作成
    for directory in dirs.values():
        ensure_dir(directory)

    # 元のデータセットを保存
    with open(os.path.join(dirs["original"], filepath), "w") as f:
        json.dump(dataset, f)

    # トークン化したデータセットを保存
    tokenized_dataset = [[token for token in sequence] for sequence in dataset]
    with open(os.path.join(dirs["tokenize"], filepath), "w") as f:
        json.dump(tokenized_dataset, f)

    # 前処理したデータセットを保存
    preprocessed_dataset = [[token2id[token] for token in sequence] for sequence in tokenized_dataset]
    with open(os.path.join(dirs["preprocessed"], filepath), "w") as f:
        json.dump(preprocessed_dataset, f)

# データセットの作成と保存
dataset_dir = "/app/Transformer-self-implementation/kakko/kakko_dataset/original"
datasets = glob.glob(f"{dataset_dir}/*.json")
for dataset_path in datasets:
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    preprocess_and_save_dataset(dataset, os.path.basename(dataset_path))
