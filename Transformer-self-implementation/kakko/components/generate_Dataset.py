# ファイル名：/app/Transformer-self-implementation/kakko/components/generate_Dataset.py

import os
import json
from torch.utils.data import Dataset, DataLoader
import glob
import random
from typing import List, Tuple

# 括弧の種類とキーワード
tokens = ["(", ")", "[", "]", "{", "}", "input", ",output" , "," ]

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

BRACKETS = [('(', ')'), ('[', ']'), ('{', '}')]

def ensure_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        print(f"ディレクトリ {directory} の作成に失敗しました。エラー: {e}")


def tokenize_string(string):
    string = string.replace(":", "").replace("\"", "")
    tokenized_string = []
    start = 0
    is_input_found = False
    for i in range(len(string)):
        if string[i] in tokens:
            if start < i:
                token = string[start:i]
                if token == "input":
                    is_input_found = True
                if is_input_found:
                    tokenized_string.append(token)
            tokenized_string.append(string[i])
            start = i+1

    result = []
    temp = []
    count = 0
    for token in tokenized_string:
        if "input" in token:
            count += 1
            if count > 1:
                result.append(temp)
                temp = []
        temp.append(token)
    result.append(temp)  # 最後のリストを追加

    return result

def preprocess_and_save_dataset(dataset, filepath):
    for directory in dirs.values():
        ensure_dir(directory)

    try:
        with open(os.path.join(dirs["original"], filepath), "w") as f:
            json.dump(dataset, f)
        print(f"{filepath} の保存に成功しました。")
    except Exception as e:
        print(f"{filepath} の保存に失敗しました。エラー: {e}")

    with open(os.path.join(dirs["original"], filepath), "w") as f:
        json.dump(dataset, f)

    dataset_string = json.dumps(dataset)
    tokenized_dataset = tokenize_string(dataset_string)
    with open(os.path.join(dirs["tokenize"], filepath), "w") as f:
        json.dump(tokenized_dataset, f)

    preprocessed_dataset = [[token2id[token] for token in token_list if token in token2id] for token_list in tokenized_dataset]
    with open(os.path.join(dirs["preprocessed"], filepath), "w") as f:
        json.dump(preprocessed_dataset, f)


def generate_bracket_sequence(depth: int) -> str:
    if depth == 0:
        return ""
    
    num_brackets = random.randint(1, 3)
    brackets = ""
    for _ in range(num_brackets):
        bracket = random.choice(BRACKETS)
        inner = generate_bracket_sequence(depth - 1) if random.random() < 0.5 else ""
        brackets += bracket[0] + inner + bracket[1]
    
    return brackets

def split_sequence(seq: str) -> Tuple[str, str]:
    if all(c in (')', ']', '}') for c in seq):
        return seq[:-1], seq[-1:]
    
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] in ('(', '[', '{'):
            return seq[:i+1], seq[i+1:]
    
    return seq, ""

def generate_brackets(n_samples: int, max_depth: int, min_len: int, max_len: int) -> List[str]:
    dataset = []
    for _ in range(n_samples):
        while True:
            depth = random.randint(1, max_depth)
            sequence = generate_bracket_sequence(depth)
            if min_len <= len(sequence) <= max_len:
                break
        input_seq, output_seq = split_sequence(sequence)
        dataset.append(f"input:{input_seq},output:{output_seq}")
    
    return dataset


num_samples = 10  # データセットのサンプル数
max_depth = 5  # 括弧の最大深さ
min_len = 5  # シーケンスの最小長
max_len = 10  # シーケンスの最大長

# データセットの生成
dataset = generate_brackets(num_samples, max_depth, min_len, max_len)

# データセットの前処理と保存
preprocess_and_save_dataset(dataset, "bracket_dataset.json")
print("データセットが保存された場所:", os.path.join(dirs["original"], "bracket_dataset.json"))
print("保存するデータセット:", dataset)
