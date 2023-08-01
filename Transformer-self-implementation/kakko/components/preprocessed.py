# ファイル名：/app/Transformer-self-implementation/kakko/components/preprocessed.py
import os
import json
from torch.utils.data import Dataset, DataLoader
import glob
import random
from typing import List, Tuple

# 括弧の種類とキーワード
tokens = ["(", ")", "[", "]", "{", "}", "input", ",output" ,","]

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

# 開括弧と閉括弧の対応関係
brackets = {'(': ')', '[': ']', '{': '}'}

BRACKETS = [('(', ')'), ('[', ']'), ('{', '}')]

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def tokenize_string(string):
    # ":"やダブルクォーテーションを削除
    string = string.replace(":", "").replace("\"", "")
    tokenized_string = []
    start = 0
    for i in range(len(string)):
        if string[i] in tokens:
            if start < i:
                tokenized_string.append(string[start:i])
            tokenized_string.append(string[i])
            start = i+1

    # 'input'を含むトークンが2回目以降に出現したときに、その前のトークンでリストを分割
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
    # 保存先ディレクトリの存在を確認し、存在しなければ作成
    for directory in dirs.values():
        ensure_dir(directory)

    # 元のデータセットを保存
    with open(os.path.join(dirs["original"], filepath), "w") as f:
        json.dump(dataset, f)

    # トークン化したデータセットを保存
    dataset_string = json.dumps(dataset)
    tokenized_dataset = tokenize_string(dataset_string)
    with open(os.path.join(dirs["tokenize"], filepath), "w") as f:
        json.dump(tokenized_dataset, f)

    # 前処理したデータセットを保存
    preprocessed_dataset = [[token2id[token] for token in token_list if token in token2id] for token_list in tokenized_dataset]
    with open(os.path.join(dirs["preprocessed"], filepath), "w") as f:
        json.dump(preprocessed_dataset, f)


# 以下の関数を追加
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
        # シーケンスを"input:~~,output:~~"の形式の文字列に変換
        dataset.append(f"input:{input_seq},output:{output_seq}")
    
    return dataset


# "generate_dataset"の呼び出しを"generate_brackets"の呼び出しに置き換え
num_samples = 3  # データセットのサンプル数
max_depth = 5  # 括弧の最大深さ
min_len = 5  # シーケンスの最小長
max_len = 10  # シーケンスの最大長
# ファイル名：/app/Transformer-self-implementation/kakko/components/preprocessed.py

def tokenize_dataset(dataset: List[str]) -> List[List[str]]:
    """
    与えられたデータセット（"input:~~,output:~~"の形式の文字列のリスト）をトークン化します。
    具体的には、各文字列をカンマで分割し、それぞれの部分を個別にトークン化します。
    """
    tokenized_dataset = []
    for string in dataset:
        # 各文字列をトークン化
        tokenized_string = tokenize_string(string)
        # トークン化した文字列を追加
        tokenized_dataset.append(tokenized_string)
    return tokenized_dataset


# データセットの生成
dataset = generate_brackets(num_samples, max_depth, min_len, max_len)

# データセットのトークン化
tokenized_dataset = tokenize_dataset(dataset)

# データセットの前処理と保存
preprocess_and_save_dataset(dataset, "bracket_dataset.json")
