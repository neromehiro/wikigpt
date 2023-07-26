# file: training/train_model.py
#  python /app/training/train_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.utils import to_categorical
import numpy as np
import os
import string

def prepare_sequences(encoded_tokens, seq_length):
    # 入力シーケンスと目標シーケンスを準備
    input_sequences = []
    target_tokens = []
    for i in range(len(encoded_tokens) - seq_length):
        input_sequences.append(encoded_tokens[i:i+seq_length])
        target_tokens.append(encoded_tokens[i+seq_length])
    return np.array(input_sequences), np.array(target_tokens)

def define_model(seq_length, output_dim):
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_length, 1)))  
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_model(model, input_sequences, target_tokens, epochs, batch_size):
    # モデルを訓練
    model.fit(input_sequences, target_tokens, epochs=epochs, batch_size=batch_size)

# エンコードされたトークンのリストを準備
encoded_tokens = []

# ディレクトリとファイル名のリストを準備
folders = list(string.ascii_uppercase[:17])
folders[-1] = folders[-1] + '/wiki_48'
files = [f'wiki_{str(i).zfill(2)}' for i in range(100)]

# 全てのファイルを読み込み
for folder in folders:
    for file in files:
        file_path = f"encoded_extracted/{folder}/{file}"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                encoded_tokens += [int(line.strip()) for line in f]

seq_length = 100  # ここでシーケンス長を定義
model = define_model(seq_length, len(set(encoded_tokens)))

# 入力シーケンスと目標シーケンスを準備
input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)

# ターゲットトークンをone-hotエンコーディング
target_tokens = to_categorical(target_tokens)

# モデルを訓練
train_model(model, input_sequences, target_tokens, epochs=10, batch_size=32)

# モデルを保存
model.save('my_model.h5')