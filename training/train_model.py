# file: training/train_model.py
# python /app/training/train_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.utils import to_categorical
import numpy as np
import os
import string

def prepare_sequences(encoded_tokens, seq_length):
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

print("Training started.")

def train_model(model, input_sequences, target_tokens, epochs, batch_size):
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        model.fit(input_sequences, target_tokens, epochs=epochs, batch_size=batch_size)
        print(f"Training finished for: {file_path}")
    else:
        print(f"No data for training in: {file_path}")


    
# ディレクトリとファイル名のリストを準備
folders = list(string.ascii_uppercase[:17])
folders[-1] = folders[-1] + '/wiki_48'
files = [f'wiki_{str(i).zfill(2)}' for i in range(100)]

seq_length = 100  # シーケンス長を定義
vocab_size = 0  # ボキャブラリーのサイズを初期化

# 全てのファイルを読み込み、vocab_sizeを計算
vocab_size = 0
for dirpath, dirnames, filenames in os.walk("encoded_extracted"):
    for file in filenames:
        file_path = os.path.join(dirpath, file)
        with open(file_path, "r") as f:
            encoded_tokens = [int(line.strip()) for line in f]
            vocab_size = max(vocab_size, max(encoded_tokens))  # ボキャブラリーのサイズを更新

model = define_model(seq_length, vocab_size + 1)

print("Checking file paths.")

# encoded_extracted ディレクトリおよびそのサブディレクトリ内のすべてのファイルを訓練に使用
for dirpath, dirnames, filenames in os.walk("encoded_extracted"):
    for file in filenames:
        file_path = os.path.join(dirpath, file)
        print(f"Processing file: {file_path}")
        with open(file_path, "r") as f:
            encoded_tokens = [int(line.strip()) for line in f]
            if len(encoded_tokens) > 0:
                input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                if len(input_sequences) > 0 and len(target_tokens) > 0:
                    target_tokens = to_categorical(target_tokens, num_classes=vocab_size + 1)
                    train_model(model, input_sequences, target_tokens, epochs=100, batch_size=32)
                else:
                    print(f"Input sequences or target tokens not properly prepared for: {file_path}")
            else:
                print(f"No data in: {file_path}")

print("Finished checking file paths.")

# モデルを保存
model.save('my_model.h5')

print("Training finished.")