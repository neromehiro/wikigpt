# python /app/training/train_5minits.py
from tensorflow.keras import layers, models
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


def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def define_model(seq_length, output_dim):
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=64, mask_zero=True)(inputs)

    x = transformer_block(x, head_size=128, num_heads=4, ff_dim=4*128)
    x = transformer_block(x, head_size=128, num_heads=4, ff_dim=4*128)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(output_dim, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )

    return model


print("Training started.")

def train_model(model, input_sequences, target_tokens, epochs, batch_size):
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")
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

# For 5 minutes training
# Select only one file for training
# For 5 minutes training
# Select five files for training
file_paths = [
    "encoded_extracted/AQ/wiki_42",
    "encoded_extracted/AQ/wiki_43",
    "encoded_extracted/AQ/wiki_44",
    "encoded_extracted/AQ/wiki_45",
    "encoded_extracted/AQ/wiki_46",
]
for file_path in file_paths:
    with open(file_path, "r") as f:
        encoded_tokens = [int(line.strip()) for line in f]
    if len(encoded_tokens) > seq_length:
        input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
        if len(input_sequences) > 0 and len(target_tokens) > 0:
            target_tokens = to_categorical(target_tokens, num_classes=vocab_size + 1)
            model = define_model(seq_length, vocab_size + 1)
            train_model(model, input_sequences, target_tokens, epochs=3, batch_size=64)
        else:
            print(f"Input sequences or target tokens not properly prepared for: {file_path}")
    else:
        print(f"Not enough data in: {file_path}")

# モデルを保存
model.save('my_model.h5')

print("Training finished.")
