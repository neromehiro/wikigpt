# ファイル名：/app/Transformer-self-implementation/kakko/components/train_transformer.py

from tensorflow.keras import layers, models
import numpy as np
import os
import time
import glob
import json
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

# 元のトークンとIDの対応付け
tokens = ["(", ")", "[", "]", "{", "}", "input", ",output" ,","]
token2id = {token: i for i, token in enumerate(tokens)}

# データセットの保存先ディレクトリ
encode_dir_path = "/app/Transformer-self-implementation/kakko/kakko_dataset/preprocessed/"

# モデル保存先ディレクトリ
model_save_path = "/app/Transformer-self-implementation/kakko/models/"

TRAINING_MODES = {
    "1min": {"epochs": 1, "batch_size": 128, "num_files": 5, "learning_rate": 0.01},
    "10min": {"epochs": 3, "batch_size": 256, "num_files": 10, "learning_rate": 0.01},
    "1hour": {"epochs": 7, "batch_size": 512, "num_files": 50, "learning_rate": 0.001},
    "6hours": {"epochs": 20, "batch_size": 1024, "num_files": 300, "learning_rate": 0.001},
}

def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def select_mode():
    mode = input("Select a mode from: " + ", ".join(TRAINING_MODES.keys()) + "\n")
    while mode not in TRAINING_MODES:
        print(f"Invalid mode. Please select a mode from: {', '.join(TRAINING_MODES.keys())}")
        mode = input()
    return TRAINING_MODES[mode]["epochs"], TRAINING_MODES[mode]["batch_size"], TRAINING_MODES[mode]["num_files"], TRAINING_MODES[mode]["learning_rate"]

def save_model(model, model_path):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    model.save(model_path)

def prepare_sequences(encoded_tokens, seq_length):
    input_sequences = []
    target_tokens = []
    for i in range(len(encoded_tokens) - seq_length):
        input_sequences.append(encoded_tokens[i:i+seq_length])
        target_tokens.append(encoded_tokens[i+seq_length])
    return np.array(input_sequences), np.array(target_tokens)

def define_model(seq_length, output_dim, learning_rate):
    inputs = layers.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=output_dim, output_dim=64, mask_zero=True)(inputs)
    x = layers.GRU(64)(x)  # ここをTransformerに変更する必要があります
    outputs = layers.Dense(output_dim, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    return model

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def train_model(model, input_sequences, target_tokens, epochs, batch_size, model_path):
    if len(input_sequences) > 0 and len(target_tokens) > 0:
        print(f"Shapes: {input_sequences.shape}, {target_tokens.shape}")
        # Create a tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_tokens))
        # Shuffle and batch the dataset
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        # Initialize the TimeHistory callback
        time_callback = TimeHistory()
        # Create the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(model_path, save_best_only=False)
        # Record the start time
        start_time = time.time()
        # Split the data into batches and train the model
        model.fit(dataset, epochs=epochs, callbacks=[time_callback, checkpoint_callback])
        # Record the end time
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Training finished for: {model_path}. Time taken: {elapsed_time} seconds.")
        # Calculate and print the average time per epoch
        average_epoch_time = sum(time_callback.times) / len(time_callback.times)
        print(f"Average time per epoch: {average_epoch_time} seconds.")
    else:
        print(f"No data for training in: {model_path}")



def main():
    epochs, batch_size, num_files, learning_rate = select_mode()
    seq_length = 1

    # 全てのトークンをvocab_setに追加
    vocab_set = set(tokens)

    all_input_sequences = []
    all_target_tokens = []

    # For all files in the directory
    for dirpath, dirnames, filenames in os.walk(encode_dir_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            encoded_tokens_list = load_dataset(file_path)
            for encoded_tokens in encoded_tokens_list:
                # Prepare sequences for each file
                if len(encoded_tokens) > seq_length:
                    input_sequences, target_tokens = prepare_sequences(encoded_tokens, seq_length=seq_length)
                    all_input_sequences.extend(input_sequences)
                    all_target_tokens.extend(target_tokens)
                else:
                    print(f"Not enough data in: {file_path}")

    vocab_size = len(vocab_set)
    model = define_model(seq_length, vocab_size + 1, learning_rate)  # Define the model once

    # Convert lists to numpy arrays
    all_input_sequences = np.array(all_input_sequences)
    all_target_tokens = np.array(all_target_tokens)

    # Train the model
    for epoch in range(epochs):
        model_path = f"{model_save_path}mymodel{epoch}.h5" 
        train_model(model, all_input_sequences, all_target_tokens, epochs=1, batch_size=batch_size, model_path=model_path)

    print("Training finished.")

if __name__ == "__main__":
    main()
