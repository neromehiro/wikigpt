# file: generate/generate_original_text.py
# python /app/generate/generate_original_text.py

from keras.models import load_model
import numpy as np
import pickle
import MeCab
import os
import random

def generate_output_filename(directory):
    # 指定されたディレクトリ内のテキストファイルの数をカウント
    num_files = sum(1 for file in os.listdir(directory) if file.endswith('.txt'))

    # 新しく生成されるテキストファイルの名前を生成（連番をつける）
    output_filename = os.path.join(directory, f"output_txt{num_files + 1}.txt")

    return output_filename

def generate_text(model, id_to_token, token_to_id, seed_text, num_tokens_to_generate, output_directory, temperature=1.0, p=0.9):
    # MeCabでテキストをトークン化
    mecab = MeCab.Tagger("-Owakati")
    seed_tokens = [token_to_id[token] for token in mecab.parse(seed_text).split() if token in token_to_id]
    # トークンが100になるまでパディングを追加
    while len(seed_tokens) < 100:
        seed_tokens.append(0)  # ここでは0をパディングとして使用します

    generated_tokens = list(seed_tokens)
    unknown_index = token_to_id.get('<unknown>', None)
    for _ in range(num_tokens_to_generate):
        input_sequence = np.array(generated_tokens[-100:]).reshape(1, 100, 1)
        predictions = model.predict(input_sequence, verbose=0)[0]
        predictions = np.exp(np.log(predictions) / temperature)
        predictions = predictions / np.sum(predictions)
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_predictions = predictions[sorted_indices]
        cumulative_probs = np.cumsum(sorted_predictions)

        idx = np.searchsorted(cumulative_probs, p)
        candidate_indices = sorted_indices[:idx+1]
        # トレーニングデータに存在するトークンのみを候補とする
        candidate_indices = [idx for idx in candidate_indices if idx in id_to_token]
        # '<unknown>' に対応するインデックスを候補から除外
        if unknown_index is not None and unknown_index in candidate_indices:
            candidate_indices.remove(unknown_index)
        # 候補が空になった場合（すべての候補がトレーニングデータに存在しないトークンだった場合）は、トレーニングデータに存在する最も確率の高いトークンを選ぶ
        if len(candidate_indices) == 0:
            candidate_indices = [idx for idx in sorted_indices if idx in id_to_token and idx != unknown_index]

        next_token = np.random.choice(candidate_indices)
        generated_tokens.append(next_token)

    # 生成されたテキストにseed_textを追加
    generated_text = seed_text + ' ' + ' '.join(id_to_token.get(token_id, '<unknown>') for token_id in generated_tokens)

    output_filename = generate_output_filename(output_directory)
    
    with open(output_filename, 'w') as file:
        file.write(generated_text)

    return generated_text

# 使用例
output_directory = "/app/output"
model = load_model('my_model.h5')
with open('id_to_token.pkl', 'rb') as file:
    id_to_token = pickle.load(file)
with open('token_to_id.pkl', 'rb') as file:
    token_to_id = pickle.load(file)

# 使用例
seed_text = "星"
generated_text = generate_text(model, id_to_token, token_to_id, seed_text, num_tokens_to_generate=1000, output_directory=output_directory)
# print(generated_text)
