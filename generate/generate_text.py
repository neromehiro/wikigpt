# file: generate/generate_text.py
from keras.models import load_model
import numpy as np
import pickle
import MeCab
import os

def generate_output_filename(directory):
    # 指定されたディレクトリ内のテキストファイルの数をカウント
    num_files = sum(1 for file in os.listdir(directory) if file.endswith('.txt'))

    # 新しく生成されるテキストファイルの名前を生成（連番をつける）
    output_filename = os.path.join(directory, f"output_txt{num_files + 1}.txt")

    return output_filename

def generate_text(model, id_to_token, seed_tokens, num_tokens_to_generate, output_directory):
    # 生成するトークンのリストを初期化
    generated_tokens = list(seed_tokens)

    # 指定された数のトークンを生成
    for _ in range(num_tokens_to_generate):
        # 直近のトークンを入力シーケンスとして使用
        input_sequence = np.array(generated_tokens[-100:]).reshape(1, 100, 1)
        # 次のトークンを予測
        predicted_token_id = np.argmax(model.predict(input_sequence, verbose=0), axis=-1)
        # 予測されたトークンをリストに追加
        generated_tokens.append(predicted_token_id[0])

    # トークンIDを実際のトークンに変換
    generated_text = ' '.join(id_to_token.get(token_id, '<unknown>') for token_id in generated_tokens)
    
    # ファイル名を生成
    output_filename = generate_output_filename(output_directory)
    
    # テキストをファイルに書き込み
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
mecab = MeCab.Tagger("-Owakati")
text = "これはテストです"
tokens = mecab.parse(text).split()
unknown_token_id = max(token_to_id.values()) + 1  # 未知のトークンIDを設定
seed_tokens = [token_to_id.get(token, unknown_token_id) for token in tokens]  
# トークンが100になるまでパディングを追加
while len(seed_tokens) < 100:
    seed_tokens.append(unknown_token_id)
generated_text = generate_text(model, id_to_token, seed_tokens, num_tokens_to_generate=1000, output_directory=output_directory)
print(generated_text)