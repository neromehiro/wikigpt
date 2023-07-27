# file: generate/generate_original_text.py
# python /app/generate/generate_original_text.py

from keras.models import load_model
import numpy as np
import os
from transformers import AutoTokenizer

tokenizer_jp = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
token_to_id = tokenizer_jp.get_vocab()
id_to_token = {id_: token for token, id_ in token_to_id.items()}
# トレーニングデータに存在しないトークンの確率を0に設定
trained_tokens = set(id_to_token.keys()) 

def generate_output_filename(directory):
    # 指定されたディレクトリ内のテキストファイルの数をカウント
    num_files = sum(1 for file in os.listdir(directory) if file.endswith('.txt'))

    # 新しく生成されるテキストファイルの名前を生成（連番をつける）
    output_filename = os.path.join(directory, f"output_txt{num_files + 1}.txt")

    return output_filename

def generate_text(model, id_to_token, token_to_id, seed_text, num_tokens_to_generate, output_directory, temperature=1.0, p=0.9):
    # BERTトークナイザでテキストをトークン化
    seed_tokens = tokenizer_jp.encode(seed_text, return_tensors='pt').tolist()[0]
    # トークンが100になるまでパディングを追加
    while len(seed_tokens) < 100:
        seed_tokens.append(tokenizer_jp.pad_token_id)

    generated_tokens = list(seed_tokens)
    for _ in range(num_tokens_to_generate):
        input_sequence = np.array(generated_tokens[-100:]).reshape(1, 100, 1)
        predictions = model.predict(input_sequence, verbose=0)[0]
        predictions = np.exp(np.log(predictions) / temperature)
        predictions = predictions / np.sum(predictions)

        next_token = np.random.choice(len(predictions), p=predictions)
        generated_tokens.append(next_token)
    # 生成されたテキストにseed_textを追加
    generated_text = seed_text + ' ' + ' '.join(id_to_token.get(token_id, '<unknown>') for token_id in generated_tokens if id_to_token[token_id] not in ['[CLS]', '[SEP]', '[PAD]'])

    output_filename = generate_output_filename(output_directory)
    
    with open(output_filename, 'w') as file:
        file.write(generated_text)

    return generated_text

# 使用例
output_directory = "/app/output"
model = load_model('/app/models/mymodel4.h5')

# 使用例
seed_text = "星"
generated_text = generate_text(model, id_to_token, token_to_id, seed_text, num_tokens_to_generate=1000, output_directory=output_directory)
# print(generated_text)
