# file: check/check.py
# python /app/check/check.py

from transformers import AutoTokenizer

tokenizer_jp = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

token_to_id = tokenizer_jp.get_vocab()
id_to_token = {id_: token for token, id_ in token_to_id.items()}

print("token_to_id:")
for i, (token, id_) in enumerate(token_to_id.items()):
    print(f"Type of token: {type(token)}, Type of id: {type(id_)}, token: {token}, id: {id_}")
    if i >= 10:
        break

print("\nid_to_token:")
for i, (id_, token) in enumerate(id_to_token.items()):
    print(f"Type of id: {type(id_)}, Type of token: {type(token)}, id: {id_}, token: {token}")
    if i >= 10:
        break
