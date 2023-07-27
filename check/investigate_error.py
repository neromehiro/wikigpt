# /app/check/investigate_error.py

import pickle
from transformers import BertJapaneseTokenizer

tokenizer_jp = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

def check_tokens(tokenizer, id_to_token):
    for id, token in id_to_token.items():
        if token == '大統領':
            print(f'Found "大統領" with id: {id}')
            print(f'Decoded token: {tokenizer.decode([id])}')

if __name__ == '__main__':
    with open('id_to_token.pkl', 'rb') as f:
        id_to_token = pickle.load(f)
        
    check_tokens(tokenizer_jp, id_to_token)
