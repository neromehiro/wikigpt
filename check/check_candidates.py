# ファイル名: /app/check/check_candidates.py

import os
import torch
from transformers import BertTokenizer

def check_candidates(seed_text):
    tokenizer_jp = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model_path = os.path.join("models", "model.pth")

    # Load id_to_token and token_to_id dictionaries
    model = torch.load(model_path)
    id_to_token = model['id_to_token']
    token_to_id = model['token_to_id']

    # Convert seed text to token ids
    tokenized_text = tokenizer_jp.tokenize(seed_text)
    tokenized_ids = [token_to_id[token] if token in token_to_id else token_to_id['[UNK]'] for token in tokenized_text]

    # Get next token probabilities
    logits = model['model'](torch.tensor([tokenized_ids]))
    probabilities = torch.nn.functional.softmax(logits[0, -1], dim=0)
    
    # Get candidate indices
    candidate_indices = probabilities.topk(10).indices.tolist()

    print(f"candidate_indices: {candidate_indices}")

    for idx in candidate_indices:
        print(f"Checking candidate index: {idx}")
        if idx in id_to_token:
            print(f"Index {idx} is in id_to_token. Token: {id_to_token[idx]}")
        else:
            print(f"Index {idx} is not in id_to_token.")

if __name__ == "__main__":
    seed_text = "テスト"
    check_candidates(seed_text)
