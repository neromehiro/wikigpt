import random
from typing import List, Tuple

# 括弧の種類と対応する閉じ括弧
BRACKETS = {'(': ')', '【': '】', '{': '}'}

def generate_bracket_sequence(max_depth: int) -> str:
    if max_depth == 0:
        return ""
    
    num_brackets = random.randint(1, 3)
    brackets = ""
    for _ in range(num_brackets):
        bracket = random.choice(list(BRACKETS.keys()))
        inner = generate_bracket_sequence(max_depth - 1) if random.random() < 0.5 else ""
        brackets += bracket + inner
    
    return brackets

def close_brackets(seq: str) -> str:
    stack = []
    output_seq = ""
    
    for char in seq:
        if char in BRACKETS.keys():  # Opening brackets
            stack.append(char)
        elif char in BRACKETS.values():  # Closing brackets
            if stack and BRACKETS[stack[-1]] == char:
                stack.pop()
            else:
                output_seq += char
    
    while stack:
        opening_bracket = stack.pop()
        output_seq += BRACKETS[opening_bracket]
    
    return output_seq

def generate_brackets(n_samples: int, max_depth: int, min_len: int, max_len: int) -> List[str]:
    dataset = []
    for _ in range(n_samples):
        while True:
            sequence = generate_bracket_sequence(random.randint(1, max_depth))
            if min_len <= len(sequence) <= max_len:
                break
        input_seq = sequence
        output_seq = close_brackets(sequence)
        dataset.append(f"input:{input_seq},output:{output_seq}")
    
    return dataset

num_samples = 10  # データセットのサンプル数
max_depth = 5  # 括弧の最大深さ
min_len = 5  # シーケンスの最小長
max_len = 10  # シーケンスの最大長

# データセットの生成
dataset = generate_brackets(num_samples, max_depth, min_len, max_len)
for data in dataset:
    print(data)
