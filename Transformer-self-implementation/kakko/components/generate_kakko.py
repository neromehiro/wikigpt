# python /app/Transformer-self-implementation/kakko/components/generate_kakko.py
import random
import json
import os

def generate_bracket_string(max_depth, length, bracket_types):
    assert max_depth <= len(bracket_types), "Not enough bracket types for the given depth"

    brackets = [bracket for bracket in bracket_types]
    stack = []
    string = ''
    
    for _ in range(length):
        if len(string) >= 29:  # Reserve 1 space for closing bracket
            break

        if not stack or (len(stack) < max_depth and random.choice([True, False])):
            depth = len(stack)  # Determine the bracket type based on depth
            stack.append(brackets[depth][0])  # Opening bracket
            string += brackets[depth][0]
        elif stack:  # Ensure stack is not empty before trying to pop
            # Find the depth of the last opened bracket
            depth = next(i for i, b in enumerate(brackets) if b[0] == stack[-1]) 
            stack.pop()  # Pop opening bracket
            string += brackets[depth][1]  # Closing bracket
    
    # Close all remaining open brackets
    while stack and len(string) < 30:
        # Find the depth of the last opened bracket
        depth = next(i for i, b in enumerate(brackets) if b[0] == stack[-1]) 
        stack.pop()
        string += brackets[depth][1]
    
    return string

# Specify the bracket types
bracket_types = [('(', ')'), ('[', ']'), ('{', '}')]

# Define absolute path for the dataset directory
dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(dir_path, 'kakko_dataset')

# Make the target directory if it doesn't already exist
os.makedirs(dataset_dir, exist_ok=True)

# Generate and save datasets
for i in range(1, 4):
    # Generate dataset
    data = [generate_bracket_string(3, 100, bracket_types) for _ in range(10000)]
    # Save dataset to json file
    with open(os.path.join(dataset_dir, f'dataset{i}.json'), 'w') as f:
        json.dump(data, f)
