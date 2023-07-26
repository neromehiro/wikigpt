# file: tokenize_text.py
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def tokenize_text(text):
    # テキストを単語に分割
    tokens = word_tokenize(text)
    return tokens
