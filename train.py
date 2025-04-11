import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk import pos_tag
import pandas as pd
import re
from tokenize_text import TokenizeText

df = pd.read_csv('ef_POStagged_original_corrected.csv')
print(df.head())  # Print the first 5 rows

text = value = df["original"].iloc[0]
tokenizer = TokenizeText()
print(tokenizer.tokenize(text))