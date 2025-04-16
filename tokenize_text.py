import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk import pos_tag
import re

class TokenizeText:
    def tokenize(self, text):
        cleaned = re.sub(r'<.*?>', '', text)
        tokens = word_tokenize(cleaned)
        tagged_tokens = pos_tag(tokens)
        pos = [n[1] for n in tagged_tokens]
        return pos
    


