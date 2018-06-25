import string
from nltk.corpus import stopwords

def clean_text(text):
    stwords = set(stopwords.words("english"))
    translator = str.maketrans("","", string.punctuation)
    text = str.translate(text, translator)
    tokens = [token for token in text.split() if token not in stwords]
    print(tokens)

clean_text("this is a test sentence.")
