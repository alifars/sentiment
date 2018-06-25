import pandas as pd
import string
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, MaxPooling1D, Embedding, Conv1D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_texts = test_df["Tweet"]
train_texts = train_df["text"]

labels = train_df["class_label"]
print(labels.shape)
encoded_labels = pd.get_dummies(labels)

encoded_labels.drop(encoded_labels.columns[2], axis=1, inplace=True)


def clean_text(text):
    stwords = set(stopwords.words("english"))
    translator = str.maketrans("","", string.punctuation)
    text = str.translate(text, translator)
    tokens = [token for token in text.split() if token not in stwords]
    return " ".join(tokens)

def create_toknizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer




max_length = max([len(s.split()) for s in train_texts])
print("maximum length: {}".format(max_length))

def encode_texts(tokenizer, max_length, texts):
    encoded = tokenizer.texts_to_sequences(texts)
    #encoded = tokenizer.texts_to_matrix(texts, 'tfidf')
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    #plot_model(model, show_shapes=True)
    return model






def write_predictions(predictions):
    with open("subm.csv", "w") as f:
        f.write("Id" + "," + "Expected" + "\n")
        for i, pred in enumerate(predictions):
            if pred[0] > pred[1]:
                label = 0
            else:
                label = 1

            f.write(str(i) + "," + str(label) + "\n")

train_tokenizer = create_toknizer(train_texts)
vocab_size = len(train_tokenizer.word_index) + 1

print("vocab size: {}".format(vocab_size))
train_encoded_texts = encode_texts(train_tokenizer, max_length=max_length, texts=train_texts)
test_encoded_texts = encode_texts(train_tokenizer, max_length=max_length, texts=test_texts)

model = define_model(vocab_size=vocab_size, max_length=max_length)
model.fit(train_encoded_texts, encoded_labels, epochs=10, batch_size=10)
preds = model.predict(test_encoded_texts)
write_predictions(preds)


