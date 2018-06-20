from keras.datasets import imdb
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np

np.random.seed(23)

(X_train, y_train) , (X_test, y_test) = imdb.load_data(num_words=1000)

tokenizer = Tokenizer(num_words=1000)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')

num_classes = 2
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)


#
model = Sequential()
model.add(Dense(100, input_dim=1000, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

print(model.summary())
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, batch_size=25)

score = model.evaluate(X_test, y_test)
print(score)
