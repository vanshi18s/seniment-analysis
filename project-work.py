import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)  

max_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

model = Sequential()
model.add(Embedding(10000, 128))  
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
model.fit(x_train, y_train, epochs=5, batch_size=batch_size, validation_split=0.2)

test_score, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_score:.4f}, Test Accuracy: {test_accuracy:.4f}')
example_review = "This movie was fantastic! I loved it."
movie_review_encoded = [1, 2, 3, 4, 5]  
movie_review_encoded = sequence.pad_sequences([movie_review_encoded], maxlen=max_length)
prediction = model.predict(movie_review_encoded)
print("Predicted Sentiment: ", "Positive" if prediction[0][0] > 0.5 else "Negative")