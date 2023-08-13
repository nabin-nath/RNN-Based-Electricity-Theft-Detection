import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/Major Project - II/merged_data.csv')

# dataset shuffling
df = df.sample(frac=1).reset_index(drop=True)

# test and train dataset preparation
trainX, trainy = df.iloc[:30000, 2:], df.iloc[:30000, 1:2]
testX, testy = df.iloc[30001:, 2:], df.iloc[30001:, 1:2]
     
#input scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(trainX)
scaled_train = scaler.transform(trainX)
scaled_test = scaler.transform(testX)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
     
# train model
verbose, epochs, batch_size = 1, 15, 64
n_timesteps, n_features, n_outputs = trainX.shape[1], 1, trainy.shape[1]
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model fitting
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

# evaluation
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
