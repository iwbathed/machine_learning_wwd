import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix

# Loading saved csv
df = pd.read_pickle("audio/final_audio_data_csv/audio_data.csv")

# Making our data training-ready
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40)

y = np.array(df["class_label"].tolist())
y = to_categorical(y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training

model = Sequential([
    Dense(256, input_shape=X_train[0].shape),
    Activation('relu'),
    Dropout(0.5),
    Dense(256),
    Activation('relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(X_train, y_train, epochs=100)
model.save("audio/saved_model/WWD_3.h5")
print("Model Score: \n")
score = model.evaluate(X_test, y_test)
print(score)

# Evaluating  model
print("Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm,
                      # y_true=y_test, y_pred=y_pred,
                      X=X_train, y_true=y_pred,
                      labels=["Does not have Wake Word", "Has Wake Word"])
