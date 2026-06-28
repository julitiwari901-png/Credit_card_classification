# %%
#from google.colab import files
#uploaded=files.upload()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df=pd.read_csv('creditcard.csv')
print(df)

# %%
df.describe()

# %%
df.info()

# %%
df.duplicated().sum()

# %%
df=df.drop_duplicates()

# %%
df = df.dropna() # Drop rows with any NaN values

# %%
df.isna().sum()

# %%
X=df.drop('Class',axis=1)
y=df['Class']

# %%
X.shape

# %%
y.shape

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# %%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# %%
y_train.value_counts()

# %%
y_test.value_counts()

# %%
'''from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report'''

# %%
''''log=LogisticRegression()
log.fit(X_train,y_train)'''

# %%
'''y_pred = log.predict(X_test)

print(y_pred)'''

# %%
'''acc = accuracy_score(y_test, y_pred)
print("Accuracy:",acc)


cr = classification_report(y_test, y_pred)
print("Classification Report:")
print (cr)'''

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

# %%
model=RandomForestClassifier()
model.fit(X_train,y_train)

# %%
y_pred=model.predict(X_test)
print(y_pred)

# %%
cr=classification_report(y_test,y_pred)
print("Classification Report:")
print(cr)

# %%
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

# %%
model = Sequential()

# %%
model = Sequential()

model.add(Input(shape=(30,))) # Changed from 12 to 30
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# %%
model.summary()

# %%

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# %%
earlystop = EarlyStopping(monitor = 'val_loss', patience = 50, restore_best_weights = True)

# %%
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), callbacks = earlystop)

# %%
y_pred = model.predict(X_test)


