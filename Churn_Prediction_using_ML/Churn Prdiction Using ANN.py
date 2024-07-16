# Churn Prediction Unsing ANN (Artificial Neural Network)

# Importing Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - data preprocessing
# Importing Dataset
dataset = pd.read_csv(r'E:\Naresh IT Data Science Course\Course Study Material\November Month Course Material\6th n 7th nov\6th, 7th- Introduction to Deep Learning\Practicle - CPU\ANN_ 1st\Churn_Modelling.csv')

# Spliting data into dependent and independent variable
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values
print(X)
print(y)

# Encoding the categorical data

# Label Encoding the "Gender" Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
print(X)

# One hot encoding the "Geography" column 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])],remainder= 'passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential() # create a neural network model in a sequential manner

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

ann.add(tf.keras.layers.Dense(units=5, activation = 'relu'))

ann.add(tf.keras.layers.Dense(units= 4, activation= 'relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units= 1, activation= 'sigmoid'))

# Part 3 - Training the ANN
ann.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train,y_train,batch_size= 32, epochs =300)

# Part 4 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# compute accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)