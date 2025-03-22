# importing Libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load Dataset
house_data = pd.read_csv(r'E:\FSDS With GEN AI_NIT\3. March\20th -SLR WORKSHOP\20th- slr\SLR - House price prediction\House_data.csv')
house_data.head()

space = house_data['sqft_living']
space.head()

price = house_data['price']
price.head()

#Splitting data into Dependent and Independent Variable
X = np.array(space).reshape(-1,1)
y = np.array(price)

#Split the dataset in test and train datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state=0)

#Buid a Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fit the training data into model
regressor.fit(X_train, y_train)

# Predict values of y based on X_test
y_pred = regressor.predict(X_test)

#Plot the graph for train dataset
plt.scatter(X_train, y_train,color = 'red',)
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title("Visiual for Training Dataset")
plt.xlabel("Space(Sqft)")
plt.ylabel("Price")
plt.show()


#Plot the graph for test dataset
plt.scatter(X_test, y_test,color = 'red',)
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title("Visiual for Test Dataset")
plt.xlabel("Space(Sqft)")
plt.ylabel("Price")
plt.show()

#Find out Bias and variance for the model
bias = regressor.score(X_train,y_train)
print(bias)

variance = regressor.score(X_test,y_test)
print(variance)
