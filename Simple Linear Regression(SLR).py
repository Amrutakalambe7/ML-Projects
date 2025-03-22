import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"E:\FSDS With GEN AI_NIT\3. March\20th -SLR WORKSHOP\Salary_Data.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train = X_train.reshape(-1,1)

X_test = X_test.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train) #Regrssor is the model consist Linear Regrssion algorithm

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(test Set)')
plt.xlabel('years of experience')
plt.ylabel("Salary")
plt.show()
          
plt.scatter(X_train, y_train,color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(train Set)')
plt.xlabel('years of experience')
plt.ylabel("Salary")
plt.show()

# find out slope and intercept
print(f"coefficient: {regressor.coef_}")
print(f"Intercept: {regressor.intercept_}")

comparision = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparision)

#Future Prediction for 20 Yrs Experience
y_20 = 26777.39134119764 + 9360.26128619*20
print(y_20)

#Find Bias and VAriance
bias = regressor.score(X_train, y_train)
print(bias)

variance = regressor.score(X_test, y_test)
print(variance)

# as we got bias = 0.9423 and variance = 0.9740 the model is good model

#Find out Descriptive Statistics

dataset.mean() # Mean for entire Dataset
dataset['Salary'].mean() # mean for Salary Column

#median
dataset.median() # Median for entire Dataset
dataset['Salary'].median() # Median for Salary Column

#mode
dataset.mode() # Mode for entire Dataset
dataset['Salary'].mode() # Mode for Salary Column

#variance
dataset.var()

dataset['Salary'].var()
dataset['YearsExperience'].var()

#Standard Deviation
dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()

from scipy.stats import variation
variation(dataset.values)
variation(dataset['Salary'])
variation(dataset['YearsExperience'])

#correlation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness
dataset.skew()
dataset['Salary'].skew()

#Standard error
dataset.sem()
dataset['Salary'].sem()

# z-Score
import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

# degree of freedom
a = dataset.shape[0]
b = dataset.shape[1]

degrees_freedom = b-a
print(degrees_freedom)

#sum of 
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y_pred = y_pred[0:6]  # Match the length of y
SSE = np.sum((y - y_pred) ** 2)
print(SSE)


# SST 
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#r_square
r_square = 1 - (SSR/SST)
print(r_square)

print(regressor)

#Find Bias and Variance
bias = regressor.score(X_train,y_train)
print(bias)

varinace = regressor.score(X_test,y_test)
print(variance)




