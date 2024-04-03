# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.
# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
![image](https://github.com/ThivakarR/rnn-stock-price-prediction/assets/118707074/3d6be3be-2ec0-419d-8e1e-eb7aa0e089c8)

## Design Steps

### Step 1:
import required header files

### Step 2:
read the csv file using pd.read_csv

### Step 3:
use minmaxscaler to set range of feature

### Step 4:
train the dataset

### Step 5:
compile the training set

### Step 6:
fit the training set

## Program
#### Name: YUVARAJ B
#### Register Number:212222230182
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values

type(train_set)

train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape

length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(60,input_shape=(60,1)))
model.add(layers.Dense(1))
model.compile(optimizer='adam',loss='mse')

print("NAME : R.Thivakar REG NO. : 212222240109")
model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Name: YUVARAJ B          Register Number:212222230182")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
## Output

### True Stock Price, Predicted Stock Price vs time
![exp 5 1](https://github.com/ThivakarR/rnn-stock-price-prediction/assets/118707074/50733549-9475-4553-87cf-270ef970380e)

### Mean Square Error
![exp 5 2](https://github.com/ThivakarR/rnn-stock-price-prediction/assets/118707074/4678375b-026d-4fff-8041-58cc2e84cad9)

## Result
Thus the program is successfully created and executed.

## Problem Statement and Dataset


## Design Steps

### Step 1:
Write your own steps

### Step 2:

### Step 3:



## Program
#### Name:
#### Register Number:

Include your code here

## Output

### True Stock Price, Predicted Stock Price vs time

Include your plot here

### Mean Square Error

Include the mean square error

## Result
