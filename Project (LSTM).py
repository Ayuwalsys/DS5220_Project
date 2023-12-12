#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[1124]:


import matplotlib.pyplot as plt


# In[395]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# In[675]:


df = pd.read_csv("store_sales_cleaned_combined.csv")
df
#trainonly = pd.read_csv("train.csv")


# In[ ]:





# In[1119]:


#df.set_index('date', inplace=True)

# Plotting the lines
plt.figure(figsize=(16, 8), dpi=150)
df['sales'].plot(linestyle='-')
df['onpromotion'].plot(linestyle='-')
df['oil_price'].plot(linestyle='-')


# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Values')
plt.xticks(rotation = 45)
plt.title('Sales and Features Over Time')
plt.legend()  # Show legend with labels

# Show the plot
plt.show()


# In[1123]:


#df.set_index('date', inplace=True)

# Plotting the lines
plt.figure(figsize=(20, 15))
df['sales'][0:500].plot(linestyle='-')
df['onpromotion'][0:500].plot(linestyle='-')
df['oil_price'][0:500].plot(linestyle='-')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Values')
plt.xticks(rotation = 45)
plt.title('Sales and Features Over Time')
plt.legend()  # Show legend with labels

# Show the plot
plt.show()


# In[984]:


data_features = df.iloc[:, 1:4]


# In[1011]:





# In[ ]:





# In[331]:


df[df['sales'] < 100000]


# In[87]:


def get_x(data):
    sequence_length = 6
    sequences = [data[i:i+sequence_length] for i in range(len(data)-sequence_length-1)]
    X = np.array(sequences)
    return X

#used to look at the dates on data
def get_x_df(data):
    sequence_length = 6
    sequences = [data[i:i+sequence_length] for i in range(len(data)-sequence_length-1)]
    #X = np.array(sequences)
    return sequences


# In[1092]:





# In[ ]:





# In[93]:


def get_y(data):
    sequence_length = 6
    sequences = [data[i:i+sequence_length] for i in range(len(data)-sequence_length)]
    y = []
    for i in range(1,len(sequences)):
        y.append(sequences[i].iloc[5]['sales'])
    return np.array(y)

#used to look at the dates on data
def get_y_df(data):
    y = []
    for i in range(1,len(sequences)):
        y.append([sequences[i].iloc[5]['date'],sequences[i].iloc[5]['sales']])
    return y


# In[85]:


def normalize(df):
    df_norm = (df-df.min())/(df.max()-df.min())
    return df_norm


# In[1161]:


X = get_x(normalize(data_features))
y = get_y(normalize(data_features))


# In[1162]:


#last 15 days of testing data
test_x = X[1661:1676]
test_y = y[1661:1676]


# In[1190]:


def make_df(y_pred,test_y):
    dic={'prediction': y_pred.reshape(len(y_pred)), 'actual': test_y}
    pred_act = pd.DataFrame(dic)
    return pred_act


def unnormalize(test_df, original_df):
    unnorm_df = test_df * (original_df.max()-original_df.min()) + original_df.min()
    return unnorm_df

def calc_metrics(df):
    rmse = np.sqrt(sum((df['actual']-df['prediction'])**2)/len(df))
    mse = sum((df['actual']-df['prediction'])**2)/len(df)
    mae = sum(np.abs(df['actual']-df['prediction']))/len(df)
    rmse_norm = rmse/df['actual'].max()
    mse_norm = mse/df['actual'].max()
    mae_norm = mae/df['actual'].max()
    df1 = pd.DataFrame([{"RMSE": rmse, "MSE": mse,"MAE": mae, "RMSE normalized": rmse_norm,"MSE normalized": mse_norm, "MAE normalized": mae_norm}])
    return df1


# In[1229]:


def calc_metrics_new(df):
    rmse = np.sqrt(sum((df['actual']-df['prediction'])**2)/len(df))
    mae = sum(np.abs(df['actual']-df['prediction']))/len(df)
    rmsle = np.sqrt(sum((np.log(df['actual']+1)-np.log(df['prediction']+1))**2)/len(df))
    mape = sum(np.abs((df['actual']-df['prediction'])/df['actual']))/len(df) * 100
    overEstim=0
    underEstim=0
    for i in range(0,len(predictions_actual)):
        if predictions_actual.iloc[i]['actual'] < predictions_actual.iloc[i]['prediction']:
            overEstim += (1-0.5) * np.abs(predictions_actual.iloc[i]['actual'] - predictions_actual.iloc[i]['prediction'])
        else:
            underEstim += (0.5) * np.abs(predictions_actual.iloc[i]['actual'] - predictions_actual.iloc[i]['prediction'])
    quantile_loss = overEstim + underEstim
    df1 = pd.DataFrame([{"RMSE": rmse, "MAE": mae, "RMSLE": rmsle, "MAPE": mape, "Quantile Loss:": quantile_loss}])
    return df1


# In[ ]:





# In[ ]:





# In[1166]:


X_train = X[0:1665]
y_train = y[0:1665]


# In[1231]:


model = Sequential()

model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1,activation='linear'))

#optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)

#model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics = ['mse'])  # Use 'mse' for regression tasks

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, shuffle=True)


# In[ ]:





# In[1232]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model accuracy')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[1233]:


y_pred = model.predict(test_x)


# In[ ]:





# In[1234]:


predictions_actual = unnormalize(make_df(y_pred, test_y),data_features[1661:1676]['sales'])
#predictions_actual


# In[1235]:


metrics = calc_metrics(predictions_actual)
metrics
#100 33647.063257
#50 33847.687786
#150 38013.115445
#200 35788.6477
#100 32038.095687


# In[1236]:


metrics1 = calc_metrics_new(predictions_actual)
metrics1


# In[1237]:


plt.figure(figsize=(16, 8), dpi=150)
  
predictions_actual['actual'].plot(label='sales_actual')
predictions_actual['prediction'].plot(label='sales_prediction', color='orange')

# adding Label to the x-axis
plt.xlabel('Time')
  
# adding legend to the curve
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1174]:


model = Sequential()

model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(1,activation='linear'))

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics = ['mse'])
#model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mse'])  # Use 'mse' for regression tasks

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)


# In[1175]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model accuracy')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[1176]:


y_pred = model.predict(test_x)


# In[1177]:


predictions_actual = unnormalize(make_df(y_pred, test_y),data_features[1661:1676]['sales'])


# In[1178]:


#predictions_actual


# In[1179]:


metrics = calc_metrics(predictions_actual)
metrics
#100,100 34081.645677
#100,50 31662.477867
#100, 150 33939.70692
#100, 50, 100 31640.174052
#100,50,50 33651.689035
#100,50,100 31483.57225
#100,100 29112.232536
#100,100,100 29790.953227


# In[1230]:


metrics1 = calc_metrics_new(predictions_actual)
metrics1


# In[1181]:


plt.figure(figsize=(16, 8), dpi=150)
  
predictions_actual['actual'].plot(label='sales_actual')
predictions_actual['prediction'].plot(label='sales_prediction', color='orange')

# adding Label to the x-axis
plt.xlabel('Time')
  
# adding legend to the curve
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




