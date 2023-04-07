import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
import joblib

#PATH :str =  "./my_model.pth" ;
PATH :str =  "./my_model.pth" ;
global timeseries;
global X_train;
lookback = 7;
global train_size;
global X_test;

def myread(filepath='new_data.csv'):
    global timeseries
    df = pd.read_csv(filepath)
    df=df.iloc[::-1]
    #timeseries = df[["Close","vader_sentiment"]].values.astype('float32')
    timeseries = df[["Close", "vader_sentiment"]].values.astype('float32')
    timeseries,_,_,_ = normalization_init(timeseries,timeseries[:, 0] .reshape(-1, 1))
    # return df,timeseries

def train_test_split():
    # train-test split for time series
    global  train_size
    train_size = int(len(timeseries) * 0.8)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]
    return train,test


def create_dataset(dataset, lookback,seq_length = 7):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback - seq_length +1):
        feature = np.array(dataset[i:i + lookback])
        target = np.array(dataset[i + seq_length:i + lookback + seq_length,0:1])
        X.append(feature)
        y.append(target)
        X_=np.array(X)
        y_=np.array(y)
    return torch.tensor(X_), torch.tensor( y_)

def my_dataset(train,test):
    global lookback
    # lookback = 5
    X_train, y_train = create_dataset(train, lookback=lookback)

    X_test, y_test = create_dataset(test, lookback=lookback)


    return  X_test,X_train,y_test,y_train


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def my_train():
    global  X_train,X_test
    myread()
    the_train,the_test=train_test_split()
    X_test,X_train,y_test,y_train=my_dataset(train=the_train,test=the_test)
    model = AirModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=21)

    n_epochs = 10000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 500 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))



    torch.save(model.state_dict(), PATH)


def normalization_init(X : pd.DataFrame, Y:pd.DataFrame):

    mm_x=MinMaxScaler() # Importing sklearn's preprocessing container
    mm_y=MinMaxScaler()
    # X=X.values    # Convert pd's series format to np's array format
    # Y=Y.values
    X=mm_x.fit_transform(X) # Normalisation of data and labels etc.
    Y=mm_y.fit_transform(Y)

    joblib.dump(mm_x, 'scalar03')
    joblib.dump(mm_y, 'scalar04')
    return X,Y,mm_x,mm_y


def normalization_use(X_t):
    mm_x = joblib.load('scalar03')
    # X_t = X_t.values
    X_test = mm_x.transform(X_t)


    return X_test

def normalization_read(Y_t):
    mm_y = joblib.load('scalar04')
    Y_test = mm_y.inverse_transform(Y_t)
    return Y_test

def myPredict(series:pd.DataFrame, modelName: str = "LSTM", priceFlag=True, volumeFlag=True,
              time_length: int = 1):  # timeSeries : list or Tensor
  
    # price_output =[] ,volume_output =[]
    if modelName == "LSTM":
        if priceFlag :
            series_price = series [:,[0,5]]
            mm_x = joblib.load('scalar03')

            series_price = mm_x.transform(series_price)
            series_price,_ = create_dataset (series_price,7)

            model = AirModel()  # TODO
            # X_input = torch.tensor(timeSeries)
            model.load_state_dict(torch.load("my_model.pth"))  # TODO
            model.eval()  # TODO
            with torch.no_grad():  # toDO
                y_output = model(series_price)
                print(y_output.shape)
                y_output=y_output[-1, :, :]
                print(y_output.shape)
                mm_y = joblib.load('scalar04')
                y_output = mm_y.inverse_transform(y_output)

                price_output = y_output.flatten().tolist()

                price_output = price_output [0:time_length]
        if volumeFlag :
            model2 = AirModel()  # TODO
            # X_input = torch.tensor(timeSeries)
            series_volume = series[:,[4,5]]
            mm_x2 = joblib.load('scalar01')
            series_volume = mm_x2.transform(series_volume)
            series_volume,_ = create_dataset (series_volume,7)

            model2.load_state_dict(torch.load("my_mode2.pth"))  # TODO
            model2.eval()  # TODO
            with torch.no_grad():  # toDO
                y_output2 = model(series_volume)
                y_output2=y_output2[-1, :, :]
                mm_y2 = joblib.load('scalar02')
                y_output2 = mm_y2.inverse_transform(y_output2)
                volume_output = y_output2.flatten().tolist()
                price_output = price_output[0:time_length]

    # else :
    #     ;


    return price_output, volume_output

