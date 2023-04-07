from ctypes.wintypes import tagRECT
from datetime import time
from pickle import TRUE
import socket
import myutil
import struct
import Getdata
import SentenceProcessing
import pickle
##############################################################################################
#################################   File Part    #############################################
##############################################################################################
from cgi import test
from msilib import MSIMODIFY_INSERT
import myutil
import torch
import pandas as pd
import numpy as np

####### need to update before using the prediction function, if you want to predict the data in the future
period = 14
Getdata.Getdata(period) # Get the newest period*3 days of date
df = pd.read_csv('data.csv')
timeseries = df[["Close","Open","High","Low","Volume","vader_sentiment"]].values.astype('float32')
'''
# dataset is the feature you want to use
# lookback means how long/many the data you wann to use as an input
def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = np.array(dataset[i:i + lookback])
        target = np.array(dataset[i + 1:i + lookback + 1,0:1])
        X.append(feature)
        y.append(target)
        X_=np.array(X)
        y_=np.array(y)
    return torch.tensor(X_), torch.tensor(y_)

myinput,mynotiong= create_dataset(timeseries,5)
'''
##############################################################################################
#################################  Server part  ##############################################
##############################################################################################

with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as server:

# Set the SO_REUSEADDR option, ensure that the next connection can be set immediately
# after closing the previous connection
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind(('127.0.0.1',1234))
    server.listen()
    print("The server started")
    while True:
        # accept a client connection
        client, client_addr = server.accept()
        print(client_addr,'connected.')

        ###### Receive the message and handle it ########
        data = client.recv(1024)
        # convert the bytes object to a string 
        data = data.decode('utf-8')

        mode,target,timelength = SentenceProcessing.decoding(data)
        
        price_output,volume_output = myutil.myPredict(timeseries,'LSTM',True,True,7)
        # target
        if target == 'volume':
            target_list = volume_output
        else:
            target_list = price_output
        # keep two decimal places
        target_list = [round(num, 2) for num in target_list]

        # time length
        target_list = target_list[:timelength]
        feedback = pickle.dumps(target_list)
        client.send(feedback)
        print('wait')
        
        #################################################
        # close the client connection,ensuring the success of next connection
        client.close()
