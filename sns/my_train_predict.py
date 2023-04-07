import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import myutil

if __name__ == "__main__":
    # myutil.myread()
    # the_train,the_test=myutil.train_test_split()
    # myutil.X_test,myutil.X_train,y_test,y_train=myutil.my_dataset(train=the_train,test=the_test)
    myutil.my_train()

model = myutil.AirModel()  #TODO: Predicted required codes

model.load_state_dict(torch.load(myutil.PATH)) # TODO: Predicted required codes
model.eval() # Set to evaluation mode, as we will not train the model again # TODO : Predicted required codes


with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(myutil.timeseries[:,:-1]) * np.nan
    y_pred = model(myutil.X_train)  #!!!!!!!!!!! TODO: Predicted required code, predicted value = model(input value)
    y_pred = y_pred[:, -1, :]
    print(y_pred.shape,myutil.X_train.shape)
    y_pred = myutil.normalization_read(y_pred)
    print(y_pred.shape)
    train_plot[myutil.lookback + myutil.lookback - 1:myutil.train_size] = myutil.normalization_read(model(myutil.X_train)[:, -1, :])
    # shift test predictions for plotting
    test_plot = np.ones_like(myutil.timeseries[:,:-1]) * np.nan
    test_plot[myutil.train_size + myutil.lookback + + myutil.lookback - 1:len(myutil.timeseries)] = myutil.normalization_read(model(myutil.X_test)[:, -1, :])
# plot
plt.plot(myutil.normalization_read(myutil.timeseries[:, 0] .reshape(-1, 1)))
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()