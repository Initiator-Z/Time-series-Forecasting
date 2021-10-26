import pandas as pd
import numpy as np
import glob
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch import hstack
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import os
import gc
from os import listdir
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_data(path):
    data = pd.read_csv(path)
    data["SETTLEMENTDATE"] = pd.to_datetime(data["SETTLEMENTDATE"], format="%Y-%m-%d %H:%M:%S")
    return data

def multivariate_sequence(X1, X2, window_size):
    seq1, seq2, label1, label2 = list(), list(), list(), list()
    for t in range(window_size, len(X1)):
        seq1.append(X1[t-window_size:t])
        seq2.append(X2[t-window_size:t])
        label1.append(X1[t:t+1])
        label2.append(X2[t:t+1])
    seq1, seq2, label1, label2 = np.asarray(seq1), np.asarray(seq2), np.asarray(label1), np.asarray(label2)
    rrp, demand = list(), list()
    for seqs in seq1:
        seqs = seqs.reshape((len(seqs), 1))
        rrp.append(seqs)
    for seqs in seq2:
        seqs = seqs.reshape((len(seqs), 1))
        demand.append(seqs)
    del seq1, seq2
    sequences = list()
    labels = list()
    for i in range(len(rrp)):
        seq = np.hstack((rrp[i], demand[i]))
        lab = np.hstack((label1[i], label2[i]))
        labels.append(lab)
        sequences.append(seq)
    del rrp, demand, label1, label2
    return np.asarray(sequences), np.asarray(labels)

class columnDataset(Dataset):
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target
    def __len__(self):
        return len(self.feature)
    def __getitem__(self, idx):
        item = self.feature[idx]
        label = self.target[idx]
        return item, label

class CNN_ForecastNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(CNN_ForecastNet, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=1, stride=1)
        self.conv1d2 = nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(1024, 50)
        self.fc3 = nn.Linear(50, out_channels)
        self.fc4 = nn.Linear(50, out_channels)

    def forward(self, x):
        x1 = self.conv1d1(x)
        x1 = self.relu(x1)
        x1 = x1.view(x1.shape[0], -1)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc3(x1)

        x2 = self.conv1d2(x)
        x2 = self.relu(x2)
        x2 = x2.view(x2.shape[0], -1)
        x2 = self.fc2(x2)
        x2 = self.relu(x2)
        x2 = self.fc4(x2)
        return x1, x2

def training_loop(model, train_loader, test_loader, device):
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 500
    train_losses = []
    test_losses = []
    for t in range(num_epochs):
        if t<=3 or t%10==0:
            print('epochs {}/{}'.format(t+1, num_epochs))
        running_loss = .0
        model.train()
        for idx, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device=device)
            y_train = y_train.to(device=device)
            y_pred = model(X_train)
            y_pred = torch.stack(y_pred, 2)
            #print('y_pred_train'+str(y_pred.shape))
            loss = loss_fn(y_pred.reshape(y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2]), y_train.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
        train_loss = running_loss/len(train_loader)
        train_losses.append(train_loss.detach().cpu().numpy())
        print(f'train_loss {train_loss}')
        running_loss = .0
        model.eval()
        with torch.no_grad():
            for idx, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test.to(device=device)
                y_test = y_test.to(device=device)
                y_pred = model(X_test)
                y_pred = torch.stack(y_pred, 2)
                #print('y_pred_test'+str(y_pred.shape))
                loss = loss_fn(y_pred.reshape(y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2]), y_test.reshape(y_test.shape[0]*y_test.shape[1]*y_test.shape[2]))
                running_loss += loss
            test_loss = running_loss/len(test_loader)
            test_losses.append(test_loss.detach().cpu().numpy())
            print(f'test_loss {test_loss}')
        gc.collect()
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='test_loss')
    plt.title('MSE Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    return model.eval()

def get_state(filename):
    pt = filename.find('.')
    name = filename[10:pt]
    return name

def write_predictions(pred_RRP, pred_Demand, modelname, state, step_size):
    all_predictions = np.hstack((pred_RRP, pred_Demand))
    df = pd.DataFrame(all_predictions, columns=['RRP', 'TOTALDEMAND'])
    outdir, outname = './prediction', modelname+'_'+state+'_'+str(step_size)+'steps'+'_multi'+'.csv'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, outname)
    df.to_csv(fullname, index=False, header=True)
    print('Writing complete!')

def plot_predictions_errors(pred_RRP, pred_Demand, test_RRP, test_Demand, rmse, modelname, state, step_size):
    plt.figure(figsize=(8,5))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.plot(pred_RRP, color='blue', label='Pred_RRP')
    plt.plot(pred_Demand, color='green', label='Pred_Demand')
    plt.plot(test_RRP, color='black', label='RRP')
    plt.plot(test_Demand, color='gray', label='Demand')
    plt.plot([rmse]*len(tests), color='red', label='RMSE='+str(rmse))
    plt.legend()
    plt.savefig('D:/Project/notebooks/Capstone/code/models/plots/'+modelname+'_'+state+'_'+str(step_size)+'steps_multi'+'.jpg')
    plt.show()
    print('Plotting complete!')

def validate(model, test_loader, device):
    predictions, tests = list(), list()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device=device)
            y = y.to(device=device)
            y_pred = model(X)
            y_pred = torch.stack(y_pred, 2)
            predictions.extend(v for v in y_pred.float().cpu().numpy())
            tests.extend([v for v in y.cpu().numpy()])
    return np.asarray(predictions), np.asarray(tests)

lag_size = [50, 200, 1000, 2000]
path = 'D:/Project/notebooks/Capstone/data/merged/separated/' # path containing csv files with filenames: 'half_hour_nsw.csv' etc.

if __name__ == '__main__':
    for filename in listdir(path):
        for size in lag_size:
            fullpath = path+filename
            state = get_state(filename)
            data = load_data(fullpath)
            col_name = 'RRP'
            X1 = data[['RRP']].values.astype(float)
            X2 = data[['TOTALDEMAND']].values.astype(float)

            record_size = X1.shape[0]
            window_size = size
            test_size = size*6
            device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            batch_size = 64

            scaler = StandardScaler()
            scaler2 = StandardScaler()

            X1_normalized = scaler.fit_transform(X1)
            X2_normalized = scaler2.fit_transform(X2)
            del data, X1, X2

            seqs, labels = multivariate_sequence(X1_normalized, X2_normalized, window_size)
            del X1_normalized, X2_normalized

            print(seqs.shape, labels.shape, '\n')
            X_train = torch.from_numpy(seqs[:-test_size]).float()
            y_train = torch.from_numpy(labels[:-test_size]).float()
            X_test = torch.from_numpy(seqs[-test_size:]).float()
            y_test = torch.from_numpy(labels[-test_size:]).float()
            del seqs, labels

            train_data = columnDataset(X_train.reshape(X_train.shape[0], X_train.shape[1], 2), y_train)
            test_data = columnDataset(X_test.reshape(X_test.shape[0], X_test.shape[1], 2), y_test)
            print(train_data.feature.shape, train_data.target.shape)
            print(test_data.feature.shape, test_data.target.shape)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            del X_train, y_train, X_test, y_test, train_data, test_data

            model = CNN_ForecastNet(in_channels=window_size).to(device=device)
            model = training_loop(model, test_loader, train_loader, device=device)

            predictions, tests = validate(model, test_loader, device=device)
            del model
            predictions, tests = np.squeeze(predictions), np.squeeze(tests)
            pred_RRP, pred_Demand, test_RRP, test_Demand = list(), list(), list(), list()
            for value in predictions:
                pred_RRP.append(value[0])
                pred_Demand.append(value[1])
            for value in tests:
                test_RRP.append(value[0])
                test_Demand.append(value[1])

            pred_RRP, pred_Demand, test_RRP, test_Demand = np.asarray(pred_RRP), np.asarray(pred_Demand), np.asarray(test_RRP), np.asarray(test_Demand)
            pred_RRP, pred_Demand, test_RRP, test_Demand = np.expand_dims(pred_RRP, axis=1), np.expand_dims(pred_Demand, axis=1), np.expand_dims(test_RRP, axis=1), np.expand_dims(test_Demand, axis=1)
            pred_RRP, pred_Demand, test_RRP, test_Demand = scaler.inverse_transform(pred_RRP), scaler2.inverse_transform(pred_Demand), scaler.inverse_transform(test_RRP), scaler2.inverse_transform(test_Demand)

            rmse_RRP = sqrt(mean_squared_error(pred_RRP, test_RRP))
            rmse_Demand = sqrt(mean_squared_error(pred_Demand, test_Demand))
            rmse = rmse_RRP + rmse_Demand
            rmse = float("{:.4f}".format(rmse))

            write_predictions(pred_RRP, pred_Demand, 'CNN', state, window_size)
            plot_predictions_errors(pred_RRP, pred_Demand, test_RRP, test_Demand, rmse, 'CNN', state, window_size)
