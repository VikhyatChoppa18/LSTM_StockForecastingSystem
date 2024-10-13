import logging
import configparser

import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import streamlit as st
import pandas as pd
import plotly.express as plot

from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
config = configparser.ConfigParser()
config.read('./.config')

api_key = config.get('alpha_vantage_api','api_key')

db_user = config.get('stock_db','user')
db_password = config.get('stock_db','password')
db_host = config.get('stock_db','host')
db_database = config.get('stock_db','database')
db_port = config.get('stock_db','port')

connection = mysql.connector.connect(user=db_user,password=db_password,host=db_host,
                                     database=db_database,port=db_port)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

engine = create_engine(f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_database}")

log = logging.getLogger()

st.html("""<html>
           <title>
           <head>Exploratory Data Analysis of ETF holdings</head></title>
           <body>
           <p>The current web page contains the visualizations related the selected stock symbol.</p>
           </body>
           </html>""")


class Stock_vis:
    def __init__(self):
        pass

    @staticmethod
    def get_data():
        try:
            symbol = st.selectbox("Select Symbol", options=('AAPL', 'IBM', 'AMZN'), placeholder="Select your symbol",
                                  label_visibility="visible")
            button = st.button(label="Submit symbol")

            if button:
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}&outputsize=full&datatype=csv'
                stock_data = pd.read_csv(url)
                log.info("Retrieving data")
                log.info(f'Data inserted into successfully')

                return stock_data
        except Exception as error:
            log.info(str(error))

    def visualize(self,data):
        try:
        #Visualizing the data
            data = data.dropna(inplace=True)
            #Check for duplicates
            data = data.drop_duplicates(keep='first')
                #Visualizingthe data
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['timestamp'] = data['timestamp'].dt.strftime('%A')
            plot_data = data.groupby('timestamp')['open'].mean().reset_index()
            tab1,tab2 = st.tabs(["Barplot","Lineplot"])

            with tab1:
            #Stacked bar plot
                st.markdown("Open,close and high with respect to weekdays by volume")
                st_bar_plot = plot.bar(data, x='timestamp', y=['close', 'open', 'high'])
                st.plotly_chart(st_bar_plot)
            with tab2:
            #Line plot
                st.markdown("Days with highest open values")
                line_plot = plot.line(plot_data, x='timestamp', y='open')
                st.plotly_chart(line_plot)

            return data
        except Exception as error:
            log.error(str(error))

    def insert_db(self,data):
        if data is not None:
            data.to_sql(name='stock_data',con=engine,if_exists='replace',index=False)
            st.success('Data Inserted into DB')

    def process_data(self,data):
        st.subheader('Performing feature Engineering')
        # data = pd.read_sql('SELECT * FROM stock_data', engine)
        data = pd.DataFrame(data)
        st.table(data.head())

        #Boxplot to find out outliers
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        fig = plot.box(data[numeric_columns],title="Box plot of stock data")
        st.plotly_chart(fig)
        #Data is as follows
        data = data[['timestamp','close']]
        data['timestamp']= pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        days_count = 7
        for i in range(1, days_count+1):
            data[f'Close(t-{i})'] = data['close'].shift(i)
        data.dropna(inplace=True)


        #Applying the encoding
        #Data to be encoded
        st.text('Data  is ready to train the model')
        #Standardizing the data
        np_data = data.to_numpy()
        standardize_data = MinMaxScaler(feature_range=(-1,1)).fit_transform(np_data)
        #Splitting the data
        X = standardize_data[:,1:]
        y = standardize_data[:,0]
        X = dc(np.flip(X,axis=1))
        data_split_index = int(len(X)*0.95)
        X_train = X[:data_split_index]
        X_test = X[data_split_index:]
        y_train = y[:data_split_index]
        y_test = y[data_split_index:]

        #Reshaping the data
        X_train = X_train.reshape((-1,days_count,1))
        X_test = X_test.reshape((-1,days_count,1))

        y_train = y_train.reshape((-1,1))
        y_test = y_test.reshape((-1,1))
        X_train_ = torch.tensor(X_train).float()
        X_test_ = torch.tensor(X_test).float()

        y_train_ = torch.tensor(y_train).float()
        y_test_ = torch.tensor(y_test).float()

        return X_train_,X_test_,y_train_,y_test_




class TimeSeriesDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

#Model
class LSTM(nn.Module):
    def __init__(self, input_size,hidden_size,num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_stacked_layers
                              ,batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batchsize = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers,batchsize,self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers,batchsize,self.hidden_size).to(device)

        out,_ = self.lstm(x,(h0,c0))
        out = self.fc(out[:, -1, :])
        return out

learn_rate = 0.001
epochs = 10
loss_function = nn.MSELoss()

def train_epoch(model,optimizer, epoch,train_load):
    model.train(False)
    print(f'Epoch: {epoch + 1}')
    run_loss = 0.0

    for batch_index, batch in enumerate(train_load):
        xbatch, ybatch = batch[0].to(device), batch[1].to(device)

        output = model(xbatch)
        loss = loss_function (output, ybatch)
        run_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:
            avg_loss = run_loss / 100
            print('Batch {0}, Loss: {1: .3f}'.format(batch_index+1,avg_loss))
            run_loss = 0.0
    print()
    return avg_loss

def validate_epoch(model,test_load):
     model.train(False)
     run_loss = 0.0
     for batch_index, batch in enumerate(test_load):
         xbatch,ybatch = batch[0].to(device), batch[1].to(device)

         with torch.no_grad():
             output = model(xbatch)
             loss = loss_function(output,ybatch)
             run_loss  += loss.item()

     avg_loss = run_loss / len(test_load)

     print('Val Loss: {0:.3f}'.format(avg_loss))
     print('###############################################')
     print()
     return avg_loss


def main():
    sto = Stock_vis()
    data = sto.get_data()
    if data is not None:
        sto.visualize(data)
        sto.insert_db(data)
        result = sto.process_data(data)
        X_train, X_test, y_train, y_test = result

        train_data = TimeSeriesDataset(X_train,y_train)
        test_data = TimeSeriesDataset(X_test,y_test)
        #Data loading in batches
        batchSize = 17
        train_load = DataLoader(train_data,batchSize,shuffle=True)
        test_load = DataLoader(test_data,batchSize,shuffle=False)
        for _,batch in enumerate(train_load):
            xbatch= batch[0].to(device)
            ybatch = batch[1].to(device)
            print(xbatch.shape,ybatch.shape)
            break
        st.subheader('Building the model')
        model = LSTM(1, 4, 1)
        model.to(device)

        st.subheader("Training the model")
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        model_metrics = st.empty()
        train_val_progress = st.progress(0)
        for epoch in range(epochs):
            train_ = train_epoch(model,optimizer,epoch, train_load)
            validate_ = validate_epoch(model,test_load)
            model_metrics.text(f'')
            model_metrics.text(f"epoch: {epoch + 1}/{epochs}\n"
                         f"Train Loss: {train_:.3f}\n"
                         f"Val Loss: {validate_:.3f}")
            train_val_progress.progress((epoch + 1) / epochs)
        st.success("Model trained on data")

        with torch.no_grad():
            predicted = model(X_train.to(device)).to('cpu').numpy()

        predicted = predicted.flatten()
        y_train = y_train.numpy().flatten()

        dataframe = pd.DataFrame({
            'index': range(len(predicted)),
            'predicted': predicted,
            'actual': y_train
        })

        fig = plot.line(dataframe, x='index', y=['predicted', 'actual'], title='Predicted vs Actual',color_discrete_sequence=['yellow','brown'])
        fig.show()
        st.plotly_chart(fig)



if __name__=="__main__":
    main()





