# LSTM Stock Forecasting

The current app allows users to select their respective choice of the stock, then the app provides 
the actual and the predicted output of the model such that the future performance of the stock can be predicted.system 
This is a complete microservice built using FastAPI, Python and Docker that fetches live data from AlphaVantage API this system predicts the closing price prediction of the stock.

The system is built on the follwing:\n
  - 1)This system is built on LSTM model built from scratch using PyTorch and python.
  - 2)Streamlit for the frontend.
The model predicts the closing price with 96% accuracy making it reliable source to consider the closing price from live data.
  
## Table of Contents
- [Features](#features)
- [Application Installation](#installation)
- [Tools Used](#tools-used)



## Features
- F 1: Timestamp
- F 2: closing price prediction of stock

## Application installation
To set up the project, follow these steps:

1. Clone the repository:
2. Build the docker image
<code>docker build -t app_lstm .</code>
3. Run the docker container
<code>docker run -p 8501:8501 app_lstm</code>

## Tools Used
<p align="left">
<a href="https://pytorch.org/" target="_blank" rel="noreferrer">
 <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/>
</a>
<a href="https://streamlit.io/" target="_blank" rel="noreferrer">
 <img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" alt="streamlit" width="40" height="40"/>
</a>
<a href="" target="_blank" rel="noreferrer">
 <img src="https://github.com/Venkata-Ch/LSTM_App/blob/65860ae206f960db552639815e438f7bab54e806/assets/docker-logo-blue.png" alt="streamlit" width="100" height="40"/>
</a>
</p>




![Screenshot 2](https://github.com/Venkata-Ch/LSTM_App/blob/65860ae206f960db552639815e438f7bab54e806/assets/Screenshot%20from%202024-10-09%2017-29-01.png)
*Model building*

![Screenshot 2](https://github.com/Venkata-Ch/LSTM_App/blob/65860ae206f960db552639815e438f7bab54e806/assets/Screenshot%20from%202024-10-09%2017-29-18.png)
*Model results*

![Screenshot 3](https://github.com/Venkata-Ch/LSTM_App/blob/e00f5dfb15a5d827e06b77aeae56c24f5a2b9186/assets/Screenshot%202024-10-13%20175823.png)
*Model Closing price prediction results*

#### Walkthrough



[Screencast from 10-04-2024 02_09_57 PM.webm](https://github.com/user-attachments/assets/2d909304-d679-4088-9324-08a2635f2537)

