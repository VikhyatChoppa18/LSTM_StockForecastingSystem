# LSTM Stock Forecasting

THe current app allows users to select their respective choice of the stock, then the app provides 
the actual and the predicted output of the model such that the future performance of the stock can be predicted.
## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Tools Used](#tools-used)



## Features
- F 1: Timestamp
- F 2: close value of stock

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
<a href="https://streamlit.io/" target="_blank" rel="noreferrer">
 <img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" alt="streamlit" width="40" height="40"/>
</a>
</p>


![Screenshot 1](assets/Screenshot from 2024-10-04 14-09-40.png)
*Data gathering and insertion*


![Screenshot 2](assets/Screenshot from 2024-10-09 17-29-01.png)
*Model building*

![Screenshot 3](assets/Screenshot from 2024-10-09 17-29-18.png)
*Model results*

#### Walkthrough




