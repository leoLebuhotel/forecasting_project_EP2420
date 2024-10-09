
# Forecasting Service Metrics for Video-on-Demand and Key-Value Store Services

## Project Overview

This project aims to forecast service-level metrics of **Video-on-Demand (VoD)** and **Key-Value (KV) store services** using machine learning models. We explored various forecasting approaches like **linear regression**, **Recurrent Neural Networks (RNNs)**, including **Long Short-Term Memory (LSTM)** networks, and time series models like **ARIMA** and **Exponential Smoothing**. The data for this project was collected from a **KTH testbed** and the **FedCSIS 2020 challenge dataset**. 

The goal was to predict future service-level metrics such as response time and latency based on past and present data.

## Tasks Overview

### Task I: Linear Regression for Forecasting

In this task, we used **linear regression models** to predict service metrics based on historical data. The dataset was pre-processed through **standardization** and **outlier removal**, followed by **feature selection** using tree-based methods. Models were evaluated using **Normalized Mean Absolute Error (NMAE)** across different lag and horizon values.

### Task II: Recurrent Neural Networks (RNNs) and LSTM Models

We explored the use of **RNNs** and **LSTM models** to handle non-linearities and long-term dependencies in time series data. The data was structured into sequences with different lags and horizons, and the **Keras Sequential** model was used for building and tuning the LSTM models. Evaluation was performed using **MSE** and **NMAE** metrics.

### Task III: Time Series Analysis

Traditional **time series analysis** was conducted using models like **Auto-Regressive (AR)** and **Moving Average (MA)**. The **Augmented Dickey-Fuller (ADF) test** and **Auto-Correlation Function (ACF)** were applied to assess stationarity and correlation across different lags in the KTH and FedCSIS datasets.

### Task IV: Forecasting with ARIMA and Exponential Smoothing

We applied **ARIMA** and **Exponential Smoothing** to the standardized datasets. The models were tested across various parameter values to determine the optimal setup. Performance was measured using **Mean Absolute Error (MAE)** and compared across different time series horizons.

## Code Structure

The code for each task is organized in separate Jupyter notebooks:

- [**task_I.ipynb**](code/task_I.ipynb): Data pre-processing, feature selection, and linear regression model training.
 - [**task_II.ipynb**](code/task_II.ipynb): Recurrent Neural Networks (RNNs) and LSTM Models.
- [**task_III.ipynb**](code/task_III.ipynb): Time series analysis, including ADF tests and ACF computation.
- [**task_IV.ipynb**](rcode/task_IV.ipynb): ARIMA and Exponential Smoothing models implementation.

## Results

Key insights from the project include:
- **LSTM models** demonstrated better short-term forecasting capabilities due to their proficiency in capturing time-dependent patterns.
- **ARIMA models** proved useful for capturing both short-term and long-term dependencies.
- **Exponential Smoothing** provided reliable performance for datasets with high variability and noise.

You can find a full report [here](EP2420___Project_2__Final_Report.pdf).

## Requirements

- **Python 3.x**
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `keras`, `keras-tuner`
