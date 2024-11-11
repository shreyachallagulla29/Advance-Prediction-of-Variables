## Advanced Prediction of Process Variables Using Deep Learning
This project aims to predict critical process variables using deep learning, specifically using a Long Short-Term Memory (LSTM) model. By leveraging sequences of historical data, we predict future values of these variables and explore different input-output window configurations for optimal performance.

### Table of Contents
Project Overview

Model Architecture

Data Preprocessing

Experiments and Configurations

Results and Visualization

Getting Started

Dependencies

### Project Overview
This project focuses on predicting future process variables using LSTM-based deep learning techniques. By testing different input-output configurations, we aim to understand how sequence length impacts prediction accuracy. The LSTM model is used to handle the sequential nature of the data, making it ideal for time series forecasting tasks.

### Model Architecture
We employed an LSTM model for sequential data modeling, with the following key features:

LSTM Layers: Captures long-term dependencies and patterns in the data.

Dense Layers: Maps LSTM outputs to target predictions.
The model was trained and validated on multiple configurations of the time-series data, as described below.

### Data Preprocessing
Our data consists of time-series measurements sampled at regular intervals. To prepare it for the LSTM model, we transformed the dataset into input-output sequences as follows:

Initial Configuration: We used 2 hours of data as input to predict the next 20 minutes. This corresponds to 120 rows as input and 20 rows as output, with each sequence containing a total of 140 rows.

Sampling Adjustments: To explore how a larger input-output window affects model performance, we expanded the time window to 6 hours of input and 1 hour of output by selecting every third row from the original dataset. This downsampling reduced the number of sequences from 900 to 300.
To maintain consistency across datasets, we created three parallel datasets with staggered sequences:

  -(0, 3, 6, …)
  -(1, 4, 7, …)
  -(2, 5, 8, …)
  
This helped preserve the data size while still benefiting from a longer time window.

Experiments and Configurations
The following configurations were explored in this project:

### Configuration 1:

Input: 2 hours (120 rows)

Output: 20 minutes (20 rows)

Number of Sequences: 900

Purpose: To test model performance with shorter sequences and frequent updates.

### Configuration 2:

Input: 6 hours (one row every 3 intervals, creating 120 rows)

Output: 1 hour (20 rows)

Number of Sequences: 300 sequences per staggered sequence set (total ~900 across three sets)

Purpose: To test model performance with longer, more spaced-out data.

### Results and Visualization
The performance of each configuration was evaluated using common metrics and visualizations to demonstrate prediction accuracy:

Plots: Visualizations of predicted versus actual values for each configuration, providing insights into the model’s performance on various input-output windows.

Comparison Metrics: Statistical metrics like RMSE or MAE (if applicable) were calculated to quantify performance.
Our findings indicate that increasing the input window (i.e., Configuration 2) resulted in satisfactory model performance, even with a reduced number of sequences.

### Getting Started
To run this project:

Clone the repository.

Install the required dependencies listed below.

Preprocess the data as per the steps outlined in Data Preprocessing.

Run the scripts with your preferred configuration.


### Dependencies
Python 3.x

TensorFlow / Keras

Pandas
Matplotlib (for visualization)
