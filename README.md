# Human Activity recognition using UCI HAR

This project focuses on Human Activity Recognition (HAR) using various deep learning models. The goal is to classify human activities based on sensor data. We have implemented and evaluated three different deep learning architectures: a 1D Convolutional Neural Network (CNN), a Long Short-Term Memory (LSTM) network, and a hybrid model combining both CNN and LSTM layers.

## Dataset

The dataset used in this project is the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones). It contains sensor readings from a smartphone's accelerometer and gyroscope, collected while subjects performed six different activities:

*   WALKING
*   WALKING_UPSTAIRS
*   WALKING_DOWNSTAIRS
*   SITTING
*   STANDING
*   LAYING

The dataset is pre-processed and split into training and testing sets.

## Models

We have implemented the following deep learning models for this task:

### 1. 1D Convolutional Neural Network (CNN)

The CNN model is designed to effectively capture the local patterns and features from the sensor data. The architecture consists of several 1D convolutional layers followed by max-pooling layers to extract hierarchical features.

**Accuracy:** 92.8%

### 2. Long Short-Term Memory (LSTM)

The LSTM model is a type of recurrent neural network (RNN) that is well-suited for sequence data. It is designed to learn long-term dependencies in the time-series sensor data.

**Accuracy:** 91%

### 3. Hybrid CNN-LSTM Model

This model combines the strengths of both CNNs and LSTMs. The CNN layers are used to extract spatial features from the sensor data, which are then fed into the LSTM layers to capture temporal dependencies.

**Accuracy:** 96%

## Results

The following table summarizes the performance of the different models on the test set:

| Model              | Accuracy |
| ------------------ | -------- |
| CNN                | 95%      |
| LSTM               | 94%      |
| Hybrid CNN-LSTM    | 96%      |

## How to Run the Code

The project is organized into several Jupyter notebooks:

*   `final_dataAnalysis_HAR.ipynb`: Performs exploratory data analysis on the dataset.
*   `final_HAR_CNN.ipynb`: Implements and evaluates the 1D CNN model.
*   `final_lstm2.ipynb`: Implements and evaluates the LSTM model.
*   `final_hybrid_lstm_cnn.ipynb`: Implements and evaluates the hybrid CNN-LSTM model.

To run the notebooks, you will need to have Python and the following libraries installed:

*   TensorFlow
*   Keras
*   Pandas
*   NumPy
*   Scikit-learn
*   Seaborn
*   Matplotlib

You can install these dependencies using pip:

```bash
pip install tensorflow pandas numpy scikit-learn seaborn matplotlib
```

After installing the dependencies, you can run the Jupyter notebooks to see the data analysis, model training, and evaluation.
.
