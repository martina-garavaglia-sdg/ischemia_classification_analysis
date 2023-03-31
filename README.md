# ischemia_classification_analysis

[![Build Status](https://github.com/martina-garavaglia-sdg/ischemia_classification_analysis.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/martina-garavaglia-sdg/ischemia_classification_analysis.jl/actions/workflows/CI.yml?query=branch%3Amaster)


Affordable hardware and effective data compression means acquiring vast amount of high-resolution data is becoming common practice in modern healthcare pipelines. While data acquisition is gradually becoming effortless, massive, highly-variable data sets are still hard to process. However, deep learning methods are particularly effective when trained on large, high-quality data sets. In particular, such techniques showed tremendous potential in healthcare applications and have been successfully applied to a wide variety of tasks.  Moreover, the nonlinear nature of deep learning algorithms allows them to detect patterns and correlations that may be difficult for humans or traditional machine-learning algorithms to identify, and perform with high accuracy in extremely specialized diagnostic tasks. Despite the promising results, there are still challenges to be addressed in the usage of deep learning in healthcare, such as ensuring the reliability and explainability of the models' output, guaranteeing robustness to noisy inputs, and overcoming regulatory and ethical issues. 

## Dataset
we consider the ECG200 dataset (see [ECG200](http://www.timeseriesclassification.com/description.php?Dataset=ECG200)), a benchmark data set for time series classification. Each series traces the electrical activity recorded during one heartbeat. Time series are labelled as normal or abnormal heartbeats (myocardial ischemia). Importantly, alterations of the hearbeat signal due to ischemia can be extremely varied. This variability and the complexity of the mechanics underlying heart dynamics make the ECG200 dataset suitable for testing novel techniques and architectures such as parametric machines. The ECG200 dataset consists of predetermined train and test set. Both sets are composed by $100$ time series, each comprised of $96$ observations. The difference in ECG between a normal heartbeat and a myocardial ischemia can be better seen by analyzing the average of all measurements. 

## Preprocessing
The ECG200 dataset comes already split into training and test sets. Each set contains $100$ samples of either normal or abnormal heartbeats. The ECG200 samples are bounded and do not present outliers. For this reason, we designed an extremely simple preprocessing pipeline encoding labels as one-hot vectors and adding a channel dimension to the data, as it is customary in deep-learning practice.

## Methods
 we consider two types of architectures: the dense and time machines. In the dense machine case, we choose the sigmoid function as nonlinearity, which constrains output values between zero and one. The model also includes a dense output layer.
For time machine, we divide the global space into six subspaces, each of size 16. Moreover, we use the sigmoid function as nonlinearity and a timeblock of  length 16. The model also includes a convolutional output layer.

## Results
Using dense machines, the results surpass the state of the art achieved by other models with an accuracy of 0.9.
With time machine, the achieved accuracy is 0.91, and once again, above the state of the art.

# Regularization
Regularization is a technique commonly used in deep learning to prevent overfitting. Overfitting occurs when a model is too complex and overspecializes its parameters to perfectly fit the training data. This specialization results in poor generalization to data endowed with different features or following a different distribution than the training ones. Regularization involves adding a penalty term to the loss function of the neural network during training. As an example, the penalty term can be proportional to the squared magnitude of the weights in the network, which encourages the network to learn smaller weights and thus learn smoother solutions.

In our scenario, the regularization term is more sophisticated. The smoothing process is only applied to the temporal dimension, which involves squaring the difference between model weights in two consecutive data points in the time series.
When performing regularization on the time dimension, the goal is to encourage the model to learn smooth patterns over time. By regularizing only on the time dimension, the model is encouraged to learn patterns that are consistent over time, which can improve its ability to predict future outcomes.



