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

## Regularization
Regularization is a technique commonly used in deep learning to prevent overfitting. Overfitting occurs when a model is too complex and overspecializes its parameters to perfectly fit the training data. This specialization results in poor generalization to data endowed with different features or following a different distribution than the training ones. Regularization involves adding a penalty term to the loss function of the neural network during training. As an example, the penalty term can be proportional to the squared magnitude of the weights in the network, which encourages the network to learn smaller weights and thus learn smoother solutions.

In our scenario, the regularization term is more sophisticated. The smoothing process is only applied to the temporal dimension, which involves squaring the difference between model weights in two consecutive data points in the time series.
When performing regularization on the time dimension, the goal is to encourage the model to learn smooth patterns over time. By regularizing only on the time dimension, the model is encouraged to learn patterns that are consistent over time, which can improve its ability to predict future outcomes.

## Explainability
The word explainability refers to the ability to understand and interpret the decision-making process of a machine-learning model in an input-dependent fashion. In this sense, our aim is to devise a technique that could effectively communicate the workings of deep learning models to individuals who lack expertise in the field, by utilizing sensitivity maps.

## Sensitivity maps
Each machine consists of a non-sequential juxtaposition of linear and nonlinear components. We use the sigmoid function to illustrate the construction of explainability maps for parametric machines that we shall call \textit{sensitivity maps}. However, the following construction holds for any pointwise nonlinear function. For the sake of intuition, we can think about the sigmoid function (i.e. the activation function we utilize in~\cref{classification}) as a piecewise linear function defined on three intervals: first, the function is nearly flat and tends towards zero. Then, the function has a positive slope, and hence positive derivative. Finally, the function returns to be constant and of value one. The input data traverse this function, crossing through these three sections. The data that pass through the outermost sections will not contribute to the output because they have almost zero derivative. Instead, points mapped to regions of positive slope contribute actively to the model's output. Practically, we compute the derivative of the nonlinear function with respect to the machine's output before the nonlinearity is applied to the input. We call this construction a sensitivity map.

In time machines, the sensitivity map provides insights into individual time series by identifying the specific time points and learning depths where the model is more sensitive to the signal.

## Uncertainty measure
The sensitivity map provides us with a general indication of how the model is learning from our observations and what the critical points are for each individual. 
Although very informative for an expert, it cannot be used in a medical context, so it is necessary to develop a strategy to make these interpretations user-friendly.
The idea is to initially develop an algorithm that reduces the dimensionality of the sensitivity matrices. This will be followed by reducing the cardinality of the data and analyzing it using a graph-based approach. The aim is to cluster the training observations based on their loss function and gain insights into their distribution. To achieve this goal, we utilize UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction, and Mapper for cardinality reduction.

In our case, we reduce the dimensionality of the data using the sensitivity matrices generated from the training data, selecting 40 components for the reduction process. For the Mapper algorithm, we use a simple x-axis projection as the filter function, and used the k-means clustering algorithm with $k=3$ to identify clusters. The two observed clusters divide the observations based on the degree of loss, providing an indication of their reliability in terms of classification. Specifically, the cluster with lower loss is considered more reliable, while the cluster with greater loss is less reliable. The next step is to identify, given a new observation, which of these two clusters it will belong to, so as to be able to say something about the degree of uncertainty of the classification.


