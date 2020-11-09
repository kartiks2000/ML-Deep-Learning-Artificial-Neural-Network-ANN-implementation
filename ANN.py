# ARTIFICIAL NEURAL NETWORK (ANN)


# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
# OR   pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# If something doesnt work try to ugrade pip if u using pip
# pip install --upgrade pip


# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf


# Tensorflow and Theano both can work on both CPU aswell as GPU.
# Tenserflow and Theano are Numerical Computation Liberary, but these are used for development and research. Hence we need to write a lot of code to implement. Thats why we will use Keras liberary.
# keras is a liberary which is emplemented using tenserflow and theano, it makes implementation of NN very easy.


# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Seperating Independent and dependent variables.
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values



# Encoding categorical data
# We only need to encode feature matrix X column "Country" and "Gender"
# We dont need to Encode the y vector because it is already in binary(0 or 1)
from sklearn.preprocessing import LabelEncoder
# Label Encoding the "Country" column in the features matrix/independent vairiable.
le1 = LabelEncoder()
X[:, 1] = le1.fit_transform(X[:, 1])
# Label Encoding the "Gender" column in the features matrix/independent vairiable.
le2 = LabelEncoder()
X[:, 2] = le2.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" (Country) column
# now emplimenting Dummy Encoding so that ML algorithm model do not attribute an order in to the categorical vairiable so that we can have the correct computation while modeling dataset
# We dont need to make dummy vairiable for "Gender" Column Basically to avoid Dummy Variable trap as it would be unecessary as already there are two choices 0 and 1, and we knoe when we dummy encode we remove any one of the choice for correct calculations.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Basically to avoid Dummy Variable trap we need to eliminate one dummy vairiable used for "Country" column.
# So thats why we will remove first dummy column at index 0.
X = X[:,1:]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




# Part 2 - Building the ANN


# Importing the Keras libraries and packages
# keras uses tenserflow backend
import keras
# Sequential package is used to initialize nueral network
from keras.models import Sequential
# Dense used to make layers in our ANN
from keras.layers import Dense


# Initializing the ANN
ann = tf.keras.models.Sequential()

# Defining the layers of our ANN
# Adding the input layer and the first hidden layer
# We need to pass the number od nodes in this layer and the activation function which we want to use.
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
# We need to pass the number od nodes in this layer and the activation function which we want to use.
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
# We need to pass the number od nodes in this layer and the activation function which we want to use.
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))





# Part 3 - Training the ANN

# Compiling the ANN
# we need to pass the "optimizer", it is the algorithm we want to use to find the opimal values of weight.
# we need to pass the "loss", which is actually the LOSS function we want to use.
# we need to pass the "metrics" it is just a crieterian we use to evaluate our model. 
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
# we need to pass our training set.
# we need to pass the batch size, the number of observations after which the back-propagation (Updation of Weights) will take place.
# we need to pass the number of Epochs, Epochs are the number of interation of the same dataset to be trained again and again on this ANN.
# we pass epochs so that the ANN train again and again to get better and betetr on each training(iteration) to predict accurate results.
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)




# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)
# Changing result into boolean
y_pred = (y_pred > 0.5)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





