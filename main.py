# José San Martin

# Machine Learning Model Implementations


# All model implementations in the "models" directory. Below I will run them using test data
import models.ConvolutinalNeuralNet as my_cnn
import models.ArtificialNeuralNet as my_ann
from models.ConvolutinalNeuralNet import readArray
from models.SGD import sgd
import matplotlib.pyplot as plt
import pandas as pd



# Convolutional Neural Network


# Setup Data

X = my_cnn.readArray('data/cnn/x')
y = readArray('data/cnn/y').flatten()
T = {'x': X, 'y': y}

# Setup network and plot risk vs epochs
net = my_cnn.Network((2, 16, 16, 2))
loss = my_cnn.Loss()
LT = sgd(net, loss, T, batch_size=20, max_iter=200)
plt.figure()
plt.plot(LT)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('risk', fontsize=18)
plt.show()



# Artificial Neural Network

# Prepare Data

dataset = pd.read_csv('data/ann/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # RowNumber, CustomerId, Surname not included
y = dataset.iloc[:, 13].values.flatten() # Exited is the response variable
T = {'x': X, 'y': y}

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # Encoder for the country variable
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # Encoder for the gender variable
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Split up Countries into dummy variables to remove ordinal ranking
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Remove 1 dummy variable to avoid dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = X_train[:,:2]
T = {'x': X_train, 'y': y_train}




# Setup network and plot risk vs epochs
net = my_ann.Network(relu_layers=2, output_size=2)
net.setWeights(range(len(net.getWeights())))
loss = my_ann.Loss()
LT = sgd(net, loss, T, batch_size=10, max_iter=200)
plt.figure()
plt.plot(LT)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('risk', fontsize=18)
plt.show()







# Logistic Regression