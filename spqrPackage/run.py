# IMPORTS
from SPQR import SPQR
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as sy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# TEST FOR SINGLE DIMENSION X VARIABLE
np.random.seed(919)
x = np.random.binomial(1, 0.5, 10000)
y = np.random.beta(a=x+2, b=3-x)
np_array = np.asmatrix([y,x])
np_array = np.transpose(np_array)
x = np_array[:,1]
y = np_array[:,0]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

SPQR_Object = SPQR(x_train, y_train)
print(SPQR_Object.model)

x_pred = np.array([0,1])
y_pred = np.array([.1, .5])

cdf = SPQR_Object.predict(x_pred, y_pred, prediction_type="CDF")
print("CDF: ", cdf)
pdf = SPQR_Object.predict(x_pred, y_pred, prediction_type="PDF")
print("PDF: ", pdf)
pdf = SPQR_Object.predict(x_pred[0], y_pred[0], prediction_type="PDF")
print("PDF single element: ", pdf)
qf = SPQR_Object.predict(x_pred[0], tau=[0.25,.75], prediction_type="QF")
print("QF: ", qf)
SPQR_Object.plot_GOF(x_test, y_test)
pdf, sequence = SPQR_Object.plot_PDF(x_pred, 101)
dist0 = sy.beta(2+0, 3-0)
dist0 = dist0.pdf(sequence)
dist1 = sy.beta(2+1, 3-1)
dist1 = dist1.pdf(sequence)
plt.plot(sequence, pdf[0])
plt.plot(sequence,dist0,color='red')
plt.show()
plt.plot(sequence, pdf[1])
plt.plot(sequence,dist1,color='red')
plt.show()

# TEST FOR MULTI DIMENSION X VARIABLE
x1 = np.random.uniform(size=10000)
x2 = np.random.uniform(size=10000)
x = np.array([x1,x2])
x = np.transpose(x)
y = np.random.beta(a=x[:,0]+2, b=3-x[:,1])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.2, random_state=909, shuffle=True
)
# Testing the multi dimensional case with a different model then the base one
class Model(nn.Module):
    def __init__(self, output_features, input_features = 1, h1 = 60, h2 = 45, h3=30):
      super(Model, self).__init__()
      self.fc1 = nn.Linear(input_features, h1)
      self.fc2 = nn.Linear(h1, h2)
      self.fc3 = nn.Linear(h2, h3)
      self.out = nn.Linear(h3, output_features)

    def forward(self, x):
      x = x.view(x.shape[0], -1)
      x = F.sigmoid(self.fc1(x))
      x = F.sigmoid(self.fc2(x))
      x = F.sigmoid(self.fc3(x))
      x = self.out(x)
      x = F.softmax(x, dim=1)

      return x

model = Model(15, 2)

# Testing with different inputs from base ones
SPQR_Object = SPQR(x_train, y_train, nk=15, activation="sigmoid", seed=909, epochs=90, order=3, lr=.005, model=model, x_valid=x_valid, y_valid=y_valid)

x_pred = np.array([[.1,.2],[.2,.2],[.3,.1]])
y_pred = np.array([.1, .25,.5])

cdf = SPQR_Object.predict(x_pred, y_pred, prediction_type="CDF")
print("CDF: ", cdf)
pdf = SPQR_Object.predict(x_pred, y_pred, prediction_type="PDF")
print("PDF: ", pdf)
pdf = SPQR_Object.predict(x_pred[0:1], y_pred[0], prediction_type="PDF")
print("PDF single element: ", pdf)
qf = SPQR_Object.predict(x_pred[0:1], tau=[0.25,.75], prediction_type="QF")
print("QF: ", qf)
SPQR_Object.plot_GOF(x_test, y_test)
