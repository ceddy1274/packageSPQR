
# SPQR
In this project, we use semi-parametric quantile regression (SPQR) as a Python package used to model a probability density function (PDF) with known M-splines and weights estimated by a neural network. This package contains four callable functions: a fit function that is called automatically when an SPQR object is initialized, a predict function to predict the PDF, the cumulative density function (CDF), or the quantile function (QF), and two functions used for viewing the PDF or the goodness-of-fit (GOF) function.
Below are two examples of how to use SPQR. The first example is for when the x variable has a single dimension. The second example is for when x has more than one dimension. This package is based off of methods in
Xu and Reich (2021) [[1]](#1). This package also reproduces functionality from the SPQR R package found here https://github.com/stevengxu/SPQR.

## How to use
To begin, first download and import the packages.
```python
# DOWNLOADS
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install dms_variants
!pip install -i https://test.pypi.org/simple/ SPQR==1.5

# IMPORTS
from SPQR import SPQR
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as sy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
```
When running the package, it is minimally required to have an x_train variable and a y_train variable. Both are numpy arrays where x represents to covariates and y represents the response vectors. In the first example x is a Bernoulli trial performed 10,000 times and y is a Beta distribution with the alpha parameter being x+2 and the beta parameter being 3-x. After creating the data, we need to put the data into a matrix inorder to ensure it is formatted correctly for the train_test_split function. Once that is completed, we have the minimum requirements needed to create an SPQR object and train our model.

```python
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
```
![Loss plot](https://raw.githubusercontent.com/ceddy1274/packageSPQR/main/loss_plot.png)<br />

The above image are the results from creating an SPQR object. During intialization training happens automatically. We can determine the model is performing better every 10 epochs because the  loss from training is decreasing. Validation is used alongside training to ensure the model is not overfitting. We can tell the model is overfitting when validation starts to rise while training is still decreasing. When the model has completed training and validation, we can began using the predict functions. To do this, we need to create x_pred and y_pred variables. The x_pred variable represents the Bernoulli trial which could be either 0 or 1. The y_pred variable is a variable for which quantile to evaulate at. For example, the y_pred variable will evaulate the PDF and CDF at the .1 and .5 quantile.

```python
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

tau = [0.05, .15, .25, .35, .45, .55, .65, .75, .85, .95]
qf = SPQR_Object.predict(x_pred[1], tau=tau, prediction_type="QF")
plt.plot(tau, qf)
```
![Results single dimension](https://raw.githubusercontent.com/ceddy1274/packageSPQR/main/results_single_dimension.png)<br />
![Results single dimension 2](https://raw.githubusercontent.com/ceddy1274/packageSPQR/main/results_single_dimension2.png)<br />
![Results single dimension 3](https://raw.githubusercontent.com/ceddy1274/packageSPQR/main/results_single_dimension3.png)<br />
![Results single dimension 4](https://raw.githubusercontent.com/ceddy1274/packageSPQR/main/results_single_dimension4.png)<br />

We have displayed 4 examples of using the predict function and 3 examples using plots. First we use the predict function to caculate the CDF. Then we use the predict function to caculate the PDF function. We also provide an example of caculate a single PDF element. The last example uses the QF at the .25 and .75 quantiles. After using the predict functions, we use the plot_GOF function with the two test variables to see how well the model was trained. The next plot is for when we want to plot the PDF. We plot the PDF at x=0 and x=1 with the original Beta distribution displayed over it. The final plot is for the QF where tau is all of the percentiles which we want to display on the y axis. We need to gather the QF using the predict function. The result will be what we display on the x axis. 

One thing we should note is that the last example was a very basic case of how to use the SPQR package. A more realistic case would be a multi dimensional case with more user provided variables. For this example, we will focus less on the data generating process and functions and more on minor details. For example let's display the old basic model used in the last example using this line of code.

```python
print(SPQR_Object.model)
```
![Model](https://raw.githubusercontent.com/ceddy1274/packageSPQR/main/model.png)<br />

We can see from the results that this is a simple 3 layer model with a 30 neuron hidden layer, 20 neuron hidden layer, and 10 neuron output layer. Most users will likely want to use their own model. To create one a user could run this code.

```python
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
```
This is a more complex model with 3 hidden layers and 1 output layer. Now a user can create a new SPQR object with more custom parameters.

```python
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

# Testing with different inputs from base ones
SPQR_Object = SPQR(x_train, y_train, nk=15, activation="sigmoid", seed=919, epochs=90, order=3, lr=.005, model=model, x_valid=x_valid, y_valid=y_valid)
```
Now we have trained and validated a new SPQR object, we can use the prediction functions along with the GOF plot to ensure that the model was trained well.

```python
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
SPQR_Object.plot_CDF(x_test, y_test)
```
Through both these examples, we can see how a user can utilize the SPQR package. By abstracting away lots of the details, this package can use the known M-splines and the covariates to predict weights with the help of a custom loss function. Along with caculating predictions, the plot functions provided ensure the neural network is doing a good job of predicting the weights. 

## References
<a id="1">[1]</a> Xu, S.G. and Reich, B.J., 2021. Bayesian
nonparametric quantile process regression and estimation of marginal
quantile effects. Biometrics.





