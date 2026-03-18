#import libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from dms_variants.ispline import Msplines, Isplines
from sklearn.model_selection import train_test_split

class SPQR():
  """The purpose of this class is to implement Semi Parametric Quantile Regression (SPQR) in the Python programming language.
     SPQR is imlemented as a PyTorch custom loss function for a Neural Network.
     """
  def __init__(self, x, y, nk=10, activation="relu", seed=42, epochs=100, order=3, lr=0.01, model=None, x_valid=None, y_valid=None):
      """IMPORTANT NOTES FOR THIS CLASS:
         1) This class is broken up into sections with the start of each new section having a comment with START OF...
         2) Functions and variables use snake_case while classes use PascalCase
         3) A private function uses a '_' as the first character (only the predict function is a public function)
         """
      # Intialize class member variables that don't need a function
      self.x_train = x
      self.y_train = y
      self.x_valid = x_valid
      self.y_valid = y_valid
      self.nk = nk
      self.activation = activation
      self.seed = seed
      self.epochs = epochs
      self.order = order
      self.lr = lr
      self.model = model

      # Error check the memeber variables just intialized as some can be none, incorrect, etc.
      self._split_train_pair()
      self._convert_x_vars_to_torch_tensors()
      self._set_model_if_none_provided()

      # Intialize specific member variables that need a seperate function
      self.seed = torch.manual_seed(self.seed)
      self.mesh = self._create_mesh_sequence()
      self.y_train_msplines = self._create_basis(self.y_train)
      self.y_valid_msplines = self._create_basis(self.y_valid)
      self.optim = self._create_optim()
      self.nll = SPQR.LossSPQR()

      self._fit()

  # START OF functions to error check member variables that didn't need a function to be intialized
  def _split_train_pair(self):
      """If neither validation function is provided, this function splits x_train and y_train.
         This way some of the data can be used for validation
         """
      if(self.x_valid is None and self.y_valid is None):
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
          self.x_train, self.y_train, test_size=0.2, random_state=self.seed, shuffle=True)
      # If one is provided but not the other, throw an error
      elif(self.x_valid is None or self.y_valid is None):
        raise ValueError("x_valid and y_valid must be provided together")

  def _convert_x_vars_to_torch_tensors(self):
      """The x variables are numpy arrays, however they must be torch tensors for the model to work"""
      self.x_train = torch.from_numpy(self.x_train).float()
      self.x_valid = torch.from_numpy(self.x_valid).float()

  def _set_model_if_none_provided(self):
      """This function intializes a base model user can use if none is provided"""
      if(self.model is None):
        self._set_input_features()
        self.output_features = self.nk
        self.model = SPQR.Model(self.output_features, self.input_features)
      else:
        self.input_features = self.model.fc1.in_features

  def _set_input_features(self):
    """Helper function for setting the input features that the base model will use based on dimensions of x_train data"""
    try:
      self.input_features = self.x_train.shape[1]
    except IndexError:
      self.input_features = 1

  # START OF functions used to intialize some member variables
  def _create_mesh_sequence(self):
    """Set mesh (knot sequence from 0, knot 1, ..., knot n, 1)"""
    mesh = np.arange(1/(self.nk-2),1-1/(self.nk-2),1/(self.nk-2))
    mesh = np.append(mesh,1-1/(self.nk-2))
    mesh = np.append(mesh,1)
    mesh = np.append(0, mesh)

    return mesh

  def _create_basis(self, y):
    """The basis/msplines are known and when multipled by each weight, integrated, and then summed togehter.
       The result is the I spline used to model the CDF.
       """
    # Must be performed because y must be a nd.array of dim=1
    y_scale = np.array(y).flatten()
    msplines = Msplines(self.order, self.mesh, y_scale)
    msplines_torch_tensor_array = self._convert_msplines_to_torch_tensors(msplines)
    # Must convert the outside list just created to a torch tensor
    msplines_torch_tensor_array = torch.stack(msplines_torch_tensor_array)
    # The dimensions for the torch tensor up to this point are incorrect, so we transform them
    msplines_torch_tensor_array = msplines_torch_tensor_array.t()

    return msplines_torch_tensor_array

  def _convert_msplines_to_torch_tensors(self, msplines):
    """Helper function to convert msplines to torch tensor"""
    mspline_torch_tensor_array = []
    for i in range(1, msplines.n+1):
      mspline_torch_tensor_array.append(torch.from_numpy(msplines.M(i)))

    return mspline_torch_tensor_array

  def _create_optim(self):
    """Based on what activation type, model, and lr user specifies, this function creates an optimizer"""
    if self.activation == 'relu':
      return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    elif self.activation == 'tanh':
      return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    elif self.activation == 'sigmoid':
      return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    else:
      return ValueError("Invalid activation type, must be relu, tanh, or sigmoid")

  # START OF functions used to fit the model
  def _fit(self):
    """This function is used to fit the intialized SPQR object by training and validating the loss"""
    for i in range(self.epochs):
      # Get loss from the training data
      loss = self._train()

      # Every 10 epochs print the loss and validation loss
      if i % 10 == 0:
          validation_loss = self._validation()
          print(f'Epoch: {i} Loss: {loss} Validation Loss: {validation_loss}')

  def _train(self):
    """The train function uses the model, loss function, x_train data, and y_train_msplines to train the NN"""
    self.model.train()
    # Move forward and get a prediction
    coef_pred = self.model.forward(self.x_train)
    # Measure the loss/error of training
    loss = self.nll(coef_pred, self.y_train_msplines)
    # Perform backpropagation
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

    return loss

  def _validation(self):
    """Validation is similar to training, except it is used to see when the model is overfitting.
       Backpropgation is not needed during validation so only the loss is caculated without effecting the NN
       """
    self.model.eval()
    # Move forward and get a prediction
    coef_pred = self.model.forward(self.x_valid)
    # Measure the loss/error of training
    loss = self.nll(coef_pred, self.y_valid_msplines)

    return loss

  # START OF functions used for prediction
  def predict(self, x, y=None, tau=None, nY=101, prediction_type="QF"):
    """Function to generate values of the CDF, PDF, and QF. For the QF, these values can be used to plot the QF."""
    # Pre process x and y and ensure they are both one dimension
    x = np.atleast_1d(x)
    x = torch.from_numpy(x).float()
    if y is not None:
      y = np.atleast_1d(y)
    self._error_check_prediction_type(prediction_type)

    if prediction_type == "CDF":
      """Generates a cumulative density function"""
      self._error_check_cdf_inputs(x, y)
      cdf = self._get_cdf(x, y)

      return cdf

    if prediction_type == "PDF":
      """Generates a probability density function"""
      self._error_check_pdf_inputs(x, y)
      pdf = self._get_pdf(x, y)

      return pdf

    if prediction_type == "QF":
      """Generates the quantile function. To do this, we must first get the CDF"""
      self._error_check_qf_inputs(y, tau)
      x = self._check_for_scalar_x(x)
      sequence = self._generate_sequence(nY)
      # Get the CDF and use it to find the quantile function
      cdf = self._get_cdf(x, sequence)
      qf = np.interp(tau, cdf, sequence)
      # Check for edge cases of tau == 0 and or tau == 1
      qf = self._check_for_edge_cases_on_quantile(qf, tau)

      return qf


  def _error_check_prediction_type(self, prediction_type):
    """Helper function to ensure prediction_type is valid"""
    if prediction_type not in ["CDF", "PDF", "QF"]:
       raise ValueError("prediction_type must be CDF, PDF, or QF")

  def _error_check_cdf_inputs(self, x, y):
    """Helper function to ensure cdf inputs are correct"""
    if y is None:
      raise ValueError("y must be provided for the CDF function")
    if (len(y) != x.shape[0]):
      raise ValueError("x and y should have the same number of observations. To check this ensure they are both numpy arrays, then run x.shape and y.shape") # RMJ changed

  def _error_check_pdf_inputs(self, x, y):
    """Helper function to ensure pdf inputs are correct"""
    if y is None:
      raise ValueError("Y must be provided for the PDF prediction function")
    if len(y) != x.shape[0]:
      raise ValueError("x and y should have the same number of observations. To check this ensure they are both numpy arrays, then run x.shape and y.shape")

  def _error_check_qf_inputs(self, y, tau):
    """Helper function to ensure qf tau input is correct"""
    if tau is None:
      raise ValueError("tau must be provided for the quantile function")
    for i in range(len(tau)):
         if tau[i] < 0 or tau[i] > 1:
           raise ValueError("tau must be between 0 and 1")
    if y is not None:
      print("WARNING: y does nothing when provided into quantile function")

  def _check_for_scalar_x(self, x):
    """Helper function to ensure qf x input is correct"""
    # Error checking user input for x
    try:
      if x.shape[0] > self.input_features:
        raise ValueError("x must be a single value")
    except TypeError:
      # Inorder to avoid an error due to x being a scalar, we must unsqueeze the value
      x = x.unsqueeze(0)

    return x

  def _error_check_for_pred_weights(self, x):
    """Helper function to ensure correct dimensions of x for model training"""
    try:
      pred_weights = self.model(x)
    except RuntimeError:
      raise ValueError("If receiving a matrix multiplication error, ensure that the dimensions of the x variable are correct. To check this, make the x variable a numpy array. Then, ensure that when running x.shape the dimensions are (n,p), where n is the number of rows and p is the number of columns. Also ensure that the number of columns used to train the model is equal to the number of columns used on the x data for predictions.")
    else:
      return pred_weights

  def _get_cdf(self, x, y):
    """This is a helper function used to get the prediction of the cdf.
       It is similar to the basis function except now we are getting an Ispline.
       """
    isplines = self._create_ispline(y)
    pred_weights = self._error_check_for_pred_weights(x)
    cdf = torch.sum(isplines*pred_weights, dim=1)
    cdf = cdf.detach().numpy()

    return cdf

  def _create_ispline(self, y):
    """Helper function to generate isplines for the get_cdf function.
       It is similar to the basis function except now we are getting an Ispline.
       """
    # Must be performed because y must be a nd.array of dim=1
    y_scale = np.array(y).flatten()
    isplines = Isplines(self.order, self.mesh, y_scale)
    isplines_torch_tensor_array = self._convert_isplines_to_torch_tensors(isplines)
    # Must convert the outside list just created to a torch tensor
    isplines_torch_tensor_array = torch.stack(isplines_torch_tensor_array)
    # The dimensions for the torch tensor up to this point are incorrect, so we transform them
    isplines_torch_tensor_array = isplines_torch_tensor_array.t()
    #Preprocess isplines
    isplines_torch_tensor_array = isplines_torch_tensor_array.float()
    isplines_torch_tensor_array = isplines_torch_tensor_array[:,:self.nk]

    return isplines_torch_tensor_array

  def _convert_isplines_to_torch_tensors(self, isplines):
    """Helper function to convert isplines to torch tensors."""
    isplines_torch_tensor_array = []
    for i in range(1, isplines.n+1):
      isplines_torch_tensor_array.append(torch.from_numpy(isplines.I(i)))

    return isplines_torch_tensor_array

  def _get_pdf(self, x, y):
    """Helper function to get values of PDF"""
    msplines = self._create_basis(y)
    pred_weights = self._error_check_for_pred_weights(x)
    pdf = torch.sum(msplines*pred_weights, dim=1)
    pdf = pdf.detach().numpy()

    return pdf

  def _generate_sequence(self, nY):
    """Helper function to generate a sequence"""
    sequence = list(range(1,nY,1))
    sequence = np.array([item / nY for item in sequence])

    return sequence

  def _check_for_edge_cases_on_quantile(self, qf, tau):
    """Helper function for quantile to make sure edge case of 0 and 1 are exactly 0 and 1"""
    for i in range(len(tau)):
      if tau[i] == 0:
        qf[i] = 0
      elif tau[i] == 1:
        qf[i] = 1

    return qf

  #START OF plotting functions
  def plot_GOF(self, x, y):
    """The goal of this function is to plot the sorted CDF function for determining goodness of fit"""
    x = np.atleast_1d(x) # Ensure x is at least 1d
    x = torch.from_numpy(x).float() # Convert x to torch tensor
    if y is not None:
      y = np.atleast_1d(y) # Ensure y is at least 1d
    self._error_check_cdf_inputs(x,y)
    uniform = np.sort(np.random.uniform(size=len(y)))
    cdf = self._get_cdf(x, y)
    cdf = np.sort(cdf)
    plt.scatter(cdf, uniform)
    plt.plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=2)
    plt.show()

  def _get_pdf_no_y(self, x, sequence):
    """Helper function to get pdf without y used for the plot PDF function"""
    msplines = self._create_basis(sequence)
    pred_weights = self._error_check_for_pred_weights(x)
    pdf = torch.matmul(pred_weights, msplines.t().float())
    pdf = pdf.detach().numpy()

    return pdf

  def plot_PDF(self, x, nY=101):
    """The goal of this function is to plot the PDF function"""
    x = np.atleast_1d(x)
    x = torch.from_numpy(x).float()
    sequence = self._generate_sequence(nY)
    pdf = self._get_pdf_no_y(x, sequence)
    
    return pdf, sequence

  # START OF Helper classes
  class Model(nn.Module):
    """This class is a basic model used for the NN if the user doesn't provide their own"""
    def __init__(self, output_features, input_features = 1, h1 = 30, h2 =20):
      super(SPQR.Model, self).__init__()
      self.fc1 = nn.Linear(input_features, h1)
      self.fc2 = nn.Linear(h1, h2)
      self.out = nn.Linear(h2, output_features)

    def forward(self, x):
      x = x.view(x.shape[0], -1)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.out(x)
      x = F.softmax(x, dim=1)

      return x

  class LossSPQR(nn.Module):
    """This class is used to create a PyTorch custom loss function"""
    def __init__(self):
        super(SPQR.LossSPQR, self).__init__()

    # Forward function uses the SPQR gradient descent equation
    def forward(self, basis, coef):
        y_hat_scaled = torch.sum(basis*coef, dim=1)
        return -torch.sum(torch.log(y_hat_scaled))