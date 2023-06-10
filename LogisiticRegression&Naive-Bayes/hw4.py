import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        #X = (X - np.mean(X,axis=0)) / ( np.max(X,axis=0) - np.min(X,axis=0))
        #y = (y - np.mean(y,axis=0)) / ( np.max(y,axis=0) - np.min(y,axis=0))
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


        self.theta = np.random.randn(X.shape[1])
        for _ in range(self.n_iter): 
          z = np.dot(X, self.theta.T)  
          h = self._sigmoid(z)
          gradient = np.dot(X.T, (h - y))
          self.theta -= self.eta * gradient  
          self.Js.append(self._cost(X, y, h)) 
          self.thetas.append(self.theta.copy()) 

          if len(self.Js) > 1 and abs(self.Js[-1] - self.Js[-2]) < self.eps:
              break  # Break the loop if the difference in cost values is less than eps

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def _sigmoid(self,z):
        """
        Compute the sigmoid function.

        Parameters
        ----------
        z : array-like
          Linear combination of inputs and parameters.

        Returns
        -------
        sigmoid : array-like
          Computed sigmoid values.
        """
        return 1 / (1 + np.exp(-z))
    
    
    def _cost(self, X, y, h):
      """
      Compute the logistic cost function.

      Parameters
      ----------
      X : array-like, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of examples and
        n_features is the number of features.
      y : array-like, shape = [n_examples]
        Target values.
      h : array-like
        Predicted probabilities.

      Returns
      -------
      cost : float
        Computed logistic cost value.
      """
      m = len(y)  
      cost = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))

      return cost

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        z = np.dot(X, self.theta.T)  
        h = self._sigmoid(z) 
        preds = np.where(h >= 0.5, 1, 0) 
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds
    
    def accuracy(self,X,y):
        return np.count_nonzero(self.predict(X) == y) / y.shape[0]

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = []

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Calculate the number of examples per fold
    fold_size = len(X) // folds

    # Perform cross validation
    for i in range(folds):
        # Split the data into training and validation sets
        start = i * fold_size
        end = (i + 1) * fold_size
        X_train = np.concatenate((X_shuffled[:start], X_shuffled[end:]), axis=0)
        y_train = np.concatenate((y_shuffled[:start], y_shuffled[end:]), axis=0)
        X_val = X_shuffled[start:end]
        y_val = y_shuffled[start:end]

        # Fit the algorithm on the training set
        algo.fit(X_train, y_train)

        # Predict the labels for the validation set
        y_pred = algo.predict(X_val)

        # Calculate accuracy for the current fold
        accuracy = np.mean(y_pred == y_val)
        cv_accuracy.append(accuracy)

    # Calculate the average accuracy across all folds
    cv_accuracy = np.mean(cv_accuracy)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((data - mu)**2) / (2 * sigma**2))
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None
        

    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.costs = []
        
        self.mus = np.zeros(self.k)
        _ = np.random.choice(data.shape[0], size=self.k)
        self.mus = np.random.uniform(np.min(data), np.max(data), self.k)
        
        self.weights = np.ones(self.k) / self.k

        self.responsibilities = np.ones((data.shape[0], self.k)) / self.k
        self.sigmas = np.random.rand(self.k)
        
        
        

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        # Calculate and update the responsibilites numpy array.
        new_responsabilities = self.weights * norm_pdf(data,self.mus,self.sigmas)
        new_responsabilities = new_responsabilities / new_responsabilities.sum(axis = 1, keepdims=True)
        self.responsibilities = new_responsabilities
        
        
    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.mus = np.sum(self.responsibilities * data, axis=0) / np.sum(self.responsibilities, axis = 0)
        self.weights = np.sum(self.responsibilities, axis=0) / data.shape[0]        
        self.sigmas = np.sqrt(np.sum(self.responsibilities * ((data - self.mus) ** 2), axis=0) / np.sum(self.responsibilities, axis = 0))
        
    
     
    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        
        # Initialize the params of the data.
        self.init_params(data)
        
        for i in range(self.n_iter):
          
          # Expectation Step
          self.expectation(data)
          
          # Maximization step
          self.maximization(data)
          
          # Convergence checking
          self.costs.append(self.cost(data))
          if i > 0 and np.abs((self.costs[-2] - self.costs[-1])) < self.eps:
            break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas
    
    def cost(self,data):
      """
      Calculates the cost based on the provided data.

      Parameters:
        self (object): The instance of the class.
        data (numpy.ndarray): The input data.

      Returns:
        float: The calculated cost.
    """
      total_cost = 0
      
      for i in range(data.shape[0]):
        cost_value = 0
        sample = data[i]
        for j in range(self.k):
          cost_value -= np.log2(self.weights[j] * norm_pdf(sample,self.mus[j],self.sigmas[j]))
        total_cost += np.log2(cost_value)
      
      return total_cost
    

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = 0
    
    for weight, mu, sigma in zip(weights, mus, sigmas):
        pdf += weight * norm_pdf(data, mu, sigma)
    
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """      
                
        mus = {}
        weights = {}
        sigmas = {}
        self.classes = {}

        
        self.prior = np.unique(y, return_counts=True)[1] / y.shape[0]
        
        for i in range(len(self.prior)):
          self.classes[i] = {}
          
          for j in range(X.shape[1]):
            self.classes[i]['prior'] = self.prior[i]
            EM_by_label = EM(self.k)
            EM_by_label.fit(X[y.flatten() == i][:, j].reshape(-1,1))
            weights[j], mus[j] , sigmas[j] = EM_by_label.get_dist_params()
          
          self.classes[i]['weights'] , self.classes[i]['mus'] , self.classes[i]['sigmas'] = weights, mus, sigmas

          mus = {}
          weights = {}
          sigmas = {}
          

        
    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        predictions = []
        posteriors = []
        likelihood = 1

        
        for data_point in X:
          class_posteriors = []
          
          for label in range(len(self.classes)):
            
            for feature in range(X.shape[1]):
              
              likelihood = likelihood * (gmm_pdf(data_point[feature], self.classes[label]['weights'][feature], self.classes[label]['mus'][feature],self.classes[label]['sigmas'][feature]))
            
            posterior = likelihood * self.prior[label]
            class_posteriors.append(posterior)
            likelihood = 1 
          
          if (class_posteriors[0] < class_posteriors[1]):
            posteriors.append(1) 
          else:
            posteriors.append(0)   
            
        predictions = posteriors
        
        return np.array(predictions)
      

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = []
    lor_test_acc = []
    bayes_train_acc = []
    bayes_test_acc = []

    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)

    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)

    lor_train_acc_ = np.count_nonzero(lor.predict(x_train) == y_train) / x_train.shape[0]
    lor_train_acc.append( lor_train_acc_)

    lor_test_acc_ = np.count_nonzero(lor.predict(x_test) == y_test) / x_test.shape[0]
    lor_test_acc.append(lor_test_acc_)

    bayes_train_acc_ = np.count_nonzero(naive_bayes.predict(x_train) == y_train) / x_train.shape[0]
    bayes_train_acc.append(bayes_train_acc_)
    
    bayes_test_acc_ = np.count_nonzero(naive_bayes.predict(x_test) == y_test) / x_test.shape[0]
    bayes_test_acc.append(bayes_test_acc_)

  # Plot decision boundaries for each model
    plot_decision_regions(x_train, y_train, lor, title="Logistic Regression Decision Boundary")
    plot_decision_regions(x_train, y_train, naive_bayes, title="Naive Bayes Decision Boundary")

    # Plot cost vs. iteration number for Logistic Regression
    plt.plot(range(len(lor.Js)), lor.Js)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Logistic Regression: Cost by number of Iteration")
    plt.show()

    print("Decision boundary of Logistic Regression: This graph visualizes how the Logistic Regression model has learnt to classify the data. If the decision boundary effectively separates the data points of different classes, it suggests that the model is making good predictions.")
    print("Cost function of Logistic Regression over iterations: This graph shows the learning process of the Logistic Regression model. If the cost decreases rapidly and then plateaus, it indicates that the model has converged to a solution and the error has been minimized.")
    print("Decision boundary of Naive Bayes Gaussian: This graph shows the classification decisions made by the Naive Bayes Gaussian model. If the decision boundary successfully separates data points of different classes, the model is performing well.")
    print("Cost function of Naive Bayes Gaussian over iterations: Similar to the second graph, this shows the learning process of the Naive Bayes model. A rapid decrease followed by a plateau indicates successful learning and minimal error.")
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    plt.show()
