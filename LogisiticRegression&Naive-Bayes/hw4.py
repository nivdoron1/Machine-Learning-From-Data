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
        X = (X - np.mean(X,axis=0)) / ( np.max(X,axis=0) - np.min(X,axis=0))
        y = (y - np.mean(y,axis=0)) / ( np.max(y,axis=0) - np.min(y,axis=0))
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
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((data - mu)**2) / (2 * sigma**2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of Gaussians in each dimension.
    n_iter : int
      Passes over the training dataset in the EM process.
    eps: float
      Minimal change in the cost to declare convergence.
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.response = {}
        self.weights =[]
        self.mus =[]
        self.sigmas = []
        self.J_history = []
        self.costs = []

    def init_params(self, data):
        """
        Initialize distribution params.
        """
        # self.means = []
        # self.stds = []
        # self.response = {}
        n_samples, n_features = data.shape

        self.weights = np.ones(self.k) / self.k
        self.mus = np.zeros(self.k)
        self.sigmas = np.ones(self.k)

        for i in range(self.k):
            random_indices = np.random.choice(range(len(data)), size=self.k, replace=False)
            self.mus[i] = data[random_indices[i]]
        
        # Initialize response dictionary
        self.response = {}


    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities.
        """
        n_samples = data.shape[0]
        n_features = data.shape[1]
        
        self.response = {}
        
        for i in range(self.k):
            # Calculate the likelihood of each sample for the current Gaussian component
            likelihood = norm_pdf(data, self.mus[i], self.sigmas[i])
            
            # Multiply the likelihood by the corresponding weight
            weighted_likelihood = likelihood * self.weights[i]
            
            # Store the weighted likelihood as the responsibility for the current Gaussian component
            self.response[i] = weighted_likelihood
    
        # Normalize the responsibilities
        for i in range(n_samples):
            sample_sum = sum([self.response[j][i] for j in range(self.k)])
            for j in range(self.k):
                self.response[j][i] /= sample_sum

    def maximization(self, data):
        """
        M step - updating distribution params
        """
        n_samples = data.shape[0]
        
        # Update weights
        for i in range(self.k):
            self.weights[i] = np.mean(self.response[i])
        
        # Normalize weights
        total_weight = np.sum(self.weights)
        self.weights /= total_weight        
        # Update means
        for i in range(self.k):
            resp = self.response[i].reshape(-1, 1)
            weighted_sum = np.sum(resp * data, axis=0)
            self.mus[i] = weighted_sum / np.sum(resp)
        
        # Update covariances (variances)
        for i in range(self.k):
            resp = self.response[i].reshape(-1, 1)
            weighted_diff = data - self.mus[i]
            weighted_sum = np.sum(resp * (weighted_diff ** 2), axis=0)
            self.sigmas[i] = np.sqrt(weighted_sum / np.sum(resp))


    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization functions to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        self.J_history = []
        
        for _ in range(self.n_iter):
            # E-step
            self.expectation(data)
            
            # M-step
            self.maximization(data)
            
            # Calculate the current cost (negative log-likelihood)
            cost = self.calculate_cost(data)
            
            # Store the cost in the history
            self.J_history.append(cost)
            
            # Check for convergence
            if len(self.J_history) > 1 and np.abs(self.J_history[-1] - self.J_history[-2]) < self.eps:
                break
                
    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas
    
    def calculate_cost(self, data):
        """
        Calculate the current cost (negative log-likelihood).
        """
        cost = 0.0
        n_samples = data.shape[0]
        
        for i in range(n_samples):
            sample_likelihood = 0.0
            for j in range(self.k):
                likelihood = norm_pdf(data[i], self.mus[j], self.sigmas[j])
                sample_likelihood += self.weights[j] * likelihood
            cost += -np.log(sample_likelihood)
        
        return cost



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
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_components = len(weights)
    pdf = np.zeros_like(data)

    for i in range(n_components):
        component_pdf = weights[i] * norm_pdf(data, mus[i], sigmas[i])
        pdf += component_pdf

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.classes = np.unique(y)
        self.priors = np.zeros(len(self.classes))
        self.gmms = []

        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.priors[i] = len(X_cls) / len(X)

            em_obj = EM(k=self.k)
            em_obj.fit(X_cls.reshape(-1, 1))
            self.gmms.append(em_obj)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

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
        preds = []
        for x in X:
            likelihoods = []

            for i, cls in enumerate(self.classes):
                gmm = self.gmms[i]
                likelihood = gmm_pdf(x, weights=gmm.get_dist_params()[0],mus=gmm.get_dist_params()[1],sigmas=gmm.get_dist_params()[2])
                likelihoods.append(likelihood)

            likelihoods = np.array(likelihoods)
            posterior_probs = self.priors * likelihoods
            prediction = self.classes[np.argmax(posterior_probs, axis=0)]
            preds.append(prediction)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

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

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################


    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    # Fit Logistic Regression model
    lr_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lr_model.fit(x_train, y_train)


    # Fit Naive Bayes model
    nb_model = NaiveBayesGaussian(k=k)
    nb_model.fit(x_train, y_train)
    result_dict = {}
    list_models = [lr_model]
    # Evaluate models
    for model in list_models:
        model_train_acc = model.accuracy(x_train,y_train)
        model_test_acc = model.accuracy(x_train,y_train)
        #print(f"{model.__class.__name__} training accuracy: {train_preds}, test accuracy: {test_preds}")
        plot_decision_regions(X=x_train, y=y_train, classifier=model,
                              title= " Decision Boundaries")
        #result_dict[f'{model.__class.__name__.lower()}_train_acc'] = model_train_acc
        #result_dict[f'{model.__class.__name__.lower()}_test_acc'] = model_test_acc

        plt.plot(np.arange(len(lr_model.Js)), lr_model.Js)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss as a function of iterations')
        plt.show()

        if model == list_models[0]:
            lor_train_acc = model_train_acc
            lor_test_acc = model_test_acc
        else:
            bayes_train_acc = model_train_acc
            bayes_test_acc = model_test_acc
    print("Decision boundary of Logistic Regression: This graph visualizes how the Logistic Regression model has learnt to classify the data. If the decision boundary effectively separates the data points of different classes, it suggests that the model is making good predictions.")
    print("Cost function of Logistic Regression over iterations: This graph shows the learning process of the Logistic Regression model. If the cost decreases rapidly and then plateaus, it indicates that the model has converged to a solution and the error has been minimized.")

    print("Decision boundary of Naive Bayes Gaussian: This graph shows the classification decisions made by the Naive Bayes Gaussian model. If the decision boundary successfully separates data points of different classes, the model is performing well.

    print("Cost function of Naive Bayes Gaussian over iterations: Similar to the second graph, this shows the learning process of the Naive Bayes model. A rapid decrease followed by a plateau indicates successful learning and minimal error.")
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }

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
