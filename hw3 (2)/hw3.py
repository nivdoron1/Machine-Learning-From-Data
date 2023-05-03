import numpy as np


class conditional_independence():

    def __init__(self):
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)

        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        # We need to find P(X=x, Y=y) values such that X and Y are not independent
        # We can do this by assuming certain relationships between the variables
        # For example, let's assume a relationship between X and Y: P(Y=1 | X=0) > P(Y=1)
        self.X_Y = {
            (0, 0): None,
            (0, 1): None,
            (1, 0): None,
            (1, 1): None,
        }  # P(X=x, Y=y)

        # Joint probabilities for X=x, C=c
        self.X_C = {
            (0, 0): None,
            (0, 1): None,
            (1, 0): None,
            (1, 1): None,
        }  # P(X=x, C=c)

        # Joint probabilities for Y=y, C=c
        self.Y_C = {
            (0, 0): None,
            (0, 1): None,
            (1, 0): None,
            (1, 1): None,
        }  # P(Y=y, C=c)

        # Conditional probabilities for X=x, Y=y, C=c based on the condition that X and Y are conditionally independent
        # given C
        self.X_Y_C = {
            (0, 0, 0): self.X_C[(0, 0)] * self.Y_C[(0, 0)] / self.C[0],
            (0, 0, 1): self.X_C[(0, 1)] * self.Y_C[(0, 1)] / self.C[1],
            (0, 1, 0): self.X_C[(0, 0)] * self.Y_C[(1, 0)] / self.C[0],
            (0, 1, 1): self.X_C[(0, 1)] * self.Y_C[(1, 1)] / self.C[1],
            (1, 0, 0): self.X_C[(1, 0)] * self.Y_C[(0, 0)] / self.C[0],
            (1, 0, 1): self.X_C[(1, 1)] * self.Y_C[(0, 1)] / self.C[1],
            (1, 1, 0): self.X_C[(1, 0)] * self.Y_C[(1, 0)] / self.C[0],
            (1, 1, 1): self.X_C[(1, 1)] * self.Y_C[(1, 1)] / self.C[1]
        }

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are dependent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        if np.isclose(X[0] * Y[0], X_Y[0, 0]) and np.isclose(X[0] * Y[1], X_Y[0, 1]) and np.isclose(X[1] * Y[0], X_Y[
            1, 0]) and np.isclose(X[1] * Y[1], X_Y[1, 1]):
            return False
        return True

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        return all(np.isclose(X_C[x, c] * Y_C[y, c], X_Y_C[x, y, c]) for x in [0, 1] for y in [0, 1] for c in
                   [0, 1])


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = k * np.log(rate) - rate - np.log(np.math.factorial(k))
    return log_p


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represents the log-likelihood value of rates[i]
    """
    likelihoods = np.zeros(len(rates))

    for i, rate in enumerate(rates):
        likelihoods[i] = np.sum([poisson_log_pmf(k, rate) for k in samples])

    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood
    """
    likelihoods = get_poisson_log_likelihoods(samples, rates)
    max_index = np.argmax(likelihoods)
    return rates[max_index]


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = np.mean(samples)
    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and std for the given x.
    """
    p = None
    p = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    return p


class NaiveNormalClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditional normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value: The class to calculate the parameters for.
        """
        self.class_value = class_value
        self.dataset = dataset
        self.feature_means, self.feature_stds = self._calculate_parameters()

    def _calculate_parameters(self):
        """
        Calculate the mean and standard deviation for each feature of the given class.

        Returns
        - feature_means: A numpy array of the means for each feature in the dataset
        - feature_stds: A numpy array of the standard deviations for each feature in the dataset
        """
        class_data = self.dataset[self.dataset[:, -1] == self.class_value]
        feature_means = np.mean(class_data[:, :-1], axis=0)
        feature_stds = np.std(class_data[:, :-1], axis=0)

        return feature_means, feature_stds

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        num_class_instances = np.sum(self.dataset[:, -1] == self.class_value)
        total_instances = self.dataset.shape[0]
        prior = num_class_instances / total_instances
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        likelihood = np.prod(normal_pdf(x[:-1], self.feature_means, self.feature_stds))
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions.
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability
        for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        pred = 0 if posterior0 > posterior1 else 1
        return pred


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.

    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.

    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    num_correct = 0
    for instance in test_set:
        prediction = map_classifier.predict(instance)
        true_label = instance[-1]
        if prediction == true_label:
            num_correct += 1

    acc = num_correct / test_set.shape[0]
    return acc


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    k = mean.shape[0]
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    if cov_det == 0:
        raise ValueError("The covariance matrix is singular.")

    constant = 1 / (np.power(2 * np.pi, k / 2) * np.sqrt(cov_det))
    x_minus_mean = x - mean
    exponent = -0.5 * np.dot(np.dot(x_minus_mean.T, cov_inv), x_minus_mean)

    return constant * np.exp(exponent)


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.mean, self.cov = self._compute_parameters()

    def _compute_parameters(self):
        # Extract data for the given class value
        class_data = self.dataset[self.dataset[:, -1] == self.class_value]

        # Remove the class label column
        class_data = class_data[:, :-1]

        # Calculate mean and covariance matrix
        mean = np.mean(class_data, axis=0)
        cov = np.cov(class_data, rowvar=False)

        return mean, cov

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        num_class_instances = np.sum(self.dataset[:, -1] == self.class_value)
        total_instances = self.dataset.shape[0]
        prior = num_class_instances / total_instances
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        pred = 0 if posterior0 > posterior1 else 1
        return pred


class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        x_features = x[:-1]  # Exclude the class label
        likelihood_class0 = self.ccd0.get_instance_likelihood(x_features)
        likelihood_class1 = self.ccd1.get_instance_likelihood(x_features)

        if likelihood_class0 > likelihood_class1:
            pred = 0
        else:
            pred = 1
        return pred


EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes
        distribution for a specific class. The probabilites are computed with laplace smoothing.

        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_prior(self):
        """
        Returns the prior porbability of the class
        according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under
        the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return acc
