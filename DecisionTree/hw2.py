import numpy as np
import matplotlib.pyplot as plt
import math

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:, -1]  
    unique_labels, counts = np.unique(labels, return_counts=True)  
    probabilities = counts / len(labels)  
    gini = 1 - np.sum(probabilities**2)  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.
    Input:
    - data: any dataset where the last column holds the labels.
    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:, -1]  
    unique_labels, counts = np.unique(labels, return_counts=True)  
    probabilities = counts / len(labels)  
    entropy = -np.sum(probabilities * np.log2(probabilities)) 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.
    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    n_samples = len(data)
    impurity = impurity_func(data)

    for value in set(data[:, feature]):
        subset = data[data[:, feature] == value]
        n_subset = len(subset)
        subset_impurity = impurity_func(subset)
        groups[value] = subset
        goodness += (n_subset / n_samples) * (impurity - subset_impurity)

    if gain_ratio:
        split_info = 0
        for value, subset in groups.items():
            p = len(subset) / n_samples
            split_info -= p * math.log2(p)
            
        if split_info > 0:
            goodness /= split_info

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.
        Returns:
        - pred: the prediction of the node
        """
        pred = None
        labels = self.data[:, -1]  # extract class labels from the data
        unique, counts = np.unique(labels, return_counts=True)  # count occurrences of each label
        max_count_idx = np.argmax(counts)  # find the index of the label with the highest count
        pred = unique[max_count_idx]  # use the corresponding label as the prediction
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values
        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):
    """
    Splits the current node according to the impurity_func. This function finds
    the best feature to split according to and create the corresponding children.
    This function should support pruning according to chi and max_depth.
    Input:
    - The impurity function that should be used as the splitting criteria
    This function has no return value
    """
    if self.depth == self.max_depth or len(self.data) == 0:
        return

    if len(np.unique(self.data[:, -1])) == 1:
        self.terminal = True
        return

    best_gain = 0
    best_feature = -1
    for feature in range(self.data.shape[1] - 1):  # Iterate over all features, excluding the target
        current_gain, _ = goodness_of_split(self.data, feature, impurity_func, self.gain_ratio)  # Unpack the tuple

        if current_gain > best_gain:
            best_gain = current_gain
            best_feature = feature

    if best_feature != -1:
        self.feature = best_feature
        unique_values = np.unique(self.data[:, best_feature])

        for value in unique_values:
            child_data = self.data[self.data[:, best_feature] == value]
            child = DecisionNode(child_data, depth=self.depth, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            child.depth=child.depth+1

            if child.depth < self.max_depth:  # Do not call split() if the child has reached max_depth
                child.split(impurity_func)

            self.add_child(child, value)

        # Chi pruning
        if self.chi != 1:
            chi_statistic = 0
            expected = 0
            observed = 0

            for value in unique_values:
                child = self.children[self.children_values.index(value)]
                observed = len(child.data)
                expected = len(self.data) * np.mean(self.data[:, -1] == child.pred)
                chi_statistic += ((observed - expected) ** 2) / expected

            chi_critical = chi_table[len(unique_values) - 1][self.chi]

            if chi_statistic < chi_critical:
                self.children = []
                self.children_values = []
                self.terminal = True



def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning
    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag
    Output: the root node of the tree.
    """
    root = DecisionNode(data, depth=0, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    queue = [root]

    while queue:
        current_node = queue.pop(0)

        # Check if the training examples in the current node are perfectly classified
        if len(np.unique(current_node.data[:, -1])) == 1:
            continue

        current_node.split(impurity)

        for child, _ in zip(current_node.children, current_node.children_values):
            if not child.terminal and child.feature is None:
                queue.append(child)

    return root


def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: a row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    # Traverse the tree until terminal is true
    node = root
    while not node.terminal:
        feature_value = instance[node.feature]
        try:
            child_index = node.children_values.index(feature_value)
            node = node.children[child_index]
        except ValueError:
            # Handle the error, e.g., return a default prediction or the most common class in the node
            return node.pred

    return node.pred


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    correct_predictions = 0
    total_predictions = len(dataset)
    
    for instance in dataset:
        prediction = predict(node, instance)
        true_label = instance[-1]  # The last element in the instance is the true label
        if prediction == true_label:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.
    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []

    best_impurity_function = calc_gini  # Replace this with the best impurity function you found
    gain_ratio_flag = False  # Replace this with the best gain_ratio flag you found

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(X_train, best_impurity_function, gain_ratio=gain_ratio_flag, max_depth=max_depth)
        train_accuracy = calc_accuracy(tree, X_train)
        test_accuracy = calc_accuracy(tree, X_test)
        training.append(train_accuracy)
        testing.append(test_accuracy)

    return training, testing


def chi_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.
    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depths = []

    best_impurity_function = calc_gini  # Replace this with the best impurity function you found
    gain_ratio_flag = False  # Replace this with the best gain_ratio flag you found

    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]

    for chi in chi_values:
        tree = build_tree(X_train, best_impurity_function, gain_ratio=gain_ratio_flag, chi=chi)
        train_accuracy = calc_accuracy(tree, X_train)
        test_accuracy = calc_accuracy(tree, X_test)
        chi_training_acc.append(train_accuracy)
        chi_testing_acc.append(test_accuracy)
        depths.append(tree.depth)

    return chi_training_acc, chi_testing_acc, depths



def count_nodes(node):
    """
    Count the number of nodes in a given tree
    Input:
    - node: a node in the decision tree.
    Output: the number of nodes in the tree.
    """
    if node is None:
        return 0

    n_nodes = 1  # Count the current node

    # Iterate through the children of the current node
    for child in node.children:
        n_nodes += count_nodes(child)  # Recursively count nodes in child subtrees

    return n_nodes
