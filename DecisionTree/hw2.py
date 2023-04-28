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
    labels = data[:, -1]  
    _ , counts = np.unique(labels, return_counts=True)  
    probabilities = counts / len(labels)  
    gini = 1 - np.sum(probabilities**2)  
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
    labels = data[:, -1]  
    _ , counts = np.unique(labels, return_counts=True)  
    probabilities = counts / len(labels)  
    entropy = -np.sum(probabilities * np.log2(probabilities)) 
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

        best_goodness_score = 0
        best_feature_idx = -1
        best_split_data = None
        num_features = self.data.shape[1] - 1

        for feature_idx in range(num_features):
            current_goodness_score, temp = goodness_of_split(self.data, feature_idx, impurity_func, self.gain_ratio)
            if best_goodness_score < current_goodness_score:
                best_split_data = temp
                best_goodness_score = current_goodness_score
                best_feature_idx = feature_idx
            self.feature = best_feature_idx

        instance_count = len(self.data)

        if best_split_data is not None:
            chi_split , chi_value = False , 0
            degree_of_freedom = len(best_split_data) - 1
            label_p = np.count_nonzero(self.data[:, -1] == "p") / instance_count
            label_e = np.count_nonzero(self.data[:, -1] == "e") / instance_count

            for _ , data_subset in best_split_data.items():
                df = len(data_subset)
                pf = np.count_nonzero(data_subset[:, -1] == "e")
                nf = np.count_nonzero(data_subset[:, -1] == "p")
                e_label_e , e_label_p = df * label_e , df * label_p
                chi_value += ((pf - e_label_e)**2 / e_label_e) + ((nf - e_label_p)**2 / e_label_p)
            

            if self.chi == 1 or chi_table[degree_of_freedom][self.chi] < chi_value:
                chi_split = True

        if self.feature != -1 and chi_split and self.depth < self.max_depth:
            for feature_value, data_subset in best_split_data.items():
                child_node = DecisionNode(data = data_subset, depth=self.depth + 1,chi=self.chi,max_depth = self.max_depth,gain_ratio=self.gain_ratio)
                self.add_child(child_node,feature_value)



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
    root = DecisionNode(data=data, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    queue = [root]
    while len(queue) > 0:
        current_node = queue.pop(0)
        if np.unique(current_node.data[:,-1]).shape[0] > 1:
            current_node.split(impurity)
            queue += current_node.children
        else:
            current_node.terminal = True

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
    node = root
    while not node.terminal:
        feature_value = instance[node.feature]
        try:
            child_index = node.children_values.index(feature_value)
            node = node.children[child_index]
        except ValueError:
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

    accuracy = correct_predictions / total_predictions
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

    best_impurity_function = calc_entropy  # Replace this with the best impurity function you found
    gain_ratio_flag = True  # Replace this with the best gain_ratio flag you found

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(X_train, best_impurity_function, gain_ratio=gain_ratio_flag, max_depth=max_depth)
        train_accuracy = calc_accuracy(tree, X_train)
        test_accuracy = calc_accuracy(tree, X_test)
        training.append(train_accuracy)
        testing.append(test_accuracy)

    return training, testing


def find_max_depth(node):
    """
    Recursive function to find the maximum depth of a decision tree.
    Input: the root node of the tree
    Output: the maximum depth of the tree
    """
    if node.terminal == True:
        return node.depth
    
    child_depths , max_depth = [] , 0
    for child in node.children:
        child_depths.append(find_max_depth(child))
        max_depth = max(child_depths)    
    return max_depth



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

    best_impurity_function = calc_entropy  
    gain_ratio_flag = True  

    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]


    for chi in chi_values:
        tree = build_tree(X_train, best_impurity_function, gain_ratio_flag, chi)
        train_accuracy = calc_accuracy(tree, X_train)
        test_accuracy = calc_accuracy(tree, X_test)
        chi_training_acc.append(train_accuracy)
        chi_testing_acc.append(test_accuracy)
        depths.append(find_max_depth(tree))

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
    
    if node.terminal:
        return 1

    n_nodes = 1 
    for child in node.children:
        n_nodes += count_nodes(child)  

    return n_nodes
