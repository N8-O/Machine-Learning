This is a Decision Tree made in Homework 1.

The following functions are found in this library.

get_DT_value(nested_dictionary,test_case)
Information_Gain(S, attribute_classifiers, labels, label_classifiers, weights, method)
ID3(S, attribute_classifiers, labels, label_classifiers, depth, weights, inf_gain='H')

PassDataToTestwithWeight(Decision_Tree,data,labels,weights)
PassDataToTest(Decision_Tree,data,labels)
ReturnDictionaryOutput(Decision_Tree,data,labels)

PassData2Adaboost(alphas,Hs,data,labels)
AdaboostHypothesis(alphas,Hs,data,labels)
PassData2BaggedDecisionTrees(Hs,data,labels)
BaggedDecisionTreesHypothesis(Hs,data,labels)



To create a decision tree call the ID3 function and the decision tree will be returned to you as a dictionary of dictionaries.
S-  is the data in dictionary form where the keys of the dictionary are the attributes of the data with associated lists of the attribute classifier.

attribute_classifiers-  are the possible classifiers that can be found in the list of the attribute key in the data dictionary. This is also a dictionary where the keys are the attributes are the keys which contain a list of the possible classifiers

labels- A list of the true labels of each piece of data

label_classifiers- A list of the possible label values

depth- How deep should the tree go?

weights- Weight vector, if uniform weight is desired just enter 1. Otherwise enter a vector of len(labels) where the ith value is the ith weight of each individial data example.

inf_gain is the method of information gain used. (Entropy-'H', Gini Index-'GI', Majority Error-'ME')
