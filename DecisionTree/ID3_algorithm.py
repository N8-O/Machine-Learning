# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:08:53 2021

@author: Nathan
"""

import numpy as np

def get_DT_value(nested_dictionary,test_case):
    # nested_dictionary is nested dictionary
    # test_case is a 
    # make sure nested dictionary is dictionary, if its not then the depth was just 0.
    if type(nested_dictionary) is dict:
        attribute = get_key(nested_dictionary)
        for key, value in nested_dictionary[attribute].items():
            if key == test_case[attribute]:
                if type(value) is dict:
                    ans = get_DT_value(value,test_case)
                    return ans
                else:
                    return value
    else:
        return nested_dictionary
        
def get_key(nested_dictionary):
    return list(nested_dictionary.keys())[0]
    
def Information_Gain(S, attribute_classifiers, labels, label_classifiers, method):
    if method == 'H':
        H_main = Entropy(labels,label_classifiers)
        denom = len(labels)
        # print('denom: ' + str(denom))
        ent_sum = 0
        
        for attribute_classifier in attribute_classifiers:
        # prepare new S, attribute_classifier, and labels for futher decision tree decisions
            new_labels = labels.copy()

            # find where in S, attribute = attribute_classifier, these labels need to be counted
            indices2keep = [i for i, x in enumerate(S) if x == attribute_classifier]
            
            # get rid of labels that are not marked with the correct attribute classifier
            new_labels = [labels[i] for i in indices2keep]
            numer = len(indices2keep)
            ent_sum = (numer/denom)*Entropy(new_labels,label_classifiers)+ent_sum
            
        IG = H_main-ent_sum
        return IG
    
    elif method == 'ME':
        # Call function for Majority Error Calculation
        ME_main = MajorityError(labels,label_classifiers)
        denom = len(labels)
        # print('denom: ' + str(denom))
        ent_sum = 0
        
        for attribute_classifier in attribute_classifiers:
        # prepare new S, attribute_classifier, and labels for futher decision tree decisions
            new_labels = labels.copy()

            # find where in S, attribute = attribute_classifier, these labels need to be counted
            indices2keep = [i for i, x in enumerate(S) if x == attribute_classifier]
            
            # get rid of labels that are not marked with the correct attribute classifier
            new_labels = [labels[i] for i in indices2keep]
            numer = len(indices2keep)
            ent_sum = (numer/denom)*MajorityError(new_labels,label_classifiers)+ent_sum
            
        IG = ME_main-ent_sum
        return IG

    elif method == 'GI':
        # Call function for Gini Index Calculation
        GI_main = GiniIndex(labels,label_classifiers)
        denom = len(labels)
        # print('denom: ' + str(denom))
        ent_sum = 0
        
        for attribute_classifier in attribute_classifiers:
        # prepare new S, attribute_classifier, and labels for futher decision tree decisions
            new_labels = labels.copy()

            # find where in S, attribute = attribute_classifier, these labels need to be counted
            indices2keep = [i for i, x in enumerate(S) if x == attribute_classifier]
            
            # get rid of labels that are not marked with the correct attribute classifier
            new_labels = [labels[i] for i in indices2keep]
            numer = len(indices2keep)
            ent_sum = (numer/denom)*GiniIndex(new_labels,label_classifiers)+ent_sum
            
        IG = GI_main-ent_sum
        return IG

    else:
        raise Exception("Sorry, entered information gain method not understood. \
                        Please enter 'H' for entropy, 'ME' for majority error, \
                        or 'GI' for gini index.")
                        
def Entropy(labels,label_classifiers):
    H = 0
    total_labels = len(labels)
    # print('total_labels: ' + str(total_labels))
    if total_labels == 0:
        return 0
    for label in label_classifiers:
        # of the labels that exist in the remaining data, which one has the most values?
        ratio = len([s for s in labels if s == label])
        ratio = ratio/total_labels
        if ratio == 0:
            H = H+0
        else:
            H = -ratio*np.log(ratio)+H
    return H
    
def MajorityError(labels,label_classifiers):
    ME = 0
    max_label_num = 0
    
    total_labels = len(labels)
    # print('total_labels: ' + str(total_labels))
    if total_labels == 0:
        return 0
    for label in label_classifiers:
            # of the labels that exist in the remaining data, which one has the most values?
            if len([s for s in labels if s == label])>=max_label_num:
                max_label = [s for s in labels if s == label]
                max_label_num_new = len(max_label)             
                max_label_num = max_label_num_new
    
    ME = 1-(max_label_num/total_labels)
    return ME

def GiniIndex(labels,label_classifiers):
    GI = 1
    total_labels = len(labels)
    # print('total_labels: ' + str(total_labels))
    if total_labels == 0:
        return 0
    for label in label_classifiers:
        # of the labels that exist in the remaining data, how much are there of each?
        num_of_labels = [s for s in labels if s == label]
        num_of_labels = len(num_of_labels)             
        GI = GI - ((num_of_labels/total_labels)**2)    
    
    return GI


def ID3(S, attribute_classifiers, labels, label_classifiers, depth, inf_gain='H'):
    # S is a dictionary of attribute lists (including 'labels')
    # attribute_classifiers is a dictionary of attribute label lists
    # labels is a list of all labels for each example
    # label_classifiers is a list of label classifiers

    # for all labels that exist
    for label in label_classifiers:
        # see if the label is the length of the number of labels
        if len(labels) == len([s for s in labels if s == label]):
            # if it is, return that label
            return label
    
    # see if there are no attributes in the attribute list
    if (len(attribute_classifiers) == 0) or (depth == 0):
        # if (len(attribute_classifiers)==0) and (depth >0):
        #     print('no more depth anyways...')
        # if there are not attributes, we are at the end of the node and need to return a label.... figure out which one!
        # initialize max label and label number
        max_label = label_classifiers[0]
        max_label_num = 0
        # for all labels that exist
        for label in label_classifiers:
            # of the labels that exist in the remaining data, which one has the most values?
            if len([s for s in labels if s == label])>=max_label_num:
                max_label = [s for s in labels if s == label]
                max_label_num_new = len(max_label)
                max_label = label
                max_label_num = max_label_num_new
                
        # return the label that has the largest presence among the data
        return max_label
    
    # If all labels are not the same and we still have attributes, keep digging!
    
    # Create new dictionary to return!
    new_dictionary = {}
    
    # See which attribute has the largest information gain
    attribute_information = {} # initialize attribute information list
    for a in attribute_classifiers:
        # calculate information for each attribute and put it in list in order
        attribute_information[a] = Information_Gain(S[a],attribute_classifiers[a],labels,label_classifiers,inf_gain)
    
    # after all information amounts are calculated, find max attribute information gain
    v = list(attribute_information.values())
    k = list(attribute_information.keys())
    max_information_attribute = k[v.index(max(v))] # get value of max info and finds its index 

    new_dictionary[max_information_attribute] = {}
    # create branch in node for each label that the max attribute can take
    for attribute_classifier in attribute_classifiers[max_information_attribute]:
        # prepare new S, attribute_classifier, and labels for futher decision tree decisions
        new_S = S.copy()
        new_attribute_classifiers = attribute_classifiers.copy()
        new_labels = labels.copy()

        # find where in S, attribute = attribute_classifier, those indices need to be put in new branch
        indices2keep = [i for i, x in enumerate(new_S[max_information_attribute]) if x == attribute_classifier]
        # remove max_info_attribute from new_S and new attribute classifier
        new_S.pop(max_information_attribute)
        new_attribute_classifiers.pop(max_information_attribute)
        # for each attribute in new_S
        for a in new_S:
            # update the information in the set for that branch with indicies to keep
            new_S[a] = [new_S[a][i] for i in indices2keep]
        
        # get rid of labels that are not marked with the correct attribute classifier
        new_labels = [new_labels[i] for i in indices2keep]
        
        # if there are no examples in the set corresponding to that value for that attribute
        if len(new_labels) ==0:
            leaf_node = max(set(labels),key=labels.count)
            new_dictionary[max_information_attribute][attribute_classifier] = {leaf_node}
            # print('here')
            # print(new_attribute_classifiers)
        else:
            new_dictionary[max_information_attribute][attribute_classifier] = ID3(new_S,new_attribute_classifiers,new_labels,label_classifiers,depth-1,inf_gain)

    # returned node
    return new_dictionary

def PassDataToTest(Decision_Tree,data,labels):
    test_case = {}
    total_correct = 0
    total_tests = len(labels)
    
    for x in range(total_tests):
        for key in data:
            test_case[key] = data[key][x]
        answer = get_DT_value(Decision_Tree,test_case)
        if answer == labels[x]:
            total_correct = total_correct+1
        
    percent_acc = total_correct/total_tests
    return percent_acc