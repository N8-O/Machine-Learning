# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:59:02 2021

@author: Nathan
"""

# Decision Tree Car Data
import csv
from ID3_algorithm import * 

buying_values = ['vhigh', 'high', 'med', 'low']
maint_values = ['vhigh', 'high', 'med', 'low']
door_values = ['2', '3', '4', '5more']
persons_values = ['2','4','more']
lug_boot_values = ['small','med','big']
safety_values = ['low','med','high']

label_classifiers = ['unacc','acc','good','vgood']

attribute_classifiers = {}
attribute_classifiers['buying'] = buying_values
attribute_classifiers['maint'] = maint_values
attribute_classifiers['doors'] = door_values
attribute_classifiers['persons'] = persons_values
attribute_classifiers['lug_boot'] = lug_boot_values
attribute_classifiers['safety'] = safety_values

def getfiledata(file):
    data = {}
    buying = []
    maint = []
    doors = []
    persons = []
    lug_boot = []
    safety = []
    labels = []
    
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_file:
            terms = line.strip().split(',')
            buying.append(terms[0])
            maint.append(terms[1])
            doors.append(terms[2])
            persons.append(terms[3])
            lug_boot.append(terms[4])
            safety.append(terms[5])
            labels.append(terms[6])
        
    data['buying'] = buying
    data['maint'] = maint
    data['doors'] = doors
    data['persons'] = persons
    data['lug_boot'] = lug_boot
    data['safety'] = safety
    return data,labels


train_file = "car/train.csv"
test_file = "car/test.csv"


data,labels = getfiledata(train_file)
test_data,test_labels = getfiledata(test_file)

inf_gains = ["H","ME","GI"]
for inf_gain in inf_gains:
    for b in range(1,7):
        
        Decision_Tree = ID3(data, attribute_classifiers, labels, label_classifiers, b, inf_gain)
        
        percent_acc = PassDataToTest(Decision_Tree,data,labels)
        print(inf_gain+" depth: "+ str(b) + '\n' + "Training Data Accuracy: "+str(percent_acc))

        percent_acc = PassDataToTest(Decision_Tree,test_data,test_labels)
        print("Testing Data Accuracy: "+str(percent_acc))