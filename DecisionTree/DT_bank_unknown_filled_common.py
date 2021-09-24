# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:59:02 2021

@author: Nathan
"""

# Decision Tree Car Data
import csv
import numpy as np
from collections import Counter

from ID3_algorithm import * 

##### NUMERICAL! 
age_values = ['less','more']
job_values = ["admin.","unemployed","management","housemaid","entrepreneur","student",
              "blue-collar","self-employed","retired","technician","services"] # removed unknown
marital_values = ["married","divorced","single"]
education_values = ["secondary","primary","tertiary"] # removed unknown
default_values = ['yes','no']
##### NUMERICAL!
balance_values = ['less','more']
housing_values = ['yes','no']
loan_values = ['yes','no']
contact_values = ["telephone","cellular"] # removed unknown
##### NUMERICAL!
day_values = ['less','more']
month_values = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
##### NUMERICAL!
duration_values = ['less','more']
##### NUMERICAL!
campaign_values = ['less','more']
##### NUMERICAL!
pdays_values = ['less','more']
##### NUMERICAL!
previous_values = ['less','more']
poutcome_values = ["other","failure","success"] # removed unknown

label_classifiers = ['yes','no']

attribute_classifiers = {}
attribute_classifiers['age'] = age_values
attribute_classifiers['job'] = job_values
attribute_classifiers['marital'] = marital_values
attribute_classifiers['education'] = education_values
attribute_classifiers['default'] = default_values
attribute_classifiers['balance'] = balance_values
attribute_classifiers['housing'] = housing_values
attribute_classifiers['loan'] = loan_values
attribute_classifiers['contact'] = contact_values
attribute_classifiers['day'] = day_values
attribute_classifiers['month'] = month_values
attribute_classifiers['duration'] = duration_values
attribute_classifiers['campaign'] = campaign_values
attribute_classifiers['pdays'] = pdays_values
attribute_classifiers['previous'] = previous_values
attribute_classifiers['poutcome'] = poutcome_values

def getfiledata(file):
    age = []
    job = []
    marital = []
    education = []
    default = []
    balance = []
    housing = []
    loan = []
    contact = []
    day = []
    month = []
    duration = []
    campaign = []
    pdays = []
    previous = []
    poutcome = []
    labels = []
        
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_file:
            terms = line.strip().split(',')
            age.append(int(terms[0]))
            job.append(terms[1])
            marital.append(terms[2])
            education.append(terms[3])
            default.append(terms[4])
            balance.append(int(terms[5]))
            housing.append(terms[6])
            loan.append(terms[7])
            contact.append(terms[8])
            day.append(int(terms[9]))
            month.append(terms[10])
            duration.append(int(terms[11]))
            campaign.append(int(terms[12]))
            pdays.append(int(terms[13]))
            previous.append(int(terms[14]))
            poutcome.append(terms[15])
            labels.append(terms[16])
    
    data = {}
    
    data['age'] = age
    data['job'] = job
    data['marital'] = marital
    data['education'] = education
    data['default'] =  default
    data['balance'] =  balance
    data['housing'] = housing
    data['loan'] = loan
    data['contact'] = contact
    data['day'] = day
    data['month'] = month
    data['duration'] = duration
    data['campaign'] = campaign
    data['pdays'] = pdays
    data['previous'] = previous 
    data['poutcome'] = poutcome
    
    # FIX NUMERICAL DATA
    age_med = np.median(data['age'])
    balance_med = np.median(data['balance'])
    day_med = np.median(data['day'])
    duration_med = np.median(data['duration'])
    campaign_med = np.median(data['campaign'])
    pdays_med = np.median(data['pdays'])
    previous_med = np.median(data['previous'])
    
    data['age'] = ['less' if ele < age_med else 'more' for ele in data['age']]
    data['balance'] = ['less' if ele < balance_med else 'more' for ele in data['balance']]
    data['day'] = ['less' if ele < day_med else 'more' for ele in data['day']]
    data['duration'] = ['less' if ele < duration_med else 'more' for ele in data['duration']]
    data['campaign'] = ['less' if ele < campaign_med else 'more' for ele in data['campaign']]
    data['pdays'] = ['less' if ele < pdays_med else 'more' for ele in data['pdays']]
    data['previous'] = ['less' if ele < previous_med else 'more' for ele in data['previous']]
    
    # FIX UNKNOWN DATA
    most_common_job = Counter(data['job'])
    most_common_education = Counter(data['education'])
    most_common_contact = Counter(data['contact'])
    most_common_poutcome = Counter(data['poutcome'])

    most_common_job = most_common_job.most_common(1)[0][0]
    most_common_education = most_common_education.most_common(1)[0][0]
    most_common_contact = most_common_contact.most_common(1)[0][0]
    most_common_poutcome = most_common_poutcome.most_common(2)[1][0]

    data['job'] = [most_common_job if ele == 'unknown' else ele for ele in data['job']]
    data['education'] = [most_common_education if ele == 'unknown' else ele for ele in data['education']]
    data['contact'] = [most_common_contact if ele == 'unknown' else ele for ele in data['contact']]
    data['poutcome'] = [most_common_poutcome if ele == 'unknown' else ele for ele in data['poutcome']]
    
    return data,labels
    
train_file = "bank/train.csv"
test_file = "bank/test.csv"


data,labels = getfiledata(train_file)
test_data,test_labels = getfiledata(test_file)

inf_gains = ["H","ME","GI"]
for inf_gain in inf_gains:
    for b in range(1,17):
        
        Decision_Tree = ID3(data, attribute_classifiers, labels, label_classifiers, b, inf_gain)
        
        percent_acc = PassDataToTest(Decision_Tree,data,labels)
        print(inf_gain+" depth: "+ str(b) + '\n' + "Training Data Accuracy: "+str(percent_acc))

        percent_acc = PassDataToTest(Decision_Tree,test_data,test_labels)
        print("Testing Data Accuracy: "+str(percent_acc))