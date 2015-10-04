from __future__ import division
import sys
import re
import pandas as pd
import math
import random

from collections import Counter, defaultdict
from node import node
import numpy as np
import matplotlib.pyplot as plt


#calculate entropy - given a list of probability, entropy = sum(-p*log(p, base=2)) with p >0
def get_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy = entropy - math.log(p,2) * p * 1.0
    return entropy

def data_entropy(df,label):
    labels = df[label].tolist()
    total_count = len(labels)
    probabilities = [count/total_count for count in Counter(labels).values()]
    return get_entropy(probabilities)


def partition_entropy(subsets, label):
    total_count = 0
    partition_entropy = 0
    for subset in subsets:
        total_count = total_count + len(subset)
    for subset in subsets:
        partial_prob = len(subset)/total_count
        partition_entropy = partition_entropy + data_entropy(subset,label) * partial_prob
    return partition_entropy

# Check of an attribute is numerical or discrete
def is_numerical(df, attribute):
    columns = list(df.columns)
    attribute_index = columns.index(attribute)
    row = df.iloc[0,].values
    attr = row[attribute_index]
    try:
        tmp = float(attr)
    except ValueError:
        return False
    return True

# If the attribute is numerical, find the best threshold to split the dataset into two part.
def numerical_attribute_threshold(df, attribute, label):
    if is_numerical(df,attribute):
        sorted = df.sort(attribute)
        sorted.index = range(0,len(sorted) )
        thresholds = []

        string1 = sorted[label].tolist()
        string2 = sorted[attribute].tolist()
        thresholds.append(string2[len(string2)-1])

        string3 = string1
        cnt = Counter(string2)

        for key in cnt.keys():
            if cnt[key] > 1:
                indices = [i for i, x in enumerate(string2) if x == key]
                for index in indices:
                    string3[index] = index

        for i in range(0, len(string2)-1):
            if string3[i] != string3[i+1]:
                threshold = 0.5*(string2[i]+string2[i+1])
                thresholds.append(threshold)

        thresholds.sort(reverse = True)
        best_threshold = thresholds[0]
        conditional_entropy = 1
        for threshold in thresholds:
            left = df[df[attribute] <= threshold]
            left_entropy = data_entropy(left,label)
            right = df[df[attribute] > threshold]
            right_entropy = data_entropy(right,label)
            tmp = (left_entropy * len(left) + right_entropy * len(right))/len(df)
            if tmp < conditional_entropy:
                conditional_entropy = tmp
                best_threshold = threshold
        return best_threshold
    else:
        return

## Partition a dataset into parts by the attribute values.
def partition_by_attr(df, attribute, label, attribute_dict):
    subsets = []
    if is_numerical(df, attribute): # if the attribute is numerical, find the threshold and split
        threshold = numerical_attribute_threshold(df, attribute, label)
        df_left = df[df[attribute] <= threshold]
        df_right = df[df[attribute] > threshold]
        df_left.index = range(0, len(df_left))
        df_right.index = range(0, len(df_right))
        subsets.append(df_left)
        subsets.append(df_right)

    else: ## if the data is discreet, split it according to values
        for item in attribute_dict[attribute]:
            subset = df[df[attribute] == item]
            subset.index = range(0, len(subset))
            subsets.append(subset)
    return subsets

## Calculate the conditional entropy of the data given the attribute
def conditional_entropy(df, attribute, label, attribute_dict):
    subsets = partition_by_attr(df, attribute, label, attribute_dict)
    return partition_entropy(subsets, label)


## Return the best attribute to split the dataset into based on the conditional entropies.
def best_candidate(df, label, attribute_dict):
    candidates = list(df.columns.values)
    candidates.remove(label)

    d_entropy = data_entropy(df,label)
    c_entropy = conditional_entropy(df, candidates[0], label, attribute_dict)
    best_candidate = candidates[0]
    for candidate in candidates:
        tmp = conditional_entropy(df, candidate, label, attribute_dict)

        if tmp < c_entropy:
            c_entropy = tmp
            best_candidate = candidate
    if c_entropy < d_entropy:
        return best_candidate
    else:
        print "no possible candidate"
        return


## Recursively build the tree.
def buildtree(df, label, m, currentNode):
    candidates_list = list(df.columns.values)
    candidates_list.remove(label)
    cnt = Counter(df[label])

    labels = attribute_dict[label]
    val0 = cnt[labels[0]]
    val1 = cnt[labels[1]]

    currentNode.setPositiveAndNegative([val0, val1], labels)
    if val0 == 0 or val1 == 0 or val0 + val1 < m or len(candidates_list) == 0:
        return

    candidate_split = best_candidate(df, label,attribute_dict)
    threshold = numerical_attribute_threshold(df, candidate_split, label)
    if threshold != None:
        currentNode.createNumericalChildren(candidate_split, threshold)
    else:
        currentNode.createRegularChildren(candidate_split, attribute_dict[candidate_split])

    subsets = partition_by_attr(df, candidate_split, label, attribute_dict)
    for index in range(0, len(subsets)):
        buildtree(subsets[index], label, m, currentNode.children[index])


# Print out the full tree into file
def printTree(currentNode, fo):
    currentNode.printNodeToFile(fo)
    for child in currentNode.children:
        printTree(child, fo)

# Generate predictions for the test data, based on previously built decision tree
def generatePredictions(df, label, rootNode, fo):
    #fo.write(" <Predictions for the Test Set Instances>\n")
    correctCount = 0
    rowsCount = 0

    for index, row in df.iterrows():
        rowsCount += 1
        #fo.write("%d: Actual: %s  " %(index+1, row[label]))
        predict_value = treeTraversal(row, rootNode, fo)
        #fo.write("Predicted: %s \n" %predict_value)
        if (predict_value == row[label]):
            correctCount += 1

    #fo.write("Number of correctly classified: %d  Total number of test instances: %d" %(correctCount, rowsCount))
    accuracyRate = correctCount / rowsCount
    fo.write(str.format("Prediction accuracy: %.2f \n" %accuracyRate))
    return accuracyRate

# get the predicted classification for the current test row
def treeTraversal(dfRow, currentNode, fo):
    if currentNode.isLeaf:
        return currentNode.printClassification()

    cur_candidate = currentNode.children[0].candidate
    input_label = dfRow[cur_candidate]

    for child in currentNode.children:
        cur_label = child.label
        cur_operator = child.operator
        if (cur_operator == '=' and cur_label == input_label) \
                or (cur_operator == '<=' and input_label <= cur_label) \
                or (cur_operator == '>' and input_label > cur_label):
            return treeTraversal(dfRow, child, fo)



# reading data in arff file format. This creates a data frame, the last column of the data is label.
def readArffFile(filename):
    f = open(filename, 'rU')
    data = []
    column_names = []
    attribute_dict = {} ## attribute_dict has key to be the name of the attribute,
                        # value is a list of possible values oof that attributes
    for line in f:
        match = re.search("\@", line)
        if match:
            attribute_name = re.search(r'(\@attribute\s)(\')(\w*)(\')(\.*)', line)
            if attribute_name:
                column_names.append(attribute_name.group(3))
                attribute_value = re.search(r'(\@attribute\s)(\')(\w*)(\')(\s*)(\{)(\s*)(.*)(\})',line)
                if attribute_value:
                    possible_values = attribute_value.group(8)
                    list_of_values = possible_values.split(", ")
                    attribute_dict[attribute_name.group(3)] = list_of_values
        else:
            str = line.split(",")
            for i in range(len(str)):
                num = str[i].strip()
                if num[0].isdigit():
                    num = float(num)
                str[i] = num
            data.append(str)
    df = pd.DataFrame(data, columns = column_names)
    f.close()
    return attribute_dict, df


def pickRandomTrainSubset(df, percentage):
    inputSize = len(df.index)
    targetSize = inputSize * percentage // 100
    samplingSet = range(0, inputSize)
    data = []
    for i in range(0, targetSize):
        index = random.choice(samplingSet)
        data.append(df.iloc[index])
        samplingSet.remove(index)
    return pd.DataFrame(data, columns = df.columns.values)

def findMinMaxAvg(valueList):
    if len(valueList)==0: return -1,-1,-1
    max = valueList[0]
    min = max
    totals = 0
    for value in valueList:
        totals += value
        if value > max: max = value
        if value < min: min = value
    return min, max, (totals/len(valueList))

train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
m = 4

attribute_dict, df = readArffFile(train_file_name)
test_attr_dict, testDf = readArffFile(test_file_name)

output_file = format("m%d.tree" %(m))
fo = open(output_file, 'w+')

avgAccuracy = []
minAccuracy = []
maxAccuracy = []
rateList = [5, 10, 20, 50, 100]
for rate in rateList:
    accuracyList = []
    fo.write(str.format("Rate=%d%%: \n" %rate))
    if rate==100: maxIteration=1
    else: maxIteration=10

    for iteration in range(0, maxIteration):
        smallDf = pickRandomTrainSubset(df, rate)
        rootNode = node(0, '', '', '=')
        buildtree(smallDf, "class", m, rootNode)
        #printTree(rootNode, fo)
        accuracyList.append(generatePredictions(testDf, "class", rootNode, fo))
    min, max, avg = findMinMaxAvg(accuracyList)
    avgAccuracy.append(avg)
    minAccuracy.append(min)
    maxAccuracy.append(max)
    fo.write(str.format("min=%.2f , max=%.2f ,  avg=%.2f \n\n" %(min, max, avg)))

# Plot the learning curve from 5% to 100%
x= [0,1,2,3,4]
my_xticks = ['5%','10%','20%','50%','100%']
plt.xticks(x, my_xticks)
plt.plot(x, avgAccuracy)
plt.plot(x, minAccuracy)
plt.plot(x, maxAccuracy)
plt.axis([0, 4, 0, 1])
plt.xlabel('Training set size')
plt.ylabel('Accuracy rate')
plt.legend(['Average accuracy', 'Min accuracy', 'Max accuracy'], loc='upper left')
plt.show()

fo.close()































