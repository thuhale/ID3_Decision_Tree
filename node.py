import sys
class node:
    depth = 0
    isNumeric = True
    candidate = ''

    def __init__(self, p_depth, p_candidate, p_label, p_operator):
        self.isLeaf = True
        self.depth = p_depth
        self.candidate = p_candidate
        self.label = p_label
        self.operator = p_operator
        self.children = []

    def setPositiveAndNegative(self, p_valueList, p_labelList):
        self.classLabels = p_labelList
        self.classValues = p_valueList

    # Create 2 numerical children
    def createNumericalChildren(self, p_childCandidate, p_value):
        self.isNumeric = True
        self.isLeaf = False
        self.children.append(node(self.depth+1, p_childCandidate, p_value, '<='))
        self.children.append(node(self.depth+1, p_childCandidate, p_value, '>'))

    # Create a lit of children based on the label List
    def createRegularChildren(self, p_childCandidate, p_labelList):
        self.isNumeric = False
        self.isLeaf = False
        for value in p_labelList:
            self.children.append(node(self.depth+1, p_childCandidate, value, '='))


    # Print the classification of this Leaf node (positive or negative)
    def printClassification(self):
        if self.classValues[0] >= self.classValues[1]: return self.classLabels[0]
        else: return self.classLabels[1]

    # write the current node to the screen
    def printNode(self):
        if (self.depth == 0): return
        for i in range(0, self.depth-1):
            print ('|\t'),
        print str.format("%s %s %s [%d %d]: " %(self.candidate, self.operator, self.label, self.classValues[0], self.classValues[1])),
        if self.isLeaf:
            print self.printClassification(),
        print ('\n'),

    # write the current node to a file
    def printNodeToFile(self, fo):
        if (self.depth == 0): return
        for i in range(0, self.depth-1):
            fo.write('|\t')
        fo.write(str.format("%s %s %s [%d %d]: " %(self.candidate, self.operator, self.label, self.classValues[0], self.classValues[1])))
        if self.isLeaf:
            fo.write(self.printClassification())
        fo.write('\n')

