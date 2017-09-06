# coding=utf-8

import sys

import numpy as np
import sklearn.tree as tree

from util      import *
from lang      import *

### decision trees

#Converts tree to term language
def exp_of_tree(clf, col_names):
    #eprint("classes = %s\n" % str(clf.classes_))
    class_values = clf.classes_
    return tree2Term_rec(clf.tree_, 0, col_names, class_values)

#Converts tree to term language
def tree2Term_rec(t, node_id, col_names, class_values):
    left_child = t.children_left[node_id]
    right_child = t.children_right[node_id]
    #if both childs are leaf nodes and the leaf values are equal, consider the node as the leaf node
    if left_child != right_child and \
       not (t.children_left[left_child] == t.children_right[left_child] and \
            t.children_left[right_child] == t.children_right[right_child] and \
            np.argmax(t.value[left_child][0]) == np.argmax(t.value[right_child][0])):
        #is internal node, which the branching is meaningful
        lt = tree2Term_rec(t, left_child,  col_names, class_values)
        rt = tree2Term_rec(t, right_child, col_names, class_values)
        col = t.feature[node_id]
        cond = ExpBinary(Lib.le,
                         ExpVar(col, col_names[col]),
                         ExpConst(t.threshold[node_id])
                         )
        return ExpCond(cond, lt, rt)
    else:
        #is leaf node or can be considered a leaf node
        #eprint("t.value = %s\n" % str(t.value[node_id]))
        values = t.value[node_id][0]
        temp_index = np.argmax(values)
        temp_score = values[temp_index]

        #equals = filter(lambda v: v == temp_score, values)
        #if len(equals) != 1:
            #eprint("HMM: %s\n" % str(values))
        
        return ExpConst(class_values[temp_index])

### converts random forest to term language
def exp_of_forest(clf, col_names):
    class_values = clf.classes_
    listTerms = [tree2Term_rec(tree.tree_, 0, col_names, class_values) for tree in clf.estimators_]

    #print listTerms
    
    classifierTerm = ExpBinary(
        Lib.ge,
        ExpAssociative(Lib.monoid_add, listTerms),
        ExpConst(len(clf.estimators_)*0.5)
    )
    return classifierTerm

### logistic regression
def exp_of_lr(clf, cols):
    listTerms = [ExpBinary(Lib.mul, ExpConst(val), ExpVar(idx, cols[idx]))
                 for idx, val in enumerate(clf.coef_[0]) if val != 0.0]

    assert(len(listTerms) > 0) # regularization removed all coeffs
    
    classifierTerm = ExpBinary(
        Lib.ge,
        ExpAssociative(Lib.monoid_add, listTerms),
        ExpConst(-clf.intercept_[0])
        #ExpConst(0.0)
    )
    return classifierTerm

### lasso
def exp_of_lasso(clf, cols):
    listTerms = [ExpBinary(Lib.mul, ExpConst(val), ExpVar(idx, cols[idx]))
                 for idx, val in enumerate(clf.coef_) if val != 0.0]

    assert(len(listTerms) > 0) # regularization removed all coeffs
    
    classifierTerm = ExpBinary(
        Lib.ge,
        ExpAssociative(Lib.monoid_add, listTerms),
        #ExpConst(clf.intercept_[0])
        ExpConst(0.0) # ???
    )
    return classifierTerm

### bayesian rule lists
def exp_of_brl(clf, cols):
    if clf.d_star:
        term = None
        for i,j in reversed(list(enumerate(clf.d_star))):
            if clf.itemsets[j] != 'null':
                cond = parse_itemsets(clf.itemsets[j], cols)
                term = ExpCond(cond, ExpConst((clf.theta[i] > 0.5)*1.), term)
            else:
                term = ExpConst((clf.theta[i] > 0.5)*1.)
        return term
    else:
        return None

def parse_itemsets(itemsets, cols):
    itemset_list_list = [parse_itemset_string(itemset_string, cols) 
                         for itemset_string in itemsets]
    #flatten list of list
    itemset_list = [conjunct for itemsets in itemset_list_list for conjunct in itemsets]
    n = len(itemset_list)
    if (n == 0):
        return ExpConst(True)
    elif (n == 1):
        return itemset_list[0]
    else:
        return ExpAssociative(Lib.monoid_and, itemset_list)


def parse_itemset_string(itemset, cols):
    from numpy import inf
    if (itemset == 'null'):
        return []
    f_range = itemset.split(" : ")
    f = f_range[0]
    i = cols.index(f)
    if (f_range[1] == 'All'):
        return []
    range_list = f_range[1].split("_to_")
    range_vals = (float(range_list[0]), float(range_list[1]))
    if (range_vals[0] == -float('inf')):
        return [ExpBinary(Lib.lt, ExpVar(i, f), ExpConst(range_vals[1]))]
    elif (range_vals[1] == float('inf')):
        return [ExpBinary(Lib.ge, ExpVar(i, f), ExpConst(range_vals[0]))]
    else:
        return [ExpBinary(Lib.ge, ExpVar(i, f), ExpConst(range_vals[0])),
                ExpBinary(Lib.lt, ExpVar(i, f), ExpConst(range_vals[1]))]


### naive bayes
#def exp_of_nb(cls, cols):
#    listTerms = [ExpBinary(Lib.mul, ExpConst(val), ExpVar(idx, cols[idx]))
#                 for idx, val in enumerate(clf.coef_) if val != 0.0]
#    print listTerms
#    classifierTerm =
#    return []
