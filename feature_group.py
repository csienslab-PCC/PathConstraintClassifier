

import logging
import numpy as np

from pysmt import fnode
from types import FunctionType

start_index = 64
LEAVES = start_index
TREE_NODES = start_index + 1
DEPTH = start_index + 2
UNIQUE_SYMBOLS = start_index + 3
THEORY = start_index + 4

class FeatureGroup(object):

    def __init__(self, feature_index, get_cache_key, extract, aggregate):

        self.feature_index = feature_index
        self.get_cache_key = get_cache_key
        self.extract = extract
        self.aggregate = aggregate
        return

class FeatureTable(object):

    start_index = 64
    BASIC = [(i, int) for i in range(start_index)]
    LEAVES = (start_index, int)
    TREE_NODES = (start_index + 1, int)
    DEPTH = (start_index + 2, int)
    UNIQUE_SYMBOLS = (start_index + 3, frozenset)

    def __init__(self):

        return

# START OF NON-VARIABLE_FEATURE_GROUP #

non_variable_feature_index = [x[0] for x in FeatureTable.BASIC] + [
    FeatureTable.LEAVES[0], FeatureTable.TREE_NODES[0], FeatureTable.DEPTH[0]
]

methods = []
exclude_method = ['is_literal', 'is_lira_op']
fnode_methods = fnode.FNode.__dict__.items()
for i, j in fnode_methods:
    if type(j) == FunctionType:
        if 'is' in i and i not in exclude_method:
            methods.append(i)

def non_variable_get_cache_key(node):

    if node.is_symbol():
        content = node._content
        cache_key = "symbol:{}-{}".format(content[0], content[2][-1])
    elif node.is_int_constant():
        
        if node.is_one():
            cache_key = "int_constant:1"
        elif node.is_zero():
            cache_key = "int_constant:0"
        else:
            cache_key = "int_constant:others"

    else:
        cache_key = node.node_type()

    return cache_key

def non_variable_extract(node):

#    features = [None for i in non_variable_feature_index]
    features = [int(getattr(node, m)) for m in methods]
    
    features.append(0 if node.args() else 1)
    features.append(1)   
    features.append(1)

    return np.ndarray(features)


def non_variable_aggregate(node_features, child_features):

    features[:66] += child_features[:66]
    if features[FeatureTable.DEPTH[0]] > child_features[FeatureTable.DEPTH[0]]:
        features[FeatureTable.DEPTH[0]] = child_features[FeatureTable.DEPTH[0]]

    return

NonVariableFeatureGroup = FeatureGroup(
    feature_index=non_variable_feature_index,
    get_cache_key=non_variable_get_cache_key,
    extract=non_variable_extract,
    aggregate=non_variable_aggregate
)

# END OF NON-VARIABLE_FEATURE_GROUP #



# START OF VARIABLE_FEATURE_GROUP #

variable_feature_index = [FeatureTable.UNIQUE_SYMBOLS[0]]
initialize_feature = [frozenset()]

def variable_get_cache_key(node):
    
    return node

def variable_extract(node):

    features = [None for i in variable_feature_index]
    features[0] = frozenset([node]) if node.is_symbol() else frozenset()

    return np.ndarray(features)


def variable_aggregate(node_features, child_features):

    node_features[FeatureTable.UNIQUE_SYMBOLS[0]] |= child_features[FeatureTable.UNIQUE_SYMBOLS[0]]
    return node_features

VariableFeatureGroup = FeatureGroup(
    feature_index=variable_feature_index,
    get_cache_key=variable_get_cache_key,
    extract=variable_extract,
    aggregate=variable_aggregate
)

# END OF VARIABLE_FEATURE_GROUP #
