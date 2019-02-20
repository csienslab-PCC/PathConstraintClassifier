

import time
import logging

from pysmt.oracles import SizeOracle, AtomsOracle





class VariableGraph(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)
        
    def __init__(self, graph={}):

        self.atoms_oracle = AtomsOracle()
        self.size_oracle = SizeOracle()
        self.graph = graph
        return

    def _add_edge(self, node_a, node_b):
        
        if node_a not in self.graph:
            self.graph[node_a] = set()
        if node_b not in self.graph:
            self.graph[node_b] = set()

        self.graph[node_a].add(node_b)
        self.graph[node_b].add(node_a)
        return
        
    def _add_node(self, node):
        if node not in self.graph:
            self.graph[node] = set()
        return

    def build_graph(self, tree, fast=False):

        self.theory = self.atoms_oracle.walk(tree)

        self.size_oracle.set_walking_measure(SizeOracle.MEASURE_SYMBOLS)
        relation_list = [
            [str(x) for x in self.size_oracle.walk(t, measure=SizeOracle.MEASURE_SYMBOLS)] for t in self.theory
        ]

        self.total_variable_degree = 0
        self.graph = {}
        for relation in relation_list:
            for node_a in relation:
                if node_a not in self.graph:
                    self._add_node(node_a)
                if len(relation) == 1:
                    continue
                for node_b in relation:
                    if node_a != node_b:
                        self._add_edge(node_a, node_b)
                if fast:
                    break

        """
        for relation in relation_list:
            for node in relation:
                if node not in self.graph:
                    self._add_node(node)
                
                if node != relation[0]:
                    self._add_edge(node, relation[0])
        """

        return 

    def _get_clause(self, node, graph):

        if graph[node] == None:
            return []

        clause = [node]
        adjacent_nodes = [x for x in graph[node]]
        graph[node] = None

        current = 0
        iter_list = [adjacent_nodes]
        while(current < len(iter_list)):

            adjacent_nodes = iter_list[current]
            for node in adjacent_nodes:
                if graph[node] != None:
                    clause.append(node)
                    iter_list.append([x for x in graph[node] if x != None])
                    graph[node] = None
            current += 1

        return clause

    def get_clause(self):

        graph = self.graph.copy()
        clause_list = []

        for node in graph:
            if graph[node] != None:
                clause_list.append(self._get_clause(node, graph))

        return clause_list

    
