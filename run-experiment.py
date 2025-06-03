# -*- coding: utf-8 -*-
"""
Hypergraph Influence Maximization Algorithms

This module implements various algorithms for influence maximization on hypergraphs,
including Neighbor Priority (NP), Hyper Degree (HHD), HCI1, HCI2, PageRank, Random,
GA, NSGA-II, NSGA-II-Init, and MOEA-IMF. The goal is to select optimal seed nodes to
maximize influence spread while minimizing cost.

Author: ASUS
Created: 2024-09-30
License: MIT
Dependencies: deap, numpy, pandas, matplotlib, hypernetx, tqdm, multiprocessing
"""

import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import queue
import math
import hypernetx as hnx
import hypernetx.algorithms.generative_models as gm
from collections import namedtuple
import time
import os
import multiprocessing as mp
from tqdm import tqdm
import warnings

class NodeStruct:
    """Represents a node in a hypergraph.

    Attributes:
        edge_list (list): List of hyperedges connected to this node.
        state (int): State of the node (0: inactive, 1: active).
        degree (int): Degree of the node (number of connected hyperedges).
        mark (int): Flag indicating if the node has been queued (0: not queued, 1: queued).
        id (int): Unique identifier for the node (used for hypergraph correction).
    """
    def __init__(self):
        self.edge_list = []
        self.state = 0
        self.degree = 0
        self.mark = 0
        self.id = 0

class EdgeStruct:
    """Represents a hyperedge in a hypergraph.

    Attributes:
        node_list (list): List of nodes contained in this hyperedge.
        state (int): State of the hyperedge (0: inactive, 1: active).
        cardinality (int): Number of nodes in the hyperedge.
    """
    def __init__(self):
        self.node_list = []
        self.state = 0
        self.cardinality = 0

def construct_hypergraph(filename: str, node_size: int, edge_size: int) -> tuple:
    """Constructs a hypergraph from a file.

    Reads hypergraph data from a file, where each line represents a hyperedge with
    the first number as the edge ID and subsequent numbers as node IDs.

    Args:
        filename (str): Path to the file containing hypergraph data.
        node_size (int): Expected number of nodes in the hypergraph.
        edge_size (int): Expected number of hyperedges in the hypergraph.

    Returns:
        tuple: Contains:
            - Node_array (list): List of NodeStruct objects representing nodes.
            - nodenum (int): Number of nodes with non-zero degree.
            - Edge_array (list): List of EdgeStruct objects representing hyperedges.
            - edgenum (int): Number of hyperedges.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file contains invalid data (e.g., non-integer values).
    """
    Node_array = [NodeStruct() for _ in range(node_size)]
    Edge_array = [EdgeStruct() for _ in range(edge_size)]
    result = []
    edgenum = 0
    nodenum = 0

    with open(filename, "r") as file:
        for line in file:
            numbers = [int(num) for num in line.strip().split()]
            result.append(numbers)

    for line_list in result:
        edgenum += 1
        Edge_array[line_list[0]].node_list.extend(line_list[1:])
        Edge_array[line_list[0]].cardinality = len(line_list) - 1
        for t in line_list[1:]:
            Node_array[t].edge_list.append(line_list[0])
            Node_array[t].degree += 1

    for i in range(node_size):
        if Node_array[i].degree != 0:
            nodenum += 1

    return Node_array, nodenum, Edge_array, edgenum

def spread(S: queue.Queue, p1: float, p2: float, Node_array: list, Edge_array: list, identical_p: bool) -> tuple:
    """Simulates influence spread in a hypergraph.

    Propagates influence from a seed set through nodes and hyperedges based on
    activation probabilities. Supports both constant and degree-based probabilities.

    Args:
        S (queue.Queue): Queue of seed nodes to start the spread.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.

    Returns:
        tuple: Contains:
            - active_node_num (int): Number of activated nodes.
            - active_edge_num (int): Number of activated hyperedges.
    """
    active_node_num = 0
    active_edge_num = 0
    if identical_p:
        while not S.empty():
            seed = int(S.get())
            if Node_array[seed].state == 0:
                Node_array[seed].state = 1
                active_node_num += 1
                for edge in Node_array[seed].edge_list:
                    if Edge_array[edge].state == 0 and random.random() < p1:
                        Edge_array[edge].state = 1
                        active_edge_num += 1
                        for node in Edge_array[edge].node_list:
                            if Node_array[node].state == 0 and random.random() < p2 and Node_array[node].mark == 0:
                                Node_array[node].mark = 1
                                S.put(node)
    else:
        while not S.empty():
            seed = int(S.get())
            if Node_array[seed].state == 0:
                Node_array[seed].state = 1
                active_node_num += 1
                for edge in Node_array[seed].edge_list:
                    if Edge_array[edge].state == 0 and random.random() < (1 / Edge_array[edge].cardinality):
                        Edge_array[edge].state = 1
                        active_edge_num += 1
                        for node in Edge_array[edge].node_list:
                            if Node_array[node].state == 0 and random.random() < (1 / Node_array[node].degree) and Node_array[node].mark == 0:
                                Node_array[node].mark = 1
                                S.put(node)
    return active_node_num, active_edge_num


def reset(Node_array: list, Edge_array: list, nodenum: int, edgenum: int) -> None:
    """Resets the state of nodes and hyperedges in the hypergraph.

    Sets all nodes and hyperedges to inactive and clears node marks.

    Args:
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        nodenum (int): Number of nodes to reset.
        edgenum (int): Number of hyperedges to reset.
    """
    for i in range(nodenum):
        Node_array[i].state = 0
        Node_array[i].mark = 0
    for i in range(edgenum):
        Edge_array[i].state = 0

def NP(k: int, p1: float, p2: float, nodenum: int, edgenum: int, Node_array: list, Edge_array: list, identical_p: bool) -> list:
    """Implements the Neighbor Priority (NP) algorithm for influence maximization.

    Selects k nodes with the highest number of unique neighbors and evaluates their
    influence spread over multiple Monte Carlo simulations.

    Args:
        k (int): Number of seed nodes to select.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.
        nodenum (int): Number of nodes in the hypergraph.
        edgenum (int): Number of hyperedges in the hypergraph.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.

    Returns:
        list: List of results, each containing:
            - Algorithm name (str)
            - Selected seed nodes (list)
            - Influence spread (float)
            - Cost of selected nodes (int)
            - Average number of activated nodes (float)
            - Average number of activated hyperedges (float)
    """
    spread_results = []
    Neighbor_array = np.zeros((nodenum, 2))
    for i in range(nodenum):
        sum1 = 0
        Neb_Mark = [0] * nodenum
        for edge in Node_array[i].edge_list:
            for node in Edge_array[edge].node_list:
                if Neb_Mark[node] == 0:
                    sum1 += 1
                    Neb_Mark[node] = 1
        Neighbor_array[i][1] = i
        Neighbor_array[i][0] = sum1
    Sorted_Neighbor_array = Neighbor_array[Neighbor_array[:, 0].argsort()[::-1]]
    S = queue.Queue()
    act_node_num = 0
    act_edge_num = 0
    for round in range(10000):
        for i in range(k):
            S.put(int(Sorted_Neighbor_array[i][1]))
            Node_array[int(Sorted_Neighbor_array[i][1])].mark = 1
        reset(Node_array, Edge_array, nodenum, edgenum)
        if round == 1:
            list1 = list(S.queue)
        a, b = spread(S, p1, p2, Node_array, Edge_array, identical_p)
        act_node_num += a
        act_edge_num += b
    total_degree = sum(Node_array[i].degree for i in list1 if i < nodenum)
    spread_results.append(['NP', list1, 0, total_degree, act_node_num / 10000, act_edge_num / 10000])
    return spread_results

def HHD(k: int, p1: float, p2: float, nodenum: int, edgenum: int, Node_array: list, Edge_array: list, identical_p: bool) -> list:
    """Implements the Hyper Degree (HHD) algorithm for influence maximization.

    Selects k nodes with the highest degree and evaluates their influence spread
    over multiple Monte Carlo simulations.

    Args:
        k (int): Number of seed nodes to select.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.
        nodenum (int): Number of nodes in the hypergraph.
        edgenum (int): Number of hyperedges in the hypergraph.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.

    Returns:
        list: List of results, each containing:
            - Algorithm name (str)
            - Selected seed nodes (list)
            - Influence spread (float)
            - Cost of selected nodes (int)
            - Average number of activated nodes (float)
            - Average number of activated hyperedges (float)
    """
    spread_results = []
    Degree_array = np.zeros((nodenum, 2))
    for i in range(nodenum):
        Degree_array[i][1] = i
        Degree_array[i][0] = Node_array[i].degree
    Sorted_Degree_array = Degree_array[Degree_array[:, 0].argsort()[::-1]]
    S = queue.Queue()
    act_node_num = 0
    act_edge_num = 0
    for round in range(10000):
        for i in range(k):
            S.put(int(Sorted_Degree_array[i][1]))
            Node_array[int(Sorted_Degree_array[i][1])].mark = 1
        reset(Node_array, Edge_array, nodenum, edgenum)
        if round == 1:
            list1 = list(S.queue)
        a, b = spread(S, p1, p2, Node_array, Edge_array, identical_p)
        act_node_num += a
        act_edge_num += b
    total_degree = sum(Node_array[i].degree for i in list1 if i < nodenum)
    spread_results.append(['HHD', list1, 0, total_degree, act_node_num / 10000, act_edge_num / 10000])
    return spread_results

def HCI1_ICM(k: int, p1: float, p2: float, nodenum: int, edgenum: int, Node_array: list, Edge_array: list, identical_p: bool) -> list:
    """Implements the HCI1 algorithm for influence maximization.
    Selects k nodes based on a heuristic combining node degree and hyperedge
    cardinality, and evaluates their influence spread over multiple Monte Carlo simulations.
    Args:
        k (int): Number of seed nodes to select.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.
        nodenum (int): Number of nodes in the hypergraph.
        edgenum (int): Number of hyperedges in the hypergraph.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.

    Returns:
        list: List of results, each containing:
            - Algorithm name (str)
            - Selected seed nodes (list)
            - Influence spread (float)
            - Cost of selected nodes (int)
            - Average number of activated nodes (float)
            - Average number of activated hyperedges (float)
    """
    spread_results = []
    HCI1_array = np.zeros((nodenum, 2))
    for i in range(nodenum):
        HCI1_array[i][1] = i
        HCI1_array[i][0] = Node_array[i].degree
        for t in Node_array[i].edge_list:
            HCI1_array[i][0] += p1 * (Edge_array[t].cardinality - 1)
    Sorted_HCI1_array = HCI1_array[HCI1_array[:, 0].argsort()[::-1]]
    S = queue.Queue()
    act_node_num = 0
    act_edge_num = 0
    for round in range(10000):
        for i in range(k):
            S.put(int(Sorted_HCI1_array[i][1]))
            Node_array[int(Sorted_HCI1_array[i][1])].mark = 1
        reset(Node_array, Edge_array, nodenum, edgenum)
        if round == 1:
            list1 = list(S.queue)
        a, b = spread(S, p1, p2, Node_array, Edge_array, identical_p)
        act_node_num += a
        act_edge_num += b
    total_degree = sum(Node_array[i].degree for i in list1 if i < nodenum)
    spread_results.append(['HCI1', list1, 0, total_degree, act_node_num / 10000, act_edge_num / 10000])
    return spread_results

def HCI2_ICM(k: int, p1: float, p2: float, nodenum: int, edgenum: int, Node_array: list, Edge_array: list, identical_p: bool) -> list:
    """Implements the HCI2 algorithm for influence maximization.

    Selects k nodes based on a heuristic combining node degree, hyperedge cardinality,
    and neighbor node degrees, and evaluates their influence spread over multiple Monte Carlo simulations.

    Args:
        k (int): Number of seed nodes to select.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.
        nodenum (int): Number of nodes in the hypergraph.
        edgenum (int): Number of hyperedges in the hypergraph.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.

    Returns:
        list: List of results, each containing:
            - Algorithm name (str)
            - Selected seed nodes (list)
            - Influence spread (float)
            - Cost of selected nodes (int)
            - Average number of activated nodes (float)
            - Average number of activated hyperedges (float)
    """
    spread_results = []
    HCI2_array = np.zeros((nodenum, 2))
    for i in range(nodenum):
        HCI2_array[i][1] = i
        HCI2_array[i][0] = Node_array[i].degree
        for t in Node_array[i].edge_list:
            sum1 = 0
            for tp in Edge_array[t].node_list:
                sum1 += p2 * (Node_array[tp].degree - 1)
            HCI2_array[i][0] += p1 * (sum1 + Edge_array[t].cardinality - 1)
    Sorted_HCI2_array = HCI2_array[HCI2_array[:, 0].argsort()[::-1]]
    S = queue.Queue()
    act_node_num = 0
    act_edge_num = 0
    for round in range(10000):
        for i in range(k):
            S.put(int(Sorted_HCI2_array[i][1]))
            Node_array[int(Sorted_HCI2_array[i][1])].mark = 1
        reset(Node_array, Edge_array, nodenum, edgenum)
        if round == 1:
            list1 = list(S.queue)
        a, b = spread(S, p1, p2, Node_array, Edge_array, identical_p)
        act_node_num += a
        act_edge_num += b
    total_degree = sum(Node_array[i].degree for i in list1 if i < nodenum)
    spread_results.append(['HCI2', list1, 0, total_degree, act_node_num / 10000, act_edge_num / 10000])
    return spread_results

def PageRank(k: int, p1: float, p2: float, nodenum: int, edgenum: int, Node_array: list, Edge_array: list,
              identical_p: bool, damping_factor: float = 0.7, max_iterations: int = 100, tolerance: float = 0.0001) -> list:
    """Implements the PageRank algorithm adapted for hypergraphs.

    Selects k nodes with the highest PageRank scores and evaluates their influence
    spread over multiple Monte Carlo simulations.

    Args:
        k (int): Number of seed nodes to select.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.
        nodenum (int): Number of nodes in the hypergraph.
        edgenum (int): Number of hyperedges in the hypergraph.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.
        damping_factor (float, optional): Damping factor for PageRank. Defaults to 0.7.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
        tolerance (float, optional): Convergence tolerance for PageRank. Defaults to 0.0001.

    Returns:
        list: List of results, each containing:
            - Algorithm name (str)
            - Selected seed nodes (list)
            - Influence spread (float)
            - Cost of selected nodes (int)
            - Average number of activated nodes (float)
            - Average number of activated hyperedges (float)
    """
    spread_results = []
    PageRank0 = np.zeros((nodenum, 2))
    for i in range(nodenum):
        PageRank0[i][1] = i
        PageRank0[i][0] = 1 / nodenum
    t = 0
    PageRank_new = np.zeros((nodenum, 2))
    while t < max_iterations:
        t += 1
        for i in range(nodenum):
            PageRank_new[i][0] = PageRank0[i][0]
            sum1 = 0
            for edge in Node_array[i].edge_list:
                for node in Edge_array[edge].node_list:
                    m = 0
                    if node != i:
                        for edge1 in Node_array[node].edge_list:
                            m += Edge_array[edge1].cardinality - 1
                        sum1 += PageRank0[node][0] / m
            PageRank_new[i][0] = damping_factor * sum1 + (1 - damping_factor) / nodenum
        error = 0
        for i in range(nodenum):
            error += (PageRank_new[i][0] - PageRank0[i][0]) ** 2
        error = math.sqrt(error / nodenum)
        if error < tolerance:
            break
        else:
            for i in range(nodenum):
                PageRank0[i][0] = PageRank_new[i][0]
    Sorted_PageRank_array = PageRank0[PageRank0[:, 0].argsort()[::-1]]
    S = queue.Queue()
    act_node_num = 0
    act_edge_num = 0
    for round in range(10000):
        for i in range(k):
            S.put(int(Sorted_PageRank_array[i][1]))
            Node_array[int(Sorted_PageRank_array[i][1])].mark = 1
        reset(Node_array, Edge_array, nodenum, edgenum)
        if round == 1:
            list1 = list(S.queue)
        a, b = spread(S, p1, p2, Node_array, Edge_array, identical_p)
        act_node_num += a
        act_edge_num += b
    total_degree = sum(Node_array[i].degree for i in list1 if i < nodenum)
    spread_results.append(['Pagerank', list1, 0, total_degree, act_node_num / 10000, act_edge_num / 10000])
    return spread_results

def Random(k: int, p1: float, p2: float, nodenum: int, edgenum: int, Node_array: list, Edge_array: list, identical_p: bool) -> list:
    """Implements a random selection algorithm for influence maximization.

    Randomly selects k nodes and evaluates their influence spread over multiple
    Monte Carlo simulations.

    Args:
        k (int): Number of seed nodes to select.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.
        nodenum (int): Number of nodes in the hypergraph.
        edgenum (int): Number of hyperedges in the hypergraph.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.

    Returns:
        list: List of results, each containing:
            - Algorithm name (str)
            - Selected seed nodes (list)
            - Influence spread (float)
            - Cost of selected nodes (int)
            - Average number of activated nodes (float)
            - Average number of activated hyperedges (float)
    """
    spread_results = []
    all_nodes = list(range(nodenum))
    sampled_nodes = random.sample(all_nodes, k)
    S = queue.Queue()
    act_node_num = 0
    act_edge_num = 0
    for round in range(10000):
        for i in range(k):
            S.put(int(sampled_nodes[i]))
            Node_array[int(sampled_nodes[i])].mark = 1
        reset(Node_array, Edge_array, nodenum, edgenum)
        if round == 1:
            list1 = list(S.queue)
        a, b = spread(S, p1, p2, Node_array, Edge_array, identical_p)
        act_node_num += a
        act_edge_num += b
    total_degree = sum(Node_array[i].degree for i in list1 if i < nodenum)
    spread_results.append(['Random', list1, 0, total_degree, act_node_num / 10000, act_edge_num / 10000])
    return spread_results



def population_init_UI(nodenum: int, ind_size: int, p1: float, Node_array: list, Edge_array: list) -> list:
    """Initializes a population for genetic algorithms using UCI, HCI, or random selection.

    Selects initial individuals based on a random choice among UCI (degree-normalized HCI),
    HCI (degree and hyperedge cardinality), or random initialization.

    Args:
        nodenum (int): Number of nodes in the hypergraph.
        ind_size (int): Size of each individual (number of seed nodes).
        p1 (float): Probability of activating a hyperedge from a node.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.

    Returns:
        list: List of node indices representing an individual.
    """
    rdnum = random.random()
    if rdnum < 0.3333:
        HCI1_array = np.zeros((nodenum, 2))
        UCI1_array = np.zeros((nodenum, 2))
        valid_indices = list(range(nodenum))
        for i in valid_indices:
            HCI1_array[i][1] = i
            HCI1_array[i][0] = Node_array[i].degree
            for t in Node_array[i].edge_list:
                HCI1_array[i][0] += p1 * (Edge_array[t].cardinality - 1)
            UCI1_array[i][1] = i
            UCI1_array[i][0] = HCI1_array[i][0] / Node_array[i].degree
        UCI_TMP = [UCI1_array[k][0] * random.random() for k in valid_indices]
        topk = sorted(range(len(UCI_TMP)), key=lambda i: UCI_TMP[i], reverse=True)[:ind_size]
        return topk
    elif 0.3333 <= rdnum <= 0.6666:
        HCI1_array = np.zeros((nodenum, 2))
        for i in range(nodenum):
            HCI1_array[i][1] = i
            HCI1_array[i][0] = Node_array[i].degree
            for t in Node_array[i].edge_list:
                HCI1_array[i][0] += (1 / Edge_array[t].cardinality) * (Edge_array[t].cardinality - 1)
        HCI_TMP = [HCI1_array[k][0] * random.random() for k in range(nodenum)]
        topk = sorted(range(len(HCI_TMP)), key=lambda i: HCI_TMP[i], reverse=True)[:ind_size]
        return topk
    else:
        return random.sample(range(nodenum), ind_size)

def Fitness_function(individual: list, p1: float, p2: float, edgenum: int, nodenum: int, Node_array: list,
                    Edge_array: list, identical_p: bool) -> tuple:
    """Evaluates the fitness of an individual for multi-objective optimization.

    Computes two objectives: influence spread (W, maximizing) and total degree (minimizing).
    Influence spread considers both first- and second-layer neighbors.

    Args:
        individual (list): List of node indices representing the seed set.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.
        edgenum (int): Number of hyperedges in the hypergraph.
        nodenum (int): Number of nodes in the hypergraph.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.

    Returns:
        tuple: Contains:
            - W (float): Estimated influence spread (including seed nodes and neighbors).
            - total_degree (int): Sum of degrees of selected nodes.
    """
    if len(individual) != len(set(individual)):
        return (0, 0)
    if identical_p:
        individual = list(map(int, individual))
        array_edge_p1 = [1] * edgenum
        array_node_p1 = [1] * nodenum
        array_edge_p2 = [1] * edgenum
        array_node_p2 = [1] * nodenum
        sum1 = 0
        for i in individual:
            for k in Node_array[i].edge_list:
                array_edge_p1[k] *= (1 - p1)
        for i in range(edgenum):
            if array_edge_p1[i] != 1:
                for k in Edge_array[i].node_list:
                    array_node_p1[k] *= (1 - p2 * (1 - array_edge_p1[i]))
        for i in individual:
            array_node_p1[i] = 1
        for i in range(nodenum):
            if array_node_p1[i] != 1:
                sum1 += (1 - array_node_p1[i])
        for i in range(nodenum):
            if array_node_p1[i] != 1:
                for k in Node_array[i].edge_list:
                    if array_edge_p1[k] == 1:
                        array_edge_p2[k] *= (1 - p1 * (1 - array_node_p1[i]))
        for i in range(edgenum):
            if array_edge_p2[i] != 1:
                for k in Edge_array[i].node_list:
                    if array_node_p1[k] == 1 and k not in individual:
                        array_node_p2[k] *= (1 - p2 * (1 - array_edge_p2[i]))
        sum2 = 0
        for i in range(nodenum):
            if array_node_p2[i] != 1:
                sum2 += (1 - array_node_p2[i])
        W = len(individual) + sum1 + sum2
        total_degree = sum(Node_array[i].degree for i in individual)
        return W, total_degree
    else:
        individual = list(map(int, individual))
        array_edge_p1 = [1] * edgenum
        array_node_p1 = [1] * nodenum
        array_edge_p2 = [1] * edgenum
        array_node_p2 = [1] * nodenum
        sum1 = 0
        for i in individual:
            for k in Node_array[i].edge_list:
                array_edge_p1[k] *= (1 - (1 / Edge_array[k].cardinality))
        for i in range(edgenum):
            if array_edge_p1[i] != 1:
                for k in Edge_array[i].node_list:
                    array_node_p1[k] *= (1 - (1 / Node_array[k].degree) * (1 - array_edge_p1[i]))
        for i in individual:
            array_node_p1[i] = 1
        for i in range(nodenum):
            if array_node_p1[i] != 1:
                sum1 += (1 - array_node_p1[i])
        for i in range(nodenum):
            if array_node_p1[i] != 1:
                for k in Node_array[i].edge_list:
                    if array_edge_p1[k] == 1:
                        array_edge_p2[k] *= (1 - (1 / Edge_array[k].cardinality) * (1 - array_node_p1[i]))
        for i in range(edgenum):
            if array_edge_p2[i] != 1:
                for k in Edge_array[i].node_list:
                    if array_node_p1[k] == 1 and k not in individual:
                        array_node_p2[k] *= (1 - (1 / Node_array[k].degree) * (1 - array_edge_p2[i]))
        sum2 = 0
        for i in range(nodenum):
            if array_node_p2[i] != 1:
                sum2 += (1 - array_node_p2[i])
        W = len(individual) + sum1 + sum2
        total_degree = sum(Node_array[i].degree for i in individual)
        return W, total_degree
    

def Fitness_function_naive(individual: list, p1: float, p2: float, edgenum: int, nodenum: int, Node_array: list,
                          Edge_array: list, identical_p: bool) -> tuple:
    """Evaluates the fitness of an individual for single-objective optimization.

    Computes the influence spread (W) for a seed set, considering both first- and
    second-layer neighbors. Used in the naive GA algorithm.

    Args:
        individual (list): List of node indices representing the seed set.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.
        edgenum (int): Number of hyperedges in the hypergraph.
        nodenum (int): Number of nodes in the hypergraph.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.

    Returns:
        tuple: Contains:
            - W (float): Estimated influence spread (including seed nodes and neighbors).
    """
    if len(individual) != len(set(individual)):
        return (0,)
    if identical_p:
        individual = list(map(int, individual))
        array_edge_p1 = [1] * edgenum
        array_node_p1 = [1] * nodenum
        array_edge_p2 = [1] * edgenum
        array_node_p2 = [1] * nodenum
        sum1 = 0
        for i in individual:
            for k in Node_array[i].edge_list:
                array_edge_p1[k] *= (1 - p1)
        for i in range(edgenum):
            if array_edge_p1[i] != 1:
                for k in Edge_array[i].node_list:
                    array_node_p1[k] *= (1 - p2 * (1 - array_edge_p1[i]))
        for i in individual:
            array_node_p1[i] = 1
        for i in range(nodenum):
            if array_node_p1[i] != 1:
                sum1 = (1 - array_node_p1[i])
        for i in range(nodenum):
            if array_node_p1[i] != 1:
                for k in Node_array[i].edge_list:
                    if array_edge_p1[k] == 1:
                        array_edge_p2[k] *= (1 - p1 * (1 - array_node_p1[i]))
        for i in range(edgenum):
            if array_edge_p2[i] != 1:
                for k in Edge_array[i].node_list:
                    if array_node_p1[k] == 1 and k not in individual:
                        array_node_p2[k] *= (1 - p2 * (1 - array_edge_p2[i]))
        sum2 = 0
        for i in range(nodenum):
            if array_node_p2[i] != 1:
                sum2 += (1 - array_node_p2[i])
        W = len(individual) + sum1 + sum2
        return (W,)
    else:
        individual = list(map(int, individual))
        array_edge_p1 = [1] * edgenum
        array_node_p1 = [1] * nodenum
        array_edge_p2 = [1] * edgenum
        array_node_p2 = [1] * nodenum
        sum1 = 0
        for i in individual:
            for k in Node_array[i].edge_list:
                array_edge_p1[k] *= (1 - (1 / Edge_array[k].cardinality))
        for i in range(edgenum):
            if array_edge_p1[i] != 1:
                for k in Edge_array[i].node_list:
                    array_node_p1[k] *= (1 - (1 / Node_array[k].degree) * (1 - array_edge_p1[i]))
        for i in individual:
            array_node_p1[i] = 1
        for i in range(nodenum):
            if array_node_p1[i] != 1:
                sum1 += (1 - array_node_p1[i])
        for i inMULTI_OBJECTIVE_OPTIMIZATION.md range(nodenum):
            if array_node_p1[i] != 1:
                for k in Node_array[i].edge_list:
                    if array_edge_p1[k] == 1:
                        array_edge_p2[k] *= (1 - (1 / Edge_array[k].cardinality) * (1 - array_node_p1[i]))
        for i in range(edgenum):
            if array_edge_p2[i] != 1:
                for k in Edge_array[i].node_list:
                    if array_node_p1[k] == 1 and k not in individual:
                        array_node_p2[k] *= (1 - (1 / Node_array[k].degree) * (1 - array_edge_p2[i]))
        sum2 = 0
        for i in range(nodenum):
            if array_node_p2[i] != 1:
                sum2 += (1 - array_node_p2[i])
        W = len(individual) + sum1 + sum2
        return (W,)
    

def replace_duplicate(lst: list, nodenum: int) -> list:
    """Replaces duplicate nodes in an individual with random unique nodes.

    Args:
        lst (list): List of node indices representing an individual.
        nodenum (int): Total number of nodes in the hypergraph.

    Returns:
        list: Individual with duplicates replaced by random unique nodes.
    """
    seen = set()
    for i in range(len(lst)):
        if lst[i] in seen:
            mx = [x for x in range(nodenum) if x not in seen]
            lst[i] = mx[random.randint(0, len(mx) - 1)]
        seen.add(lst[i])
    return [int(x) for x in lst]



def mutate(individual: list, nodenum: int, p1: float, Node_array: list, Edge_array: list, indpb: float) -> tuple:
    """Mutates an individual using UCI, HCI, or random mutation strategies.

    Randomly selects a mutation strategy (UCI, HCI, or random) and replaces a node
    with another based on the selected strategy.

    Args:
        individual (list): List of node indices representing the individual.
        nodenum (int): Number of nodes in the hypergraph.
        p1 (float): Probability of activating a hyperedge from a node.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        indpb (float): Probability of mutating each node (for random mutation).

    Returns:
        tuple: Contains the mutated individual (creator.Individual).
    """
    rdnum = random.random()
    if rdnum < 0.3333:
        HCI1_array = np.zeros((len(individual), 2))
        UCI1_array = np.zeros((len(individual), 2))
        for i in range(len(individual)):
            HCI1_array[i][1] = individual[i]
            HCI1_array[i][0] = Node_array[int(individual[i])].degree
            for t in Node_array[int(individual[i])].edge_list:
                HCI1_array[i][0] += p1 * (Edge_array[t].cardinality - 1)
            UCI1_array[i][1] = individual[i]
            UCI1_array[i][0] = HCI1_array[i][0] / Node_array[int(individual[i])].degree
            UCI1_array[i][0] *= random.random()
        Mutation_point = 0
        Low_uci = 100000
        for i in range(len(individual)):
            if Low_uci > UCI1_array[i][0]:
                Low_uci = UCI1_array[i][0]
                Mutation_point = i
        available_nodes = [n for n in range(nodenum) if n not in individual]
        selected_nodes = random.sample(available_nodes, len(individual))
        HCI1_array2 = np.zeros((len(individual), 2))
        UCI1_array2 = np.zeros((len(individual), 2))
        for i in range(len(selected_nodes)):
            HCI1_array2[i][1] = selected_nodes[i]
            HCI1_array2[i][0] = Node_array[selected_nodes[i]].degree
            for t in Node_array[selected_nodes[i]].edge_list:
                HCI1_array2[i][0] += p1 * (Edge_array[t].cardinality - 1)
            UCI1_array2[i][1] = selected_nodes[i]
            UCI1_array2[i][0] = HCI1_array2[i][0] / Node_array[selected_nodes[i]].degree
            UCI1_array[i][0] *= random.random()
        UCI_max_node = 0
        Max_uci = 0
        for i in range(len(individual)):
            if Max_uci < UCI1_array2[i][0]:
                Max_uci = UCI1_array2[i][0]
                UCI_max_node = UCI1_array2[i][1]
        individual[Mutation_point] = int(UCI_max_node)
        return creator.Individual(individual),
    elif 0.3333 <= rdnum <= 0.6666:
        HCI1_array = np.zeros((len(individual), 2))
        for i in range(len(individual)):
            HCI1_array[i][1] = individual[i]
            HCI1_array[i][0] = Node_array[int(individual[i])].degree
            for t in Node_array[int(individual[i])].edge_list:
                HCI1_array[i][0] += p1 * (Edge_array[t].cardinality - 1)
        Mutation_point = 0
        Low_hci = 100000
        for i in range(len(individual)):
            if Low_hci > HCI1_array[i][0]:
                Low_hci = HCI1_array[i][0]
                Mutation_point = i
        available_nodes = [n for n in range(nodenum) if n not in individual]
        selected_nodes = random.sample(available_nodes, len(individual))
        HCI1_array2 = np.zeros((len(individual), 2))
        for i in range(len(selected_nodes)):
            HCI1_array2[i][1] = selected_nodes[i]
            HCI1_array2[i][0] = Node_array[selected_nodes[i]].degree
            for t in Node_array[selected_nodes[i]].edge_list:
                HCI1_array2[i][0] += p1 * (Edge_array[t].cardinality - 1)
        HCI_max_node = 0
        Max_hci = 0
        for i in range(len(individual)):
            if Max_hci < HCI1_array2[i][0]:
                Max_hci = HCI1_array2[i][0]
                HCI_max_node = HCI1_array2[i][1]
        individual[Mutation_point] = int(HCI_max_node)
        return creator.Individual(individual),
    else:
        for x in range(len(individual)):
            if random.random() < indpb:
                individual[x] = random.randint(0, nodenum - 1)
        individual = replace_duplicate(individual, nodenum)
        return creator.Individual(individual),
def crossover_naive(ind1: list, ind2: list, nodenum: int) -> tuple:
    """Performs crossover between two individuals, ensuring no duplicate nodes.

    Uses two-point crossover and replaces any duplicates with random unique nodes.

    Args:
        ind1 (list): First individual (list of node indices).
        ind2 (list): Second individual (list of node indices).
        nodenum (int): Total number of nodes in the hypergraph.

    Returns:
        tuple: Contains two new individuals (creator.Individual).
    """
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    ind1 = replace_duplicate(ind1, nodenum)
    ind2 = replace_duplicate(ind2, nodenum)
    return creator.Individual(ind1), creator.Individual(ind2)

def mutate_naive(individual: list, nodenum: int, indpb: float) -> tuple:
    """Performs random mutation on an individual.

    Randomly replaces nodes with new random nodes based on the mutation probability.

    Args:
        individual (list): List of node indices representing the individual.
        nodenum (int): Total number of nodes in the hypergraph.
        indpb (float): Probability of mutating each node.

    Returns:
        tuple: Contains the mutated individual (creator.Individual).
    """
    for x in range(len(individual)):
        if random.random() < indpb:
            individual[x] = random.randint(0, nodenum - 1)
    return individual,


def export_results_to_csv(results: list, filename: str) -> None:
    """Exports algorithm results to a CSV file.

    Args:
        results (list): List of result tuples, each containing algorithm name, seed nodes,
                        fitness values, total degree, and average spread metrics.
        filename (str): Path to the output CSV file.

    Raises:
        IOError: If the file cannot be written.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Individual', 'Fitness1', 'total_degree', 'Spread Nodes', 'Spread Edges'])
        for individual, fitness1, total_degree, node_num_avg, edge_num_avg in results:
            writer.writerow([individual, fitness1, total_degree, node_num_avg, edge_num_avg])

def fast_non_dominated_sort(values1: list, values2: list) -> list:
    """Performs fast non-dominated sorting for multi-objective optimization.

    Sorts individuals into Pareto fronts based on two objectives.

    Args:
        values1 (list): First objective values (e.g., total degree).
        values2 (list): Second objective values (e.g., influence spread).

    Returns:
        list: List of Pareto fronts, where each front is a list of individual indices.
    """
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0 for _ in range(len(values1))]
    rank = [0 for _ in range(len(values1))]
    for p in range(len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(len(values1)):
            if (values1[p] < values1[q] and values2[p] > values2[q]) or \
               (values1[p] <= values1[q] and values2[p] > values2[q]) or \
               (values1[p] < values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] > values2[p]) or \
                 (values1[q] <= values1[p] and values2[q] > values2[p]) or \
                 (values1[q] < values1[p] and values2[q] >= values2[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        front[i].sort(key=lambda x: values1[x])
        i += 1
        front.append(Q)
    del front[len(front) - 1]
    return front

def calculate_diversity(PF_population: list) -> float:
    """Calculates the diversity of a Pareto front population.

    Measures diversity as 1 minus the average intersection ratio between individuals.

    Args:
        PF_population (list): List of individuals in the Pareto front.

    Returns:
        float: Diversity score (higher values indicate greater diversity).
    """
    num_individuals = len(PF_population)
    total_diversity = 0.0
    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            intersection = len(set(PF_population[i]).intersection(set(PF_population[j])))
            total_diversity += intersection / len(PF_population[i])
    avg_diversity = total_diversity * 2 / (num_individuals * (num_individuals - 1))
    return 1 - avg_diversity

def calculate_pareto_front_population(population: list) -> list:
    """Extracts the Pareto front population from a given population.

    Args:
        population (list): List of individuals with fitness values.

    Returns:
        list: List of individuals in the first Pareto front.
    """
    filtered_spread_results = [res for res in population if res.fitness.values[1] != 0 or res.fitness.values[0] != 0]
    total_degrees = [ind.fitness.values[1] for ind in filtered_spread_results]
    node_num_avgs = [ind.fitness.values[0] for ind in filtered_spread_results]
    fronts = fast_non_dominated_sort(total_degrees, node_num_avgs)
    pareto_front = [filtered_spread_results[ind] for ind in fronts[0]]
    return pareto_front

def calculate_pareto_front(population: list) -> list:
    """Extracts the indices of individuals in the first Pareto front.

    Args:
        population (list): List of individuals with fitness values.

    Returns:
        list: List of indices of individuals in the first Pareto front.
    """
    filtered_spread_results = [res for res in population if res.fitness.values[1] != 0 or res.fitness.values[0] != 0]
    total_degrees = [ind.fitness.values[1] for ind in filtered_spread_results]
    node_num_avgs = [ind.fitness.values[0] for ind in filtered_spread_results]
    fronts = fast_non_dominated_sort(total_degrees, node_num_avgs)
    return fronts[0]



def define_reference_point(pareto_front: list) -> tuple:
    """Defines a reference point for hypervolume calculation.

    Uses the maximum cost and minimum influence spread from the Pareto front.

    Args:
        pareto_front (list): List of results in the Pareto front.

    Returns:
        tuple: Reference point as (min_influence_spread, max_cost).
    """
    max_cost = max(res[3] for res in pareto_front)
    min_influence_spread = min(res[2] for res in pareto_front)
    return (min_influence_spread, max_cost)

def calculate_hypervolume(pareto_front: list) -> float:
    """Calculates the hypervolume of a Pareto front.

    Computes the hypervolume relative to a reference point for multi-objective optimization.

    Args:
        pareto_front (list): List of results in the Pareto front.

    Returns:
        float: Hypervolume value.
    """
    hypervolume = 0.0
    reference_point = define_reference_point(pareto_front)
    list_pareto_front = [(res[2], res[3]) for res in pareto_front]
    list_pareto_front = list(set(list_pareto_front))
    for point in list_pareto_front:
        width = abs(point[0] - reference_point[0])
        height = abs(point[1] - reference_point[1])
        hypervolume += width * height
    return hypervolume



#seed, filename, IND_SIZE, population_size, generations, cxpb, mutpb, p1, p2, NodeSize, EdgeSize
from deap import base, creator, tools
toolbox = base.Toolbox()  # 初始化 toolbox


def GA_naive(seed: int, ind_size: int, population_size: int, generations: int, cxpb: float, mutpb: float,
             nodenum: int, edgenum: int, Node_array: list, Edge_array: list, identical_p: bool, p1: float, p2: float) -> list:
    """Implements a naive genetic algorithm for single-objective influence maximization.

    Uses tournament selection, crossover, and mutation to optimize influence spread.

    Args:
        seed (int): Random seed for reproducibility.
        ind_size (int): Size of each individual (number of seed nodes).
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to run.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.
        nodenum (int): Number of nodes in the hypergraph.
        edgenum (int): Number of hyperedges in the hypergraph.
        Node_array (list): List of NodeStruct objects representing nodes.
        Edge_array (list): List of EdgeStruct objects representing hyperedges.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.

    Returns:
        list: List of results, each containing:
            - Algorithm name (str)
            - Best individual (list)
            - Fitness value (float)
            - Total degree of selected nodes (int)
            - Average number of activated nodes (float)
            - Average number of activated hyperedges (float)
    """
    spread_results = []
    avg_node_num = 0
    avg_edge_num = 0
    random.seed(seed)
    if not hasattr(creator, "FitnessSingle"):
        creator.create("FitnessSingle", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualType1"):
        creator.create("IndividualType1", list, fitness=creator.FitnessSingle)
    toolbox.register("random_individual", tools.initRepeat, creator.IndividualType1, lambda: random.randint(0, nodenum - 1), n=ind_size)
    toolbox.register("random_population", tools.initRepeat, list, toolbox.random_individual)
    toolbox.register("mate_naive", crossover_naive, nodenum=nodenum)
    toolbox.register("mutate_naive", mutate_naive, nodenum=nodenum, indpb=1/ind_size)
    toolbox.register("evaluate", Fitness_function_naive, p1=p1, p2=p2, edgenum=edgenum, nodenum=nodenum,
                     Node_array=Node_array, Edge_array=Edge_array, identical_p=identical_p)
    toolbox.register("select_GA", tools.selTournament, tournsize=3)
    pop = toolbox.random_population(n=population_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    fits = [ind.fitness.values[0] for ind in pop]
    g = 0
    best_ind = None
    while g < generations:
        g += 1
        offspring = toolbox.select_GA(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate_naive(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate_naive(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        max_fit = max(fits)
        best_ind = pop[fits.index(max_fit)]
    act_node_num = 0
    act_edge_num = 0
    S = queue.Queue()
    for round in range(10000):
        for item in best_ind:
            S.put(item)
        reset(Node_array, Edge_array, nodenum, edgenum)
        a, b = spread(S, p1, p2, Node_array, Edge_array, identical_p)
        act_node_num += a
        act_edge_num += b
    avg_node_num += act_node_num / 10000
    avg_edge_num += act_edge_num / 10000
    total_degree = sum(Node_array[i].degree for i in best_ind)
    spread_results.append(['GA', best_ind, best_ind.fitness.values[0], total_degree, avg_node_num, avg_edge_num])
    return spread_results

def RunGA(seed: int, filename: str, ind_size: int, population_size: int, generations: int, cxpb: float, mutpb: float,
          node_size: int, edge_size: int, identical_p: bool, p1: float, p2: float) -> tuple:
    """Runs the naive GA algorithm for various seed set sizes.

    Args:
        seed (int): Random seed for reproducibility.
        filename (str): Path to the hypergraph data file.
        ind_size (int): Maximum size of the seed set.
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to run.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.
        node_size (int): Expected number of nodes in the hypergraph.
        edge_size (int): Expected number of hyperedges in the hypergraph.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.

    Returns:
        tuple: Contains:
            - spread_results (list): List of results for each seed set size.
            - result_data (list): List of [total_degree, avg_node_num] pairs.
    """
    spread_results = []
    result_data = []
    Node_array, nodenum, Edge_array, edgenum = construct_hypergraph(filename, node_size, edge_size)
    for ind_size in range(1, ind_size + 1):
        spread_results_TP = GA_naive(seed, ind_size, population_size, generations, cxpb, mutpb, nodenum, edgenum,
                                    Node_array, Edge_array, identical_p, p1, p2)
        spread_results += spread_results_TP
    for result in spread_results:
        result_data.append([result[3], result[4]])
    return spread_results, result_data


def ga_nsga2(seed: int, filename: str, ind_size: int, population_size: int, generations: int, cxpb: float, mutpb: float,
             node_size: int, edge_size: int, identical_p: bool, p1: float, p2: float) -> tuple:
    """Implements the NSGA-II algorithm for multi-objective influence maximization.

    Optimizes influence spread and cost using NSGA-II with random initialization.

    Args:
        seed (int): Random seed for reproducibility.
        filename (str): Path to the hypergraph data file.
        ind_size (int): Size of each individual (number of seed nodes).
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to run.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.
        node_size (int): Expected number of nodes in the hypergraph.
        edge_size (int): Expected number of hyperedges in the hypergraph.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.

    Returns:
        tuple: Contains:
            - spread_results (list): List of results for the Pareto front.
            - results_df (pd.DataFrame): DataFrame with algorithm details for each generation.
            - hypervolume (float): Hypervolume of the Pareto front.
    """
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    results_df = pd.DataFrame(columns=["Algorithm", "Generation", "Individual", "Fitness1", "Fitness2"])
    rows = []
    print("nsga2")
    random.seed(seed)
    Node_array, nodenum, Edge_array, edgenum = construct_hypergraph(filename, node_size, edge_size)
    toolbox.register("evaluate_nsga", Fitness_function, p1=p1, p2=p2, edgenum=edgenum, nodenum=nodenum,
                     Node_array=Node_array, Edge_array=Edge_array, identical_p=identical_p)
    toolbox.register("select_nsga", tools.selNSGA2)
    toolbox.register("random_individual", tools.initRepeat, creator.Individual, lambda: random.randint(0, nodenum - 1), n=ind_size)
    toolbox.register("random_population", tools.initRepeat, list, toolbox.random_individual)
    toolbox.register("mate_naive", crossover_naive, nodenum=nodenum)
    toolbox.register("mutate_naive", mutate_naive, nodenum=nodenum, indpb=1/ind_size)
    population = toolbox.random_population(n=population_size)
    population = toolbox.select_nsga(population, k=256)
    fitnesses = list(map(toolbox.evaluate_nsga, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    for individual in calculate_pareto_front_population(population):
        rows.append({
            "Algorithm": 'NSGA-II',
            "Generation": 0,
            "Individual": individual,
            "Fitness1": individual.fitness.values[0],
            "Fitness2": individual.fitness.values[1]
        })
    for gen in range(generations):
        offspring = list(map(toolbox.clone, population))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate_naive(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate_naive(mutant)
                del mutant.fitness.values
        fits = list(map(toolbox.evaluate_nsga, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select_nsga(offspring + population, k=256)
        for individual in calculate_pareto_front_population(population):
            rows.append({
                "Algorithm": 'NSGA-II',
                "Generation": gen + 1,
                "Individual": individual,
                "Fitness1": individual.fitness.values[0],
                "Fitness2": individual.fitness.values[1]
            })
    PFpopulation = calculate_pareto_front_population(population)
    results_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)
    spread_results = []
    seen = set()
    spread_population = []
    for ind in PFpopulation:
        key = (ind.fitness.values[0], ind.fitness.values[1])
        if key not in seen:
            seen.add(key)
            spread_population.append(ind)
    for individual in spread_population:
        active_node_num = 0
        active_edge_num = 0
        for i in range(10000):
            S = queue.Queue()
            for node in individual:
                S.put(node)
            node_num, edge_num = spread(S, p1, p2, Node_array, Edge_array, identical_p)
            active_node_num += node_num
            active_edge_num += edge_num
            reset(Node_array, Edge_array, nodenum, edgenum)
        node_num_avg = active_node_num / 10000
        edge_num_avg = active_edge_num / 10000
        if individual.fitness.values[1] > 0:
            spread_results.append(['NSGA-II', individual, individual.fitness.values[0], individual.fitness.values[1],
                                  node_num_avg, edge_num_avg])
    spread_results = sorted(spread_results, key=lambda x: x[3])
    hypervolume = calculate_hypervolume(spread_results)
    return spread_results, results_df, hypervolume


def ga_nsga2_init(seed: int, filename: str, ind_size: int, population_size: int, generations: int, cxpb: float, mutpb: float,
                  node_size: int, edge_size: int, identical_p: bool, p1: float, p2: float) -> tuple:
    """Implements the NSGA-II algorithm with heuristic initialization for multi-objective influence maximization.

    Uses UCI/HCI-based population initialization to optimize influence spread and cost.

    Args:
        seed (int): Random seed for reproducibility.
        filename (str): Path to the hypergraph data file.
        ind_size (int): Size of each individual (number of seed nodes).
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to run.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.
        node_size (int): Expected number of nodes in the hypergraph.
        edge_size (int): Expected number of hyperedges in the hypergraph.
        identical_p (bool): If True, uses constant probabilities; otherwise, uses degree-based probabilities.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.

    Returns:
        tuple: Contains:
            - spread_results (list): List of results for the Pareto front.
            - results_df (pd.DataFrame): DataFrame with algorithm details for each generation.
            - hypervolume (float): Hypervolume of the Pareto front.
    """
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    results_df = pd.DataFrame(columns=["Algorithm", "Generation", "Individual", "Fitness1", "Fitness2"])
    rows = []
    print("nsga2_init")
    random.seed(seed)
    Node_array, nodenum, Edge_array, edgenum = construct_hypergraph(filename, node_size, edge_size)
    toolbox.register("evaluate_nsga", Fitness_function, p1=p1, p2=p2, edgenum=edgenum, nodenum=nodenum,
                     Node_array=Node_array, Edge_array=Edge_array, identical_p=identical_p)
    toolbox.register("select_nsga", tools.selNSGA2)
    toolbox.register("indices_ui", population_init_UI, nodenum=nodenum, ind_size=ind_size, p1=p1,
                     Node_array=Node_array, Edge_array=Edge_array)
    toolbox.register("individual_ui", tools.initIterate, creator.Individual, toolbox.indices_ui)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual_ui)
    toolbox.register("mate_naive", crossover_naive, nodenum=nodenum)
    toolbox.register("mutate_naive", mutate_naive, nodenum=nodenum, indpb=1/ind_size)
    population_init = toolbox.population(n=256)
    fits = list(map(toolbox.evaluate_nsga, population_init))
    for fit, ind in zip(fits, population_init):
        ind.fitness.values = fit
    population = toolbox.select_nsga(population_init, k=256)
    for individual in calculate_pareto_front_population(population):
        rows.append({
            "Algorithm": 'NSGA-II-Init',
            "Generation": 0,
            "Individual": individual,
            "Fitness1": individual.fitness.values[0],
            "Fitness2": individual.fitness.values[1]
        })
    for gen in range(generations):
        offspring = list(map(toolbox.clone, population))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate_naive(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate_naive(mutant)
                del mutant.fitness.values
        fits = list(map(toolbox.evaluate_nsga, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select_nsga(offspring + population, k=256)
        for individual in calculate_pareto_front_population(population):
            rows.append({
                "Algorithm": 'NSGA-II-Init',
                "Generation": gen + 1,
                "Individual": individual,
                "Fitness1": individual.fitness.values[0],
                "Fitness2": individual.fitness.values[1]
            })
    PFpopulation = calculate_pareto_front_population(population)
    results_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)
    spread_results = []
    seen = set()
    spread_population = []
    for ind in PFpopulation:
        key = (ind.fitness.values[0], ind.fitness.values[1])
        if key not in seen:
            seen.add(key)
            spread_population.append(ind)
    for individual in spread_population:
        active_node_num = 0
        active_edge_num = 0
        for i in range(10000):
            S = queue.Queue()
            for node in individual:
                S.put(node)
            node_num, edge_num = spread(S, p1, p2, Node_array, Edge_array, identical_p)
            active_node_num += node_num
            active_edge_num += edge_num
            reset(Node_array, Edge_array, nodenum, edgenum)
        node_num_avg = active_node_num / 10000
        edge_num_avg = active_edge_num / 10000
        if individual.fitness.values[1] > 0:
            spread_results.append(['NSGA-II-Init', individual, individual.fitness.values[0], individual.fitness.values[1],
                                  node_num_avg, edge_num_avg])
    spread_results = sorted(spread_results, key=lambda x: x[3])
    hypervolume = calculate_hypervolume(spread_results)
    return spread_results, results_df, hypervolume


def MOEA_IMF(seed: int, filename: str, ind_size: int, population_size: int, generations: int, node_size: int,
             edge_size: int, identical_p: bool, p1: float, p2: float) -> tuple:
    """Implements the MOEA-IMF algorithm for multi-objective influence maximization on hypergraphs.

    Optimizes influence spread (maximizing) and cost (minimizing) using a multi-objective evolutionary
    algorithm with adaptive crossover and mutation probabilities based on population diversity.
    Initializes the population using UCI/HCI heuristics and applies NSGA-II for selection.

    Args:
        seed (int): Random seed for reproducibility.
        filename (str): Path to the file containing hypergraph data.
        ind_size (int): Size of each individual (number of seed nodes).
        population_size (int): Number of individuals in the initial population.
        generations (int): Number of generations to run the algorithm.
        node_size (int): Expected number of nodes in the hypergraph.
        edge_size (int): Expected number of hyperedges in the hypergraph.
        identical_p (bool): If True, uses constant activation probabilities; otherwise, uses degree-based probabilities.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.

    Returns:
        tuple: Contains:
            - spread_results (list): List of results for the Pareto front, each including:
                - Algorithm name (str)
                - Individual (list of node indices)
                - Fitness1 (float, influence spread)
                - Fitness2 (float, total degree)
                - Average activated nodes (float)
                - Average activated hyperedges (float)
            - results_df (pd.DataFrame): DataFrame with details for each generation, including algorithm name,
                generation number, individual, and fitness values.
            - hypervolume (float): Hypervolume of the Pareto front.
    """
    print("MOEA_IMF_start")  # Log start of algorithm
    random.seed(seed)  # Set random seed for reproducibility

    # Construct hypergraph from file
    Node_array, nodenum, Edge_array, edgenum = construct_hypergraph(filename, node_size, edge_size)

    # Register DEAP classes for multi-objective optimization
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Maximize Fitness1, minimize Fitness2
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)  # Individual is a list with fitness attribute

    # Register toolbox functions
    toolbox.register("indices_ui", population_init_UI, nodenum=nodenum, ind_size=ind_size, p1=p1,
                     Node_array=Node_array, Edge_array=Edge_array)  # UCI/HCI-based initialization
    toolbox.register("individual_ui", tools.initIterate, creator.Individual, toolbox.indices_ui)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual_ui)
    toolbox.register("evaluate_nsga", Fitness_function, p1=p1, p2=p2, edgenum=edgenum, nodenum=nodenum,
                     Node_array=Node_array, Edge_array=Edge_array, Identical_p=identical_p)  # Fitness evaluation
    toolbox.register("mate_naive", crossover_naive, nodenum=nodenum)  # Crossover operation
    toolbox.register("mutate", mutate, nodenum=nodenum, p1=p1, Node_array=Node_array, Edge_array=Edge_array,
                     indpb=1 / ind_size)  # Mutation operation
    toolbox.register("select_nsga", tools.selNSGA2)  # NSGA-II selection

    # Initialize population
    population_init = toolbox.population(n=population_size)
    fits = list(map(toolbox.evaluate_nsga, population_init))
    for fit, ind in zip(fits, population_init):
        ind.fitness.values = fit

    # Select top individuals to form initial population
    population = toolbox.select_nsga(population_init, k=256)

    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=["Algorithm", "Generation", "Individual", "Fitness1", "Fitness2"])
    rows = []

    # Set adaptive genetic algorithm parameters
    cinit = 0.8  # Initial crossover probability
    minit = 0.2  # Initial mutation probability
    cmin = 0.6  # Minimum crossover probability
    mmax = 0.4  # Maximum mutation probability

    # Record initial generation (generation 0)
    for individual in calculate_pareto_front_population(population):
        rows.append({
            "Algorithm": 'MOEA-IMF',
            "Generation": 0,
            "Individual": individual,
            "Fitness1": individual.fitness.values[0],  # Influence spread
            "Fitness2": individual.fitness.values[1]  # Total degree
        })

    # Track diversity over generations
    list_diversity = []

    # Main evolutionary loop
    for gen in range(generations):
        # Calculate population diversity based on Pareto front
        diversity = calculate_diversity(calculate_pareto_front_population(population))
        list_diversity.append(diversity)

        # Adapt crossover and mutation probabilities based on diversity
        cxpb = cinit - (1 - diversity) * (cinit - cmin)
        mutpb = minit + (1 - diversity) * (mmax - minit)

        # Clone population to create offspring
        offspring = list(map(toolbox.clone, population))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate_naive(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate fitness for offspring
        fits = list(map(toolbox.evaluate_nsga, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        # Select next generation using NSGA-II
        population = toolbox.select_nsga(offspring + population, k=256)

        # Record Pareto front for current generation
        for individual in calculate_pareto_front_population(population):
            rows.append({
                "Algorithm": 'MOEA-IMF',
                "Generation": gen + 1,
                "Individual": individual,
                "Fitness1": individual.fitness.values[0],
                "Fitness2": individual.fitness.values[1]
            })

    # Extract final Pareto front
    PFpopulation = calculate_pareto_front_population(population)
    results_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)

    # Remove duplicate individuals in Pareto front
    spread_results = []
    seen = set()
    spread_population = []
    for ind in PFpopulation:
        key = (ind.fitness.values[0], ind.fitness.values[1])
        if key not in seen:
            seen.add(key)
            spread_population.append(ind)

    # Evaluate influence spread for Pareto front individuals
    for individual in spread_population:
        active_node_num = 0
        active_edge_num = 0
        for _ in range(10000):  # Monte Carlo simulation
            S = queue.Queue()
            for node in individual:
                S.put(node)
            node_num, edge_num = spread(S, p1, p2, Node_array, Edge_array, identical_p)
            active_node_num += node_num
            active_edge_num += edge_num
            reset(Node_array, Edge_array, nodenum, edgenum)

        node_num_avg = active_node_num / 10000
        edge_num_avg = active_edge_num / 10000
        if individual.fitness.values[1] > 0:  # Filter individuals with non-zero total degree
            spread_results.append(['MOEA-IMF', individual, individual.fitness.values[0], individual.fitness.values[1],
                                   node_num_avg, edge_num_avg])

    # Sort results by cost (total degree)
    spread_results = sorted(spread_results, key=lambda x: x[3])

    # Calculate hypervolume of Pareto front
    hypervolume = calculate_hypervolume(spread_results)

    # Save diversity metrics to CSV
    df1 = pd.DataFrame(list_diversity, columns=["Diversity"])
    df1.to_csv("D:/QXL/Hypergraph-Multi-objective optimization/algorithm/experiment/Restaurant-rev/diversity.csv",
               index=False)

    return spread_results, results_df, hypervolume


def RunRandom(seed: int, filename: str, ind_size: int, node_size: int, edge_size: int, identical_p: bool,
              p1: float, p2: float) -> tuple:
    """Runs the Random algorithm for influence maximization with varying seed set sizes.

    Constructs a hypergraph from a file and applies the Random algorithm for seed set sizes
    from 1 to ind_size, collecting influence spread results.

    Args:
        seed (int): Random seed for reproducibility.
        filename (str): Path to the file containing hypergraph data.
        ind_size (int): Maximum size of the seed set.
        node_size (int): Expected number of nodes in the hypergraph.
        edge_size (int): Expected number of hyperedges in the hypergraph.
        identical_p (bool): If True, uses constant activation probabilities; otherwise, uses degree-based probabilities.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.

    Returns:
        tuple: Contains:
            - spread_results (list): List of results for each seed set size, where each result is:
                - Algorithm name (str)
                - Selected seed nodes (list)
                - Fitness value (float, always 0 for Random)
                - Total degree of selected nodes (int)
                - Average number of activated nodes (float)
                - Average number of activated hyperedges (float)
            - result_data (list): List of [total_degree, avg_node_num] pairs for each result.
    """
    random.seed(seed)  # Set random seed for reproducibility
    spread_results = []
    result_data = []

    # Construct hypergraph
    Node_array, nodenum, Edge_array, edgenum = construct_hypergraph(filename, node_size, edge_size)

    # Run Random algorithm for each seed set size
    for k in range(1, ind_size + 1):
        spread_results_tp = Random(k, p1, p2, nodenum, edgenum, Node_array, Edge_array, identical_p)
        spread_results += spread_results_tp

    # Collect cost (total degree) and influence spread
    for result in spread_results:
        result_data.append([result[3], result[4]])  # [total_degree, avg_node_num]

    return spread_results, result_data


def Runothers(seed: int, filename: str, ind_size: int, node_size: int, edge_size: int, identical_p: bool,
              p1: float, p2: float) -> list:
    """Runs multiple heuristic algorithms for influence maximization with varying seed set sizes.

    Constructs a hypergraph and applies NP, HHD, HCI1, HCI2, and PageRank algorithms for seed set
    sizes from 1 to ind_size, collecting influence spread results.

    Args:
        seed (int): Random seed for reproducibility.
        filename (str): Path to the file containing hypergraph data.
        ind_size (int): Maximum size of the seed set.
        node_size (int): Expected number of nodes in the hypergraph.
        edge_size (int): Expected number of hyperedges in the hypergraph.
        identical_p (bool): If True, uses constant activation probabilities; otherwise, uses degree-based probabilities.
        p1 (float): Probability of activating a hyperedge from a node.
        p2 (float): Probability of activating a node from a hyperedge.

    Returns:
        list: List of results for each algorithm and seed set size, where each result is:
            - Algorithm name (str)
            - Selected seed nodes (list)
            - Fitness value (float, always 0 for heuristic algorithms)
            - Total degree of selected nodes (int)
            - Average number of activated nodes (float)
            - Average number of activated hyperedges (float)

    Raises:
        FileNotFoundError: If the hypergraph data file does not exist.
        ValueError: If ind_size, node_size, or edge_size is non-positive.
    """
    random.seed(seed)  # Set random seed for reproducibility
    spread_results = []

    # Construct hypergraph
    Node_array, nodenum, Edge_array, edgenum = construct_hypergraph(filename, node_size, edge_size)

    # Run each heuristic algorithm for seed set sizes 1 to ind_size
    for k in range(1, ind_size + 1):
        spread_results += NP(k, p1, p2, nodenum, edgenum, Node_array, Edge_array, identical_p)
        spread_results += HHD(k, p1, p2, nodenum, edgenum, Node_array, Edge_array, identical_p)
        spread_results += HCI1_ICM(k, p1, p2, nodenum, edgenum, Node_array, Edge_array, identical_p)
        spread_results += HCI2_ICM(k, p1, p2, nodenum, edgenum, Node_array, Edge_array, identical_p)
        spread_results += PageRank(k, p1, p2, nodenum, edgenum, Node_array, Edge_array, identical_p)

    return spread_results


def run_single_simulation(i,seed, filename, NodeSize, EdgeSize, IND_SIZE, Identical_p, p1, p2):
    print(f"Running simulation {i + 1}")
    result_detail = pd.DataFrame(columns=["Algorithm", "Generation", "Individual", "Fitness1", "Fitness2"])
    # spread_results最优个体（1-IND_SIZE）
    # result_data 成本和spread influence GA算法
    spread_results = []#记录帕累托前沿上个体的传播范围
    spread_results1,result_data1 = RunGA(seed, filename, IND_SIZE, 256, 50, 0.8, 0.2, NodeSize, EdgeSize, Identical_p, p1, p2)
    spread_results += spread_results1
    
    #随机函数
    spread_results2,result_data2 = RunRandom(seed, filename, IND_SIZE, NodeSize, EdgeSize, Identical_p, p1, p2)
    spread_results+=spread_results2
    
    #NSGA-II算法
    spread_results3, results_df, hypervolume_NSGA = ga_nsga2(seed, filename, IND_SIZE, 256, 50, 0.8, 0.2, NodeSize, EdgeSize, Identical_p, p1, p2)
    spread_results += spread_results3
    result_detail = pd.concat([result_detail, results_df], ignore_index=True)
    
    #NSGA-II-Init
    spread_results4, results_df4, hypervolume_NSGA_Init = ga_nsga2_init(seed, filename, IND_SIZE, 256, 50, 0.8, 0.2, NodeSize, EdgeSize, Identical_p, p1, p2)
    spread_results += spread_results4
    result_detail = pd.concat([result_detail, results_df4], ignore_index=True)
    
    
    #MOEA_IMF算法
    spread_results7, results_df7, hypervolume_MOEA = MOEA_IMF(seed,filename,IND_SIZE, 256, 50, NodeSize, EdgeSize,Identical_p,p1,p2)
    spread_results += spread_results7
    result_detail = pd.concat([result_detail, results_df7], ignore_index=True)
    
    
    result_detail.to_csv('D:/QXL/Hypergraph-Multi-objective optimization/algorithm/experiment/Restaurant-rev/result_detail-'+str(i+1)+'.csv')
    df = pd.DataFrame(spread_results)
    df.to_csv('D:/QXL/Hypergraph-Multi-objective optimization/algorithm/experiment/Restaurant-rev/spread_results-'+str(i+1)+'.csv')
    
    return  hypervolume_NSGA,hypervolume_NSGA_Init,hypervolume_MOEA


def calculate(filename: str, node_size: int, edge_size: int, identica_p: bool) -> tuple:
    """Runs multiple influence maximization algorithms across multiple simulations.

    Executes GA, Random, NSGA-II, NSGA-II-Init, MOEA-IMF, and heuristic algorithms (NP, HHD, HCI1, HCI2,
    PageRank) on a hypergraph, using multiple processes for parallel simulations, and saves results to CSV.

    Args:
        filename (str): Path to the file containing hypergraph data.
        node_size (int): Expected number of nodes in the hypergraph.
        edge_size (int): Expected number of hyperedges in the hypergraph.
        identical_p (bool): If True, uses constant activation probabilities; otherwise, uses degree-based probabilities.

    Returns:
        tuple: Contains:
            - hypervolume_results (list): List of hypervolume values for each simulation, each containing
                hypervolume for NSGA-II, NSGA-II-Init, and MOEA-IMF.
            - spread_results (list): List of results from heuristic algorithms, each including:
                - Algorithm name (str)
                - Selected seed nodes (list)
                - Fitness value (float)
                - Total degree of selected nodes (int)
                - Average number of activated nodes (float)
                - Average number of activated hyperedges (float)

    Raises:
        FileNotFoundError: If the hypergraph data file does not exist.
        ValueError: If node_size or edge_size is non-positive.
        OSError: If file operations (e.g., writing seeds or results) fail.
    """
    # Define algorithm parameters
    ind_size = 20
    p1 = 0.1
    p2 = 0.1
    num_iterations = 10
    num_processes = mp.cpu_count()

    # Generate random seeds for simulations
    seeds = [random.randint(0, 10000) for _ in range(num_iterations)]

    # Save seeds to file
    output_dir = "D:/QXL/Hypergraph-Multi-objective/algorithm/experiment/Restaurant-rev"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/random_seeds.txt", "w") as file:
        for seed in seeds:
            file.write(f"{seed}\n")

    # Run simulations in parallel
    with Pool(processes=num_processes) as pool:
        hypervolume_results = list(tqdm(
            pool.starmap(
                run_single_simulation,
                [(i, seeds[i], filename, node_size, edge_size, ind_size, identical_p, p1, p2)
                 for i in range(num_iterations)]
            ),
            total=num_iterations
        ))

    # Run heuristic algorithms
    spread_results = Runothers(2145, filename, ind_size, node_size, edge_size, identical_p, p1, p2)

    # Save heuristic results to CSV
    df = pd.DataFrame(spread_results)
    df.to_csv(f"{output_dir}/spread_results_others.csv", index=False)

    return hypervolume_results, spread_results


def main():
    """Main entry point for running hypergraph influence maximization experiments.

    Runs the calculate function with predefined parameters and saves results to files.

    """
    filename = "D:/experiment/Restaurant-rev/hyperedge.txt"
    all_hv_results = calculate(filename, node_size=10000, edge_size=10000, identical_p=True)

if __name__ == "__main__":
    main()







#test seed, filename, IND_SIZE, population_size, generations, cxpb, mutpb, NodeSize, EdgeSize, Identical_p, p1, p2
#filename='D:/QXL/Hypergraph-Multi-objective optimization/algorithm/experiment/Algebra-questions/hyperedge.txt'
#spread_results,results_df,hypervolume=ga_nsga2(42,filename,20,256,50,0.8,0.2,10000,10000,True,0.1,0.1)
#results_df.to_csv('D:/QXL/Hypergraph-Multi-objective optimization/algorithm/experiment/Algebra-questions/result.csv')
#print(spread_results[1][1])

#seed,filename,IND_SIZE, population_size, generations, NodeSize, EdgeSize,p1,p2
#filename='D:/QXL/Hypergraph-Multi-objective optimization/algorithm/experiment/Algebra-questions/hyperedge.txt'
#spread_results,results_df,hypervolume=MOEA_IMF(42,filename,20,256,50,10000,10000,True,0.1,0.1)
#results_df.to_csv('D:/QXL/Hypergraph-Multi-objective optimization/algorithm/experiment/Algebra-questions/result.csv')
#print(spread_results[1][1])

#seed, filename, IND_SIZE, population_size, generations, CXPB, MUTPB, NodeSize, EdgeSize, Identical_p, p1, p2
#spread_results,result_data=RunGA(42,filename,5,256,100,0.8,0.2,10000,10000,True,0.1,0.1)
