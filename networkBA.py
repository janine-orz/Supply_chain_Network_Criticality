import networkx as nx
import matplotlib.pyplot as plt
import random
import bezier
import string
import itertools
from matplotlib.collections import LineCollection
from scipy import sparse
from scipy import optimize
import numpy as np

'''
DONE: 
    * to show the network in a multipartite_layout
    * to set a redundance in one TIER and rename the name of every node which has a redudanced node
    * deal with the networksize = [12, 20, 8, 3]
TO DO:
    * probability still need to consider how much would be the best choice (maybe need a small test)
    * still need to consider how to present the relationship of a parallel manufactory
'''


R = 12  # Ammount of [TIER IV]: raw material
S = 20  # Ammount of [TIER III]: semifinisched products
C = 8   # Ammount of [TIER II]: components
M = 3   # Ammount of [TIER I]: modules
pR = 0.15
pS = 0.1
pC = 0.2
pM = 0
E_12 = 8 
E_23 = 20
E_34 = 12
pE = 0

# create the network of part [TIER I] with the method GNP(N,p)
def GNP(N, p, string):
    edges = itertools.combinations(range(N), 2)
    G = nx.DiGraph()
    # add nodes into the network
    G.add_nodes_from(range(N))

    # add edges into the network
    for e in edges:
        if random.random() < p:
            G.add_edge(*e)

    # rename the nodes in the network
    # change the label of nodes in TIER from 'NUMBER' into 'ALPHABET+NUMBER'
    mapping = {}
    for n in range(N):
        node = list(G.nodes)
        a = node[n]
        a = str(a)
        mapping[n] = string + a
    
    G = nx.relabel_nodes(G, mapping)
    
    return G

# combine two parts [TIER I] & [TIER II] with the method GNL(N,L)
def TRANS(R, S, C, M, pR, pS, pC, pM, E_12, E_23, E_34):
    SC = nx.Graph()
    
    # create the random network with the possibility of the part [TIER I]
    T1 = GNP(M, pM, "M-")
    # create the random network with the possibility of the part [TIER II]
    T2 = GNP(C, pC, "C-")
    T2 = RENAME(T2, C, "C-")
    # create the random network with the possibility of the part [TIER III]
    T3 = GNP(S, pS, "S-")
    T3 = RENAME(T3, S, "S-")
    # create the random network with the possibility of the part [TIER IV]
    T4 = GNP(R, pR, "R-")
    T4 = RENAME(T4, R, "R-")

    # add one nodes in both [TIER I] & [TIER II]
    # T1.add_node(M+1)
    # T2.add_node(C+1)

    # set both nodes in T1 and T2 into the Graph T
    SC.add_nodes_from(T1.nodes, layer = 3)
    SC.add_nodes_from(T2.nodes, layer = 2)
    SC.add_nodes_from(T3.nodes, layer = 1)
    SC.add_nodes_from(T4.nodes, layer = 0)

    BIPARTITE(SC, T1, T2, E_12)
    BIPARTITE(SC, T2, T3, E_23)
    BIPARTITE(SC, T3, T4, E_34)

    return SC, T1, T2, T3, T4

def BIPARTITE(SC, T1, T2, E):
    n1list = list(T1)
    n2list = list(T2)
    n2list = n2list[:-1]

    edge_count = 0
    while edge_count < E:
        u = random.choice(n2list)
        v = random.choice(n1list)
        # add more conditions how to build the network properly
        if u == v or SC.has_edge(u, v): 
            continue
        else:
            SC.add_edge(u, v)
            edge_count += 1
    
    return SC, T1, T2

def SINGLE(SC, T1, T2):
    n1list = list(T2)
    n2list = list(T1)
    a = 1
    n2list = n2list[:-a]
    nodes = []

    for k in range(C):
        u_connectivity = 0

        u = n1list[k]
        for m in range(M):
            v = n2list[m]
            if SC.has_edge(u, v):
                connectivity = 1
            else:
                connectivity = 0

            u_connectivity += connectivity
        if u_connectivity == 0:
            nodes.append(u)
    
    return nodes

def RENAME(T1, M, string):
    new_name = {}
    redundanced_node = []
    node = list(T1.nodes)
    for n in range(M):
        u = node[n]
        for m in range(M):
            v = node[m]
            if(T1.has_edge(u, v)):
                redundancy = u.strip(string)
                new_name[v] = v + "|" + redundancy
                print(new_name)
    T1 = nx.relabel_nodes(T1, new_name)
    return T1

# if there is a node without edge connecting, then add one more edge connecting u ----- v
def NOWASTE(SC, T1, T2):
    
    n1list = list(T1)
    n1list = n1list[:-1]
    nodes = SINGLE(SC, T1, T2)

    while len(nodes) > 0:
        u = random.choice(nodes)
        v = random.choice(n1list)
        SC.add_edge(u, v)
        nodes.remove(u)
    
    return SC
    
def POSITION(SC):
    pos = nx.multipartite_layout(SC, subset_key = 'layer', align = "vertical", scale = 50)

    n = list(SC.nodes)
    for i in range(len(pos)):
        key = n[i]
        pos_n = pos[key]
        pos_n[0] = pos_n[0] * 8
    
    return pos

def main(args=None):
   
    T1 = GNP(M, pM, "M")
    T1_ad = nx.adjacency_matrix(T1)
    T1_A = T1_ad.todense()
    # present T1 out in matrix
    print(T1_A)

    # T2_ad = nx.adjacency_matrix(T2)
    # T2_A = T2_ad.todense()
    # # present T2 out in matrix
    # print(T2_A)

    # set the figure_size that present the image of the network
    plt.figure(1, figsize = (14, 14))
    # pos = {'M0': (2, 0), 'M1': (2, 2), 'M2': (2, 4)}
    # nx.draw_networkx_nodes(T1, pos = pos, node_size = 300, node_color = 'black')
    # nx.draw_networkx(T1, pos = pos, with_labels = True, node_size = 250, node_color = 'w')

    # nx.draw(T2, pos = pos, node_size = 300, with_labels = True)
    # nx.draw_networkx_nodes(T2, pos = pos, node_size = 250, node_color = 'black')
    # nx.draw_networkx_nodes(T2, pos = pos, node_size = 200, node_color = "w")
    # nx.draw(T2, node_size = 500, with_labels = True)
    
    SC, T1, T2, T3, T4 = TRANS(R, S, C, M, pR, pS, pC, pM, E_12, E_23, E_34)
    SC_ad = nx.adjacency_matrix(SC)
    SC_A = SC_ad.todense()
    print(SC_A)
    pos = POSITION(SC)
    nx.draw_networkx_nodes(SC, pos = pos, node_size = 500, node_color = 'black', node_shape = 'o')
    nx.draw_networkx(SC, pos = pos, with_labels = False, node_size = 450, node_color = 'w', node_shape = 'o')
    nx.draw_networkx_labels(SC, pos = pos, font_size = 10, font_color = 'black')
    
    # nx.draw_networkx_edges(T2, pos = pos, connectionstyle = "arc3, rad = 0.5")
    # nx.draw_networkx_edges(T3, pos = pos, connectionstyle = "arc3, rad = 0.5")
    # nx.draw_networkx_edges(T4, pos = pos, connectionstyle = "arc3, rad = 0.5")
    plt.axis("equal")

    # plt.figure(2, figsize = (10, 6))
    # SC = NOWASTE(SC, T1, T2)
    # pos = nx.multipartite_layout(SC, subset_key = "layer", align = "vertical", scale = 4)
    # nx.draw_networkx_nodes(SC, pos = pos, node_size = 250, node_color = 'black')
    # nx.draw_networkx_nodes(SC, pos = pos, node_size = 200, node_color = "w")
    # nx.draw_networkx_edges(SC, pos = pos)
    # nx.draw_networkx_edges(T1, pos = pos, connectionstyle = "arc3, rad = 0.5")
    # nx.draw_networkx_edges(T2, pos = pos, connectionstyle = "arc3, rad = 0.5")
    # plt.axis("equal")

    plt.show()
    # plt.savefig('labels.png')

if __name__ == "__main__":
    main()