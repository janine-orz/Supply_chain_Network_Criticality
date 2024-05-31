import networkx as nx
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random
import bezier
import string
import itertools
import collections
import openpyxl as opxl
import os
import powerlaw
from matplotlib.collections import LineCollection
from scipy import sparse
from scipy import optimize
import numpy as np
import statsmodels.api as sm

'''
DONE: 
    * to show the network in a multipartite_layout
    * to set a redundance in one TIER and rename the name of every node which has a redudanced node
    * deal with the networksize = [12, 20, 8, 3]
TO DO:
    * probability still need to consider how much would be the best choice (maybe need a small test)
    * still need to consider how to present the relationship of a parallel manufactory
'''

R = 8  * 1   # Ammount of [TIER IV]: raw material           *  Multi-Parameter
S = 20 * 1   # Ammount of [TIER III]: semifinisched products*  Multi-Parameter
C = 4  * 1   # Ammount of [TIER II]: components             *  Multi-Parameter
M = 2  * 1   # Ammount of [TIER I]: modules                 *  (Multi-Parameter // 2)
# pR = 0.06
# pS = 0.01
# pC = 0.01
# pM = 0.001
pR = 0.05 * 1   # the probability would not change
pS = 0.03 * 1
pC = 0.00 * 1
pM = 0.00 * 1
E_12 = R  / 2 * 1 # = #(node von R) / 2                      * (Multi-Parameter // 2)
E_23 = S  / 2 * 1 # = #(node von S) / 2                      * (Multi-Parameter // 2)
E_34 = M  / 2 * 1 # = #(node von M) / 2                      * (Multi-Parameter // 2)
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

    # ------------ TO DO ------------ # 
    # we should get only one parallel redundancy for one C

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
    # SC.add_edges_from(T1.edges, layer = 3)
    SC.add_nodes_from(T2.nodes, layer = 2)
    # SC.add_edges_from(T2.edges, layer = 2)
    SC.add_nodes_from(T3.nodes, layer = 1)
    # SC.add_edges_from(T3.edges, layer = 1)
    SC.add_nodes_from(T4.nodes, layer = 0)
    # SC.add_edges_from(T4.edges, layer = 0)
    

    BIPARTITE(SC, T1, T2, E_12)
    BIPARTITE(SC, T2, T3, E_23)
    BIPARTITE(SC, T3, T4, E_34)



    return SC, T1, T2, T3, T4


# connect between two Tiers
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
                if(len(redundancy) == 1):
                    redundancy = redundancy[0]
                elif(len(redundancy) > 1):
                    idx = redundancy.find("|")
                    # print(idx)
                    if(idx == -1):
                        redundancy = redundancy
                    else:
                        redundancy = redundancy[:idx]
                new_name[v] = v + "|" + redundancy
                T1 = nx.relabel_nodes(T1, new_name)
                node = list(T1.nodes)
            new_name = {}
            # if(new_name[v] != {}):
                # v = new_name[v]

    return T1


def POSITION(SC, par):
    pos = nx.multipartite_layout(SC, subset_key = 'layer', align = "vertical", scale = 10)

    n = list(SC.nodes)
    for i in range(len(pos)):
        key = n[i]
        # print("key :", key)
        pos_n = pos[key]
        # print("pos_n :", pos_n)
        pos_n[0] = float(pos_n[0] * 10)
        pos_n[1] = float(pos_n[1] + par)
    
    return pos


def PERFORM(SC, T1, T2, T3, T4, without_node):

    n1list = list(T1.nodes)
    n2list = list(T2.nodes)
    n3list = list(T3.nodes)
    n4list = list(T4.nodes)

    # calculate all adjazent edges of without_node
    if without_node != None:
        neighbor = list(SC.neighbors(without_node))
    
    SC_copy = SC.copy()

    partition = [n1list, n2list, n3list, n4list]
    perform_G = nx.algorithms.community.partition_quality(SC_copy, partition)
    # print(perform_G)

    # delete without_ndoe from the network
    if without_node == None:        
        partition = [n1list, n2list, n3list, n4list]
        perform = nx.algorithms.community.partition_quality(SC_copy, partition)
        return perform
    elif "M" in without_node:
        n1list.remove(without_node)
        SC_copy.remove_node(without_node)
    elif "C" in without_node:
        n2list.remove(without_node)
        SC_copy.remove_node(without_node)
    elif "S" in without_node:
        n3list.remove(without_node)
        SC_copy.remove_node(without_node)
    elif "R" in without_node:
        n4list.remove(without_node)
        SC_copy.remove_node(without_node)
    else:
        return SyntaxError

    partition = [n1list, n2list, n3list, n4list]
    perform = nx.algorithms.community.partition_quality(SC_copy, partition)

    SC_copy.add_node(without_node)
    for i in range(len(neighbor)):
        v = neighbor[i]
        SC_copy.add_edge(without_node, v)

    return perform


def CRITICAL_1(mapping_perform, SC, T1, T2, T3, T4):

    perform_all = PERFORM(SC, T1, T2, T3, T4, None)[1]
    perform_all = round(perform_all, 2)
    nlist = []
    nlist = ADD_IN_LIST(nlist, T1)
    nlist = ADD_IN_LIST(nlist, T2)
    nlist = ADD_IN_LIST(nlist, T3)
    nlist = ADD_IN_LIST(nlist, T4)

    for j in mapping_perform:
        without_node = j
        perform = PERFORM(SC, T1, T2, T3, T4, without_node)[1]
        perform = round(perform, 2)
        mapping_perform[j] = str(perform_all - perform)
    
    return mapping_perform

def ADD_IN_LIST(nlist, T1):

    n1list = list(T1.nodes)
    L = len(n1list)
    for i in range(L):
        nlist.append(n1list[i])
    
    return nlist

def ADJ_IN_ONE_TIER(SC):
    node_list_a = []
    for node in SC.nodes:
        for edge in SC.edges:
            if (node == edge[0]) or (node == edge[1]):
                node_list_a.append(node)
    node_list_a = list(set(node_list_a))
    # print("2. node_list_a : ", node_list_a)
    return node_list_a

def NUM_OF_NODE_AIOT(node_list_a): # AIOT: adjacent in on tier
    number_of_node_R = 0
    number_of_node_S = 0
    number_of_node_C = 0
    for node in node_list_a:
        if (node[0] == "R"):
            number_of_node_R += 1
        elif (node[0] == "S"):
            number_of_node_S += 1
        elif (node[0] == "C"):
            number_of_node_C += 1
    # print("2. R:", number_of_node_R, "S:", number_of_node_S, "C:", number_of_node_C)
    return number_of_node_R, number_of_node_S, number_of_node_C

def NODE_LIST(SC, string):
    node_list = []
    for node in SC.nodes:
        if (node[0] == string):
            node_list.append(node)
    node_list = list(reversed(node_list))
    # print("2, node_list:" , node_list)
    return node_list

def NETW_INTO_MATRIX(SC_a, R, S, C):
    SC_ad = nx.adjacency_matrix(SC_a)
    SC_A = SC_ad.todense()
    SC_A_flip180 = SC_A.reshape(SC_A.size)
    SC_A_flip180 = SC_A_flip180[::-1]
    SC_A_flip180 = SC_A_flip180.reshape((R+S+C+M, R+S+C+M))
    
    SC_A_R = [row[:R] for row in SC_A_flip180[:R]]
    SC_A_R = np.array(SC_A_R)
    # print(SC_A_R)
    SC_A_R_flip180 = SC_A_R.reshape(SC_A_R.size)
    SC_A_R_flip180 = SC_A_R_flip180[::-1]
    SC_A_R_flip180 = SC_A_R_flip180.reshape((R, R))
    # print("SC_A_R : \n", SC_A_R_flip180)
    SC_A_S = [row[R:(S+R):1] for row in SC_A_flip180[R:(S+R):1]]
    SC_A_S = np.array(SC_A_S)
    # print(SC_A_S)
    SC_A_S_flip180 = SC_A_S.reshape(SC_A_S.size)
    SC_A_S_flip180 = SC_A_S_flip180[::-1]
    SC_A_S_flip180 = SC_A_S_flip180.reshape((S, S))
    # print("SC_A_S : \n", SC_A_S_flip180)
    SC_A_C = [row[(S+R):(S+R+C):1] for row in SC_A_flip180[(S+R):(S+R+C):1]]
    SC_A_C = np.array(SC_A_C)
    SC_A_C_flip180 = SC_A_C.reshape(SC_A_C.size)
    SC_A_C_flip180 = SC_A_C_flip180[::-1]
    SC_A_C_flip180 = SC_A_C_flip180.reshape((C, C))
    # print("SC_A_C : \n", SC_A_C_flip180)
    return SC_A_R_flip180, SC_A_S_flip180, SC_A_C_flip180

def List_at_x_axis(node_list_R, node_list_S, SC_a, SC_d, SC_e, minus_par, pos_cont2, list_on_2nd_line):
    for node in node_list_R:
        list_var = list(SC_d.neighbors(node))
        list_var_b = list(SC_a.neighbors(node))
        print("===List_at_x_axis - R=== ", list_var)
        print("===List_at_x_axis - R=== ", list_var_b)
        for node1 in list_var:
            if ("M" in node1) and (len(list_var_b) == 0):
                pos_cont2[node] = (0, (1 + minus_par))
                print("pos_cont2[", node, "] = ", pos_cont2[node])
                list_on_2nd_line.append(node)
                SC_e.add_edge(node, node1)
                minus_par = minus_par + 1
                
            elif ("C" in node1) and (len(list_var_b) == 0):
                pos_cont2[node] = (0, (1 + minus_par))
                print("pos_cont2[", node, "] = ", pos_cont2[node])
                list_on_2nd_line.append(node)
                SC_e.add_edge(node, node1)
                minus_par = minus_par + 1
            
            elif(len(list_var_b) != 0):
                print(node,"stay on x-axis")

    for node in node_list_S:
        print(node)
        list_var = list(SC_d.neighbors(node))
        list_var_b = list(SC_a.neighbors(node))
        print("===List_at_x_axis - S=== ", list_var)
        print("===List_at_x_axis - S=== ", list_var_b)
        for node1 in list_var:
            if ("M" in node1) and (len(list_var_b) == 0):
                pos_cont2[node] = (0, 1 + minus_par)
                print("pos_cont2[", node, "] = ", pos_cont2[node])
                list_on_2nd_line.append(node)
                SC_e.add_edge(node, node1)
                minus_par = minus_par + 1
            elif(len(list_var_b) != 0):
                print(node,"stay on x-axis")
    
    return list_on_2nd_line, SC_e, pos_cont2

def parallel_supply_chain(node_list_R, node_list_S, SC_a, SC_d, node_list_pos_cont, len_node_list_pos_cont, SC_e, pos_cont, pos_cont2):
    for node in node_list_R:
        list_var = list(SC_d.neighbors(node))
        list_var_b = list(SC_a.neighbors(node))
        for node1 in list_var:
            if ("M" in node1) and (len(list_var_b) == 0):
                name_node1 = node1 + "|" + node1[-1]
                SC_e.add_node(name_node1)
                pos_cont[name_node1] = (pos_cont[node1][0], pos_cont2[node][1])
                SC_e.add_edge(node, name_node1)
                SC_e.remove_edge(node, node1)
                if(node_list_pos_cont.index(node1) < len_node_list_pos_cont-1):
                    node3_idx = node_list_pos_cont.index(node1)
                    node3 = node_list_pos_cont[node3_idx+1]
                    SC_e.add_edge(name_node1, node3)
                
                if(node_list_pos_cont.index(node1) != len_node_list_pos_cont):
                    node2_idx = node_list_pos_cont.index(node1)
                    node2 = node_list_pos_cont[node2_idx-1]
                    SC_e.add_edge(node2, node)
                    pos_cont[node] = (pos_cont[node2][0], pos_cont2[node][1])
                elif(node_list_pos_cont.index(node1) == len_node_list_pos_cont):
                    pos_cont[node] = (pos_cont[node2][0], pos_cont2[node][1])
                else:
                    print("ERROR:", node, "--->", node1)
        
            elif ("C" in node1) and (len(list_var_b) == 0):
                name_node1 = node1 + "|" + node1[-1]
                SC_e.add_node(name_node1)
                pos_cont[name_node1] = (pos_cont[node1][0], pos_cont2[node][1])
                SC_e.add_edge(node, name_node1)
                SC_e.remove_edge(node, node1)

                if(node_list_pos_cont.index(node1) < len_node_list_pos_cont-1):
                    node3_idx = node_list_pos_cont.index(node1)
                    node3 = node_list_pos_cont[node3_idx+1]
                    SC_e.add_edge(name_node1, node3)
                
                if(node_list_pos_cont.index(node1) != len_node_list_pos_cont):
                    node2_idx = node_list_pos_cont.index(node1)
                    node2 = node_list_pos_cont[node2_idx-1]
                    SC_e.add_edge(node2, node)
                    pos_cont[node] = (pos_cont[node2][0], pos_cont2[node][1])
            
            else:
                    print("ERROR", node, "--->", node1)
            
    
    for node in node_list_S:
        list_var = list(SC_d.neighbors(node))
        list_var_b = list(SC_a.neighbors(node))
        for node1 in list_var:
            if ("M" in node1) and (len(list_var_b) == 0):
                name_node1 = node1 + "|" + node1[-1]
                SC_e.add_node(name_node1)
                pos_cont[name_node1] = (pos_cont[node1][0], pos_cont2[node][1])
                SC_e.add_edge(node, name_node1)
                SC_e.remove_edge(node, node1)
                if(node_list_pos_cont.index(node1) < len_node_list_pos_cont-1):
                    node3_idx = node_list_pos_cont.index(node1)
                    node3 = node_list_pos_cont[node3_idx+1]
                    SC_e.add_edge(name_node1, node3)
                
                if(node_list_pos_cont.index(node1) != len_node_list_pos_cont):
                    node2_idx = node_list_pos_cont.index(node1)
                    node2 = node_list_pos_cont[node2_idx-1]
                    SC_e.add_edge(node2, node)
                    pos_cont[node] = (pos_cont[node2][0], pos_cont2[node][1])
                elif(node_list_pos_cont.index(node1) == len_node_list_pos_cont):
                    pos_cont[node] = (pos_cont[node2][0], pos_cont2[node][1])

    return pos_cont, SC_e

def CONTRACTED_POS(pos, SC_A_R, IDX, num, node_list, list_on_2nd_line):
    # print("SC_A:\n", SC_A_R)
    node_list_rdy = []
    # print("num = ", num)
    node_hor = []
    node_ver = []

    if(IDX == R):
        test = [x for x in list_on_2nd_line if "R" in x]
        test = len(test)
        IDX = IDX - test
    elif(IDX == S):
        test = [x for x in list_on_2nd_line if "S" in x]
        test = len(test)
        IDX = IDX - test

    for i in range(IDX):
        for j in range(i, IDX):
            if SC_A_R[i][j] == 1:
                # the node in the j-th horizontal line directing to the node in the i-th vertical line
                # print("(H|", j, "---> V|", i, ")", SC_A_R[i][j]) # V: Vertical index, H: Horizontal index
                # write down the index of the horizontal line
                node_hor.append(j)
                # write down the index of the horizontal line
                node_ver.append(i)
                # with an edge from j---->i
    
    # since j ---> i
    # the i-th vertical(node) should be on the right handside of the j-th horizontal(node)
    # so we set the position of the VERTICAL nodes AT FIRST
    # count how many times the element is in the list
    result = collections.Counter(node_ver)
    # print("result = ", result)
    node_ver = set(node_ver)
    # print("node_ver : ", node_ver)
    var = -100

    # if there is no nodes connecting in the same Tier, then STOP
    if(len(node_hor) == 0)and(len(node_ver) == 0):
        # print("(node_hor == [])and(node_ver == [])")
        return pos, node_list_rdy, IDX
    
    node_mid = [x for x in node_hor if x in node_ver]
    # print(node_mid)
    node_ver = [x for x in node_ver if x not in node_mid]
    # print(node_ver)
    node_hor = [x for x in node_hor if x not in node_mid]
    # print(node_hor)
    
    # print("===VER===")
    # we use the FOR-loop to check which node should be on the right-est
    for node in node_ver:
        print("===node:", node, "===")
        if(var == -100) and (node_list[node] not in list_on_2nd_line):
            var = result[node] # how many times the node is connected in one tier
            nodevar = node
            pos[nodevar] = IDX - 1
            print("pos[", nodevar, "]", pos[nodevar])
            IDX = pos[nodevar]
            node_list_rdy.append(nodevar)
        elif(node not in node_hor) and (node_list[node] not in list_on_2nd_line):
            var = result[node]
            nodevar = node
            pos[nodevar] = IDX - 1
            print("pos[", nodevar, "]", pos[nodevar])
            # print(pos[nodevar])
            IDX = pos[nodevar]
            node_list_rdy.append(nodevar)
    
    # print("===MID===")
    for node in node_mid:
        # print("node : ", node)
        # print(node_list[node], (node_list[node] not in list_on_2nd_line))
        if(node not in node_ver) and (node not in node_hor):
            nodevar = node
            pos[nodevar] = IDX - 1
            IDX = pos[nodevar]
            node_list_rdy.append(nodevar)
            # print(pos[nodevar])

    # print("===HOR===")
    result = collections.Counter(node_hor)
    node_hor = set(node_hor)
    var = -100
    
    for node in node_hor:
        # print("node : ", node)
        # print((result[node] >= var), result[node], var)
        # print((node not in node_ver))
        # print(node_list[node], (node_list[node] not in list_on_2nd_line))
        if(var == -100) and (node not in node_ver) and (node_list[node] not in list_on_2nd_line):
            var = result[node]
            nodevar = node
            pos[nodevar] = IDX - 1
            # print(pos[nodevar])
            IDX = pos[nodevar]
            node_list_rdy.append(nodevar)
        elif(node not in node_ver) and (node_list[node] not in list_on_2nd_line):
            var = result[node]
            nodevar = node
            pos[nodevar] = IDX - 1
            # print(pos[nodevar])
            IDX = pos[nodevar]
            node_list_rdy.append(nodevar)

    return pos, node_list_rdy, IDX

def CONTRACTED_POS_CONT_AIOT(pos_cont, pos_cont_new, IDX, node_list, node_list_rdy, list_on_2nd_line, string):
    if(string == "R"):
        add_par = 0
        l = list(range(R))
    elif(string == "S"):
        add_par = R
        l = list(range(S))
    elif(string == "C"):
        add_par = R + S
        l = list(range(C))
    
    l = np.array([x for x in l if x not in node_list_rdy])
    # print(l, type(l))
    for x in node_list_rdy:
        key = node_list[x]
        pos_cont_new[key] = (pos_cont[x] + add_par, 0)
        # print("pos_cont_new[", key, "] = ", pos_cont_new[key])
    for i in l:
        # print(i)
        key = node_list[i]
        if(pos_cont.get(i) == None) and (key not in list_on_2nd_line):
            pos_cont_new[key] = (IDX - 1 + add_par, 0)
            IDX = IDX - 1
            # print("pos_cont_new[", key, "] = ", pos_cont_new[key])
    return pos_cont_new


def CONTRACTED_POS_CONT_FILL(pos_cont, SC):
    idx = R+S+C
    for node2 in SC.nodes: #reposition the nodes WITHOUT adjazent nodes in one Tier
        if("M" in node2):
            pos_cont[node2] = (idx, 0)
            idx += 1
    return pos_cont

def ADD_EDGE_TO_NEXT_TIER(node, string1, number, next_tier, SC_d, SC_b):
    if (string1 in node):
        neighbours = list(SC_b.neighbors(node))
        
        neighbours_1 = neighbours.copy()
        if(string1 == "S") and (len(neighbours) != 0):
            l = len(neighbours)
            for node1 in neighbours:
                if("R" in node1):
                    neighbours_1.remove(node1)
        elif(string1 == "C") and (len(neighbours) != 0):
            l = len(neighbours)
            for node1 in neighbours:
                if("R" in node1) or ("S" in node1):
                    neighbours_1.remove(node1)
        neighbours = neighbours_1

        if (len(neighbours) == 0):
            par = 1
        for node2 in neighbours:
            if(string1 in node2):
                par = 1
            else:
                par = 0
        if(par == 1):
            rand_idx = random.randrange(number)
            node_rand = next_tier[rand_idx]
            SC_d.add_edge(node, node_rand)
            SC_b.add_edge(node, node_rand)
    return SC_d, SC_b

def bubble_sort(length, list_pos_cont):
    for i in range(length-1):
        swapped = False
        for j in range(length-i-1):
            key1 = list_pos_cont[j][0]
            coor1 = list_pos_cont[j][1]
            key2 = list_pos_cont[j+1][0]
            coor2 = list_pos_cont[j+1][1]
            if coor1[0] > coor2[0]:
                swapped = True
                list_pos_cont[j] = (key2, coor2)
                list_pos_cont[j+1] = (key1, coor1)
        if not swapped:
            return list_pos_cont

def plot_degree_dist(G, m, color):
    plt.subplot(2,2,m)
    degree_hist = nx.degree_histogram(G)
    degree_hist = np.array(degree_hist, dtype = float)
    degree_prob = degree_hist/G.number_of_nodes()
    plt.bar(np.arange(degree_prob.shape[0]), degree_prob, color = color)
    plt.plot(np.arange(degree_prob.shape[0]), degree_prob, '-', color = color)

    plt.tick_params(axis='both', which = 'major', labelsize = 8)
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')    

def probability_mass_fct(G, SC):
    plt.subplot(2,2,2)
    deg_G = dict(G.degree()).values()
    deg_distri_G = collections.Counter(deg_G)
    deg_SC = dict(SC.degree()).values()
    deg_distri_SC = collections.Counter(deg_SC)
    
    x_G = []
    y_G = []
    for i_G in sorted(deg_distri_G):
        x_G.append(i_G)
        y_G.append(deg_distri_G[i_G]/len(G))
    
    x_SC = []
    y_SC = []
    for i_SC in sorted(deg_distri_SC):
        x_SC.append(i_SC)
        y_SC.append(deg_distri_SC[i_SC]/len(SC))

    plt.plot(x_G, y_G)
    plt.plot(x_SC, y_SC)

    plt.xlabel('Degree')
    plt.ylabel('P(k)')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.yscale('log')
    plt.xscale('log')

def multi_linear_reg(perform_SC_b, corr_x, corr_y, corr_z):
    y = np.array(list(perform_SC_b.values()), dtype = 'float')
    x = np.array([list(corr_x), list(corr_y), list(corr_z)], dtype = 'float')
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for element in x[1:]:
        X = sm.add_constant(np.column_stack((element, X)))
    results = sm.OLS(y, X, missing = 'drop').fit()
    return results

def main(args=None):
     
    SC, T1, T2, T3, T4 = TRANS(R, S, C, M, pR, pS, pC, pM, E_12, E_23, E_34)
    # print("SC: \t", SC)
    # print("SC.EDGES : \t", SC.edges)
    SC_a = SC.copy()
    # print("SC_a: \t", SC_a)
    # print("SC_a.EDGES : \t", SC_a.edges)
    SC_b = SC.copy()
    # print("SC_b: \t", SC_b)
    # print("SC_b.EDGES : \t", SC_b.edges)
    SC_b.add_edges_from(T2.edges, layer = 2)
    SC_b.add_edges_from(T3.edges, layer = 1)
    SC_b.add_edges_from(T4.edges, layer = 0)
    
    # print("SC_c: \t", SC_c)

    pos = POSITION(SC, 0)
    # print("===pos===\n", pos)

    # SC_a contains only the edges conneting two nodes in one tier
    # SC contains only the edges conneting two nodes in different tiers
    # SC_b contains only the edges conneting all nodes    
            

    mapping_perform = pos.copy()
    mapping_perform = CRITICAL_1(mapping_perform, SC_a, T1, T2, T3, T4)
    mapping_perform = CRITICAL_1(mapping_perform, SC_b, T1, T2, T3, T4)
    perform_SC_b = mapping_perform
    mapping_perform = CRITICAL_1(mapping_perform, SC, T1, T2, T3, T4)
    # print("=====after CRITICAL_1=====")
    

    # plt.figure(1, figsize = (10, 6))
    
    # nx.draw_networkx_nodes(SC, pos = pos, node_size = 500, node_color = 'black', node_shape = 'o')
    # nx.draw_networkx_nodes(SC, pos = pos, node_size = 450, node_color = 'w', node_shape = 'o')
    # nx.draw_networkx(SC, pos = pos, with_labels = False, node_size = 450, node_color = 'w', node_shape = 'o')

    # nx.draw_networkx_labels(SC, pos = pos, font_size = 10, font_color = 'black')
    # plt.axis("equal")
    
    # for node in nodes SC_a

    # plt.figure(2, figsize = (10, 6))

    edge = SC_b.edges
    edgelist= list(edge)
    length = len(edgelist)
    # print("\nSC: \t", SC)
    # print("SC.EDGES : \t", SC.edges)
    # print("\nSC_b: \t", SC_b)
    # print("SC_b.EDGES : \t", SC_b.edges)
    SC_a.remove_edges_from(edge)
    for i in range(length):
        edge_uv = edgelist[i]
        node_u = edge_uv[0]
        node_v = edge_uv[1]
        if node_u[0] == node_v[0]:
            SC_a.add_edge(node_u, node_v)
            SC_b.remove_edge(node_u, node_v)
    # print("\n1. SC_a: \t", SC_a)
    # print("SC_a.EDGES : \t", SC_a.edges)

    R_node_list = []
    S_node_list = []
    C_node_list = []
    M_node_list = []
    for u in SC.nodes:
        if("R" in u):
            R_node_list.append(u)
        elif("S" in u):
            S_node_list.append(u)
        elif("C" in u):
            C_node_list.append(u)
        elif("M" in u):
            M_node_list.append(u)
    # print("R_node_list", R_node_list)
    # print("S_node_list", S_node_list)
    # print("C_node_list", C_node_list)
    # print("M_node_list", M_node_list)

    adj_nodes = []
    for edge1 in SC_a.edges: # for edges in one tier
        node1 = edge1[0]
        node2 = edge1[1]
        par = 0
        # print("node1", node1, "node2", node2)
        for edge2 in SC.edges: # for edges in different tiers
            node3 = edge2[0]
            node4 = edge2[1]
            # print("node3", node3, "node4", node4)
            if(node1 == node3): 
                # if the starting point of two edges are the same
                # then connect the other nodes in the same tier into the next tier
                # print("adding to SC & SC_b edge (", node2, "---- ", node4,")")
                SC.add_edge(node2, node4)
                SC_b.add_edge(node2, node4)
            elif(node2 == node3):
                # print("adding to SC & SC_b edge (", node1, "---- ", node4,")")
                SC.add_edge(node1, node4)
                SC_b.add_edge(node1, node4)
    for edge in SC_b.edges:
        adj_nodes.append(edge[0])
        adj_nodes.append(edge[1])
    
    SC_c = SC_b.copy()
    edges = SC.edges()
    # print("edges : ", edges)
    SC_d = SC_c.copy()
    # print(SC_d)
    # print("adj_nodes : ", adj_nodes)
    # print("SC before adding edge", SC)
    # ADDING AN EXTRA POSSIBILITY CONTAINS ALL NODES ON THE RIGHT HANDSIDE
    S_C_M_node_list = []
    S_C_M_node_list.extend(S_node_list)
    S_C_M_node_list.extend(C_node_list)
    S_C_M_node_list.extend(M_node_list)
    C_M_node_list = []
    C_M_node_list.extend(C_node_list)
    C_M_node_list.extend(M_node_list)
    for node in SC_a.nodes:
        SC_d, SC_b = ADD_EDGE_TO_NEXT_TIER(node, "R", S+C+M, S_C_M_node_list, SC_d, SC_b) 
        # SC_d, SC_b = ADD_EDGE_TO_NEXT_TIER(node, "R", S, S_node_list, SC_d, SC_b) 
        SC_d, SC_b = ADD_EDGE_TO_NEXT_TIER(node, "S", C+M, C_M_node_list, SC_d, SC_b)
        # SC_d, SC_b = ADD_EDGE_TO_NEXT_TIER(node, "S", C, C_node_list, SC_d, SC_b)
        SC_d, SC_b = ADD_EDGE_TO_NEXT_TIER(node, "C", M, M_node_list, SC_d, SC_b)
    SC_d.remove_edges_from(edges)


    # print("\nSC: \t", SC)
    # print("SC.EDGES : \t", SC.edges)
    # print("\nSC_b: \t", SC_b)
    # print("SC_b.EDGES : \t", SC_b.edges)
    # print("\n2. SC_a: \t", SC_a)
    # print("SC_a.EDGES : \t", SC_a.edges)

    SC_ad = nx.adjacency_matrix(SC_a)
    SC_A = SC_ad.todense()
    SC_A_flip180 = SC_A.reshape(SC_A.size)
    SC_A_flip180 = SC_A_flip180[::-1]
    SC_A_flip180 = SC_A_flip180.reshape((R+S+C+M, R+S+C+M))
    
    SC_A_R = [row[:R] for row in SC_A_flip180[:R]]
    SC_A_R = np.array(SC_A_R)
    # print(SC_A_R)
    
    # print("\n3. SC_a: \t", SC_a)
    # print("SC_a.EDGES : \t", SC_a.edges)
    

    plt.figure(1)
    nx.draw_networkx_edges(SC, pos = pos)
    nx.draw_networkx_edges(SC_a, pos = pos, edge_color = 'red', connectionstyle = "arc3, rad = 0.5", arrows = True)
    nx.draw_networkx_edges(SC_d, pos = pos, edge_color = 'blue', connectionstyle = "arc3, rad = -0.5", arrows = True)
    nx.draw_networkx_nodes(SC, pos = pos, node_size = 500, node_color = 'black', node_shape = 'o')
    nx.draw_networkx_nodes(SC, pos = pos, node_size = 450, node_color = 'w', node_shape = 'o')
    nx.draw_networkx_labels(SC, pos = pos, font_size = 10, font_color = 'black')
    

    pos_cont = {}
    # check how many nodes are adjacent to another node in its tier    
    node_list_a = ADJ_IN_ONE_TIER(SC_a)
    # print(node_list_a)
    
    # number of node in one Tier that is adjacent
    number_of_node_R, number_of_node_S, number_of_node_C = NUM_OF_NODE_AIOT(node_list_a)

    idx_R = number_of_node_R
    idx_S = number_of_node_R + number_of_node_S
    idx_C = number_of_node_R + number_of_node_S + number_of_node_C
    # print(idx_R, " - ", R, "\t", idx_S, " - ", S, "\t", idx_C, " - ", C)

    node_list_R = NODE_LIST(SC, "R")
    node_list_R = node_list_R[::-1]
    # print("node_list_R:", node_list_R)
    node_list_S = NODE_LIST(SC, "S")
    node_list_S = node_list_S[::-1]
    # print("node_list_S:", node_list_S)
    node_list_C = NODE_LIST(SC, "C")
    node_list_C = node_list_C[::-1]
    node_list_M = NODE_LIST(SC, "M")
    node_list_M = node_list_M[::-1]
    # print("node_list_M:", node_list_M)
    
    pos_cont2 = {}
    list_on_2nd_line = []
    minus_par = 0
    SC_e = nx.DiGraph()
    SC_e.add_nodes_from(SC_a)
    list_on_2nd_line, SC_e, pos_cont2 = List_at_x_axis(node_list_R, node_list_S, SC_a, SC_d, SC_e, minus_par, pos_cont2, list_on_2nd_line)
    
    print(pos_cont2)

    # turn the network in to Matrix and focus only on Tier R
    SC_A_R, SC_A_S, SC_A_C = NETW_INTO_MATRIX(SC_a, R, S, C)

    pos = {}
    pos_cont = {}
    pos_cont_new = {}

    # ==========================20240531contracted_pos==========================
    # print("SC_A:\n", SC_A_R)
    node_list_rdy = []
    # print("num = ", num)
    node_hor = []
    node_ver = []
    node_hor_to_ver = {}

    IDX = S
    test = [x for x in list_on_2nd_line if "S" in x]
    test = len(test)
    IDX = IDX - test
    for i in range(IDX):
        for j in range(i, IDX):
            if SC_A_S[i][j] == 1:
                # the node in the j-th horizontal line directing to the node in the i-th vertical line
                print("(H|", j, "---> V|", i, ")", SC_A_S[i][j]) # V: Vertical index, H: Horizontal index
                # write down the index of the horizontal line
                node_hor.append(j)
                # write down the index of the horizontal line
                node_ver.append(i)
                node_hor_to_ver[j] = i
                # j:i  <=>  j-->i
                # with an edge from j---->i
    
    # since j ---> i
    # the i-th vertical(node) should be on the right handside of the j-th horizontal(node)
    # so we set the position of the VERTICAL nodes AT FIRST
    # count how many times the element is in the list
    result_ver = collections.Counter(node_ver)
    result_hor = collections.Counter(node_hor)
    # print("result = ", result)
    # print("node_ver : ", node_ver)
    var = -100

    # if there is no nodes connecting in the same Tier, then STOP
    if(len(node_hor_to_ver) == 0):
        print("(node_hor_to_ver == [])")
    
    # node_mid = [x for x in node_hor if x in node_ver]
    # # print(node_mid)
    # node_ver = [x for x in node_ver if x not in node_mid]
    # print(node_ver)
    # node_hor = [x for x in node_hor if x not in node_mid]
    # print(node_hor)
    
    # print("===VER===")
    # we use the FOR-loop to check which node should be on the right-est
    for i in range(len(node_ver)):
        node = node_ver[i]
        adjnode = node_hor[i]
        
        var_node = result_ver[node] # how many times the node is connected in one tier
        var_adjnode = result_hor[adjnode]
        if(node in node_hor)or(adjnode in node_ver):
            node_mid_h = [x for x in node_hor if x in node_ver]
            node_mid_v = [x for x in node_ver if x in node_hor]
            print("node_mid:", node_mid_h, "\n", node_mid_v)
            node_ver = [x for x in node_ver if x not in node_mid_v]
            print("node_ver:", node_ver, len(node_ver))
            node_hor = [x for x in node_hor if x not in node_mid_h]
            print("node_hor:", node_hor, len(node_hor))
            result_m_v = collections.Counter(node_mid_v)
            result_m_h = collections.Counter(node_mid_h)
            result_v = collections.Counter(node_ver)
            result_h = collections.Counter(node_hor)
            for node in node_mid_h:
                print(node_hor_to_ver.keys())
                print(node_hor_to_ver.values())
                if node in node_hor_to_ver.values():
                    print(node)
                    #-----------------20240531------------------#
                
                

        if(var_node == 1) and (var_adjnode == 1) and (node not in pos.keys()):
            pos[node] = IDX - 1
            pos[adjnode] = IDX -2
            IDX = pos[adjnode]
        elif(var_node == 1) and (var_adjnode != 1) and (node not in pos.keys()):
            print(var_node, var_adjnode)
            j_list = [j for j,val in enumerate(node_hor) if val == adjnode]
            print("j_list", j_list)
            for j in j_list:
                pos[node_ver[j]] = IDX-1
                print("pos[", S_node_list[node_ver[j]], "] = ", IDX-1)
                IDX = pos[node_ver[j]]
            pos[adjnode] = IDX-1
            print("pos[", S_node_list[adjnode], "] = ", IDX-1)
            IDX = pos[node]
        elif(var_node != 1) and (var_adjnode == 1) and (node not in pos.keys()):
            print(var_node, var_adjnode)
            j_list = [j for j,val in enumerate(node_ver) if val == node]
            print("j_list", j_list)
            pos[node] = IDX-1
            print("pos[", S_node_list[node], "] = ", IDX-1)
            IDX = pos[node]
            for j in j_list:
                pos[node_hor[j]] = IDX-1
                print("pos[", S_node_list[node_hor[j]], "] = ", IDX-1)
                IDX = pos[node_hor[j]]
            

        
        
    print(pos)

    # # print("===MID===")
    # for node in node_mid:
    #     # print("node : ", node)
    #     # print(node_list[node], (node_list[node] not in list_on_2nd_line))
    #     if(node not in node_ver) and (node not in node_hor):
    #         nodevar = node
    #         pos[nodevar] = IDX - 1
    #         IDX = pos[nodevar]
    #         node_list_rdy.append(nodevar)
    #         # print(pos[nodevar])

    # # print("===HOR===")
    # result = collections.Counter(node_hor)
    # node_hor = set(node_hor)
    # var = -100
    
    # for node in node_hor:
    #     # print("node : ", node)
    #     # print((result[node] >= var), result[node], var)
    #     # print((node not in node_ver))
    #     # print(node_list[node], (node_list[node] not in list_on_2nd_line))
    #     if(var == -100) and (node not in node_ver) and (R_node_list[node] not in list_on_2nd_line):
    #         var = result[node]
    #         nodevar = node
    #         pos[nodevar] = IDX - 1
    #         # print(pos[nodevar])
    #         IDX = pos[nodevar]
    #         node_list_rdy.append(nodevar)
    #     elif(node not in node_ver) and (R_node_list[node] not in list_on_2nd_line):
    #         var = result[node]
    #         nodevar = node
    #         pos[nodevar] = IDX - 1
    #         # print(pos[nodevar])
    #         IDX = pos[nodevar]
    #         node_list_rdy.append(nodevar)
    # ==========================20240531contracted_pos==========================
    # pos_cont, node_list_rdy, IDX = CONTRACTED_POS(pos, SC_A_R, R, number_of_node_R, node_list_R, list_on_2nd_line)
    # pos_cont = CONTRACTED_POS_CONT_AIOT(pos_cont, pos_cont_new, IDX, node_list_R, node_list_rdy, list_on_2nd_line, "R")
    print(pos_cont)

    '''
    pos = {}
    # pos_cont = {}
    pos_cont, node_list_rdy, IDX = CONTRACTED_POS(pos, SC_A_S, S, number_of_node_S, node_list_S, list_on_2nd_line)
    pos_cont = CONTRACTED_POS_CONT_AIOT(pos_cont, pos_cont_new, IDX, node_list_S, node_list_rdy, list_on_2nd_line, "S")
    print(pos_cont)


    pos = {}
    # pos_cont = {}
    pos_cont, node_list_rdy, IDX = CONTRACTED_POS(pos, SC_A_C, C, number_of_node_C, node_list_C, list_on_2nd_line)
    pos_cont = CONTRACTED_POS_CONT_AIOT(pos_cont, pos_cont_new, IDX, node_list_C, node_list_rdy, list_on_2nd_line, "C")
    pos_cont = CONTRACTED_POS_CONT_FILL(pos_cont, SC)
    print(pos_cont)

    
    # print(pos_cont)

    
    SC_f = nx.DiGraph()
    list_pos_cont = list(pos_cont.items())
    length = len(list_pos_cont)
    list_pos_cont1 = list_pos_cont.copy()
    # delete the node not at the same line
    for l in range(length):
        if(list_pos_cont[l][1][1] != 0):
            list_pos_cont1.remove(list_pos_cont[l])

    list_pos_cont = list_pos_cont1
    length = len(list_pos_cont)
    list_pos_cont = bubble_sort(length, list_pos_cont)
    # adding edges between nodes at y=0 axis
    for l in range(length-1):
        node1 = list_pos_cont[l][0]
        node2 = list_pos_cont[l+1][0]
        # print(list_pos_cont[l][1][0] == 0)
        if(list_pos_cont[l][1][1] == 0) and (list_pos_cont[l+1][1][1] == 0):
            SC_f.add_edge(node1, node2)

    pos_cont = dict(list_pos_cont)
    pos_cont.update(pos_cont2)
    print(pos_cont)
    
    node_list_pos_cont = []
    for i in list_pos_cont:
        node_list_pos_cont.append(i[0])
    len_node_list_pos_cont = len(node_list_pos_cont)
    print(len_node_list_pos_cont)

    pos_cont, SC_e = parallel_supply_chain(node_list_R, node_list_S, SC_a, SC_d, node_list_pos_cont, len_node_list_pos_cont, SC_e, pos_cont, pos_cont2)

    SC_h = nx.DiGraph()
    for edge in SC_a.edges:
        coord = (pos_cont[edge[0]][0] + 1, 0)
        key = [key for key, value in pos_cont.items() if value == coord]
        while(key == []):
            coord = (coord[0] + 1, 0)
            key = [key for key, value in pos_cont.items() if value == coord]
        print(edge[1], "--", key)
        SC_h.add_edge(edge[1], key[0])
        # print(node_list_pos_cont.index(edge[0]))
        # print(node_list_pos_cont.index(edge[0])+1)
        # print(node_list_pos_cont[node_list_pos_cont.index(edge[0])+1])

    SC_g = SC_e.copy()
    SC_g.add_edges_from(SC_f.edges)
    SC_g.add_edges_from(SC_h.edges)
    # SC_g.add_node('Start')
    # SC_g.add_node('OEM')
    # pos_cont['Start'] = (pos_cont['R-1'][0]-3, pos_cont['R-1'][1])
    # pos_cont['OEM'] = (pos_cont['M-1'][0]+3, pos_cont['M-1'][1])
    # SC_g.add_edge('Start', 'R-1')
    # SC_g.add_edge('M-1', 'OEM')
    # SC_g.add_edge('M-1|1', 'OEM')
    start = list_pos_cont[0][0]
    # print([p for p in nx.all_simple_paths(SC_g, source=start, target = 'M-1'or'M-1|1')])

    plt.figure(6)
    nx.draw_networkx_edges(SC_h, pos = pos_cont, edge_color = 'blue', connectionstyle = "arc3, rad = 0.3", arrows = True)
    nx.draw_networkx_edges(SC_e, pos = pos_cont, edge_color = 'green', connectionstyle = "arc3, rad = 0.3", arrows = True)
    nx.draw_networkx_edges(SC_f, pos = pos_cont, edge_color = 'black', arrows = True)
    nx.draw_networkx_nodes(SC_g, pos = pos_cont, node_size = 500, node_color = 'black', node_shape = 'o')
    nx.draw_networkx_nodes(SC_g, pos = pos_cont, node_size = 450, node_color = 'w', node_shape = 'o')
    nx.draw_networkx_labels(SC_g, pos = pos_cont, font_size = 10, font_color = 'black')
    
    # nx.draw_networkx_edges(SC_c, pos = pos_cont, edge_color = 'blue', connectionstyle = "arc3, rad = 0.5", arrows = True)

    '''
    '''
    
    # pos_cont = CONTRACTED_POS_CONT_FILL(pos_cont, SC, idx_R)
    plt.figure(2)
    colors = list('rgbcmyk')
    DC = nx.algorithms.degree_centrality(SC_b)
    BC = nx.algorithms.betweenness_centrality(SC_b)
    EC = nx.algorithms.eigenvector_centrality(SC_b, max_iter = 1000)
    HC = nx.hits(SC_b)[1]
    # print("DC | ", DC)
    plt.subplot(2,2,1)
    plt.title('Degree Centrality')
    for data in DC.items():
        x = data[0]
        y = data[1]
        if(y == 0):
            y = y + 10e-5
        plt.scatter(x, y, color = 'lightblue')
    # print("BC | ", BC)
    plt.subplot(2,2,2)
    plt.title('Betweenness Centrality')
    for data in BC.items():
        x = data[0]
        y = data[1]
        if(y == 0):
            y = y + 10e-5
        plt.scatter(x, y, color = 'lightblue')
    # print("EC | ", EC)
    plt.subplot(2,2,3)
    plt.title('Eigenvector Centrality')
    for data in EC.items():
        x = data[0]
        y = data[1]
        if(y == 0):
            y = y + 10e-5
        plt.scatter(x, y, color = 'lightblue')
    plt.subplot(2,2,4)
    
    corr_x = np.array(list(DC.values()))
    corr_y = np.array(list(BC.values()))
    corr_z = np.array(list(EC.values()))
    corr_w = np.array(list(HC.values()))
    # print(len(corr_x), len(corr_y), type(corr_x))
    r_DC_BC = np.corrcoef(corr_x, corr_y)[0,1]
    r_DC_EC = np.corrcoef(corr_x, corr_z)[0,1]
    r_BC_EC = np.corrcoef(corr_y, corr_z)[0,1]
    r_DC_HC = np.corrcoef(corr_x, corr_w)[0,1]
    r_BC_HC = np.corrcoef(corr_y, corr_w)[0,1]
    r_EC_HC = np.corrcoef(corr_z, corr_w)[0,1]
    plt.text(0.1, 0.5, 'DC')
    plt.text(0.1, 0.3, 'BC')
    plt.text(0.1, 0.1, 'EC')
    plt.text(0.3, 0.7, 'BC')
    plt.text(0.5, 0.7, 'EC')
    plt.text(0.7, 0.7, 'HC')
    plt.text(0.275, 0.5, '%.2f'%r_DC_BC)
    plt.text(0.475, 0.5, '%.2f'%r_DC_EC)
    plt.text(0.675, 0.5, '%.2f'%r_DC_HC)
    plt.text(0.475, 0.3, '%.2f'%r_BC_EC)
    plt.text(0.675, 0.3, '%.2f'%r_BC_HC)
    plt.text(0.675, 0.1, '%.2f'%r_EC_HC)
    
   
    # STILL NEED TO RESET THE CALCULATION FORMULAR FOR CRITICALITY

    # test if the network fit the power law Distribution
    # but we should only use it in a network with large amount of nodes
    # we would like to use it for a small test to find
    # a valuable probability for the poisson Distribution 
    # SC.add_edges_from(SC_d.edges)
    plt.figure(3, figsize = (12, 6))
    # plot_degree_dist(SC_b, 2, 'blue')
    
    
    degree_sequence = sorted([d for n, d in SC_b.degree()], reverse = True)
    # print(degree_sequence)
    degree_count = nx.degree_histogram(SC)
    
    # plt.figure(4)
    fit = powerlaw.Fit(degree_sequence)
    print(fit.power_law.alpha)
    print(fit.power_law.xmin)
    p_comp_e = fit.distribution_compare('power_law', 'exponential')
    # print(p_comp_e)
    if (p_comp_e[0] > 0) and (p_comp_e[1] >= 0.5):
        p_e = 'Power Law'
    elif (p_comp_e[0] > 0) and (p_comp_e[1] < 0.5):
        p_e = 'better Power Law, but not a certain answer'
    elif (p_comp_e[0] < 0) and (p_comp_e[1] >= 0.5):
        p_e = 'Exponential'
    elif (p_comp_e[0] < 0) and (p_comp_e[1] <0.5):
        p_e = 'better Exponential, but not a certain answer'

    p_comp_ln = fit.distribution_compare('power_law', 'lognormal')
    # print(p_comp_ln)
    if (p_comp_ln[0] > 0) and (p_comp_ln[1] >= 0.5):
        p_ln = 'Power Law'
    elif(p_comp_ln[0] > 0) and (p_comp_ln[1] < 0.5):
        p_ln = 'better Power Law, but not a certain answer'
    elif (p_comp_ln[0] < 0) and (p_comp_ln[1] >= 0.5):
        p_ln = 'Lognormal'
    elif (p_comp_ln[0] < 0) and (p_comp_ln[1] < 0.5):
        p_ln = 'better Lognormal, but not a certain answer'

    
    # print("R:", R, "S:", S, "C:", C, "M:", M, "pR", pR, "pS", pS, "pC", pC, "pM", pM, fit.fixed_xmin, fit.xmin)
    plt.subplot(2,2,1)
    plt.title('Power Law PDF Fitting') 
    fit.plot_pdf(color = 'b', marker = 'o', linewidth = 2)
    
    plt.subplot(2,2,3)
    plt.title('Power Law CCDF Fitting') 
    fit.plot_ccdf(color = 'b', marker = 'o', linewidth = 2)

    plt.subplot(2,2,4)
    plt.text(0.01, 0.9, "Checking if Poisson Distribution...")
    plt.text(0.01, 0.8, "Variance :%s"%np.var(degree_sequence))
    plt.text(0.01, 0.7, "Mean :%s"%np.mean(degree_sequence))
    plt.text(0.01, 0.55, 'Comparing with exponential Distribution...')
    plt.text(0.01, 0.45, '%s'%p_e)
    plt.text(0.01, 0.35, 'Comparing with exponential Distribution...')
    plt.text(0.01, 0.25, '%s'%p_ln)
    plt.text(0.01, 0.1, 'R:{}, S:{}, C:{}, M:{}, E_12:{}, E_23:{}, E_34:{}'.format(R, S, C, M, E_12, E_23, E_34))
    # plt.text(0.01, 0.1, 'R:{}, S:{}, C:{}, M:{}, pR:{}, pS:{}, pC:{}, pM:{}'.format(R, S, C, M, pR, pS, pC, pM))
    
    G = nx.scale_free_graph(R+S+C+M, alpha = 0.5, beta = 0.3, gamma = 0.2)
    # plot_degree_dist(G, 2, 'red')
    probability_mass_fct(G, SC_b)

    degree_sequence = sorted([d for n, d in G.degree()], reverse = True)
    # print(degree_sequence)
    degree_count = nx.degree_histogram(G)
    
    # plt.figure(4)
    fit = powerlaw.Fit(degree_sequence)
    print(fit.power_law.alpha)
    print(fit.power_law.xmin)
    plt.subplot(2,2,1)
    plt.title('Power Law PDF Fitting') 
    fit.plot_pdf(color = 'r', marker = 'o', linewidth = 2)
    plt.subplot(2,2,3)
    plt.title('Power Law CCDF Fitting') 
    fit.plot_ccdf(color = 'r', marker = 'o', linewidth = 2)

    reg_result = multi_linear_reg(perform_SC_b, corr_x, corr_y, corr_z)
    print(reg_result.summary())

    # plt.figure(4)
    # pos_G = nx.shell_layout(G, scale = 1)
    # nx.draw_networkx(G, pos = pos_G)
    # nx.draw_networkx_nodes(G, pos = pos_G, node_size = 500, node_color = 'black', node_shape = 'o')
    # nx.draw_networkx_nodes(G, pos = pos_G, node_size = 450, node_color = 'w', node_shape = 'o')
    
    # perform_pos = pos.copy()
    # for i in pos:
    #     perform_pos[i][1] += 0.1
    #     print(perform_pos[i])
    # nx.draw_networkx_labels(SC, pos = perform_pos, labels = mapping_perform)

    # STILL NEED A NEW POSITION TO SHOW THE CONTRACTED PRODUCTION NETWORK
    # print("SC.NODES : \t", SC.nodes)
    # for j in range():
    # print(SC.number_of_nodes())
    
    '''

    jls_extract_var = plt
    jls_extract_var.show()
    # plt.savefig('labels.png')


if __name__ == "__main__":
    main()