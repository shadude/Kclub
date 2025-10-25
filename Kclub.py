import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

G = nx.karate_club_graph()
pos = nx.spring_layout(G)
global_nodelist = list(G.nodes())

A = nx.to_numpy_array(G,nodelist=global_nodelist,weight = None)
k = np.array([np.sum(A,axis=0)])
m = np.sum(k)
global_B = A - 0.5*(1/m)*(k.T@k)

fin_coms = []

def split(G,node_list,depth = 0):
    B = global_B[np.ix_(node_list,node_list)]
    e,v = np.linalg.eig(B)
    
    s = v[np.argmax(e)]
    n1 = []
    n2 = []
    for i in range(s.shape[0]):
        if s[i]<=0:
            n1.append(node_list[i])
        else:
            n2.append(node_list[i])
    g1 = G.subgraph(n1)
    g2 = G.subgraph(n2)

    if np.max(e)<=0:
        fin_coms.append(node_list)
        return
    if len(n1) == 0 :
        fin_coms.append(node_list)
        return
    if len(n2) == 0 :
        fin_coms.append(node_list)
        return

    split(g1,n1)
    split(g2,n2)

split(G,global_nodelist)
print(fin_coms)
def draw(sublist):
    n = len(sublist)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0,1,n)]
    #nx.draw_networkx_nodes(G,pos)
    for i in range(n):
        nx.draw_networkx_nodes(G,pos,nodelist = sublist[i], node_color = colors[i])
    nx.draw_networkx_edges(G,pos)
draw(fin_coms)

plt.show()
