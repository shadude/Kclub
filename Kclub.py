import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

G = nx.karate_club_graph()
pos = nx.spring_layout(G)
global_nodelist = list(G.nodes())

A = nx.to_numpy_array(G,nodelist=global_nodelist,weight = None)
k = np.array([np.sum(A,axis=0)])
m = np.sum(k)
global_B = A - 0.5*(1/m)*(k.T@k)

final_communities = []
iter_tree_layers = {}
iter_tree_communities= {}


def split(G,node_list,depth = 0):
    B = global_B[np.ix_(node_list,node_list)]
    e,v = np.linalg.eig(B)
    
    s = v[:,np.argmax(e)]
    n1 = []
    n2 = []
    for i in range(s.shape[0]):
        if s[i]<=0:
            n1.append(node_list[i])
        else:
            n2.append(node_list[i])
    g1 = G.subgraph(n1)
    g2 = G.subgraph(n2)

    if depth in iter_tree_layers:
        iter_tree_layers[depth].append(node_list)
    else:
        iter_tree_layers[depth] = [node_list]

    if np.max(e)<=0:
        final_communities.append(node_list)
        if depth in iter_tree_communities:
            iter_tree_communities[depth].append(node_list)
        else:
            iter_tree_communities[depth] = [node_list]
        return
    if len(n1) == 0 :
        final_communities.append(node_list)
        if depth in iter_tree_communities:
            iter_tree_communities[depth].append(node_list)
        else:
            iter_tree_communities[depth] = [node_list]
        return
    if len(n2) == 0 :
        final_communities.append(node_list)
        if depth in iter_tree_communities:
            iter_tree_communities[depth].append(node_list)
        else:
            iter_tree_communities[depth] = [node_list]
        return
    split(g1,n1,depth+1)
    split(g2,n2,depth+1)

split(G,global_nodelist)
print(final_communities)


def draw(sublist):
    n = len(sublist)
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0,1,n)]
    nx.draw_networkx_nodes(G,pos)
    for i in range(n):
        nx.draw_networkx_nodes(G,pos,nodelist = sublist[i], node_color = colors[i])
        nx.draw_networkx_labels(G,pos,{a:i for a in sublist[i]} )
    nx.draw_networkx_edges(G,pos)

draw(final_communities)
plt.show()
for key,val in sorted(iter_tree_layers.items()):
    newval = val
    for key2,val2 in sorted(iter_tree_communities.items()):
        if key>key2:
            newval = newval + val2
    iter_tree_layers[key] = newval

for key,val in sorted(iter_tree_layers.items()):
    draw(val)
    plt.show()

node_metrics = {a:{} for a in global_nodelist}

for key,val in sorted(iter_tree_layers.items()):
    for i in val:
        Ga = G.subgraph(i)
        d = nx.degree_centrality(Ga)
        b = nx.betweenness_centrality(Ga)
        cc = nx.closeness_centrality(Ga)
        c = nx.clustering(Ga)
        for j in i:
            node_metrics[j][key] = [d[j],b[j],cc[j],c[j]]
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = fig.add_subplot(512)
#ax3 = fig.add_subplot(513)
#ax4 = fig.add_subplot(514)
ax5 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
cmap = plt.get_cmap('jet')
colors = [cmap(i) for i in np.linspace(0,1,34)]
print(len(colors))
for k,v in node_metrics.items():
    depths = list(v.keys())
    metrics = np.array(list(v.values()))
    ax1.scatter(depths,metrics[:,0],color = colors[k])
    ax1.plot(depths,metrics[:,0],color = colors[k])
    #ax2.scatter(depths,metrics[:,1],color = colors[k])
    #ax3.scatter(depths,metrics[:,2],color = colors[k])
    #ax4.scatter(depths,metrics[:,3],color = colors[k])
bounds = np.linspace(0, 33, 34)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', colors, cmap.N)
cb = mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm,
    spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
plt.show()

