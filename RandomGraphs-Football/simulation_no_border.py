import random
import itertools
from collections import Counter
import numpy as np
from scipy.spatial.distance import euclidean
import scipy.stats
from scipy.stats import expon, uniform, poisson
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from distribution import Distribution
import time
from average_graph import make_average_graph

# G = nx.grid_2d_graph(x_grid_size,y_grid_size)

#
# x_partition = [i/x_grid_size for i in range(x_grid_size + 1)]
# y_partition = [i/y_grid_size for i in range(y_grid_size + 1)]
# grid = np.array([i for i in range(x_grid_size*y_grid_size + 1)]) # each number maps to grid.
#
# grid = np.array([x_partition, y_partition])
# print(grid)

nr_graphs = 380

#
# G_grid = nx.grid_2d_graph(3,3)
# print(G_grid.nodes.data())

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return(x, y)

x_grid_size = 10
y_grid_size = 10
# MAKE SURE TO SET THESE EQUAL.

G = nx.DiGraph()
grid = []
count = 0
edgeCounts = dict()
for i in range(x_grid_size*y_grid_size):
    if i != 0 and i % x_grid_size == 0:
        # print("count: ", count, " i: ", i)
        count += 1 / x_grid_size
    grid.append((i, {"pos": (float(count), (i % x_grid_size) / x_grid_size)}))
# print(grid)

# grid.append((x_grid_size*y_grid_size + 2, {"pos": (1.0, 0.5)}))

def create_graph_no_border(P, L, T_R, T_K, D, grid):
    # T_K = 7.58 #mean
    # D = 24.86 # divide by 100
    times_keep = Distribution(expon(T_K))
    times_lose = Distribution(expon(T_R))

     # 1 / mean of exponential distribution for new time of an event
    # In general, based on the events data set, a soccer match consists of an average of 1,682 events
    endTime = 5400 # 90 minutes
    time = 0

    G.add_nodes_from(grid)


    edges = []
    # G.add_edge(1, 2, weight=4.7)

    all_edges = []

    newStart = False

    goalsAgainst = 0
    goalsFor = 0

    reset = False
    # first grid point, MENTION IN REPORT.
    startGridPoint = (x_grid_size*y_grid_size // 2 + y_grid_size // 2, {"pos": (x_grid_size // 2,x_grid_size // 2)})
    firstGridPoint = startGridPoint
    # print(firstGridPoint)
    startNodeID = startGridPoint[0]
    startPosition = startGridPoint[1]['pos']

    while time < endTime:
        if reset:
            startGridPoint = random.choice(grid) #start at a random node
            startNodeID = startGridPoint[0]
            startPosition = startGridPoint[1]['pos']
            reset = False

        r = Distribution(expon(D)).rvs()
        # r = np.random.exponential(scale= meanExp) #scale is MEAN
        phi = Distribution(uniform(scale=2*np.pi)).rvs()
        # phi = np.random.uniform(low=0.0, high=2*np.pi) #random angle
        endPosition_coords = tuple(sum(x) for x in zip(startPosition,pol2cart(r, phi)))
        newStart = False
        for nodeID, grid_positions in grid:
            if (grid_positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0] <= (grid_positions['pos'][0] + 1 / (2 * x_grid_size)):
                # print("first if normal case")
                if (grid_positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1] <= (grid_positions['pos'][1] + 1 / (2 * x_grid_size)):
                    # print("second if normal case")
                    all_edges.append(str((str(startNodeID), str(nodeID))))
                    # print(Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                    G.add_edge(startNodeID, nodeID, weight= Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                    startNodeID = nodeID
                    startPosition = grid_positions['pos']
                    # print(all_edges)
                    newStart = True
                    break
            else:
                # print("else case")
                break
        u = Distribution(uniform).rvs()
        if u <= P/(P+L):
            time = time + times_keep.rvs()
        elif P/(P+L) < u <= 1:
            reset = True
            time = time + times_lose.rvs()
        else:
            print(u, "SOMETHING WENT WRONG.")
    pos_ret = nx.get_node_attributes(G, "pos")
    return G, pos_ret

all_graphs = []
for i in tqdm(range(nr_graphs)):
    all_graphs.append(create_graph_no_border(P = 432.44, L = 125.74, T_R = 11.95, T_K = 7.58, D = 0.2486, grid=grid))


G = make_average_graph(all_graphs, VISUALIZE=True)
print(G)
nx.write_adjlist(G, "G_no_border_" + str(nr_graphs) + ".adjlist")

# print(G.edges.data())


# print(all_edges)

    # time = float(time) + expon(T)

# print(G.nodes.data())
#
# G.add_nodes_from([
#          (0, {"pos": (0, 0)}),
#          (1, {"pos": (3, 0)}),
#          (2, {"pos": (8, 0)}),
#      ])
#
# nx.geometric_edges(G, radius=4)
#

# G = create_graph(P = 432.44, L = 125.74, T_R = 11.95, T_K = 7.58, D = 0.2486)

# print(firstGridPoint)
#
# print("result: ", goalsAgainst, goalsFor)
#
# pos = nx.get_node_attributes(G, "pos")
# weights = nx.get_edge_attributes(G,'weight').values()
# vmin = min(list(weights))
# vmax = max(list(weights))
# cmap = plt.cm.Oranges
# fig = plt.figure(figsize=(5*1.54,5*1))
# # pos = {(x,y):(y,-x) for x,y in G.nodes()}
# nx.draw(G, pos=pos,
#         node_color='darkblue',
#         with_labels=True,
#         # width=list(weights),
#         edge_color=list(weights),
#         edge_cmap=cmap,
#         linewidths=0.3,
#         alpha = 0.8,
#         node_size=60)
# # nx.draw_networkx(G.subgraph(firstGridPoint[0]),
# #         pos=pos,
# #         node_color='red',
# #         with_labels=True,
# #         linewidths=0.3,
# #         alpha = 0.8,
# #         node_size=60)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
# sm._A = []
# plt.colorbar(sm)
# fig.set_facecolor('lightgreen')
# plt.show()


#
# clustering_coeff = nx.clustering(G)
# degrees = G.degree()
# print(degrees)
# maxDegree = max(degrees)[0]
# # print(maxDegree)
# # print(degrees)
# # print(clustering_coeff)
#
# degree_freq = nx.degree_histogram(G)
# s = sum(degree_freq)
# norm_degree_freq = [float(i)/s for i in degree_freq]
# print(norm_degree_freq)
#
#
#
# degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
# print(degree_sequence)

# plt.bar(*np.unique(degree_sequence, return_counts=True))
# plt.title("Degree histogram")
# plt.xlabel("Degree")
# plt.ylabel("# of Nodes")
# plt.show()

# def degreeDistributionPlot(graph):
#     degree_freq = nx.degree_histogram(graph)
#     s = sum(degree_freq)
#     norm_degree_freq = [float(i)/s for i in degree_freq]
#     degrees = range(len(degree_freq))
#     plt.plot(degrees, norm_degree_freq, color="black")
#     # plt.gca().set_aspect(1/1.61803398875)
#     # plt.xscale('log')
#     # plt.yscale('log')
#     plt.title("Degree distribution")
#     plt.xlabel('Degree')
#     plt.ylabel(r"P(k)")
#     # plt.savefig("./Figures/IntroductionFigures/youtubeGraphGoldenRatio.png")
#     plt.show()
#
# degreeDistributionPlot(G)

# def localClusteringCoefficient(graph):
#     t1 = time.time()
#     deg_tupleList = nx.degree(graph)
#     deg_dict_ = dict((deg_tupleList))
#     deg_dict = deg_dict_
#     inv_deg_dict = {}
#     for k, v in deg_dict.items():
#         inv_deg_dict[v] = inv_deg_dict.get(v, []) + [k]
#     loc_clustering_dict = {}
#     for key, value in inv_deg_dict.items():
#         total_clustering = 0
#         for v in value:
#             total_clustering += nx.clustering(graph, v)
#         avg_clustering = total_clustering / len(value)
#         loc_clustering_dict[key] = avg_clustering
#     t2 = time.time()
#     print("time to calculate loc clus coef: " + str(t2 - t1))
#     return loc_clustering_dict

# loc_clus = localClusteringCoefficient(G)
# print(loc_clus)
#
# degrees_clustering = sorted(localClusteringCoefficient(G))
# clustering_seq = []
# for degree in degrees_clustering:
#     clustering_seq.append(loc_clus[degree])
#
# print(degrees_clustering)
# print(clustering_seq)
#
# plt.plot(degrees_clustering, clustering_seq, color="r")
# plt.xlabel("degree")
# plt.ylabel("local clustering coefficient")
# plt.title("Local clustering coefficient")
# plt.show()

# NOT USE THIS:
# def makeBigClus(n_list, nr_graphs):
#     big_clusDict = {}
#     for n in n_list:
#         locClus_sum = {i : 0.0 for i in range(n)}
#         for dummy in range(nr_graphs):
#             G = create_graph()
#             loc_Clus = localClusteringCoefficient(G)
#             print(loc_Clus)
#             # print(loc_Clus)
#             for key, value in loc_Clus.items():
#                 print(key, value)
#                 if loc_Clus.get(key):
#                     locClus_sum[key] += value
#         max_key = max(k for k, v in locClus_sum.items() if v != 0.0)
#         # print(max_key)
#         # print(locClus_sum)
#         locClus_sum = dict(list(locClus_sum.items()))
#         # print(locClus_sum)
#         locClus = {k: v / nr_graphs for k, v in locClus_sum.items()}
#         # print(locClus)
#         big_clusDict[n] = locClus
#     return big_clusDict
#
# n_list = [i for i in range(maxDegree)]
#
# makeBigClus( n_list, nr_graphs = 2)
