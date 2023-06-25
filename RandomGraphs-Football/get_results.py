import numpy as np
import networkx as nx


nr_graphs = 380
nr_graphs_no_border = 380

G_no_border = nx.read_adjlist("G_no_border_" + str(nr_graphs_no_border) + ".adjlist", create_using=nx.DiGraph)
G = nx.read_adjlist("G_" + str(nr_graphs) + ".adjlist", create_using=nx.DiGraph)
G_determ = nx.read_adjlist("G_determ.adjlist", create_using=nx.DiGraph)

def degree_histogram_directed(G, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq

# G_test = nx.scale_free_graph(50)
#
# in_degree_freq = degree_histogram_directed(G_test, in_degree=True)
# out_degree_freq = degree_histogram_directed(G_test, out_degree=True)
# degrees_in = range(len(in_degree_freq))
# degrees_out = range(len(out_degree_freq))
#
# print(in_degree_freq)
# print(degrees_in)
# print(out_degree_freq)
# print(degrees_out)


def pairwise_diff(list1, list2):
    min_length = min(len(list1), len(list2))

    # Calculate the pairwise differences for the common range
    differences = [abs(list1[i] - list2[i]) for i in range(min_length)]

    # Handle the remaining elements from the longer list
    if len(list1) > len(list2):
        differences += [abs(val) for val in list1[min_length:]]
    elif len(list2) > len(list1):
        differences += [abs(val) for val in list2[min_length:]]

    return differences

# list1 = [1, 2, 3, 4, 5]
# list2 = [2, 4, 8]
#
# pairwise_diff(list1, list2)


# idea: plot pairwise difference, plot degree distribution, get difference in local clustering coefficient

def dist_TV_in(G1, G2, IN=True):
    if IN:
        degree_freq1 = degree_histogram_directed(G1, in_degree=True)
        degree_freq2 = degree_histogram_directed(G2, in_degree=True)
    else:
        degree_freq1 = degree_histogram_directed(G1, out_degree=True)
        degree_freq2 = degree_histogram_directed(G2, out_degree=True)


    s1 = sum(degree_freq1)
    norm_degree_freq1 = [float(i) / s1 for i in degree_freq1]
    # print(norm_degree_freq1)
    # print(np.mean(norm_degree_freq1), len(norm_degree_freq1))

    s2 = sum(degree_freq2)
    norm_degree_freq2 = [float(i)/s2 for i in degree_freq2]
    # print(norm_degree_freq2)
    # print(np.mean(norm_degree_freq2), len(norm_degree_freq2))

    dist_list = pairwise_diff(norm_degree_freq1, norm_degree_freq2)
    # print(dist_list)

    return sum(dist_list)/2



print(dist_TV_in(G, G_determ, IN=True))
print(dist_TV_in(G, G_determ, IN=False))


def localClusteringCoefficient(graph):
    deg_tupleList = nx.degree(graph)
    deg_dict_ = dict(deg_tupleList)
    deg_dict = deg_dict_
    inv_deg_dict = {}
    for k, v in deg_dict.items():
        inv_deg_dict[v] = inv_deg_dict.get(v, []) + [k]
    loc_clustering_dict = {}
    for key, value in inv_deg_dict.items():
        total_clustering = 0
        for v in value:
            total_clustering += nx.clustering(graph, v)
        avg_clustering = total_clustering / len(value)
        loc_clustering_dict[key] = avg_clustering
    return loc_clustering_dict

print(localClusteringCoefficient(G))
# print(localClusteringCoefficient(G_no_border))
print(localClusteringCoefficient(G_determ))

print(nx.average_clustering(G))
# print(nx.average_clustering(G_no_border))
print(nx.average_clustering(G_determ))

# Calculate the number of triangles in the graph
triangles_G = sum(nx.triangles(G.to_undirected()).values())
# Calculate the number of triplets in the graph
triplets_G = sum(nx.triads.triadic_census(G).values())
print(sum(list(nx.all_triplets(G.to_undirected()))))
# Calculate the global clustering coefficient
global_clustering_coefficient_G = 3 * triangles_G / triplets_G
print("Global Clustering Coefficient:", global_clustering_coefficient_G)

# Calculate the number of triangles in the graph
triangles_G_determ = sum(nx.triangles(G_determ.to_undirected()).values())
# Calculate the number of triplets in the graph
triplets_G_determ = sum(nx.triads.triadic_census(G_determ).values())
triplets_G_determ = sum(nx.all_triplets(G_determ))
# Calculate the global clustering coefficient
global_clustering_coefficient_G_determ = 3 * triangles_G_determ / triplets_G_determ
print("Global Clustering Coefficient:", global_clustering_coefficient_G_determ)