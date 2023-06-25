# def merge_with_insertions(list1, list2):
#     merged = []
#     interval_lengths = []
#
#     i = j = 0  # Pointers for list1 and list2
#     interval_counter_1 = 0
#     interval_counter_2 = 0
#
#     while i < len(list1) and j < len(list2):
#         if list1[i] <= list2[j]:
#             print("i", i)
#             IN_LIST1 = True
#             merged.append(list1[i])
#             print("appended " + str(list1[i]))
#             i += 1
#             interval_lengths.append(list1[j])
#         else:
#             IN_LIST1 = False
#             print("j", j)
#             merged.append(list2[j])
#             print("appended " + str(list2[j]))
#             j += 1
#             interval_lengths.append(list1[j] - list1[i])
#     # Append remaining elements from list1 and list2, if any
#     merged.extend(list1[i:])
#     merged.extend(list2[j:])
#     interval_lengths.extend([list2[-1] - list2[j]])  # Insertion points for remaining elements in list2
#
#     return merged, interval_lengths
#
# list1 = [1, 4, 5]
# list2 = [2, 3, 6]
#
# # merged, interval_lengths = merge_with_insertions(list1, list2)
#
# # print("Merged List:", merged)
# # print("Interval Lengths:", interval_lengths)
#
#
# i = 0
# j = 0
# interval_1 = 0
# interval_2 = 0
# merged_new = []
# intervals_new = [0]
# while i < len(list1) and j < len(list2):
#     if list1[i] <= list2[j]:
#         if interval_2 > 0:
#             intervals_new.append(interval_2 - intervals_new[-1])
#             interval_2 = 0
#         merged_new.append(list1[i])
#         interval_1 = list1[i]
#         i += 1
#     else:
#         intervals_new.append(interval_1 - intervals_new[-1])
#         interval_1 = 0
#         merged_new.append(list2[j])
#         interval_2 = list2[j]
#         j += 1
#
# print(merged_new)
# print(intervals_new)
import numpy as np
import scipy


def merger(L1, L2):
    M = L1 + L2
    M.sort()
    return M


def check_in_list(a, L3):
    for g in range(0, len(L3)):
        if L3[g] == a:
            return True
    return False


L1 = [1, 2, 3, 7, 7.6]
L2 = [4, 5, 6, 7.5, 9.6]

# 3, 3, 1, 0.5, 1.5,

merged = merger(L1, L2)

tags = []
intervals = []

for h in range(0, len(merged)):
    if check_in_list(merged[h], L1):
        tags.append(True)
    if check_in_list(merged[h], L2):
        tags.append(False)


def count_dups(L):
    ans = []
    if not L:
        return ans
    running_count = 1
    for i in range(len(L) - 1):
        if L[i] == L[i + 1]:
            running_count += 1
        else:
            ans.append(running_count)
            running_count = 1
    ans.append(running_count)
    return ans


# print(merged, count_dups(tags))
#
# print(np.cumsum(count_dups(tags)))


# ##
# count_dups = count_dups(tags)
# intervals_indices = np.cumsum(count_dups)
# intervals_indices = [elt - 1 for elt in intervals_indices]
# print(intervals_indices)
# print(merged)
# # print([merged[intervals_indices[i]] - merged[intervals_indices[i-1]] for i in range(1,len(intervals_indices))])
# # print(intervals)
#
# for i in range(len(intervals_indices)):
#     if i == 0:
#         intervals.append(merged[intervals_indices[i]])
#     else:
#         intervals.append(merged[intervals_indices[i]] - merged[intervals_indices[i-1]])
#
# print(intervals)

#loop over dictionary
# d = {"k11": [1,1,1], "k22": 2}
# for v in d.values():
#     print(v)



import networkx as nx
import matplotlib.pyplot as plt

# Example graphs
graph1 = nx.DiGraph()
graph1.add_edge('A', 'B', weight=2)
graph1.add_edge('B', 'C', weight=3)
pos1 = {'A': (0, 0), 'B': (1, 0), 'C': (0,1)}

graph2 = nx.DiGraph()
graph2.add_edge('A', 'C', weight=1)
graph2.add_edge('B', 'C', weight=4)
pos2 = {'A': (0, 0), 'B': (1, 0), 'C': (0, 1)}

graph3 = nx.DiGraph()
graph3.add_edge('A', 'C', weight=3)
graph3.add_edge('C', 'B', weight=3)
pos3 = {'A': (0, 0), 'B': (1, 0), 'C': (0, 1)}

# Create a list of graphs and their corresponding node positions
graphs = [(graph1, pos1), (graph2, pos2), (graph3, pos3)]

def make_average_graph(graphs, VISUALIZE):
    num_graphs = len(graphs)

    # Initialize an empty graph and an empty dictionary for accumulating positions
    average_graph = nx.DiGraph()
    average_positions = {}

    # Iterate over each graph and accumulate edge weights and positions
    for graph, pos in graphs:
        # Accumulate edge weights
        for edge in graph.edges():
            if average_graph.has_edge(*edge):
                average_graph[edge[0]][edge[1]]['weight'] += graph[edge[0]][edge[1]]['weight']
            else:
                average_graph.add_edge(*edge, weight=graph[edge[0]][edge[1]]['weight'])

        # Accumulate node positions
        for node in graph.nodes():
            if node in average_positions:
                average_positions[node][0] += pos[node][0]
                average_positions[node][1] += pos[node][1]
            else:
                average_positions[node] = [pos[node][0], pos[node][1]]

    # Divide edge weights by the number of graphs to compute the average
    for edge in average_graph.edges():
        average_graph[edge[0]][edge[1]]['weight'] /= num_graphs

    # Divide accumulated node positions by the number of graphs to compute the average
    for node in average_positions:
        average_positions[node][0] /= num_graphs
        average_positions[node][1] /= num_graphs

    # Set the average positions to the average graph
    nx.set_node_attributes(average_graph, average_positions, 'pos')

    if VISUALIZE:
        # Visualize the resulting average graph with node positions
        pos = nx.get_node_attributes(average_graph, 'pos')
        weights = nx.get_edge_attributes(average_graph, 'weight').values()
        vmin = min(list(weights))
        vmax = max(list(weights))
        cmap = plt.cm.Oranges
        fig = plt.figure(figsize=(5 * 1.54, 5 * 1))
        # pos = {(x,y):(y,-x) for x,y in G.nodes()}
        nx.draw(average_graph, pos=pos,
                node_color='darkblue',
                with_labels=True,
                # width=list(weights),
                edge_color=list(weights),
                edge_cmap=cmap,
                linewidths=0.3,
                alpha=0.8,
                node_size=60)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        plt.colorbar(sm)
        fig.set_facecolor('lightgreen')
        plt.show()

    return average_graph

G = make_average_graph(graphs, VISUALIZE=False)


all_weights = []
for elt in G.edges.data():
    print(elt)
    all_weights.append(elt[2]['weight'])

print(all_weights)
test_weights = [1,1,1,1]

print(0.5*scipy.spatial.distance.cityblock(all_weights, test_weights))

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
print(grid)

grid.append((x_grid_size*y_grid_size + 1, {"pos": (-1/x_grid_size, 0.5)}))
grid.append((x_grid_size*y_grid_size + 2, {"pos": (1.0, 0.5)}))

degrees = G.degree()
print(degrees)

# for elt in grid:
#     print(elt[0])

