import networkx as nx
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from extract_parameters import startPosition_endPosition
from average_graph import make_average_graph

n = 100
x_grid_size = n
y_grid_size = n

def make_deterministic_network(positions):
	G = nx.DiGraph()
	grid = []
	count = 0
	edgeCounts = dict()
	for i in range(x_grid_size * y_grid_size):
		if i != 0 and i % x_grid_size == 0:
			# print("count: ", count, " i: ", i)
			count += 1 / x_grid_size
		grid.append((i, {"pos": (float(count), (i % x_grid_size) / x_grid_size)}))
	# print(grid)

	# grid.append((x_grid_size * y_grid_size + 2, {"pos": (1.0, 0.5)}))

	# for nodeID, grid_positions in grid:
	# print(nodeID, grid_positions)

	G.add_nodes_from(grid)

	edges = []
	# G.add_edge(1, 2, weight=4.7)

	all_edges = []
	for pos in positions:
		x_start = pos[0][0] / 100
		y_start = pos[0][1] / 100

		x_end = pos[1][0] / 100
		y_end = pos[1][1] / 100


		for nodeID, grid_positions in grid:
			if (grid_positions['pos'][0] - 1 / (2 * x_grid_size)) < x_start <= (
					grid_positions['pos'][0] + 1 / (2 * x_grid_size)) and (grid_positions['pos'][1] - 1 / (2 * x_grid_size)) < y_start <= (
					grid_positions['pos'][1] + 1 / (2 * x_grid_size)):
				for nodeID_2, grid_positions_2 in grid:
					if (grid_positions_2['pos'][0] - 1 / (2 * x_grid_size)) < x_end <= (
							grid_positions_2['pos'][0] + 1 / (2 * x_grid_size)) and (
							grid_positions_2['pos'][1] - 1 / (2 * x_grid_size)) < y_end <= (
							grid_positions_2['pos'][1] + 1 / (2 * x_grid_size)):
						# print("Appending ", grid_positions, grid_positions_2)
						all_edges.append((nodeID, nodeID_2))
						# print(Counter(all_edges)[(nodeID, nodeID_2)])
						G.add_edge(nodeID, nodeID_2, weight=Counter(all_edges)[(nodeID, nodeID_2)])
	pos_ret = nx.get_node_attributes(G, "pos")
	return G, pos_ret

all_graphs = []
for positions in startPosition_endPosition.values():
	all_graphs.append(make_deterministic_network(positions))


G_determ = make_average_graph(all_graphs, VISUALIZE=False)
nx.write_adjlist(G_determ, "G_determ_n_" + str(x_grid_size) + ".adjlist")

# print(G.edges.data())

# G = average_graph
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
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
# sm._A = []
# plt.colorbar(sm)
# fig.set_facecolor('lightgreen')
# plt.show()
#





