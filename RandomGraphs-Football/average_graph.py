import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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
		# print(np.mean(list(weights)))
		# print(vmin, vmax)
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

# Example graphs
# graph1 = nx.DiGraph()
# graph1.add_edge('A', 'B', weight=2)
# graph1.add_edge('B', 'C', weight=3)
# pos1 = {'A': (0, 0), 'B': (1, 0), 'C': (0,1)}
#
# graph2 = nx.DiGraph()
# graph2.add_edge('A', 'C', weight=1)
# graph2.add_edge('B', 'C', weight=4)
# pos2 = {'A': (0, 0), 'B': (1, 0), 'C': (0, 1)}
#
# graph3 = nx.DiGraph()
# graph3.add_edge('A', 'C', weight=3)
# graph3.add_edge('C', 'B', weight=3)
# pos3 = {'A': (0, 0), 'B': (1, 0), 'C': (0, 1)}
#
# # Create a list of graphs and their corresponding node positions
# graphs = [(graph1, pos1), (graph2, pos2), (graph3, pos3)]
#
# make_average_graph(graphs, VISUALIZE=True)

