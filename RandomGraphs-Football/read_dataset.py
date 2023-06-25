import pandas as pd
import matplotlib.pyplot as plt
import json
import networkx as nx
from collections import Counter

# we want to read events_England.json

# Specify the path to the JSON file
file_path = './data/events/events_England.json'

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Access the data
# Example: Print the first event in the dataset
first_event = data[0]
print(first_event)

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

G.add_nodes_from(grid)

all_edges = []

reset = True

for eventID in range(len(data)):
    while data[eventID]['matchId'] == 2499719:
        if data[eventID]['eventName'] == 'Pass':
            if reset:
                startPosition_coords = (data[eventID]['positions'][0]['x'] / 100, data[eventID]['positions'][0]['y'] / 100)
                # print(startPosition_coords)
            endPosition_coords = (data[eventID]['positions'][1]['x'] / 100, data[eventID]['positions'][1]['y'] / 100)
            # print(endPosition_coords)
            for nodeID, positions in grid:
                # print(nodeID, positions['pos'][0], positions['pos'][1])
                # print(endPosition_coords)
                if reset:
                    print("entering start case")
                    if positions['pos'][0] > 0 and positions['pos'][1] > 0 and positions['pos'][0] < (
                            (x_grid_size - 1) / x_grid_size) and positions['pos'][1] < (
                            (x_grid_size - 1) / x_grid_size):
                        # print("entering normal case start")
                        print(startPosition_coords)
                        print(positions['pos'][0] - 1 / (2 * x_grid_size), positions['pos'][0] + 1 / (2 * x_grid_size))
                        if (positions['pos'][0] - 1 / (2 * x_grid_size)) < startPosition_coords[0] <= (
                                positions['pos'][0] + 1 / (2 * x_grid_size)):
                            print("first if normal case start")
                            if (positions['pos'][1] - 1 / (2 * x_grid_size)) < startPosition_coords[1] <= (
                                    positions['pos'][1] + 1 / (2 * x_grid_size)):
                                print("second if normal case start")
                                startNodeID = nodeID
                                startPosition = positions['pos']
                                startNodeID_first = nodeID
                                # print(all_edges)
                                print(reset)
                                reset = False
                                break
                    elif positions['pos'][0] == 0.0 and 0 < positions['pos'][1] < ((x_grid_size - 1) / x_grid_size):
                        # print("entering case 2")
                        if endPosition_coords[0] <= (positions['pos'][0] + 1 / (2 * x_grid_size)):
                            # print("first if case 2")
                            if (positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1] <= (
                                    positions['pos'][1] + 1 / (2 * x_grid_size)):
                                startNodeID = nodeID
                                startPosition = positions['pos']
                                startNodeID_first = nodeID
                                # print(all_edges)
                                print(reset)
                                reset = False
                                break
                    elif positions['pos'][0] > 0 and positions['pos'][1] == 0.0:
                        # print("entering case 3")
                        if (positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0] <= (
                                positions['pos'][0] + 1 / (2 * x_grid_size)):
                            # print("first if case 3")
                            if endPosition_coords[1] <= (positions['pos'][1] + 1 / (2 * x_grid_size)):
                                startNodeID = nodeID
                                startPosition = positions['pos']
                                startNodeID_first = nodeID
                                # print(all_edges)
                                print(reset)
                                reset = False
                                break
                    elif positions['pos'][0] == 0.0 and positions['pos'][1] == 0.0:
                        # print("case 4 entered")
                        if endPosition_coords[0] <= (positions['pos'][0] + 1 / (2 * x_grid_size)):
                            # print("first if statement 4")
                            if endPosition_coords[1] <= (positions['pos'][1] + 1 / (2 * x_grid_size)):
                                startNodeID = nodeID
                                startPosition = positions['pos']
                                startNodeID_first = nodeID
                                # print(all_edges)
                                print(reset)
                                reset = False
                                break
                    elif positions['pos'][0] >= ((x_grid_size - 1) / x_grid_size) and positions['pos'][1] >= (
                            (x_grid_size - 1) / x_grid_size):
                        # print("entering case 5")
                        if endPosition_coords[0] >= (positions['pos'][0] + 1 / (2 * x_grid_size)):
                            # print("first if case 5")
                            if endPosition_coords[1] >= (positions['pos'][1] + 1 / (2 * x_grid_size)):
                                startNodeID = nodeID
                                startPosition = positions['pos']
                                startNodeID_first = nodeID
                                # print(all_edges)
                                print(reset)
                                reset = False
                                break
                    elif positions['pos'][0] >= ((x_grid_size - 1) / x_grid_size) and 0 < positions['pos'][1] < (
                            (x_grid_size - 1) / x_grid_size):
                        # print("entering case 6")
                        if (positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0]:
                            # print("first if case 6")
                            if (positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1] <= (
                                    positions['pos'][1] + 1 / (2 * x_grid_size)):
                                startNodeID = nodeID
                                startPosition = positions['pos']
                                startNodeID_first = nodeID
                                # print(all_edges)
                                print(reset)
                                reset = False
                                break
                    elif positions['pos'][1] >= ((x_grid_size - 1) / x_grid_size) and 0 < positions['pos'][0] < (
                            (x_grid_size - 1) / x_grid_size):
                        # print("entering case 7")
                        if (positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0] <= (
                                positions['pos'][0] + 1 / (2 * x_grid_size)):
                            # print("first if case 7")
                            if (positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1]:
                                startNodeID = nodeID
                                startPosition = positions['pos']
                                startNodeID_first = nodeID
                                # print(all_edges)
                                print(reset)
                                reset = False
                                break
                    elif positions['pos'][1] >= ((x_grid_size - 1) / x_grid_size) and positions['pos'][0] == 0.0:
                        # print("entering case 8")
                        if endPosition_coords[0] < (positions['pos'][0] + 1 / (2 * x_grid_size)):
                            # print("first if case 8")
                            if (positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1]:
                                startNodeID = nodeID
                                startPosition = positions['pos']
                                startNodeID_first = nodeID
                                # print(all_edges)
                                print(reset)
                                reset = False
                                break
                    elif positions['pos'][1] == 0.0 and positions['pos'][0] >= ((x_grid_size - 1) / x_grid_size):
                        # print("entering case 9")
                        if (positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0]:
                            # print("first if case 9")
                            if endPosition_coords[1] < (positions['pos'][1] + 1 / (2 * x_grid_size)):
                                startNodeID = nodeID
                                startPosition = positions['pos']
                                startNodeID_first = nodeID
                                # print(all_edges)
                                print(reset)
                                reset = False
                                break
                    else:
                        print("else case start")

                elif positions['pos'][0] > 0 and positions['pos'][1] > 0 and positions['pos'][0] < ((x_grid_size - 1) / x_grid_size) and positions['pos'][1] < ((x_grid_size - 1) / x_grid_size):
                    print("entering normal case")
                    if (positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0] <= (
                            positions['pos'][0] + 1 / (2 * x_grid_size)):
                        print("first if normal case")
                        if (positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1] <= (
                                positions['pos'][1] + 1 / (2 * x_grid_size)):
                            print("second if normal case")
                            all_edges.append(str((str(startNodeID), str(nodeID))))
                            G.add_edge(startNodeID, nodeID,
                                       weight=Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                            startNodeID = nodeID
                            startPosition = positions['pos']
                            # print(all_edges)
                            newStart = True
                            break
                elif positions['pos'][0] == 0.0 and 0 < positions['pos'][1] < ((x_grid_size - 1) / x_grid_size):
                    # print("entering case 2")
                    if endPosition_coords[0] <= (positions['pos'][0] + 1 / (2 * x_grid_size)):
                        # print("first if case 2")
                        if (positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1] <= (
                                positions['pos'][1] + 1 / (2 * x_grid_size)):
                            # print("second if case 2")
                            all_edges.append(str((str(startNodeID), str(nodeID))))
                            G.add_edge(startNodeID, nodeID,
                                       weight=Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                            startNodeID = nodeID
                            startPosition = positions['pos']
                            # print(all_edges)
                            newStart = True
                            break
                elif positions['pos'][0] > 0 and positions['pos'][1] == 0.0:
                    # print("entering case 3")
                    if (positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0] <= (
                            positions['pos'][0] + 1 / (2 * x_grid_size)):
                        # print("first if case 3")
                        if endPosition_coords[1] <= (positions['pos'][1] + 1 / (2 * x_grid_size)):
                            # print("second if case 3")
                            all_edges.append(str((str(startNodeID), str(nodeID))))
                            G.add_edge(startNodeID, nodeID,
                                       weight=Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                            startNodeID = nodeID
                            startPosition = positions['pos']
                            # print(all_edges)
                            newStart = True
                            break
                elif positions['pos'][0] == 0.0 and positions['pos'][1] == 0.0:
                    # print("case 4 entered")
                    if endPosition_coords[0] <= (positions['pos'][0] + 1 / (2 * x_grid_size)):
                        # print("first if statement 4")
                        if endPosition_coords[1] <= (positions['pos'][1] + 1 / (2 * x_grid_size)):
                            # print("second if statement 4")
                            all_edges.append(str((str(startNodeID), str(nodeID))))
                            G.add_edge(startNodeID, nodeID,
                                       weight=Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                            # print("startNodeID", startNodeID, "nodeID", nodeID)
                            startNodeID = nodeID
                            # print("startNodeID after", startNodeID)
                            startPosition = positions['pos']
                            # print(all_edges)
                            newStart = True
                            break
                elif positions['pos'][0] >= ((x_grid_size - 1) / x_grid_size) and positions['pos'][1] >= (
                        (x_grid_size - 1) / x_grid_size):
                    # print("entering case 5")
                    if endPosition_coords[0] >= (positions['pos'][0] + 1 / (2 * x_grid_size)):
                        # print("first if case 5")
                        if endPosition_coords[1] >= (positions['pos'][1] + 1 / (2 * x_grid_size)):
                            # print("second if case 5")
                            all_edges.append(str((str(startNodeID), str(nodeID))))
                            G.add_edge(startNodeID, nodeID,
                                       weight=Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                            startNodeID = nodeID
                            startPosition = positions['pos']
                            # print(all_edges)
                            newStart = True
                            break
                elif positions['pos'][0] >= ((x_grid_size - 1) / x_grid_size) and 0 < positions['pos'][1] < (
                        (x_grid_size - 1) / x_grid_size):
                    # print("entering case 6")
                    if (positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0]:
                        # print("first if case 6")
                        if (positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1] <= (
                                positions['pos'][1] + 1 / (2 * x_grid_size)):
                            # print("second if case 6")
                            all_edges.append(str((str(startNodeID), str(nodeID))))
                            G.add_edge(startNodeID, nodeID,
                                       weight=Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                            startNodeID = nodeID
                            startPosition = positions['pos']
                            # print(all_edges)
                            newStart = True
                            break
                elif positions['pos'][1] >= ((x_grid_size - 1) / x_grid_size) and 0 < positions['pos'][0] < (
                        (x_grid_size - 1) / x_grid_size):
                    # print("entering case 7")
                    if (positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0] <= (
                            positions['pos'][0] + 1 / (2 * x_grid_size)):
                        # print("first if case 7")
                        if (positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1]:
                            # print("second if case 7")
                            all_edges.append(str((str(startNodeID), str(nodeID))))
                            G.add_edge(startNodeID, nodeID,
                                       weight=Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                            startNodeID = nodeID
                            startPosition = positions['pos']
                            # print(all_edges)
                            newStart = True
                            break
                elif positions['pos'][1] >= ((x_grid_size - 1) / x_grid_size) and positions['pos'][0] == 0.0:
                    # print("entering case 8")
                    if endPosition_coords[0] < (positions['pos'][0] + 1 / (2 * x_grid_size)):
                        # print("first if case 8")
                        if (positions['pos'][1] - 1 / (2 * x_grid_size)) < endPosition_coords[1]:
                            # print("second if case 8")
                            all_edges.append(str((str(startNodeID), str(nodeID))))
                            G.add_edge(startNodeID, nodeID,
                                       weight=Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                            startNodeID = nodeID
                            startPosition = positions['pos']
                            # print(all_edges)
                            newStart = True
                            break
                elif positions['pos'][1] == 0.0 and positions['pos'][0] >= ((x_grid_size - 1) / x_grid_size):
                    # print("entering case 9")
                    if (positions['pos'][0] - 1 / (2 * x_grid_size)) < endPosition_coords[0]:
                        # print("first if case 9")
                        if endPosition_coords[1] < (positions['pos'][1] + 1 / (2 * x_grid_size)):
                            # print("second if case 9")
                            all_edges.append(str((str(startNodeID), str(nodeID))))
                            G.add_edge(startNodeID, nodeID,
                                       weight=Counter(all_edges)[str((str(startNodeID), str(nodeID)))])
                            startNodeID = nodeID
                            startPosition = positions['pos']
                            # print(all_edges)
                            newStart = True
                            break
                else:
                    print("else case")


pos = nx.get_node_attributes(G, "pos")
weights = nx.get_edge_attributes(G,'weight').values()
vmin = min(list(weights))
vmax = max(list(weights))
cmap = plt.cm.Oranges
fig = plt.figure(figsize=(5*1.54,5*1))
# pos = {(x,y):(y,-x) for x,y in G.nodes()}
nx.draw(G, pos=pos,
        node_color='darkblue',
        with_labels=True,
        # width=list(weights),
        edge_color=list(weights),
        edge_cmap=cmap,
        linewidths=0.3,
        alpha = 0.8,
        node_size=60)
nx.draw_networkx(G.subgraph(startNodeID_first),
        pos=pos,
        node_color='red',
        with_labels=True,
        linewidths=0.3,
        alpha = 0.8,
        node_size=60)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
sm._A = []
plt.colorbar(sm)
fig.set_facecolor('lightgreen')
plt.show()

#compare best and worst team, compare passing networks.