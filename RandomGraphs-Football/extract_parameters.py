import matplotlib.pyplot as plt
import json
import networkx as nx
import numpy as np

# we want to read events_England.json

def get_L_T_R(l1, l2):
	def merger(l1, l2):
		M = l1 + l2
		M.sort()
		return M

	def check_in_list(a, l3):
		for g in range(0, len(l3)):
			if l3[g] == a:
				return True
		return False

	merged = merger(l1, l2)

	tags = []
	intervals = []

	for h in range(0, len(merged)):
		if check_in_list(merged[h], l1):
			tags.append(True)
		if check_in_list(merged[h], l2):
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

	intervals = []
	##
	count_dups = count_dups(tags)
	L = len(count_dups) / 2
	# print("L:", L)
	intervals_indices = np.cumsum(count_dups)
	intervals_indices = [elt - 1 for elt in intervals_indices]

	for i in range(len(intervals_indices)):
		if i == 0:
			intervals.append(merged[intervals_indices[i]])
		else:
			intervals.append(merged[intervals_indices[i]] - merged[intervals_indices[i - 1]])

	return np.mean(intervals), L

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

# Specify the path to the JSON file
file_path = './data/events/events_England.json'

# Read the JSON file
with open(file_path, 'r') as file:
	data = json.load(file)

# Access the data
# Example: Print the first event in the dataset
first_event = data[4]
# print(first_event)

matches = set()
teams = set()
c = 0

for eventID in range(len(data)):
	matches.add(data[eventID]['matchId'])
	teams.add(data[eventID]['teamId'])

matches = list(matches)
# print(len(matches))
teams = list(teams)
# print(matches)
# print(teams)



teams_match = [set() for _ in range(len(matches))]

for eventID in range(len(data)):
	for match_id in range(len(matches)):
		if data[eventID]['matchId'] == matches[match_id]:
			teams_match[match_id].add(data[eventID]['teamId'])

# print(teams_match)

passes_per_match = dict() #P
positional_params_r = dict()  #r = D
positional_params_phi = dict() #phi
time_between_passes = dict() # T_K
time_differences_posession = dict()
time_differences_posession_counter = dict()
startPosition_endPosition = dict()
T_R_list = dict()
# positions_per_pair = dict()
for match_id in range(len(matches)):
	for team in teams_match[match_id]:
		passes_per_match[(matches[match_id],team)] = 0
		positional_params_r[(matches[match_id],team)] = 0
		positional_params_phi[(matches[match_id],team)] = 0
		time_between_passes[(matches[match_id],team)] = 0
		time_differences_posession[(matches[match_id],team)] = []
		time_differences_posession_counter[(matches[match_id], team)] = 0
		startPosition_endPosition[(matches[match_id], team)] = []
		T_R_list[(matches[match_id], team)] = []
		# positions_per_pair[(matches[match_id], team)] = []
# passes_per_match = {str(match) + str(team): 0 for match in matches}
# # print(params_per_match)
# positional_params_r = {str(match): 0 for match in matches}
# positional_params_phi = {str(match): 0 for match in matches}

for eventID in range(len(data)):
	for match_id in range(len(matches)):
		if data[eventID]['eventName'] == 'Pass' and data[eventID]['matchId'] == matches[match_id]:
			for team in teams_match[match_id]:
				if data[eventID]['teamId'] == team:
					positional_params_r[(matches[match_id],team)] += np.sqrt((data[eventID]['positions'][0]['x'] - data[eventID]['positions'][1]['x'])**2 + (data[eventID]['positions'][0]['y'] - data[eventID]['positions'][1]['y'])**2)
					positional_params_phi[(matches[match_id],team)] += np.arctan2(data[eventID]['positions'][0]['y'] - data[eventID]['positions'][1]['y'], data[eventID]['positions'][0]['x'] - data[eventID]['positions'][1]['x'])
					passes_per_match[(matches[match_id],team)] += 1
					time_between_passes[(matches[match_id],team)] = data[eventID]['eventSec']
					time_differences_posession[(matches[match_id],team)].append(data[eventID]['eventSec'])
					startPosition_endPosition[(matches[match_id], team)].append([(data[eventID]['positions'][0]['x'], data[eventID]['positions'][0]['y']), (data[eventID]['positions'][1]['x'], data[eventID]['positions'][1]['y'])])
		# elif data[eventID]['eventName'] == 'Shot':
		# 	for dic in data[eventID]['tags']:
		# 		if dic['id'] == 101:
		# 			for team in teams_match[match_id]:
		# 				if data[eventID]['teamId'] == team:
		# 					# print(data[eventID])
		# 					startPosition_endPosition[(matches[match_id], team)].append(
		# 					 	[(data[eventID]['positions'][0]['x'], data[eventID]['positions'][0]['y']),(100,50)])
	# for match in matches[:3]:
	# 	positional_params_r[str(match)] = positional_params_r[str(match)] / passes_per_match[str(match)]
	# 	positional_params_phi[str(match)] = positional_params_phi[str(match)] / passes_per_match[str(match)]

#TODO: loop over team ID too. (done)
#TODO: average time between two passes. (done)
#TODO: ball lost? time of ball position. (done)
#TODO: ball regain? time of ball regain. (done)

for match_id in range(len(matches)):
	# print("teams_match[match_id]:", list(teams_match[match_id])[0])
	for team in teams_match[match_id]:
		# print("team", team)
		positional_params_r[(matches[match_id],team)] = positional_params_r[(matches[match_id],team)] / passes_per_match[(matches[match_id],team)]
		positional_params_phi[(matches[match_id], team)] = positional_params_phi[(matches[match_id], team)] / \
														 passes_per_match[(matches[match_id], team)]

		time_between_passes[(matches[match_id], team)] = time_between_passes[(matches[match_id],team)] / passes_per_match[(matches[match_id], team)]
		T_R_list[(matches[match_id], team)] = get_L_T_R(time_differences_posession[(matches[match_id],list(teams_match[match_id])[0])], time_differences_posession[(matches[match_id],list(teams_match[match_id])[1])])[0]
		time_differences_posession_counter[(matches[match_id], team)] = get_L_T_R(time_differences_posession[(matches[match_id],list(teams_match[match_id])[0])], time_differences_posession[(matches[match_id],list(teams_match[match_id])[1])])[1]

# print(len(time_differences_posession_counter))
# for match in matches[:3]:
# 	positional_params_r[str(match)] = positional_params_r[str(match)] / passes_per_match[str(match)]
# 	positional_params_phi[str(match)] = positional_params_phi[str(match)] / passes_per_match[str(match)]
#
# l1 = time_differences_posession[(2499719, 1609)]
# l2 = time_differences_posession[(2499719, 1631)]

# print(startPosition_endPosition[(2499719, 1609)])

# positions = startPosition_endPosition[(2499719, 1609)] # list of list of tuples


# print(time_differences_posession[(2499719, 1609)])
# print(time_differences_posession[(2499719, 1631)])

##
# print(startPosition_endPosition)

# print(intervals)
# print("T_R:", T_R_list) #T_R
# print("L:", time_differences_posession_counter)
# print("P:", passes_per_match)
# print("D:", positional_params_r)
# # print(positional_params_phi)
# print("T_K:",time_between_passes)


T_R = np.mean(list(T_R_list.values()))
L = np.mean(list(time_differences_posession_counter.values()))
P = np.mean(list(passes_per_match.values()))
D = np.mean(list(positional_params_r.values()))
T_K = np.mean(list(time_between_passes.values()))

# print(T_R, L, P, D, T_K)