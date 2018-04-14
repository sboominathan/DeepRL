import numpy as np

action_distributions = [

	{0: 0.5138393615214807, 1: 0.48616063847851926},
	{0: 0.4975, 1: 0.5025},
	{0: 0.5469514295556321, 1: 0.4530485704443679},
	{0: 0.5004, 1: 0.4996},
	{0: 0.5, 1: 0.5},
	{0: 0.50975, 1: 0.49025},
	{0: 0.4903, 1: 0.5097},
	{0: 0.5, 1: 0.5},
	{0: 0.47113373309161904, 1: 0.528866266908381},
	{0: 0.5, 1: 0.5},
	{0: 0.586779498642823, 1: 0.4132205013571771},
	{0: 0.5, 1: 0.5},
	{0: 0.49995, 1: 0.50005},
	{0: 0.5007774489642374, 1: 0.49922255103576263},
	{0: 0.50035, 1: 0.49965},
	{0: 0.5138393615214807, 1: 0.48616063847851926},

]

def get_action_variance(action_distributions):
	possible_actions = action_distributions[0].keys()
	variances = {}
	for action in possible_actions:
		action_values = [dist[action] for dist in action_distributions]
		print(action_values)
		variances[action] = np.var(np.array(action_values))
	return variances

print(get_action_variance(action_distributions))


