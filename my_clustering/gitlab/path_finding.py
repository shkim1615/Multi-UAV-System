from torch import tensor
import numpy as np
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import itertools
import numpy as np

def random_choice(agents, targets):
    n_agents, n_targets = len(agents), len(targets)
    number= n_targets // n_agents
    
    return [targets[i:i+number] for i in range(0, n_targets, number)]
