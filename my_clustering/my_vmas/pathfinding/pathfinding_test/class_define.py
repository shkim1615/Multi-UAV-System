import torch
from torch import tensor
class State:
    def __init__(self, x, y):
        self.pos = tensor([x, y], dtype=torch.float32)

class Agent(State):
    def __init__(self, name, x, y, radius = 10):
        self.name = name
        self.state = State(x, y)
        self.radius = radius
    
    # def __repr__(self):
    #     return f"{self.name}: ({self.state.pos})"

class Target:
    def __init__(self, name, x, y, cost, radius = 5):
        self.name = name
        self.state = State(x, y)
        self.cost = cost
        self.radius = radius
        
    def __repr__(self):
        return f"{self.name}: ({self.state.pos}, {self.cost})"