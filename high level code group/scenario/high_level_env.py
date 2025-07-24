import gym
import numpy as np
from gym import spaces

class MultiAgentTargetEnv(gym.Env):
    def __init__(self, num_agents=4, num_targets=10, grid_size=100, max_steps=500):
        super(MultiAgentTargetEnv, self).__init__()
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_count = 0
        
        # Action space: (dx, dy) for each agent
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents, 2), dtype=np.float32)
        
        # Observation space: agent positions, agent targets, and all targets
        self.observation_space = spaces.Box(
            low=0, high=grid_size, shape=(num_agents * 2 + num_agents * 2 + num_targets * 2,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.step_count = 0
        
        # Initialize agent positions randomly
        self.agent_positions = np.random.uniform(0, self.grid_size, size=(self.num_agents, 2))
        
        # Initialize target positions randomly
        self.target_positions = np.random.uniform(0, self.grid_size, size=(self.num_targets, 2))
        
        # Assign each agent a random target
        self.agent_targets = np.array([self.target_positions[i % self.num_targets] for i in range(self.num_agents)])
        
        return self._get_observation()
    
    def _get_observation(self):
        return np.concatenate([
            self.agent_positions.flatten(),
            self.agent_targets.flatten(),
            self.target_positions.flatten()
        ], dtype=np.float32)
    
    def step(self, actions):
        self.step_count += 1
        rewards = np.zeros(self.num_agents)
        done = False
        
        for i in range(self.num_agents):
            # Apply movement
            self.agent_positions[i] += np.clip(actions[i], -1, 1)
            self.agent_positions[i] = np.clip(self.agent_positions[i], 0, self.grid_size)  # Keep inside bounds
            
            # Compute distance to target
            prev_distance = np.linalg.norm(self.agent_positions[i] - self.agent_targets[i])
            new_distance = np.linalg.norm(self.agent_positions[i] - self.agent_targets[i])
            
            # Reward for moving closer
            if new_distance < prev_distance:
                rewards[i] += 0.1
                
            # If agent reaches the target (distance <= 1), assign a new target
            if new_distance <= 1.0:
                rewards[i] += 10
                target_idx = np.where((self.target_positions == self.agent_targets[i]).all(axis=1))[0]
                if target_idx.size > 0:
                    self.target_positions[target_idx[0]] = [-1, -1]  # Mark as found
                new_target_idx = np.random.randint(0, self.num_targets)
                while (self.target_positions[new_target_idx] == [-1, -1]).all():
                    new_target_idx = np.random.randint(0, self.num_targets)  # Ensure valid target
                self.agent_targets[i] = self.target_positions[new_target_idx]
                
        # Small penalty per step
        rewards -= 0.01
        
        # Check if all targets are found
        if np.all(self.target_positions == [-1, -1]):
            done = True
            rewards += (self.max_steps - self.step_count)  # Reward remaining steps
            
        # Terminate if max steps reached
        if self.step_count >= self.max_steps:
            done = True
        
        return self._get_observation(), rewards, done, {}
    
    def render(self, mode='human'):
        print("Agent Positions:", self.agent_positions)
        print("Target Positions:", self.target_positions)
    
    def close(self):
        pass

# 환경 테스트
env = MultiAgentTargetEnv()
obs = env.reset()
done = False
while not done:
    actions = np.random.uniform(-1, 1, size=(env.num_agents, 2))
    obs, rewards, done, _ = env.step(actions)
    env.render()
