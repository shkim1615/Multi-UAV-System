# gym을 사용한 계층 구조의 기본 형태를 보여줌. 
# 작동은 잘 하나. 내가 원하는 학습의 형태는 아님. 
# 구조만 참고할 것

import gym
import numpy as np
import random

# =======================
# LowLevelEnv: 서브목표까지 이동하는 환경
# =======================
class LowLevelEnv(gym.Env):
    def __init__(self, start_pos, target_pos, grid_size=10, max_steps=50):
        super(LowLevelEnv, self).__init__()
        self.grid_size = grid_size
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        self.state = self.start_pos.copy()
        self.max_steps = max_steps
        self.steps = 0
        
        # 행동: 0: up, 1: down, 2: left, 3: right
        self.action_space = gym.spaces.Discrete(4)
        # 관측: [agent_x, agent_y, target_x, target_y]
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(4,), dtype=np.float32)
    
    def reset(self):
        self.state = self.start_pos.copy()
        self.steps = 0
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate([self.state, self.target_pos])
    
    def step(self, action):
        # 행동에 따라 이동: 간단히 격자 내에서 한 칸 이동
        if action == 0:  # up
            self.state[1] = min(self.grid_size - 1, self.state[1] + 1)
        elif action == 1:  # down
            self.state[1] = max(0, self.state[1] - 1)
        elif action == 2:  # left
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 3:  # right
            self.state[0] = min(self.grid_size - 1, self.state[0] + 1)
        
        self.steps += 1
        done = False
        reward = -1  # 매 스텝마다 소규모 페널티
        
        # 목표에 도달하면 보상
        if np.array_equal(self.state, self.target_pos):
            done = True
            reward = 10  # 목표 도달 보상
        
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, {}
        
# =======================
# HighLevelEnv: 여러 타겟 중 하나를 선택해 하위 환경 실행
# =======================
class HighLevelEnv(gym.Env):
    def __init__(self, targets, grid_size=10, max_low_steps=50):
        super(HighLevelEnv, self).__init__()
        self.grid_size = grid_size
        self.targets = targets  # 예: [[x1, y1], [x2, y2], ...]
        self.remaining_targets = targets.copy()
        self.agent_pos = np.array([0, 0])  # 초기 위치 (0,0)
        self.max_low_steps = max_low_steps
        
        # 고수준 행동: 남아있는 타겟 중 하나의 인덱스 선택
        self.action_space = gym.spaces.Discrete(len(targets))
        # 관측: 에이전트 위치 + 남은 타겟들의 위치 (남은 타겟이 부족하면 -1로 패딩)
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, 
                                                shape=(2 + len(targets)*2,), dtype=np.float32)
    
    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.remaining_targets = self.targets.copy()
        return self._get_obs()
    
    def _get_obs(self):
        # 에이전트 위치와 남은 타겟들의 위치를 평탄화하여 반환
        obs = list(self.agent_pos)
        for t in self.remaining_targets:
            obs.extend(t)
        # 남은 타겟 수가 적으면 패딩 (-1)
        for _ in range(len(self.targets) - len(self.remaining_targets)):
            obs.extend([-1, -1])
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        # 고수준 행동: 남아있는 타겟 리스트에서 action 인덱스 선택
        if action >= len(self.remaining_targets):
            # 잘못된 행동에 대해서 큰 페널티
            return self._get_obs(), -10, True, {}
        
        chosen_target = self.remaining_targets[action]
        print(f"High-Level: 선택한 타겟 {chosen_target}")
        
        # LowLevelEnv를 생성해, 현재 위치에서 선택한 타겟까지 이동
        low_env = LowLevelEnv(start_pos=self.agent_pos, target_pos=chosen_target,
                              grid_size=self.grid_size, max_steps=self.max_low_steps)
        low_obs = low_env.reset()
        total_low_reward = 0
        done = False
        steps = 0
        
        # 여기서는 예시로 low-level에서 랜덤 정책을 사용
        while not done:
            action_ll = low_env.action_space.sample()
            low_obs, reward, done, _ = low_env.step(action_ll)
            total_low_reward += reward
            steps += 1
        
        # Low-level 실행 후 에이전트 위치 업데이트
        self.agent_pos = low_env.state.copy()
        
        # 목표에 도달했으면, 해당 타겟을 제거하고 보너스 지급
        if np.array_equal(self.agent_pos, np.array(chosen_target)):
            bonus = 20
            self.remaining_targets.pop(action)
        else:
            bonus = 0
        
        # 고수준 보상: 소요 스텝에 따른 페널티와 목표 달성 보너스
        high_reward = -steps + bonus
        
        # 모든 타겟이 달성되면 에피소드 종료
        done_high = len(self.remaining_targets) == 0
        
        return self._get_obs(), high_reward, done_high, {}
        
# =======================
# 메인 함수: 계층적 환경 테스트
# =======================
def main():
    # 예시 타겟들 (여기서는 3개의 타겟)
    targets = [[8, 8], [3, 7], [6, 2]]
    high_env = HighLevelEnv(targets=targets, grid_size=10, max_low_steps=50)
    
    obs = high_env.reset()
    total_reward = 0
    done = False
    
    while not done:
        available = len(high_env.remaining_targets)
        # 여기서는 고수준 정책 대신 랜덤 선택 (실제 학습 시에는 고수준 정책 네트워크가 결정)
        action = random.randrange(available)
        print("\n고수준 행동: 타겟 인덱스", action, "선택 (남은 타겟:", high_env.remaining_targets,")")
        obs, reward, done, _ = high_env.step(action)
        total_reward += reward
        print("고수준 보상:", reward, "| 누적 보상:", total_reward)
        print("새 관측값:", obs)
    
    print("\n에피소드 종료. 총 누적 보상:", total_reward)
    
if __name__ == '__main__':
    main()
