# vmas의 이미 설정된 환경을 가져와서 억지로 덮어씌우기 한 형태
# 작동도 안하고. 에이전트가 자동으로 액션과 관찰 공간을 가지는 것을 해결하지 못함. 
# 아마 작동이 되게 만들어도 torchrl에 변환되서 들어가면서 agent의 쓸모없는 action, obs space가 들어가서 문제를 일으킬 것. 
# 이런 형태가 가능은 하다는 참고 정도. 

import gym
import numpy as np
import random
import vmas  # VMAS 라이브러리 (설치되어 있어야 함)

# --------------------------------------------------
# HighLevelVMASEnvWrapper: VMAS 환경을 래핑하여 상위 환경 구현
# --------------------------------------------------
class HighLevelVMASEnvWrapper(gym.Env):
    def __init__(self, targets, grid_size=10, max_low_steps=50):
        super(HighLevelVMASEnvWrapper, self).__init__()
        self.grid_size = grid_size
        # VMAS 환경 생성 (예: 간단한 멀티에이전트 격자 환경)
        # 실제 환경 이름과 파라미터는 VMAS 문서를 참조하세요.
        self.vmas_env = vmas.make("simple_ma", grid_size=self.grid_size, num_agents=1)
        
        # 고수준 목표로 사용할 타겟 리스트 (예: [[8, 8], [3, 7], [6, 2]])
        self.targets = [np.array(t, dtype=int) for t in targets]
        self.remaining_targets = self.targets.copy()
        self.max_low_steps = max_low_steps
        
        # 고수준 액션: 남아 있는 타겟 중 하나를 선택 (인덱스)
        self.action_space = gym.spaces.Discrete(len(self.targets))
        
        # 고수준 관측: 현재 에이전트 위치 + 모든 타겟의 위치 (남은 타겟은 그대로, 제거된 타겟은 (-1, -1)로 패딩)
        obs_length = 2 + len(self.targets) * 2  # 예: 에이전트 (x,y) + 각 타겟 (x,y)
        self.observation_space = gym.spaces.Box(low=-1, high=self.grid_size-1, shape=(obs_length,), dtype=np.float32)
    
    def reset(self):
        # VMAS 환경 리셋 (여기서는 내부적으로 에이전트 위치가 초기화됨)
        self.vmas_env.reset()
        self.remaining_targets = self.targets.copy()
        return self._get_obs()
    
    def _get_obs(self):
        # 예시로 단일 에이전트(에이전트 0)의 위치를 가져온다고 가정합니다.
        agent_obs = self.vmas_env.get_agent_obs(0)  # 예: np.array([x, y])
        obs = list(agent_obs)
        for target in self.remaining_targets:
            obs.extend(target.tolist())
        # 남은 타겟 수가 부족하면 (-1, -1) 패딩
        for _ in range(len(self.targets) - len(self.remaining_targets)):
            obs.extend([-1, -1])
        return np.array(obs, dtype=np.float32)
    
    def step(self, high_level_action):
        # 고수준 액션: 남아있는 타겟 리스트의 인덱스를 선택
        if high_level_action >= len(self.remaining_targets):
            # 잘못된 액션에 대해서 큰 페널티 부여 및 에피소드 종료
            return self._get_obs(), -100, True, {}
        
        chosen_target = self.remaining_targets[high_level_action]
        print("High-level: 선택한 타겟", chosen_target)
        
        low_steps = 0
        success = False
        
        # 하위 정책 실행: 최대 max_low_steps 동안 VMAS 환경에서 행동 실행
        while low_steps < self.max_low_steps:
            current_pos = self.vmas_env.get_agent_obs(0)
            if np.array_equal(current_pos, chosen_target):
                success = True
                break
            
            # 간단한 그리디(low-level) 정책: Manhattan 거리를 줄이는 방향 선택
            diff = chosen_target - current_pos
            if abs(diff[0]) > abs(diff[1]):
                low_action = 3 if diff[0] > 0 else 2  # 오른쪽 혹은 왼쪽
            else:
                low_action = 0 if diff[1] > 0 else 1  # 위쪽 혹은 아래쪽
            
            # VMAS 환경은 일반적으로 에이전트별 행동을 딕셔너리로 받습니다.
            actions = {0: low_action}
            self.vmas_env.step(actions)
            low_steps += 1
        
        # 고수준 보상: 도달 성공 여부 및 소요 스텝 수에 따른 보상 산출
        if success:
            reward = 20 - low_steps  # 도달하면 보너스, 적은 스텝일수록 높은 보상
            # 도달한 타겟은 목록에서 제거
            self.remaining_targets.pop(high_level_action)
        else:
            reward = -low_steps  # 실패시 소요 스텝에 따른 페널티
        
        done = len(self.remaining_targets) == 0
        
        return self._get_obs(), reward, done, {}

# --------------------------------------------------
# 간단한 테스트 실행 코드
# --------------------------------------------------
def main():
    targets = [[8, 8], [3, 7], [6, 2]]
    env = HighLevelVMASEnvWrapper(targets=targets, grid_size=10, max_low_steps=50)
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        available = len(env.remaining_targets)
        # 여기서는 고수준 정책 대신 랜덤 선택 (실제 학습 시에는 고수준 정책 네트워크가 결정)
        action = random.randrange(available)
        print("\n고수준 액션: 타겟 인덱스", action, "선택 (남은 타겟:", env.remaining_targets, ")")
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        print("고수준 보상:", reward, "| 누적 보상:", total_reward)
        print("새 관측값:", obs)
    
    print("\n에피소드 종료. 총 누적 보상:", total_reward)

if __name__ == "__main__":
    main()
