import torch
from vmas import make_env, Scenario

class HighLevelVMASScenario(Scenario):
    def __init__(self, num_agents=1, num_targets=3, grid_size=10):
        super().__init__()
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.grid_size = grid_size

    def build(self, world):
        """VMAS 시뮬레이션을 초기화하는 함수"""
        # 에이전트 생성
        for i in range(self.num_agents):
            agent = world.add_agent()
            agent.pos = torch.rand(2) * self.grid_size  # 랜덤 초기 위치

        # 타겟 생성
        self.targets = torch.rand((self.num_targets, 2)) * self.grid_size

    def reset(self, world):
        """환경 리셋 시 호출되는 함수"""
        for agent in world.agents:
            agent.pos = torch.rand(2) * self.grid_size  # 초기 위치 랜덤 배치

        self.targets = torch.rand((self.num_targets, 2)) * self.grid_size

    def step(self, world):
        """한 스텝 진행"""
        for agent in world.agents:
            # 상위 환경에서 받은 액션을 "어떤 타겟을 선택할지"로 해석
            high_level_action = agent.action[0].item()  # 타겟 선택 인덱스

            # 선택한 타겟의 위치
            target_pos = self.targets[high_level_action]

            # 간단한 이동 정책 (타겟으로 한 스텝씩 이동)
            direction = (target_pos - agent.pos).sign()  # 부호에 따라 방향 결정
            agent.pos += direction * 0.1  # 속도 조정 가능

            # 목표 도달 체크
            if torch.norm(agent.pos - target_pos) < 0.1:
                agent.reward = 10  # 목표 도달 시 보상
            else:
                agent.reward = -1  # 매 스텝 작은 페널티

