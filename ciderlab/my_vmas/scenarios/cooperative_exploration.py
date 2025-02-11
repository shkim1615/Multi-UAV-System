# 완전히 새로 제작
import torch

from my_vmas.simulator.scenario import BaseScenario
from my_vmas.simulator.utils import ScenarioUtils, Color
from my_vmas.simulator.core import World, Agent, Sphere, Landmark
from my_vmas.simulator.sensors import Lidar

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # 월드 크기
        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.2)      # 객체들 간 최소 거리
        
        # 드론
        self.n_agents = kwargs.pop("n_agents", 4)
        self.collisions = kwargs.pop("collisions", True)
        
        # 타겟
        self.n_targets = kwargs.pop("n_targets", 10)
        self._covering_range = kwargs.pop("covering_range", 0.25)                   # 타겟 내 일정 거리
        
        # 보상
        self.shared_reward = kwargs.pop("shared_reward", True)                      # 공유 보상
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", 0)
        self.time_penalty = kwargs.pop("time_penalty", 0)                           # 시간 패널티
        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 1.0)             # 타겟 커버 시 보상 가중치
        
        # 라이다 센서는 나중에 추가
        
        # kwargs에 남아있는 키가 없는지 확인하여 전부 다 사용했는지 확인
        ScenarioUtils.check_kwargs_consumed(kwargs)
        
        # 추가 설정
        self.min_collision_distance = 0.005
        self.agent_radius = 0.05
        self.target_radius = self.agent_radius

        self.target_color = Color.GREEN
        
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            collision_force=500,                # collision_force: 충동 시 에이전트가 겹치지 않도록 하는 반발력
            substeps=2,                         # substeps: 전체 시뮬레이션 시간 단계를 더 작은 단위(서브스텝)로 나누어 물리 연산을 반복함으로써, 시뮬레이션의 안정성과 정확도를 높임
            drag=0.25,                          #           메인 스텝 1번 당 2번의 서브 스텝을 진행하여 물리 효과를 보다 부르럽게 처리
        )                                       # drag: 물체가 이동할 때 발생하는 마찰이나 공기 저항 같은 감쇠 효과
        
        # 에이전트 색상 
        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        
        # 에이전트 생성
        # 현재 라이다 제거 버전
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )
            
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(),
            )
            # 각 에이전트 별 보상 초기화 느낌인데..
            agent.collision_reward = torch.zeros(batch_dim, device=device)
            agent.covering_reward = agent.collision_reward.clone()
            world.add_agent(agent)
            
        self._targets = []
        for i in range(self.n_targets):
            target = Landmark(
                name=f"target_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=self.target_radius),
                color=self.target_color,
            )
            world.add_landmark(target)
            self._targets.append(target)
            
        # 타겟과 관련된 보상
        self.covered_targets = torch.zeros(batch_dim, self.n_targets, device=device)    # 커버된 타겟 확인용
        self.shared_covering_reward = torch.zeros(batch_dim, device=device)             # 커버된 타겟 보상
        
        return world
    
    def reset_world_at(self, env_index = None):
        return super().reset_world_at(env_index)
    
    def reward(self, agent):
        return super().reward(agent)
    
    def observation(self, agent):
        return super().observation(agent)