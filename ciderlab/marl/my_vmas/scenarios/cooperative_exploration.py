# 완전히 새로 제작
import torch

from my_vmas.simulator.scenario import BaseScenario
from my_vmas.simulator.utils import ScenarioUtils, Color
from my_vmas.simulator.core import World, Agent, Sphere, Landmark, Entity
from my_vmas.simulator.sensors import Lidar

import typing
from typing import Callable, Dict, List

from torch import Tensor

from vmas.simulator.utils import X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # 월드 크기
        self.x_semidim = kwargs.pop("x_semidim", 2)
        self.y_semidim = kwargs.pop("y_semidim", 2)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.3)      # 객체들 간 최소 거리
        
        # 드론
        self.n_agents = kwargs.pop("n_agents", 4)
        self.collisions = kwargs.pop("collisions", True)
        
        # 라이다
        self._lidar_range = kwargs.pop("lidar_range", 0.35)
        self.n_lidar_rays_agents = kwargs.pop("n_lidar_rays_agents", 15)            # 라이다 선 개수
        
        # 타겟
        self.n_targets = kwargs.pop("n_targets", 1)
        self._covering_range = kwargs.pop("covering_range", 0.25)                   # 타겟 내 일정 거리
        self.targets_respawn = kwargs.pop("targets_respawn", True)
        
        # 보상
        self.shared_reward = kwargs.pop("shared_reward", True)                      # 공유 보상
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        self.time_penalty = kwargs.pop("time_penalty", -0.01)                           # 시간 패널티
        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 1.00)             # 타겟 커버 시 보상 가중치
        
        # kwargs에 남아있는 키가 없는지 확인하여 전부 다 사용했는지 확인
        ScenarioUtils.check_kwargs_consumed(kwargs)
        
        # 추가 설정
        self.min_collision_distance = 0.005
        self.agent_radius = 0.05
        self.target_radius = self.agent_radius

        self.target_color = tuple(Color.GREEN.value)

        
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
        
        # 엔터지에서 에이전트 객체만 선택
        entity_filter_agents: Callable[[Entity], bool] = lambda e: e.name.startswith(
            "agent"
        )
        
        # 에이전트 생성
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
                sensors=(
                    [
                        Lidar(
                            world,
                            angle_start=0.05,
                            angle_end=2 * torch.pi + 0.05,
                            n_rays=self.n_lidar_rays_agents,
                            max_range=self._lidar_range,
                            entity_filter=entity_filter_agents,
                            render_color=tuple(Color.BLUE.value),
                        )
                    ]
                ),
            )
            # 각 에이전트 별 보상 초기화
            agent.agent_collision_rew = torch.zeros(batch_dim, device=device)
            agent.covering_reward = agent.agent_collision_rew.clone()               # 현재 에이전트 본인의 발견에 대한 개인 보상은 없음
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
        
        # 타겟 삭제 옵션
        self.fixed_costs = torch.tensor([10, 10, 10, 20, 20, 20, 30, 30, 40, 40], device=device)
        self.selected_costs = self.fixed_costs[: self.n_targets]
        self.finish_cost_targets = self.selected_costs.unsqueeze(0).repeat(batch_dim, 1)
        self.cost_targets = torch.zeros(batch_dim, self.n_targets, device=device)
        
        
        return world
    
    def reset_world_at(self, env_index = None):
        placable_entities = self._targets[: self.n_targets] + self.world.agents
        
        batch_dim = self.world.batch_dim
        device = self.world.device    
        
        
        if env_index is None:
            # self.finish_cost_targets = self.self.fixed_costs.unsqueeze(0).repeat(batch_dim, 1)
            self.cost_targets = torch.zeros(batch_dim, self.n_targets, device=device)
            
            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets),
                False,
                device=self.world.device,
            )
        else:
            # 특정 환경(env_index)만 초기화
            # 기존 전체 텐서에서 해당 인덱스만 업데이트
            # self.finish_cost_targets[env_index] = self.fixed_costs
            self.cost_targets[env_index] = torch.zeros(1, self.n_targets, device=device)
            
            self.all_time_covered_targets[env_index] = False
            
        ScenarioUtils.spawn_entities_randomly(
            entities=placable_entities,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=self._min_dist_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
        )
    
    def reward(self, agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]
        
        if is_first:
            # 시간 패널티 보상 초기화
            self.time_rew = torch.full(
                (self.world.batch_dim,),
                self.time_penalty,
                device=self.world.device,
            )
            
            # 에이전트와 타겟의 위치 수집
            self.agents_pos = torch.stack(
                [a.state.pos for a in self.world.agents], dim=1
            )
            self.targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)
            
            # 에이전트 충돌 보상 초기화
            for a in self.world.agents:
                a.agent_collision_rew[:] = 0    
            
            # 에이전트 간 충돌 감지 패널티 부여
            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
            
            # 에이전트와 타겟 간의 거리 계산  
            self.agents_targets_dists = torch.cdist(self.agents_pos, self.targets_pos)
            
            # 타겟 커버 초기화
            self.covered_targets = torch.zeros(self.world.batch_dim, self.n_targets, device=self.world.device)
            
            # 공유 보상
            self.covered_targets = (self.agents_targets_dists < self._covering_range).any(dim=1)  
            self.shared_covering_rew = self.covered_targets.float().sum(dim=1) * self.covering_rew_coeff
            
            # 에이전트 개별적인 보상에 대한 것은 없음
            
        if is_last:
            self.cost_targets += self.covered_targets.float()
            
            # 각 타겟에 대해 누적 비용이 해당 타겟의 임계치 이상인 경우, 타겟을 활성 영역 밖으로 이동
            for i, target in enumerate(self._targets):
                # mask는 (batch_dim,) 크기의 불린 텐서: 각 배치에서 i번째 타겟의 누적 비용이 임계치 이상인 경우 True
                mask = self.cost_targets[:, i] >= self.finish_cost_targets[:, i]
                if mask.any():
                    # get_outside_pos(None)는 (batch_dim, dim_p) 크기의 텐서를 반환합니다.
                    target.state.pos[mask] = self.get_outside_pos(None)[mask]   
                    target.active = False
                
                                 
            
        return agent.agent_collision_rew + self.shared_covering_rew + self.time_rew
    
    def get_outside_pos(self, env_index):
        return torch.empty(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)
        
    def observation(self, agent: Agent):
        # 에이전트의 상태 정보
        pos = agent.state.pos         # 예: [x, y]
        vel = agent.state.vel         # 예: [vx, vy]

        # 충돌 정보를 추가
        collision_info = []
        if self.collisions:
            collision_info = [agent.sensors[0]._max_range - agent.sensors[0].measure()]

        # 각 타겟에 대해 에이전트 기준 상대 위치 계산
        target_rel_positions_list = [target.state.pos - pos for target in self._targets]
        # print(target_rel_positions)
        # 각 타겟의 상대 위치
        if target_rel_positions_list:
            target_rel_positions = torch.cat(target_rel_positions_list, dim=-1)
        else:
            target_rel_positions = torch.tensor([], device=pos.device)

        # 최종 관찰 벡터 구성: 에이전트의 상태, 라이다 측정, (옵션) 충돌 정보, 타겟 상대 위치
        observation = torch.cat(
            [pos, vel] + collision_info + [target_rel_positions],
            dim=-1,
        )
        
        # 관찰 정보가 너무 많은 건 아닌지. 상대 정보 말고 타겟 정보만 넣어줘볼까?
        # observation = torch.cat(
        #     [pos, vel] + collision_info + self._targets,
        #     dim=-1,
        # )
        return observation
    
    def done(self):
        return (self.cost_targets >= self.finish_cost_targets).all(dim=1)
        
    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        for target in self._targets:
            range_circle = rendering.make_circle(self._covering_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*self.target_color)
            geoms.append(range_circle)

        return geoms
