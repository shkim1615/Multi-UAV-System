#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    
class Scenario(BaseScenario):
    
    # discovery랑 꽤나 큰 차이가 있는데 일단 냅두자.
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # batch_dim: 배치 처리 크기. 멀티 에이전트 시뮬레이션을 한 번에 여러 환경에서 병렬로 실행할 때 사용
        # **kwargs: 시뮬레이션 설정을 유연하게 받기 위한 키워드 인자
        
        # 이거 하나는 베이스 시나리오에 있는거고
        self.plot_grid = False
        
        ## ====================================================================================================
        # 여기부터는 kwargs에서 입력을 받으면 거기에서 꺼내오는 거고
        # 근데 순서는 상관이 없는 건지. 존재하지 않으면 어떻게 되는건지 모르겠음
        self.n_agents = kwargs.pop("n_agents", 4)
        self.collisions = kwargs.pop("collisions", True)
        
        ## 추가
        self.n_targets = kwargs.pop("n_targets", 10)
        self.target_cost = kwargs.pop("target_cost", 4)

        self.world_spawning_x = kwargs.pop(
            "world_spawning_x", 1
        )  # X-coordinate limit for entities spawning
        
        # 아래의 두 항목이 적용되는 것이 드론이나 탐색 지점이냐
        # 월드 내 객체가 생성될 x, y 범위
        self.world_spawning_y = kwargs.pop(
            "world_spawning_y", 1
        )  # Y-coordinate limit for entities spawning
        
        # 월드의 경계 제한을 활성화할지 여부
        self.enforce_bounds = kwargs.pop(
            "enforce_bounds", False
        )  # If False, the world is unlimited; else, constrained by world_spawning_x and world_spawning_y.

        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)

        self.lidar_range = kwargs.pop("lidar_range", 0.35)
        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)

        self.shared_reward = kwargs.pop("shared_reward", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        ## ====================================================================================================

        # 위에서 가져온 값으로 뭐 기준을 세우는 듯
        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005

        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
        else:
            self.x_semidim = None
            self.y_semidim = None

        # 제약 조건 확인
        # 공동 목표, 개별 목표에 따라 설정이 달라지는데
        # assert: 조건 확인과 디버깅을 위해 사용되는 도구로, 조건이 만족되지 않으면 바로 예외를 발생시켜 코드의 잘못된 상태를 조기에 발견할 수 있도록 도와줌
        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert (
                not self.collisions
            ), "If agents share goals they cannot be collidables"
        # agents_with_same_goal == n_agents: all agent same goal
        # agents_with_same_goal = x: the first x agents share the goal
        # agents_with_same_goal = 1: all independent goals
        if self.split_goals:
            assert (
                self.n_agents % 2 == 0
                and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=2,                     # 물리엔진의 시뮬레이션 단계
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )

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
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)

        # 각 에이전트를 생성하고 월드에 추가
        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Agent는 시뮬레이션 내에서 개별 행동을 담당하는 객체
            # Constraint: all agents have same action range and multiplier
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
                            n_rays=self.n_lidar_rays,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,     # 다른 에이전트만 감지하도록 필터링
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )
            agent.pos_reward = torch.zeros(batch_dim, device=device)
            agent.agent_collision_reward = agent.pos_reward.clone()
            world.add_agent(agent)
            
        self.target_radius = self.agent_radius
        self.target_color = Color.GREEN

        self.targets = []
        for i in range(self.n_targets):
            target = Landmark(
                name=f"target_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=self.target_radius),
                color=self.target_color,
            )
            world.add_landmark(target)
            self.targets.append(target)


        # 보상 변수를 초기화하여 각 에이전트의 성과를 추적
        # pos_rew: 위치 기반 보상
        # final_rew: 최종 보상
        # 무슨 말이지 각각의 에이전트 개별로 보상을 축적한다는 말인가. 이거는 협동 게임이 아닌가
        # 여기가 일단 보상 체계의 일종인 것 같은데, 나한테 맞게 나중에 세팅을 해야 함.
        self.pos_reward = torch.zeros(batch_dim, device=device)
        self.final_reward = self.pos_reward.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        # 에이전트 무작위 배치
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,                     # 에이전트 간 최소 거리 제한을 통해 겹치지 않게 함
            (-self.world_spawning_x, self.world_spawning_x),
            (-self.world_spawning_y, self.world_spawning_y),
        )

        # 현재 배치된 에이전트 위치 저장
        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        # env_index가 주어질 경우 해당 환경에만 적용
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        # 목표 위치 무작위 배치
        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(        # 최소 거리 조건을 만족하면서 겹치지 않는 좌표 찾기
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            )
            goal_poses.append(position.squeeze(1))
            # 새로 할당된 목표 위치를 기존의 위치들과 병합하여 다른 위치와의 충돌 방지
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        # 목표 위치 할당
        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i

            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )
                
    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            # 시간 패널티 보상 초기화
            self.time_reward = torch.full(
                (self.world.batch_dim,),
                self.time_penalty,
                device=self.world.device,
            )
            
            # 에이전트와 타겟의 위치 수집
            self.agents_pos = torch.stack(
                [a.state.pos for a in self.world.agents], dim=1
            )
            self.targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)
            self.agents_targets_dists = torch.cdist(self.agents_pos, self.targets_pos)
            self.agents_per_target = torch.sum(
                (self.agents_targets_dists < self._covering_range).type(torch.int),
                dim=1,
            )
            self.covered_targets = self.agents_per_target >= self._agents_per_target

            self.shared_covering_reward[:] = 0
            for a in self.world.agents:
                self.shared_covering_reward += self.agent_reward(a)
            self.shared_covering_reward[self.shared_covering_reward != 0] /= 2

        # 충돌 패널티
        agent.collision_reward[:] = 0
        for a in self.world.agents:
            if a != agent:
                agent.collision_reward[
                    self.world.get_distance(a, agent) < self.min_collision_distance
                ] += self.agent_collision_penalty

        # 타겟 커버 보상인 거 같은데
        self.all_time_covered_targets += self.covered_targets
        for i, target in enumerate(self._targets):
            target.state.pos[self.covered_targets[:, i]] = self.get_outside_pos(
                None
            )[self.covered_targets[:, i]]
                    
        covering_reward = (
            agent.covering_reward
            if not self.shared_reward
            else self.shared_covering_reward
        )

        return agent.collision_reward + covering_reward + self.time_reward
    
    def get_outside_pos(self, env_index):
        return torch.empty(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)

    def agent_reward(self, agent):
        agent_index = self.world.agents.index(agent)

        agent.covering_reward[:] = 0
        targets_covered_by_agent = (
            self.agents_targets_dists[:, agent_index] < self._covering_range
        )
        num_covered_targets_covered_by_agent = (
            targets_covered_by_agent * self.covered_targets
        ).sum(dim=-1)
        agent.covering_reward += (
            num_covered_targets_covered_by_agent * self.covering_rew_coeff
        )
        return agent.covering_reward

    def observation(self, agent: Agent):
        lidar_1_measures = agent.sensors[0].measure()
        return torch.cat(
            [agent.state.pos, agent.state.vel, lidar_1_measures]
            + ([agent.sensors[1].measure()] if self.use_agent_lidar else []),
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "covering_reward": (
                agent.covering_reward
                if not self.shared_reward
                else self.shared_covering_rew
            ),
            "collision_rew": agent.collision_rew,
            "targets_covered": self.covered_targets.sum(-1),
        }
        return info

    def done(self):
        return self.all_time_covered_targets.all(dim=-1)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        for target in self._targets:
            range_circle = rendering.make_circle(self._covering_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*self.target_color.value)
            geoms.append(range_circle)
        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self._comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms
