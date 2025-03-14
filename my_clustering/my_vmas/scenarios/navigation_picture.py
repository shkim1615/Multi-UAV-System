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

import sys
sys.path.append('/home/ksh-server/workspace/ICUFN/my_clustering')
from my_vmas.pathfinding.path_finding import random_choice
from my_vmas.my_interactive_rendering import render_interactively

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    # 타겟 개수, 타겟들 분배 등에 관한 설정이 아예 없음
    # 탐색이 완료된 에이전트의 대기 위치
    # 에이전트의 초기 생성 위치
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        
        # kwargs는 무조건 트레이닝에서 다시 지정할 것. 
        # 훈련 시에 어떤 조건으로 돌아가는지 헷갈리더라
        self.n_agents = kwargs.pop("n_agents", 4)
        self.collisions = kwargs.pop("collisions", True)

        self.world_spawning_x = kwargs.pop(
            "world_spawning_x", 1 * 2
        )  # X-coordinate limit for entities spawning
        self.world_spawning_y = kwargs.pop(
            "world_spawning_y", 1 * 2
        )  # Y-coordinate limit for entities spawning
        self.enforce_bounds = kwargs.pop(
            "enforce_bounds", True
        )  # If False, the world is unlimited; else, constrained by world_spawning_x and world_spawning_y.

        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)

        self.lidar_range = kwargs.pop("lidar_range", 0.35) # 0.35
        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)  # 12

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        
        # 타겟 설정
        self.n_targets_per_agent = kwargs.pop("n_targets_per_agent", 3)
        self.n_targets = self.n_agents * self.n_targets_per_agent
        self.target_radius = kwargs.pop("target_radius", self.agent_radius)
        self.covering_range = kwargs.pop("covering_range", self.target_radius * 1.2)
        self.target_temp_color = Color.BLACK
        
        # 타임 패널티
        self.time_penalty = kwargs.pop("time_penalty", -0.01)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005
        
        # agents 대기 상태 위치
        dist = 0.2
        self.finished_pos = [[Tensor([[dist, dist]])], Tensor([[dist, -dist]]), Tensor([[-dist, -dist]]), Tensor([[-dist, dist]])]

        # 월드 크기의 제한이 없는 상태
        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
        else:
            self.x_semidim = None
            self.y_semidim = None

        # 모든 목표는 각 에이전트 마다 독립적으로 작용
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
            substeps=2,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )

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

        # Add agents
        # 에이전트 갯수만큼 반복하여 에이전트 하나씩 설정
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

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
                            entity_filter=entity_filter_agents,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            agent.cur_cost = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)
            # 이거 보니까 여기서 설정하는 목표는 그냥 이 에이전트의 어떤 구조를 만드는 것 같은데
            # agent.goal = goal
            # 정확한 건 navigation을 다시 보면서 확인할 것

        # Add goals
        # 미리 받은 타겟의 수 만큼 생성을 해두고 거기서 처음 goal을 지정해서 agent.goal에 적용할 것.
        # 지정된 개수만큼 타겟 생성
        # 아니야. 에이전트의 목표는 일단 비워두고 이게 맞는 거 같아.
        # 나중에 필요한 것이
        # agent.goal = target in targets
        # target_pos = 뭐시기
        
        # 타겟 당 코스트 지정
        self.targets_cur_cost = torch.zeros(self.n_targets)
        self.targets_cost = torch.tensor([10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40], device=device)
        
        self.targets = []
        
        for i in range(self.n_targets):
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=self.target_temp_color,
            )
            goal.cost = self.targets_cost[i]
            world.add_landmark(goal)
            self.targets.append(goal)
            
        self.finished_targets = []
        for i in range(len(self.finished_pos)):
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=self.target_temp_color,
            )
            goal.cost = torch.tensor([10])
            world.add_landmark(goal)
            self.finished_targets.append(goal)
            
        
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        # 클러스터링, tsp 등의 기법을 사용해서 경로 탐색을 한 후 정보를 넘겨주려면
        # 미리 에이전트의 위치를 포함해서 계산을 해야만 함.
        # 꼼수 사용. 여기서 생성을 하되 그 정보를 바탕으로 이 안에서 경로 생성을 할 것. 
        
        # 모든 에이전트를 랜덤 위치에 스폰하는 것인데
        # 에이전트에는 이미 목표라는 속성이 같이 들어가 있단 말이지
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_spawning_x, self.world_spawning_x),
            (-self.world_spawning_y, self.world_spawning_y),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        ) 
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)
            
        # 어파치 1차원 그냥 리셋 시키길
        self.targets_cur_cost = torch.zeros(self.n_targets)

        # 타겟들 위치 할당        
        goal_poses = []
        for _ in self.targets:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)
            
        # 타겟들의 위치 분배
        # discovery에 보면 이런 식으로 위치를 할당함
        for i in range(len(self.targets)):
            self.targets[i].set_pos(goal_poses[i], batch_index=env_index)
            
        for i in range(len(self.finished_targets)):
            self.finished_targets[i].set_pos(self.finished_pos[i][0], batch_index=env_index)
            
        # goal_poses에 지금 위치까지 랜덤으로 정해진 모든 타겟의 정보가 저장되어 있음
        # 이 정보를 바탕으로 경로를 찾아서 이제 어... 진행을 해야 함
        
        #################################################################################################################################################
        # path finding
        # 무난하게 시나리오랑 같이 넣어둡시다
       
        self.agents_path = random_choice(self.world.agents, self.targets)
        # print(self.agents_path)
        
        
        #################################################################################################################################################

        # 본인에게 주어진 타겟에 대해서 본인과 동일한 색으로 칠해서 보기 좋게 업그레이드
        for i in range(len(self.world.agents)):
            for j in range(self.n_targets_per_agent):
                self.agents_path[i][j].color = Color.RED

        # 목표까지의 거리 계산?
        # 목표 까지의 거리를 계산하여 보상 시스템으 초기화
        # 처음 위치 기준으로 거리 계산을 해서 보상을 주는구나????
        # 매번 이전 스텝의 데이터를 가지고 가지 않아도 계산이 가능하게끔?
        # 그럼 이 기능은 스텝 과정 중에도 들어가야 함.
        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i

            # self.targets[i].color = agent.color                                                  # 이거는 통째로 설정을 하는 것이..?
            agent.goal = self.agents_path[i][0]
            # agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

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

        # 무서우니까 보상 체계는 건드리지 말고
        # 다음 단계로 넘어가기 위해 보상을 건드려야겠군
        if is_first:
            self.time_rew = torch.full(
                (self.world.batch_dim,),
                self.time_penalty,
                device=self.world.device,
            )
            
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

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
        
        # 마지막 에이전트를 확인할 때 타겟의 변경 등 추가 사항 점검 필요
        # 스텝 단계가 따로 없음. 여기서 해결 해야 함
        if is_last:
            for agent_idx, agent in enumerate(self.world.agents):
                # print(agent.on_goal)
                # print(agent.on_goal.item())
                if agent.on_goal.item():
                    # print(agent.on_goal.item())
                    if agent.goal != self.finished_targets[agent_idx]:
                        self.agents_path[agent_idx].pop(0)
                        if len(self.agents_path[agent_idx]) > 0:
                            agent.goal = self.agents_path[agent_idx][0]
                        else:
                            agent.goal = self.finished_targets[agent_idx]
                        # agent.goal.set_pos = agent.goal.state.pos
                        agent.on_goal = torch.tensor([False])
                        
                        agent.pos_shaping = (
                                torch.linalg.vector_norm(
                                    agent.state.pos - agent.goal.state.pos,
                                    dim=1,
                                )
                                * self.pos_shaping_factor
                            )
            
            # 타겟들의 비용 지불 여부 확인

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew + self.time_rew

    # 에이전트가 목표 근처에 있으면 보상
    # 거리 기반 보상. 가까이 갈 수록 큰 보상
    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        # 이렇게 하면 작동하지 않을까
        agent.stop_rew = 0
        if agent.distance_to_goal < self.covering_range:
            if agent.cur_cost < agent.goal.cost:
                agent.cur_cost += 1
                agent.stop_rew += 0.1
            else:
                agent.on_goal = torch.tensor([True])
                agent.cur_cost = 0
        else:
            agent.on_goal = torch.tensor([False])
        # agent.on_goal = agent.distance_to_goal < self.covering_range

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew + agent.stop_rew

    def observation(self, agent: Agent):
        goal_poses = []
        
        # observe_all_goals == False 상태
        # 모든 에이전트, 모든 타겟들 사이의 모든 거리 
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        # 에이전트와 본인의 목표 사이의 거리만
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
            
        # 에이전트 위치, 속도, 방향, 목표 타겟들, 에이전트 센서 값
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ]
            + goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1,
        )

    # 모든 타겟 탐색을 완료하면 종료하도록 조건 수정 필요
    def done(self):
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
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


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, clf_epsilon=0.2, clf_slack=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon  # Exponential CLF convergence rate
        self.clf_slack = clf_slack  # weights on CLF-QP slack variable

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """
        QP inputs:
        These values need to computed apriri based on observation before passing into QP

        V: Lyapunov function value
        lfV: Lie derivative of Lyapunov function
        lgV: Lie derivative of Lyapunov function
        CLF_slack: CLF constraint slack variable

        QP outputs:
        u: action
        CLF_slack: CLF constraint slack variable, 0 if CLF constraint is satisfied
        """
        # Install it with: pip install cvxpylayers
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        goal_pos = (-1.0) * (observation[:, 4:6] - agent_pos)

        # Pre-compute tensors for the CLF and CBF constraints,
        # Lyapunov Function from: https://arxiv.org/pdf/1903.03692.pdf

        # Laypunov function
        V_value = (
            (agent_pos[:, X] - goal_pos[:, X]) ** 2
            + 0.5 * (agent_pos[:, X] - goal_pos[:, X]) * agent_vel[:, X]
            + agent_vel[:, X] ** 2
            + (agent_pos[:, Y] - goal_pos[:, Y]) ** 2
            + 0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) * agent_vel[:, Y]
            + agent_vel[:, Y] ** 2
        )

        LfV_val = (2 * (agent_pos[:, X] - goal_pos[:, X]) + agent_vel[:, X]) * (
            agent_vel[:, X]
        ) + (2 * (agent_pos[:, Y] - goal_pos[:, Y]) + agent_vel[:, Y]) * (
            agent_vel[:, Y]
        )
        LgV_vals = torch.stack(
            [
                0.5 * (agent_pos[:, X] - goal_pos[:, X]) + 2 * agent_vel[:, X],
                0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) + 2 * agent_vel[:, Y],
            ],
            dim=1,
        )
        # Define Quadratic Program (QP) based controller
        u = cp.Variable(2)
        V_param = cp.Parameter(1)  # Lyapunov Function: V(x): x -> R, dim: (1,1)
        lfV_param = cp.Parameter(1)
        lgV_params = cp.Parameter(
            2
        )  # Lie derivative of Lyapunov Function, dim: (1, action_dim)
        clf_slack = cp.Variable(1)  # CLF constraint slack variable, dim: (1,1)

        constraints = []

        # QP Cost F = u^T @ u + clf_slack**2
        qp_objective = cp.Minimize(cp.sum_squares(u) + self.clf_slack * clf_slack**2)

        # control bounds between u_range
        constraints += [u <= u_range]
        constraints += [u >= -u_range]
        # CLF constraint
        constraints += [
            lfV_param + lgV_params @ u + self.clf_epsilon * V_param + clf_slack <= 0
        ]

        QP_problem = cp.Problem(qp_objective, constraints)

        # Initialize CVXPY layers
        QP_controller = CvxpyLayer(
            QP_problem,
            parameters=[V_param, lfV_param, lgV_params],
            variables=[u],
        )

        # Solve QP
        CVXpylayer_parameters = [
            V_value.unsqueeze(1),
            LfV_val.unsqueeze(1),
            LgV_vals,
        ]
        action = QP_controller(*CVXpylayer_parameters, solver_args={"max_iters": 500})[
            0
        ]

        return action


# if __name__ == "__main__":
#     render_interactively(
#         __file__,
#         control_two_agents=True,
#     )

# 타임 패널티 추가, 타겟 인지 범위 확장
# 타임 패널티가 없으니 하나만 움직이고 나머지는 멈춰있음
# 타겟 인지 범위는 너무 넓음. 좀 줄일 것. 현재 1.3배
# 라이다 세팅 원복. 타겟 인지 범위 1.2배
# 벡터화 개같이 어려움.
# 시간이 없으니까 이제는 포기하고 그냥 코스트를 추가해서 실험을 하자고