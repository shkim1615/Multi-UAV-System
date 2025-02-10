#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
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

        self.shared_rew = kwargs.pop("shared_rew", True)
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
        # assert가 뭐임
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
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # 각 에이전트마다 별도의 목표 지점을 생성하고 연결
            # Landmark는 단순한 위치 목표로 충돌하지 않음
            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        # 보상 변수를 초기화하여 각 에이전트의 성과를 추적
        # pos_rew: 위치 기반 보상
        # final_rew: 최종 보상
        # 무슨 말이지 각각의 에이전트 개별로 보상을 축적한다는 말인가. 이거는 협동 게임이 아닌가
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    # env_index를 통해서 병렬 실행 중인 환경들 중에 특정 환경 선태 가능
    # None이 들어가면 모든 환경 초기화
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
        # 현재 에이전트가 첫 번째 에이전트인지 확인
        # 현재 받은 매개변수 에이전트가 첫 번째 에이전트인지 확인해서 나온 boolean 값을 넣음
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            # 각 에이전트의 위치 기반 보상 계산
            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0                # 충돌 상태 초기화

            # 모든 에이전트 목표 도달 확인
            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

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

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    # 에이전트의 위치 기반 보상
    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        # 목표 지점을 에이전트의 상대 좌표로 기록
        goal_poses = []
        if self.observe_all_goals:                                          # 모든 에이전트의 목표에 대한 상대 위치 관찰
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:                                                               # 해당 에이전트의 목표에 대한 상대 위치만 관찰
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
            
        # 관찰 데이터 결합
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

## 여기는 사용 안하는듯..?
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


if __name__ == "__main__":
    render_interactively(
        "cooperative_exploration",
        control_two_agents=True,
    )
