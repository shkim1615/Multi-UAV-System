from torch import tensor
import torch
from sklearn.cluster import KMeans
import numpy as np
from vmas.simulator.core import Landmark
from vmas.simulator.utils import Color

def print_data():
    agents_data = [tensor([[-0.0075,  0.5364]]), tensor([[-0.8230, -0.7359]]), tensor([[-0.3852,  0.2682]]), tensor([[-0.0198,  0.7929]])]
    # targets_data = [tensor([[-0.0887,  0.2646]]), tensor([[-0.3022, -0.1966]]), tensor([[0.3953, 0.6000]]), tensor([[ 0.9055, -0.9277]])]
    targets_data = [tensor([[-0.0887,  0.2646]]), tensor([[-0.3022, -0.1966]]), tensor([[0.3953, 0.6000]]), tensor([[ 0.9055, -0.9277]]), 
                    tensor([[-0.6058, -0.0803]]), tensor([[0.2823, 0.4182]]), tensor([[-0.5896,  0.6298]]), tensor([[0.6618, 0.7357]]), 
                    tensor([[-0.2421, -0.7303]]), tensor([[ 0.4424, -0.0980]]), tensor([[-0.2639,  0.2779]]), tensor([[ 0.6133, -0.0806]])]
    return agents_data, targets_data

def trans_pos(agents_data, targets_data):
    agents_positions = [t.squeeze(0).numpy().tolist() for t in agents_data]
    targets_positions = [t.squeeze(0).numpy().tolist() for t in targets_data]
    return agents_positions, targets_positions

# print(trans_pos(agents_data, targets_data))

def random_choice(agents, targets):
    n_agents, n_targets = len(agents), len(targets)
    number= n_targets // n_agents
    
    return [targets[i:i+number] for i in range(0, n_targets, number)]
    
# print(random_choice(agents_data, targets_data))

def targets_grouping(agents, targets):
    # agents: 리스트, 각 요소는 에이전트의 좌표를 나타내는 tensor (예: tensor([[x, y]]))
    # targets: 리스트, 각 요소는 타겟의 좌표를 나타내는 tensor (예: tensor([[x, y]]))
    
    n_agents = len(agents)
    n_targets = len(targets)
    # 에이전트당 타겟 개수 (여기서는 12개/4에이전트 = 3개)
    number = n_targets // n_agents

    # 타겟 좌표들을 하나의 텐서로 결합 후 numpy 배열로 변환
    target_coords = torch.cat(targets, dim=0).numpy()  # shape: (n_targets, 2)

    # KMeans를 사용해 n_agents=4개의 클러스터로 타겟들을 분류
    kmeans = KMeans(n_clusters=n_agents, random_state=0)
    kmeans.fit(target_coords)
    labels = kmeans.labels_  # 각 타겟에 대한 클러스터 레이블 (0~3)

    # 클러스터별로 타겟을 그룹화
    groups = {i: [] for i in range(n_agents)}
    for label, target in zip(labels, targets):
        groups[label].append(target)

    # 각 에이전트에 클러스터를 할당하기 위해, 에이전트와 클러스터 센트로이드 간의 거리를 계산
    # (여기서는 가장 가까운 센트로이드를 가진 에이전트에게 해당 클러스터를 할당)
    centroids = kmeans.cluster_centers_  # shape: (n_agents, 2)
    agent_assignments = {}
    for i, agent in enumerate(agents):
        # 에이전트 좌표 numpy 배열로 변환 (예: [x, y])
        agent_coord = agent.numpy().flatten()
        # 각 센트로이드와의 유클리드 거리 계산
        distances = np.linalg.norm(centroids - agent_coord, axis=1)
        assigned_cluster = np.argmin(distances)
        agent_assignments[i] = {
            'agent_coord': agent,
            'assigned_cluster': assigned_cluster,
            'targets': groups[assigned_cluster]
        }
    
    return agent_assignments

def targets_grouping_cost_ver(agents, targets, cost_weight=1.0):
    """
    agents: 각 요소가 tensor([[x, y]]) 형태의 에이전트 좌표
    targets: 각 요소가 tensor([[x, y]]) 형태의 타겟 좌표
    cost_weight: 코스트의 영향력을 조절하는 파라미터 (기본값 1.0)
    
    각 타겟에 대해 1~10 범위의 랜덤 코스트를 부여하고,
    타겟의 피처 벡터를 [x, y, cost * cost_weight]로 구성하여 클러스터링 수행.
    클러스터링 후, 각 에이전트는 자신과 가장 가까운 클러스터(좌표 기준)를 할당받습니다.
    """
    n_agents = len(agents)
    n_targets = len(targets)
    number = n_targets // n_agents  # 에이전트 당 할당해야 하는 타겟 개수 (예: 12/4 = 3)

    # 각 타겟에 대해 랜덤 코스트 생성 (1~10 사이)
    random_costs = np.random.randint(1, 11, size=n_targets)

    # 타겟 정보를 딕셔너리 형태로 구성 (좌표와 코스트)
    target_info = []
    for t, cost in zip(targets, random_costs):
        target_info.append({'coord': t, 'cost': cost})
    
    # 클러스터링을 위해 각 타겟의 피처 벡터 구성: [x, y, cost * cost_weight]
    target_coords = torch.cat(targets, dim=0).numpy()  # shape: (n_targets, 2)
    cost_feature = random_costs.reshape(-1, 1) * cost_weight  # shape: (n_targets, 1)
    features = np.hstack([target_coords, cost_feature])  # shape: (n_targets, 3)

    # KMeans 클러스터링 (클러스터 수 = 에이전트 수)
    kmeans = KMeans(n_clusters=n_agents, random_state=0)
    kmeans.fit(features)
    labels = kmeans.labels_  # 각 타겟에 대한 클러스터 레이블

    # 클러스터별로 타겟 정보를 그룹화 (코스트 정보 포함)
    groups = {i: [] for i in range(n_agents)}
    for label, info in zip(labels, target_info):
        groups[label].append(info)

    # 에이전트 할당: 에이전트의 좌표와 클러스터 센트로이드(여기서는 좌표만 사용) 간 거리를 계산
    centroids = kmeans.cluster_centers_[:, :2]  # 코스트 차원은 제외하고 2차원 좌표만 사용
    agent_assignments = {}
    for i, agent in enumerate(agents):
        agent_coord = agent.numpy().flatten()
        distances = np.linalg.norm(centroids - agent_coord, axis=1)
        assigned_cluster = np.argmin(distances)
        agent_assignments[i] = {
            'agent_coord': agent,
            'assigned_cluster': assigned_cluster,
            'targets': groups[assigned_cluster]  # 각 타겟은 {'coord': tensor, 'cost': int} 형태
        }

    return agent_assignments