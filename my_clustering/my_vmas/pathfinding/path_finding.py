from torch import tensor
import numpy as np
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import itertools
import numpy as np

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
def targets_clustering(agents, targets, finished_targets, cost_weight=1.0):
    """
    타겟 12개의 위치와 코스트를 이용하여, 탐색 비용(거리 + 코스트 패널티)을 고려한 클러스터링을 수행합니다.
    
    탐색 비용은 두 타겟 i, j에 대해:
       distance(i,j) = ||pos_i - pos_j|| + ((cost_i + cost_j) / 2) * cost_weight
    로 정의됩니다.
    
    이 거리 행렬을 이용하여 AgglomerativeClustering (계층적 군집화)를 수행하고,
    결과를 4개의 클러스터(리스트의 리스트)로 반환합니다.
    
    agents, finished_targets는 이번 클러스터링에서는 사용하지 않음.
    
    Args:
        agents: 사용하지 않음 (함수 시그니처 유지용)
        targets: Target 객체들의 리스트. 각 객체는 target.state.pos (torch.Tensor, shape: (1,2))와 target.cost (스칼라)를 가짐.
        finished_targets: 사용하지 않음 (함수 시그니처 유지용)
        cost_weight (float): 코스트가 거리 비용에 미치는 영향력.
    
    Returns:
        list: 4개의 클러스터(리스트의 리스트). 각 클러스터는 할당된 Target 객체들을 포함.
    """
    n = len(targets)
    # 각 타겟의 위치와 코스트를 추출 (위치는 2차원 벡터, 코스트는 스칼라)
    positions = [target.state.pos.flatten().cpu().numpy() for target in targets]  # 수정: .cpu() 추가
    costs = [target.cost for target in targets]
    
    # n x n 크기의 거리(탐색 비용) 행렬 계산
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            euclidean = np.linalg.norm(positions[i] - positions[j])
            cost_penalty = ((costs[i] + costs[j]) / 2) * cost_weight
            d = euclidean + cost_penalty
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # AgglomerativeClustering은 precomputed metric을 사용 (scikit-learn 최신 버전 기준)
    clustering = AgglomerativeClustering(
        n_clusters=4, 
        metric="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(dist_matrix)
    
    # labels에 따라 타겟들을 클러스터별로 분류
    clusters = [[] for _ in range(4)]
    for target, label in zip(targets, labels):
        clusters[label].append(target)
    
    return clusters

def solve_open_tsp(points, start, finish):
    """
    주어진 points(2차원 좌표 리스트)에 대해, start에서 시작하여 finish에서 끝나는 open TSP 문제를 
    브루트포스로 해결합니다.
    
    각 경로의 비용은 아래와 같이 계산합니다:
       cost = distance(start, first_point) + 
              Σ(distance(point_i, point_{i+1})) + 
              distance(last_point, finish)
    
    Args:
        points (list of np.array): 타겟들의 2차원 좌표 리스트.
        start (np.array): 시작 좌표.
        finish (np.array): 종료 좌표.
    
    Returns:
        best_perm (tuple): points의 인덱스로 구성된 최적 순서.
        best_cost (float): 최적 순서에 따른 총 이동 비용.
        
        만약 points가 빈 리스트라면, start와 finish 사이의 거리를 반환합니다.
    """
    if not points:
        return (), np.linalg.norm(start - finish)
    
    best_cost = float('inf')
    best_perm = None
    for perm in itertools.permutations(range(len(points))):
        cost = np.linalg.norm(start - points[perm[0]])
        for i in range(len(perm)-1):
            cost += np.linalg.norm(points[perm[i]] - points[perm[i+1]])
        cost += np.linalg.norm(points[perm[-1]] - finish)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return best_perm, best_cost

def open_tsp_assignment(agents, clusters, finished_targets):
    """
    각 에이전트(시작, 도착 위치가 있음)와 클러스터(타겟 그룹)를 매칭시켜 open TSP 경로를 결정합니다.
    
    처리 과정:
      1. 4개의 클러스터(클러스터링 결과)는 순서(label)는 임의로 부여되어 있으므로,
         에이전트와 클러스터 간의 할당을 모든 경우(순열)로 고려합니다.
      2. 각 에이전트에 대해, 할당된 클러스터의 타겟들을 방문하는 최적의 순서를 open TSP 문제로 해결합니다.
         (시작: agent.state.pos, 종료: finished_targets 해당 에이전트의 위치)
      3. 모든 에이전트의 open TSP 비용의 합이 최소가 되는 할당 및 방문 순서를 선택합니다.
      4. 최종 반환값은 2차원 리스트로, 각 인덱스가 에이전트를 나타내며 내부 리스트에 해당 에이전트가 방문할 타겟 객체들이 순서대로 들어갑니다.
         그리고 최적 총 비용(best_total_cost)도 함께 반환합니다.
    
    Args:
        agents (list): Agent 객체 리스트. 각 객체는 agent.state.pos (torch.Tensor)를 가짐.
        clusters (list): 클러스터링 결과, 4개의 타겟 그룹 (각 그룹은 Target 객체 리스트)
        finished_targets (list): FinishedTarget 객체 리스트. 각 객체는 finished_target.state.pos (torch.Tensor)를 가짐.
    
    Returns:
        best_routes_2d (list of list): 각 에이전트별 방문 순서에 따른 타겟 객체들의 2차원 리스트.
                                       예) best_routes_2d[agent_idx] = [target_obj1, target_obj2, ...]
        best_total_cost (float): 모든 에이전트의 open TSP 비용 합계 (나중에 확인용으로 사용).
    """
    n_agents = len(agents)
    best_total_cost = float('inf')
    best_assignment = None
    best_routes = {}  # 임시 저장: {agent_idx: {'cluster_index': ..., 'route_order': ..., 'cost': ..., 'cluster_targets': ...}}
    
    # 모든 할당(에이전트와 클러스터의 순열)을 고려 (총 4! 가지)
    for perm in itertools.permutations(range(n_agents)):
        total_cost = 0.0
        routes = {}
        for agent_idx, cluster_idx in enumerate(perm):
            agent = agents[agent_idx]
            start = agent.state.pos.flatten().cpu().numpy()
            finish = finished_targets[agent_idx].state.pos.flatten().cpu().numpy()
            
            # 할당된 클러스터 내의 타겟들의 좌표 추출
            cluster = clusters[cluster_idx]
            points = [target.state.pos.flatten().cpu().numpy() for target in cluster]
            
            # open TSP 해결: start → cluster의 타겟 순회 → finish
            route_order, cost = solve_open_tsp(points, start, finish)
            total_cost += cost
            routes[agent_idx] = {
                'cluster_index': cluster_idx,
                'route_order': route_order,   # cluster 리스트 내에서의 순서 (인덱스 튜플)
                'cost': cost,
                'cluster_targets': cluster
            }
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_assignment = perm
            best_routes = routes
    
    # best_routes 딕셔너리를 이용해 2차원 리스트 형태로 각 에이전트별 방문 타겟 객체들을 재배열
    best_routes_2d = []
    for agent_idx in range(n_agents):
        route_info = best_routes[agent_idx]
        cluster_targets = route_info['cluster_targets']
        # route_order는 cluster_targets 내 인덱스 순서
        ordered_targets = [cluster_targets[i] for i in route_info['route_order']]
        best_routes_2d.append(ordered_targets)
    
    return best_routes_2d, best_total_cost