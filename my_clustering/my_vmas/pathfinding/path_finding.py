from torch import tensor
import torch
from sklearn.cluster import KMeans

agents_data = [tensor([[-0.0075,  0.5364]]), tensor([[-0.8230, -0.7359]]), tensor([[-0.3852,  0.2682]]), tensor([[-0.0198,  0.7929]])]
targets_data = [tensor([[-0.0887,  0.2646]]), tensor([[-0.3022, -0.1966]]), tensor([[0.3953, 0.6000]]), tensor([[ 0.9055, -0.9277]])]

def trans_pos(agents_data, targets_data):
    agents_positions = [t.squeeze(0).numpy().tolist() for t in agents_data]
    targets_positions = [t.squeeze(0).numpy().tolist() for t in targets_data]
    return agents_positions, targets_positions

# print(trans_pos(agents_data, targets_data))

def random_choice(agents, targets):
    n_agents, n_targets = len(agents), len(targets)
    number= n_targets // n_agents
    # print(agents)
    # print(targets)
    # print(n_agents, n_targets)
    # print(number)
    
    # temp = [targets[i:i+number] for i in range(0, n_targets, number)]
    # print(temp)
    # temp[0].pop(0)
    # print(temp)
    
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