from torch import tensor
import numpy as np
import numpy as np
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment

# 동일한 갯수로 분배
def random_choice(agents, targets):
    n_agents, n_targets = len(agents), len(targets)
    number= n_targets // n_agents
    
    return [targets[i:i+number] for i in range(0, n_targets, number)]
 
 # 클러스터링
def numpy_pos(agents, targets, finished_position):
    agents_pos = []
    for agent in agents:
        x, y = agent.state.pos.squeeze().tolist()
        agents_pos.append([x, y])
    agents_pos = np.array(agents_pos)
    
    targets_pos = []
    targets_cost = []
    for target in targets:
        x, y = target.state.pos.squeeze().tolist()
        targets_pos.append([x, y])
        cost = target.cost.item()
        targets_cost.append(cost)
    targets_pos = np.array(targets_pos)
    targets_cost = np.array(targets_cost)

    finished_pos = []
    for pos in finished_position:
        x, y = pos.state.pos.squeeze().tolist()
        finished_pos.append([x, y])
    finished_pos = np.array(finished_pos)
    
    return agents_pos, targets_pos, targets_cost, finished_pos

def kmeans_clust(targets_pos, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(targets_pos)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    return labels, centers

# 이건 보류
def kmeans_cost_division_clust(targets_pos, targets_cost, n_clusters):
    # 타겟 증식
    for i, cost in enumerate(targets_cost):
        count = cost // 10 - 1
        for _ in range(count):
            targets_pos = np.concatenate((targets_pos, np.expand_dims(targets_pos[i], axis=0)), axis=0)
            
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(targets_pos)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    return labels, centers, targets_pos

def zip_clusters(agents, targets, labels):
    clusters = [[] for _ in range(len(agents))]
    for target, label in zip(targets, labels):
        clusters[label].append(target)
    return clusters

def closest_centers(agents_pos, centers, clusters):
    cost_matrix = np.linalg.norm(agents_pos[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)

    agent_indices, center_indices = linear_sum_assignment(cost_matrix)
    
    closest_clusters = [clusters[i] for i in center_indices]
    
    return agent_indices, center_indices, closest_clusters

# open tsp
def compute_euclidean_distance_matrix(locations):
    n = len(locations)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx, dy = locations[i] - locations[j]
                matrix[i][j] = int(np.hypot(dx, dy) * 1000)  # 소수점 줄이기 위해 스케일 업
    return matrix

def dist_matrix(closest_clusters, agents_pos, finished_pos):
    agents_goals = []
    for i in range(len(closest_clusters)):
        agent_goal = [agents_pos[i].tolist(), finished_pos[i].tolist()]
        for target in closest_clusters[i]:
            x, y = target.state.pos.squeeze().tolist()
            agent_goal.append([x, y])
        agents_goals.append(agent_goal)  

    distance_matrix = []
    for locations in agents_goals:
        locations = np.array(locations)
        matrix = compute_euclidean_distance_matrix(locations)
        matrix = matrix.tolist()
        distance_matrix.append(matrix)
    
    return distance_matrix

def create_data_model(distance_matrix):
    data = {}
    data["distance_matrix"] = distance_matrix
    data["num_vehicles"] = 1
    data["starts"] = [0]
    data["ends"] = [1]
    return data

def create_data_models(distance_matrices):
    data_matrix = []
    for distance_matrix in distance_matrices:
        data = create_data_model(distance_matrix)
        data_matrix.append(data)
    return data_matrix

def make_solutions(distance_matrices):
    """Entry point of the program."""
    # Instantiate the data problem.
    data_matrix = create_data_models(distance_matrices)
    
    solutions = []
    managers, routings = [], []

    for data in data_matrix:

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["starts"], data["ends"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # print(f"[DEBUG] from_node: {from_node}, to_node: {to_node}, matrix size: {len(data['distance_matrix'])}")
            if from_node >= len(data['distance_matrix']) or to_node >= len(data['distance_matrix']):
                print(f"[DEBUG] from_node: {from_node}, to_node: {to_node}, matrix size: {len(data['distance_matrix'])}")
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            10000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        managers.append(manager)
        routings.append(routing)
        solutions.append(solution)
        
    return data_matrix, managers, routings, solutions

def get_routes(data, manager, routing, solution):
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))

    return route

def kmeans_tsp(agents, targets, finished_position):
    agents_pos, targets_pos, targets_cost, finished_pos = numpy_pos(agents, targets, finished_position)
    n_clusters = len(agents)
    
    labels, centers = kmeans_clust(targets_pos, n_clusters)
    clusters = zip_clusters(agents, targets, labels)
    
    agent_indices, center_indices, closest_clusters = closest_centers(agents_pos, centers, clusters)
    
    distance_matrices = dist_matrix(closest_clusters, agents_pos, finished_pos)
    data_matrix = create_data_models(distance_matrices)
    
    data_matrix, managers, routings, solutions = make_solutions(distance_matrices)
    
    routes = []
    for i in range(len(solutions)):
        if solutions[i]:
            route = get_routes(data_matrix[i], managers[i], routings[i], solutions[i])
            routes.append(route[1:-1])
            
    agents_goal = []

    for cluster, route in zip(closest_clusters, routes):
        reordered = [cluster[i-2] for i in route]
        agents_goal.append(reordered)
        
    return agents_goal