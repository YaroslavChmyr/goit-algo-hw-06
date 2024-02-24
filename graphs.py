import heapq
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Створення порожнього графа
G = nx.Graph()

# Додавання вершин (міста)
cities = ["Київ", "Харків", "Дніпро", "Одеса", "Львів", "Івано-Франківськ","Чернівці", "Варшава"]
G.add_nodes_from(cities)

# Додавання ребер (дороги) та їх ваги (відстані)
roads = [("Київ", "Харків", {'distance': 490}),
         ("Київ", "Дніпро", {'distance': 495}),
         ("Київ", "Львів", {'distance': 540}),
         ("Київ", "Одеса", {'distance': 480}),
         ("Харків", "Дніпро", {'distance': 220}),
         ("Дніпро", "Одеса", {'distance': 480}),
         ("Львів", "Івано-Франківськ", {'distance': 130}),
         ("Івано-Франківськ", "Чернівці", {'distance': 135}),
         ("Львів", "Варшава", {'distance': 430}),
         ]
G.add_edges_from(roads)

# Візуалізація графа
pos = nx.spring_layout(G)  # Визначення позицій вершин для графічного представлення
nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=9, font_weight="bold")
labels = nx.get_edge_attributes(G, 'distance')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Транспортна мережа")
plt.show()

# Аналіз основних характеристик графа
print("Кількість вершин:", G.number_of_nodes())
print("Кількість ребер:", G.number_of_edges())
print("Список вершин:", list(G.nodes()))
print("Список ребер:", list(G.edges()))
print("Ступінь вершин:", dict(G.degree()))


# Побудова маршруту за алгоритмом DFS
route_dfs = []

def dfs_recursive(graph, vertex, visited=None):
    if visited is None:
        visited = set()
    visited.add(vertex)
    route_dfs.append(vertex)
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    return route_dfs

print(f"Маршрут за алгоритмом DFS: {dfs_recursive(G, 'Варшава')}")


# Побудова маршруту за алгоритмом BFS
def bfs_iterative(graph, start):
    visited = set()
    queue = deque([start])
    route_bfs = []
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            route_bfs.append(vertex)
            visited.add(vertex)
            queue.extend(set(graph[vertex]) - visited)
    return route_bfs

print(f"Маршрут за алгоритмом BFS: {bfs_iterative(G, 'Варшава')}")


# Реалізуємо алгоритм Дейкстри для знаходження найкоротшого шляху
def dijkstra(graph, start):

    distances = {node: float('inf') for node in graph.nodes()}
    distances[start] = 0

    queue = [(0, start)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue
        for neighbor, edge_data in graph[current_node].items():
            distance = current_distance + edge_data['distance']
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances

# Знаходимо відстані до всіх вершин від початкової вершини 'Варшава' за алгоритмом Дейкстри
start_node = 'Варшава'
distances = dijkstra(G, start_node)

print(f"Відстань від Варшави до решти населених пунктів: {distances}")