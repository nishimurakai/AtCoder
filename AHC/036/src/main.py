import numpy as np
import pandas as pd
from collections import deque, defaultdict

N, M, L_T, L_A, L_B = map(int, input().split())

# 入力を格納するリスト
edges = []

for _ in range(M):

    # 入力をスペース区切りで分割
    u, v = map(int, input().split())
    
    # 分割された値をリストに追加
    edges.append([u, v])

T = list(map(int, input().split()))

# 座標入力を格納するリスト
x = []
y = []

for _ in range(N):

    # 入力をスペース区切りで分割
    x_, y_ = map(int, input().split())
    
    # 分割された値をリストに追加
    x.append(x_)
    y.append(y_)

# def bfs(start, graph, n):
#     distances = [-1] * n
#     distances[start] = 0
#     queue = deque([start])
    
#     while queue:
#         node = queue.popleft()
#         for neighbor in graph[node]:
#             if distances[neighbor] == -1:
#                 distances[neighbor] = distances[node] + 1
#                 queue.append(neighbor)
    
#     max_distance = max(distances)
#     farthest_node = distances.index(max_distance)
#     return farthest_node, max_distance, distances

# def find_min_diameter_tree(n, edges):
#     graph = defaultdict(list)
#     for u, v in edges:
#         graph[u].append(v)
#         graph[v].append(u)
    
#     # 任意の頂点から最遠の頂点を見つける
#     farthest_node, _, _ = bfs(0, graph, n)
    
#     # その最遠の頂点からさらに最遠の頂点を見つける
#     opposite_node, diameter, distances = bfs(farthest_node, graph, n)
    
#     # 最小直径の木の中心を見つける
#     path = []
#     current = opposite_node
#     while current != farthest_node:
#         path.append(current)
#         for neighbor in graph[current]:
#             if distances[neighbor] == distances[current] - 1:
#                 current = neighbor
#                 break
#     path.append(farthest_node)
    
#     center = path[len(path) // 2]
    
#     return center, diameter, graph

def bfs_path(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        current, path = queue.popleft()
        
        if current == goal:
            return path
        
        if current not in visited:
            visited.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    
    return None

# グラフを作成する
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

# パスを見つける
# path = bfs_path(graph, 0, 4)

# if path:
#     print(f"パス: {' -> '.join(map(str, path))}")
# else:
#     print("パスが見つかりませんでした。")

def missing_elements(list1, list2):
    # list1の各要素がlist2に存在するかを確認し、存在しない要素をリストに追加
    missing = [item for item in list1 if item not in list2]
    return missing

def split_list_into_chunks(list1, chunk_size=4):
    # list1をchunk_sizeのサイズで分割
    return [list1[i:i + chunk_size] for i in range(0, len(list1), chunk_size)]


# 初期設定
now = 0
A = []
while len(A) <= L_A:
    A.append(now)
    now = (now + 1) % N
B = [-1 for _ in range(L_B)]

now = 0
for i in range(L_T):
    
    # パスを見つける
    path = bfs_path(graph, now, T[i])

    chunks = split_list_into_chunks(path, L_B)
    print(chunks)
    for j in range(len(chunks)):
        B[j] = chunks[j][0]
    # for j in range(len(path) - 1):
    #     print(f"{path[j]} -> ", end="")
    # print(f"{T[i]}\n")

    now = T[i]

