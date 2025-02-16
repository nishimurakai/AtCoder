N, M = map(int, input().split())

edge_list: list[tuple[int, int]] = {}

ans: int = 0

for i in range(M):
    u, v = map(int, input().split())
    if (u, v) in edge_list or (v, u) in edge_list or u == v:
        ans += 1
    else:
        edge_list[(u, v)] = 1

print(ans)
