import random

MAX_ITER = 100000
MAX_RAND_ITER = 100

random.seed(42)
DX = [0, 1, 0, -1]
DY = [1, 0, -1, 0]
DIR = ["R", "D", "L", "U"]

# read input
N, M, V = map(int, input().split())
s = [list(map(int, list(input()))) for _ in range(N)]
t = [list(map(int, list(input()))) for _ in range(N)]


def rotate(center_x: int, center_y: int, now_x: int, now_y: int, rot: int) -> tuple:
    # 時計回りに90度回転
    if rot == 1:
        return center_x + center_y - now_y, center_y - center_x + now_x
    # 反時計回りに90度回転
    elif rot == 2:
        return center_x - center_y + now_y, center_y + center_x - now_x
    else:
        return now_x, now_y


def search_target_is_reachable(
    center_x: int, center_y: int, x: int, y: int, t: list
) -> int:
    d = -1
    for i in range(3):
        x, y = rotate(center_x, center_y, x, y, i)
        if (x >= 0) and (x < N) and (y >= 0) and (y < N) and t[x][y] == 1:
            d = i
    return d


def search_source_is_reachable(
    center_x: int, center_y: int, x: int, y: int, s: list
) -> int:
    d = -1
    for i in range(3):
        x, y = rotate(center_x, center_y, x, y, i)
        if (x >= 0) and (x < N) and (y >= 0) and (y < N) and s[x][y] == 1:
            d = i
    return d


def init_random_tree():
    tree = [
        [random.randint(0, i), random.randint(1, min(N, V))]
        for i in range(min(N, V) - 1)
    ]
    return tree


def init_tree():
    tree = [[0, i] for i in range(1, min(N, V))]
    return tree


def init_random_rx_ry():
    rx = random.randint(0, N - 1)
    ry = random.randint(0, N - 1)
    return rx, ry


def eval_initial_tree(tree: list, best_score: int = 100000) -> int:
    _s = [s[i][:] for i in range(N)]
    _t = [t[i][:] for i in range(N)]

    # decide the initial position
    rx, ry = 0, 0

    xs = [0 for _ in range(len(tree))]  # 全ての葉のx座標
    ys = [tree[i][1] for i in range(len(tree))]  # 全ての葉のy座標
    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか

    for turn in range(MAX_ITER):
        # 乱択
        d = random.randint(0, 3)
        dx, dy = DX[d], DY[d]
        if 0 <= rx + dx < N and 0 <= ry + dy < N:
            rx += dx
            ry += dy
            # 全ての頂点を移動
            for i in range(len(tree)):
                xs[i] += dx
                ys[i] += dy
        # 全ての頂点において行動を決定
        # 乱択
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            rot = -1
            if holdings[i]:
                rot = search_target_is_reachable(rx, ry, x, y, _t)
            if rot == -1 or not holdings[i]:
                rot = random.randint(0, 2)
            # 回転の中心（今回は全てrootノードより，rx, ry）
            rotate_center_x, rotate_center_y = rx, ry
            # grab or release takoyaki
            x, y = rotate(rotate_center_x, rotate_center_y, x, y, rot)
            xs[i] = x
            ys[i] = y

        # 乱択
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            change = False
            if 0 <= x and x < N and 0 <= y and y < N:
                if _s[x][y] == 1 and _t[x][y] == 0 and not holdings[i]:
                    change = True
                    _s[x][y] = 0
                    holdings[i] = True
                elif _s[x][y] == 0 and _t[x][y] == 1 and holdings[i]:
                    change = True
                    _s[x][y] = 1
                    holdings[i] = False
        if _s == _t:
            break
        if turn > best_score:
            break
    return turn


def search_best_tree():
    best_tree = []
    best_score = 100000
    for _ in range(MAX_RAND_ITER):
        tree = init_random_tree()
        score = eval_initial_tree(tree, best_score)
        if score < best_score:
            best_tree = tree
            best_score = score
    return best_tree


# def simulate(rx: int, ry: int, s: list, t: list, S: list) -> str:

# 最終的な答えを出力
# design the tree
tree = init_tree()
print(len(tree) + 1)
for p, L in tree:
    print(p, L)

# decide the initial position
rx, ry = 0, 0
print(rx, ry)

xs = [0 for _ in range(len(tree))]  # 全ての葉のx座標
ys = [tree[i][1] for i in range(len(tree))]  # 全ての葉のy座標
holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか

Ss = []

for turn in range(MAX_ITER):
    S = []
    # 乱択
    dir = random.randint(0, 3)
    dx, dy = DX[dir], DY[dir]
    if 0 <= rx + dx < N and 0 <= ry + dy < N:
        rx += dx
        ry += dy
        S.append(DIR[dir])
        # 全ての頂点を移動
        for i in range(len(tree)):
            xs[i] += dx
            ys[i] += dy
    else:
        S.append(".")
    # 全ての頂点において行動を決定
    # 乱択
    for i in range(len(tree)):
        x = xs[i]
        y = ys[i]
        rot = -1
        if holdings[i]:
            rot = search_target_is_reachable(rx, ry, x, y, t)
        else:
            rot = search_source_is_reachable(rx, ry, x, y, s)
        if rot == -1:
            rot = random.randint(0, 2)
        if rot == 0:
            S.append(".")
        elif rot == 1:
            S.append("L")
        else:
            S.append("R")
        # 回転の中心（今回は全てrootノードより，rx, ry）
        rotate_center_x, rotate_center_y = rx, ry
        # grab or release takoyaki
        x, y = rotate(rotate_center_x, rotate_center_y, x, y, rot)
        xs[i] = x
        ys[i] = y

    S.append(".")  # vertex 0 (root) is not a leaf
    # 乱択
    for i in range(len(tree)):
        x = xs[i]
        y = ys[i]
        change = False
        if 0 <= x and x < N and 0 <= y and y < N:
            if s[x][y] == 1 and t[x][y] == 0 and not holdings[i]:
                change = True
                s[x][y] = 0
                holdings[i] = True
            elif s[x][y] == 0 and t[x][y] == 1 and holdings[i]:
                change = True
                s[x][y] = 1
                holdings[i] = False
        if change:
            S.append("P")
        else:
            S.append(".")
    # output the command
    Ss.append(S)
    if s == t:
        break

for i in range(len(Ss)):
    print(("".join(Ss[i])))
