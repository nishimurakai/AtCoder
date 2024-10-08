import math
import random
import time

start_time = time.perf_counter()

t_start = 0.1
t_final = 2.9

time_passed = time.perf_counter() - start_time
temp = t_start * math.pow((t_final / t_start), time_passed)

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


def num_diff_list(s: list, t: list) -> int:
    return sum([1 for i in range(N) for j in range(N) if s[i][j] != t[i][j]])


def eval_func(s: list, t: list, turn: int) -> int:
    return 1000 * num_diff_list(s, t) + turn


def eval_func_senni(s: list, t: list, turn: int) -> int:
    return num_diff_list(s, t) + turn


def calc_score_path(rx: int, ry: int, s: list, t: list, paths: list) -> str:
    # init_tree()は [0,1] [0,2] [0,3] ... [0,V-1] のようなスター型の木を返す
    tree = init_tree()

    xs = [0 for _ in range(len(tree))]  # 全ての葉のx座標
    ys = [tree[i][1] for i in range(len(tree))]  # 全ての葉のy座標
    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか
    history_xs = [xs[:]]
    history_ys = [ys[:]]
    history_holdings = [holdings[:]]
    score = 0

    for path in paths:
        # 乱択
        if path[0] == ".":
            dx = 0
            dy = 0
        elif path[0] == "R":
            dx = 1
            dy = 0
        elif path[0] == "D":
            dx = 0
            dy = 1
        elif path[0] == "L":
            dx = -1
            dy = 0
        else:
            dx = 0
            dy = -1
        rx += dx
        ry += dy
        # 全ての頂点を移動
        for i in range(len(tree)):
            xs[i] += dx
            ys[i] += dy
        # 全ての頂点において行動を決定
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            if path[i] == ".":
                rot = 0
            elif path[i] == "L":
                rot = 1
            else:
                rot = 2
            # 回転の中心（今回は全てrootノードより，rx, ry）# 要変更
            rotate_center_x, rotate_center_y = rx, ry
            # grab or release takoyaki
            x, y = rotate(rotate_center_x, rotate_center_y, x, y, rot)
            xs[i] = x
            ys[i] = y

        history_xs.append(xs[:])
        history_ys.append(ys[:])

        # 行動の通りにpickをする
        score += 1
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            if 0 <= x and x < N and 0 <= y and y < N:
                if s[x][y] == 1 and t[x][y] == 0 and not holdings[i]:
                    s[x][y] = 0
                    holdings[i] = True
                elif s[x][y] == 0 and t[x][y] == 1 and holdings[i]:
                    s[x][y] = 1
                    holdings[i] = False
            history_holdings.append(holdings[:])
            if s == t:
                break

    return eval_func(s, t, score)


def calc_score_path_senni(rx: int, ry: int, s: list, t: list, paths: list) -> str:
    # init_tree()は [0,1] [0,2] [0,3] ... [0,V-1] のようなスター型の木を返す
    tree = init_tree()

    xs = [0 for _ in range(len(tree))]  # 全ての葉のx座標
    ys = [tree[i][1] for i in range(len(tree))]  # 全ての葉のy座標
    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか
    history_xs = [xs[:]]
    history_ys = [ys[:]]
    history_holdings = [holdings[:]]
    score = 0

    for path in paths:
        # 乱択
        if path[0] == ".":
            dx = 0
            dy = 0
        elif path[0] == "R":
            dx = 1
            dy = 0
        elif path[0] == "D":
            dx = 0
            dy = 1
        elif path[0] == "L":
            dx = -1
            dy = 0
        else:
            dx = 0
            dy = -1
        rx += dx
        ry += dy
        # 全ての頂点を移動
        for i in range(len(tree)):
            xs[i] += dx
            ys[i] += dy
        # 全ての頂点において行動を決定
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            if path[i] == ".":
                rot = 0
            elif path[i] == "L":
                rot = 1
            else:
                rot = 2
            # 回転の中心（今回は全てrootノードより，rx, ry）# 要変更
            rotate_center_x, rotate_center_y = rx, ry
            # grab or release takoyaki
            x, y = rotate(rotate_center_x, rotate_center_y, x, y, rot)
            xs[i] = x
            ys[i] = y

        history_xs.append(xs[:])
        history_ys.append(ys[:])

        # 行動の通りにpickをする
        score += 1
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            if 0 <= x and x < N and 0 <= y and y < N:
                if s[x][y] == 1 and t[x][y] == 0 and not holdings[i]:
                    s[x][y] = 0
                    holdings[i] = True
                elif s[x][y] == 0 and t[x][y] == 1 and holdings[i]:
                    s[x][y] = 1
                    holdings[i] = False
            history_holdings.append(holdings[:])
            if s == t:
                break

    return eval_func(s, t, score)


def change_paths(paths: list) -> list:
    operation = random.choice(["add", "delete", "modify"])

    if operation == "add":
        new_action = [
            random.choice([".", "L", "R", "D", "U"])
            if i == 0
            else random.choice([".", "L", "R"])
            for i in range(V)
        ]
        insert_index = random.randint(0, len(paths))
        paths.insert(insert_index, new_action)
    elif operation == "delete":
        if len(paths) <= 1:
            return paths
        delete_index = random.randint(0, len(paths) - 1)
        paths.pop(delete_index)
    else:
        if len(paths) <= 1:
            return paths
        modify_index = random.randint(0, len(paths) - 1)
        new_action = [
            random.choice([".", "L", "R", "D", "U"])
            if i == 0
            else random.choice([".", "L", "R"])
            for i in range(V)
        ]
        paths[modify_index] = new_action
    return paths


def paths_yakinamasi(rx: int, ry: int, s: list, t: list, paths: list) -> list:
    score = calc_score_path_senni(
        rx=0, ry=0, s=s.copy(), t=t.copy(), paths=paths.copy()
    )
    best_paths = paths.copy()
    while time.perf_counter() - start_time < t_final:
        new_paths = change_paths(paths.copy())
        new_score = calc_score_path_senni(rx, ry, s.copy(), t.copy(), new_paths)

        time_passed = time.perf_counter() - start_time
        temp = t_start * math.pow((t_final / t_start), time_passed)

        if new_score < score:
            best_paths = new_paths
            paths = new_paths
            score = new_score
        elif random.random() < math.exp((score - new_score) / temp):
            paths = new_paths
            score = new_score

    return best_paths


def create_init_paths(rx: int, ry: int, s: list, t: list) -> list:
    # design the tree
    tree = init_tree()
    # decide the initial position
    rx, ry = 0, 0

    xs = [0 for _ in range(len(tree))]  # 全ての葉のx座標
    ys = [tree[i][1] for i in range(len(tree))]  # 全ての葉のy座標
    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか

    paths = []

    for turn in range(MAX_ITER):
        path = []
        # 乱択
        dir = random.randint(0, 3)
        dx, dy = DX[dir], DY[dir]
        if 0 <= rx + dx < N and 0 <= ry + dy < N:
            rx += dx
            ry += dy
            path.append(DIR[dir])
            # 全ての頂点を移動
            for i in range(len(tree)):
                xs[i] += dx
                ys[i] += dy
        else:
            path.append(".")
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
                path.append(".")
            elif rot == 1:
                path.append("L")
            else:
                path.append("R")
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
            if 0 <= x and x < N and 0 <= y and y < N:
                if s[x][y] == 1 and t[x][y] == 0 and not holdings[i]:
                    s[x][y] = 0
                    holdings[i] = True
                elif s[x][y] == 0 and t[x][y] == 1 and holdings[i]:
                    s[x][y] = 1
                    holdings[i] = False
        # output the command
        paths.append(path)
        if s == t:
            break
    return paths


def make_output(rx: int, ry: int, s: list, t: list, paths: list) -> str:
    # init_tree()は [0,1] [0,2] [0,3] ... [0,V-1] のようなスター型の木を返す
    tree = init_tree()

    xs = [0 for _ in range(len(tree))]  # 全ての葉のx座標
    ys = [tree[i][1] for i in range(len(tree))]  # 全ての葉のy座標
    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか

    picks = []

    for path in paths:
        # 乱択
        if path[0] == ".":
            dx = 0
            dy = 0
        elif path[0] == "R":
            dx = 1
            dy = 0
        elif path[0] == "D":
            dx = 0
            dy = 1
        elif path[0] == "L":
            dx = -1
            dy = 0
        else:
            dx = 0
            dy = -1
        rx += dx
        ry += dy
        # 全ての頂点を移動
        for i in range(len(tree)):
            xs[i] += dx
            ys[i] += dy
        # 全ての頂点において行動を決定
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            if path[i] == ".":
                rot = 0
            elif path[i] == "L":
                rot = 1
            else:
                rot = 2
            # 回転の中心（今回は全てrootノードより，rx, ry）# 要変更
            rotate_center_x, rotate_center_y = rx, ry
            # grab or release takoyaki
            x, y = rotate(rotate_center_x, rotate_center_y, x, y, rot)
            xs[i] = x
            ys[i] = y

        # 行動の通りにpickをする
        pick = []
        pick.append(".")  # rootノードは何もしない
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            change = False
            if 0 <= x and x < N and 0 <= y and y < N:
                if s[x][y] == 1 and t[x][y] == 0 and not holdings[i]:
                    s[x][y] = 0
                    holdings[i] = True
                    change = True
                elif s[x][y] == 0 and t[x][y] == 1 and holdings[i]:
                    s[x][y] = 1
                    holdings[i] = False
                    change = True
            if change:
                pick.append("P")
            else:
                pick.append(".")
            if s == t:
                break
        picks.append(pick)
    for i in range(len(picks)):
        print(("".join(paths[i])) + ("".join(picks[i])))


# paths = create_init_paths(rx=0, ry=0, s=s.copy(), t=t.copy())
paths = []
paths.append(
    [
        random.choice([".", "L", "R", "D", "U"])
        if i == 0
        else random.choice([".", "L", "R"])
        for i in range(V)
    ]
)

paths = paths_yakinamasi(rx=0, ry=0, s=s.copy(), t=t.copy(), paths=paths.copy())

print(calc_score_path(rx=0, ry=0, s=s.copy(), t=t.copy(), paths=paths.copy()))

# make_output(rx=0, ry=0, s=s.copy(), t=t.copy(), paths=paths.copy())
