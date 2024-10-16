import math
import random
import time
from collections import defaultdict
from functools import wraps

from bitarray import bitarray

# 実行時間を記録するための辞書
execution_times = defaultdict(list)


# 実行時間を測定するデコレータ
# def measure_time(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.perf_counter()
#         result = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         execution_time = end_time - start_time
#         execution_times[func.__name__].append(execution_time)
#         return result

#     return wrapper


# 実行時間の集計と表示
# def print_execution_times():
#     for func_name, times in execution_times.items():
#         total_time = sum(times)
#         average_time = total_time / len(times)
#         print(f"Function {func_name} executed {len(times)} times")
#         print(f"Total execution time: {total_time:.10f} seconds")
#         print(f"Average execution time: {average_time:.10f} seconds")
#         print()


start_time = time.perf_counter()

t_start = 0.1
t_final = 2.5

time_passed = time.perf_counter() - start_time
temp = t_start * math.pow((t_final / t_start), time_passed)

TOTAL_TURN = 1
MAX_ITER = 50000//TOTAL_TURN
# MAX_RAND_ITER = 100

random.seed(42)
DX = [0, 1, 0, -1, 0]
DY = [1, 0, -1, 0, 0]
DIR = ["R", "D", "L", "U", "."]

# read input
N, M, V = map(int, input().split())
s = [list(map(int, list(input()))) for _ in range(N)]
t = [list(map(int, list(input()))) for _ in range(N)]

# ROOT_N = math.floor(math.sqrt(N))
ROOT_N = 1

len_tree = V - 1

s_str = "".join(["".join(map(str, s[i])) for i in range(N)])
t_str = "".join(["".join(map(str, t[i])) for i in range(N)])

bit_s = bitarray(s_str)
bit_t = bitarray(t_str)

all_ones = bitarray(len(bit_s))
all_ones.setall(1)

# 全てのタイムステップにおける根の座標
history_rx = [0 for _ in range(MAX_ITER)]
history_ry = [0 for _ in range(MAX_ITER)]

# 全てのタイムステップにおける葉の座標
history_ds = []

# 全てのタイムステップにおける葉の持っているかどうか
history_holdings = []

# 全てのタイムステップにおける盤面
history_s = [bitarray("0" * N * N) for _ in range(MAX_ITER)]
len_history_s = 0

# @measure_time
def init_tree():
    tree = [[0, i%(ROOT_N)+1] for i in range(V-1)]
    return tree

# @measure_time
def num_diff_list(bit_now: bitarray) -> int:
    return (~bit_now & bit_t).count()


# @measure_time
def eval_func(a: bitarray, num) -> int:
    return 1000000 * num_diff_list(a) + num


# @measure_time
def calc_score_paths(
    bit_now: bitarray,
    paths: list,
) -> str:
    
    rx = 0
    ry = 0
    # init_tree()は [0,1] [0,2] [0,3] ... [0,V-1] のようなスター型の木を返す
    tree = init_tree()

    ds = [0 for _ in range(len(tree))]  # 全ての葉の回転方向
    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか
    turn = 0

    for path in paths:
        turn += 1
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
        # 全ての頂点において行動を決定
        for i in range(1, len(tree)):
            if path[i] == "R":
                ds[i-1] = (ds[i-1] + 1) % 4    
            else:
                ds[i] = (ds[i-1] + 3) % 4
            x = rx + DX[ds[i-1]] * (i-1+1)
            y = ry + DY[ds[i-1]] * (i-1+1)
            if 0 <= x and x < N and 0 <= y and y < N:
                if (
                    (bit_now[x * N + y] == 1)
                    and (bit_t[x * N + y] == 0)
                    and (not holdings[i-1])
                ):
                    bit_now[x * N + y] = 0
                    holdings[i-1] = True
                elif (
                    (bit_now[x * N + y] == 0)
                    and (bit_t[x * N + y] == 1)
                    and (holdings[i-1])
                ):
                    bit_now[x * N + y] = 1
                    holdings[i-1] = False
        if bit_now == bit_t:
            len_history_s = turn
            break

    return eval_func(bit_now, turn)


# @measure_time
def update_history_rx_history_ry(index: int, paths: list) -> None:
    global history_rx, history_ry
    for i in range(index, len_history_s):
        if paths[i][0] == "R":
            if history_ry[i - 1] + 1 < N:
                history_ry[i] = history_ry[i - 1] + 1
            else:
                paths[i][0] = "."
        elif paths[i][0] == "L":
            if history_ry[i - 1] - 1 >= 0:
                history_ry[i] = history_ry[i - 1] - 1
            else:
                paths[i][0] = "."
        elif paths[i][0] == "D":
            if history_rx[i - 1] + 1 < N:
                history_rx[i] = history_rx[i - 1] + 1
            else:
                paths[i][0] = "."
        elif paths[i][0] == "U":
            if history_rx[i - 1] - 1 >= 0:
                history_rx[i] = history_rx[i - 1] - 1
            else:
                paths[i][0] = "."
    # for i in range(index, len_history_s):
    #     if paths[i][num] == "R":
    #         history_xs[i][num] = history_rx[i] + DX[0] * (num+1)
    #         history_ys[i][num] = history
    #     elif paths[i][num] == "L":
    #         history_xs[i][num], history_ys[i][num] = rotate(history_rx[i-1], history_ry[i-1], history_xs[i-1][num], history_ys[i-1][num], 2)
    #     else:
    #         history_xs[i][num] = history_xs[i - 1][num]


# @measure_time
def update_history_s_and_holdings(paths:list, index: int, num:int) -> None:
    global history_xs, history_ys, history_s, history_holdings, len_history_s, bit_t
    bit_now = history_s[index].copy()
    x = 0
    y = 0
    for i in range(index, len_history_s):
        if paths[i][num] == "R":
            history_ds[i][num-1] = (history_ds[i-1][num-1] + 1) % 4
            x = history_rx[i] + DX[history_ds[i][num-1]] * (num)
            y = history_ry[i] + DY[history_ds[i][num-1]] * (num)
        elif paths[i][num] == "L":
            history_ds[i][num-1] = (history_ds[i-1][num-1] + 3) % 4
            x = history_rx[i] + DX[history_ds[i][num-1]] * (num)
            y = history_ry[i] + DY[history_ds[i][num-1]] * (num)
        if 0 <= x and x < N and 0 <= y and y < N:
            if (
                (bit_now[x * N + y] == 1)
                and (bit_t[x * N + y] == 0)
                and (not history_holdings[i][num-1])
            ):
                bit_now[x * N + y] = 0
                history_holdings[i][num-1] = True
            elif (
                (bit_now[x * N + y] == 0)
                and (bit_t[x * N + y] == 1)
                and (history_holdings[i][num-1])
            ):
                bit_now[x * N + y] = 1
                history_holdings[i][num-1] = False
        history_s[i] = bit_now & all_ones
        if (bit_now == bit_t):
            len_history_s = i+1
            break


# @measure_time
def change_action(old_act: str) -> str:
    if old_act == "R":
        return random.choice([".", "L"])
    elif old_act == "L":
        return random.choice([".", "R"])
    else:
        return random.choice(["R", "L"])


def change_root_action(old_act: str, x:int,y:int) -> int:
    if old_act == "R":
        if (x == 0) and (y == 0):
            return random.choice([".", "D"])
        elif (x == 0) and (y == N-1):
            return random.choice([".", "D", "L"])
        elif (x == N-1) and (y == 0):
            return random.choice([".", "U"])
        elif (x == N-1) and (y == N-1):
            return random.choice([".", "U", "L"])
        elif (x == 0):
            return random.choice([".", "D", "L"])
        elif (x == N-1):
            return random.choice([".", "U", "L"])
        elif (y == 0):
            return random.choice([".", "D", "U"])
        elif (y == N-1):
            return random.choice([".", "D", "U", "L"])
        else:
            return random.choice([".", "L", "D", "U"])
    elif old_act == "L":
        if (x == 0) and (y == 0):
            return random.choice([".", "D", "R"])
        elif (x == 0) and (y == N-1):
            return random.choice([".", "D"])
        elif (x == N-1) and (y == 0):
            return random.choice([".", "U", "R"])
        elif (x == N-1) and (y == N-1):
            return random.choice([".", "U"])
        elif (x == 0):
            return random.choice([".", "D", "R"])
        elif (x == N-1):
            return random.choice([".", "U", "R"])
        elif (y == 0):
            return random.choice([".", "D", "U", "R"])
        elif (y == N-1):
            return random.choice([".", "D", "U"])
        else:
            return random.choice([".", "R", "D", "U"])
    elif old_act == "D":
        if (x == 0) and (y == 0):
            return random.choice([".", "R"])
        elif (x == 0) and (y == N-1):
            return random.choice([".", "L"])
        elif (x == N-1) and (y == 0):
            return random.choice([".", "U", "R"])
        elif (x == N-1) and (y == N-1):
            return random.choice([".", "U", "L"])
        elif (x == 0):
            return random.choice([".", "R", "L"])
        elif (x == N-1):
            return random.choice([".", "U", "R", "L"])
        elif (y == 0):
            return random.choice([".", "R", "U"])
        elif (y == N-1):
            return random.choice([".", "L", "U"])
        else:
            return random.choice([".", "R", "L", "U"])
    # U
    elif old_act == "U":
        if (x == 0) and (y == 0):
            return random.choice([".", "D", "R"])
        elif (x == 0) and (y == N-1):
            return random.choice([".", "D", "L"])
        elif (x == N-1) and (y == 0):
            return random.choice([".", "R"])
        elif (x == N-1) and (y == N-1):
            return random.choice([".", "L"])
        elif (x == 0):
            return random.choice([".", "D", "R", "L"])
        elif (x == N-1):
            return random.choice([".", "R", "L"])
        elif (y == 0):
            return random.choice([".", "D", "R"])
        elif (y == N-1):
            return random.choice([".", "D", "L"])
        else:
            return random.choice([".", "R", "L", "D"])
    else:
        if (x == 0) and (y == 0):
            return random.choice(["D", "R"])
        elif (x == 0) and (y == N-1):
            return random.choice(["D", "L"])   
        elif (x == N-1) and (y == 0):
            return random.choice(["U", "R"])
        elif (x == N-1) and (y == N-1):
            return random.choice(["U", "L"])
        elif (x == 0):
            return random.choice(["D", "R", "L"])
        elif (x == N-1):
            return random.choice(["U", "R", "L"])
        elif (y == 0):
            return random.choice(["D", "R", "U"])
        elif (y == N-1):
            return random.choice(["D", "L", "U"])
        else:
            return random.choice(["R", "L", "D", "U"])

# @measure_time
def change_paths(paths: list) -> list:
    global history_rx, history_ry, history_xs, history_ys, history_holdings, history_s, len_history_s
    # operation = random.choice(["insert", "delete", "modify"])
    operation = random.choice(["add", "modify"])
    if (len_history_s <= 1):
        operation = "add"
    else:
        operation = random.choice(["add", "modify"])
    index = 0
    act_index = 0
    old_act = 0
    if operation == "add" or len_history_s == 1:
        new_action = ["." for _ in range(len_tree+1)]
        paths.append(new_action)
        new_rx = history_rx[-1]
        history_rx.append(new_rx)
        new_ry = history_ry[-1]
        history_ry.append(new_ry)
        new_ds = history_ds[-1]
        history_ds.append(new_ds)
        history_holdings.append(history_holdings[-1])
        history_s[len_history_s] = history_s[len_history_s-1] & all_ones
        len_history_s += 1
    # elif operation == "delete":
    #     if len(paths) <= 1:
    #         return paths
    #     index = random.randint(0, len_history_s)
    #     paths = paths[:index] + paths[index+1:]
    #     update_history_rx_history_ry(index, paths)
    #     for i in range(len_tree):
    #         update_history_s_and_holdings(paths, index, i+1)
    else:
        # print(f"len(paths):{len(paths)}")
        # print(f"len_history_s:{len_history_s}")
        index = random.randint(0, len_history_s)
        act_index = random.randint(1, len_tree - 1)
        old_act = paths[index][act_index]
        
        if act_index == 0:
            new_action = change_root_action(old_act, history_rx[index-1], history_ry[index-1])
            paths[index][act_index] = new_action
            update_history_rx_history_ry(index, paths)
        else:
            new_action = change_action(old_act)
            paths[index][act_index] = new_action
            history_ds[index][act_index-1] = (history_ds[index][act_index-1] + 1) % 4 if new_action == "R" else (history_ds[index][act_index-1] + 3) % 4
        update_history_s_and_holdings(paths, index, act_index)

    return paths, operation, index, act_index, old_act

# @measure_time
def paths_yakinamasi() -> list:
    global len_history_s
    paths = create_init_paths(rx=0,ry=0,bit_now=bit_s.copy())
    best_score = eval_func(history_s[len_history_s - 1], len_history_s)
    score = best_score
    best_paths = paths.copy()
    while time.perf_counter() - start_time < t_final:

        new_paths, operation, index, act_index, old_act = change_paths(paths.copy())
        new_score = eval_func(history_s[len_history_s - 1], len_history_s)

        time_passed = time.perf_counter() - start_time
        temp = t_start * math.pow((t_final / t_start), time_passed)

        if new_score < best_score:
            best_paths = new_paths.copy()
            best_score = new_score
            score = new_score
            paths = new_paths
        elif random.random() < math.exp((score - new_score) / temp):
            paths = new_paths.copy()
            score = new_score
        else:
            if operation == "add":
                history_rx.pop()
                history_ry.pop()
                history_ds.pop()
                history_holdings.pop()
                len_history_s -= 1
            else:
                if act_index == 0:
                    update_history_rx_history_ry(index, paths)
                    len_history_s -= 1
                else:
                    update_history_s_and_holdings(paths, index, act_index)
    return best_paths

def calc_dx(x:int, d:int, bit_now:bitarray) -> int:
    if (x == 0) and (d == -1):
        return 0
    elif d == -1:
        ret = bit_now[x * N:(x + 1) * N].count()
    elif (x == N-1) and (d == 1):
        return 0
    else:
        return random.choice([1, -1])
    
def calc_dy(y:int, d:int, bit_now:bitarray) -> int:
    if (y == 0) and (d == -1):
        return 0
    elif y == N-1:
        return -1
    else:
        return random.choice([1, -1])

# DX = [0, 1, 0, -1, 0]
# DY = [1, 0, -1, 0, 0]
# DIR = ["R", "D", "L", "U", "."]
def choice_root_dir():
    choices = [0, 1, 2, 3]
    return random.choice(choices)

def choice_rotate():
    choices = [0, 1]
    return random.choice(choices)

def set_position(self, x, y):
    if 0 <= x < self.m and 0 <= y < self.n:
        self.current_x = x
        self.current_y = y
        self.update_counts()  # 位置を変更したらカウントを更新

def create_init_history(paths: list) -> None:
    global history_rx, history_ry, history_holdings, history_s, len_history_s
    history_rx[0] = 0
    history_ry[0] = 0
    history_ds.append([0 for _ in range(len_tree)])
    history_holdings.append([False for _ in range(len_tree)])
    history_s[0] = bit_s.copy()
    len_history_s = 1
    if paths[0][0] == "R":
        history_rx[1] = 1
        history_ry[1] = history_ry[0]
    elif paths[0][0] == "L":
        history_rx[1] = -1
        history_ry[1] = history_ry[0]
    elif paths[0][0] == "D":
        history_ry[1] = 1
        history_rx[1] = history_rx[0]
    elif paths[0][0] == "U":
        history_ry[1] = -1
        history_rx[1] = history_rx[0]
    else:
        history_rx[1] = history_rx[0]
        history_ry[1] = history_ry[0]
    new_ds = [0 for _ in range(len_tree)]
    for i in range(len_tree):
        if paths[0][i+1] == "R":
            new_ds[i] = (history_ds[0][i] + 1) % 4
        elif paths[0][i+1] == "L":
            new_ds[i] = (history_ds[0][i] + 3) % 4
        else:
            new_ds[i] = history_ds[0][i]
    bit_now = history_s[0].copy()
    holdings = history_holdings[0].copy()
    for i in range(len_tree):
        x = history_rx[1] + DX[new_ds[i]] * (i%(ROOT_N)+1)
        y = history_ry[1] + DY[new_ds[i]] * (i%(ROOT_N)+1)
        if 0 <= x and x < N and 0 <= y and y < N:
            if (
                (bit_now[x * N + y] == 1)
                and (bit_t[x * N + y] == 0)
                and (not holdings[i])
            ):
                bit_now[x * N + y] = 0
                holdings[i] = True
            elif (
                (bit_now[x * N + y] == 0)
                and (bit_t[x * N + y] == 1)
                and (holdings[i])
            ):
                bit_now[x * N + y] = 1
                holdings[i] = False
    history_ds.append(new_ds)
    history_holdings.append(holdings)
    history_s[1] = bit_now & all_ones
    len_history_s += 1

# @measure_time
# def create_init_paths() -> list:
#     paths = []
#     new_action = [
#         random.choice(["R", "D", "."])
#         if i == 0
#         else random.choice([".", "L", "R"])
#         for i in range(len_tree+1)
#     ]
#     paths.append(new_action)
#     create_init_history(paths)
#     # print(f"history_rx:{history_rx[1]}")
#     # print(f"history_ry:{history_ry[1]}")
#     # print(f"history_ds:{history_ds}")
#     # print(f"history_holdings:{history_holdings}")
#     # print(f"history_s:{history_s}")
#     # print(f"len_history_s:{len_history_s}")
#     return paths

# @measure_time
def create_init_paths(rx: int, ry: int, bit_now: bitarray) -> list:
    global \
        history_rx, \
        history_ry, \
        history_holdings, \
        history_s, \
        len_history_s
    # design the tree
    tree = init_tree()

    ds = [0 for _ in range(len(tree))]  # 全ての葉の回転方向

    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか

    history_rx[0] = rx
    history_ry[0] = ry
    history_ds.append(ds.copy())
    history_holdings.append(holdings.copy())
    history_s[0] |= bit_now
    len_history_s = 1

    paths = []

    for turn in range(MAX_ITER):
        path = []
        # DX = [0, 1, 0, -1, 0]
        # DY = [1, 0, -1, 0, 0]
        # DIR = ["R", "D", "L", "U", "."]
        # dir = random.randint(0, 4)
        dir = choice_root_dir()
        dx, dy = DX[dir], DY[dir]
        if 0 <= rx + dx < N and 0 <= ry + dy < N:
            rx += dx
            ry += dy
            path.append(DIR[dir])
        else:
            path.append(".")
        # 全ての頂点において行動を決定
        # 乱択

        for i in range(len(tree)):
            rot = random.randint(0, 2)
            if rot == 0:
                path.append(".")
            elif rot == 1:
                path.append("R")
                ds[i] = (ds[i] + 1) % 4
            else:
                path.append("L")
                ds[i] = (ds[i] + 3) % 4
            x = rx + DX[ds[i]] * (i%(ROOT_N)+1)
            y = ry + DY[ds[i]] * (i%(ROOT_N)+1)
            if 0 <= x and x < N and 0 <= y and y < N:
                if (
                    (bit_now[x * N + y] == 1)
                    and (bit_t[x * N + y] == 0)
                    and (not holdings[i])
                ):
                    bit_now[x * N + y] = 0
                    holdings[i] = True
                elif (
                    (bit_now[x * N + y] == 0)
                    and (bit_t[x * N + y] == 1)
                    and (holdings[i])
                ):
                    bit_now[x * N + y] = 1
                    holdings[i] = False
        # output the command
        paths.append(path)
        history_rx[turn] = rx
        history_ry[turn] = ry
        history_ds.append(ds.copy())
        history_holdings.append(holdings.copy())
        history_s[turn] |= bit_now
        # if bit_now == bit_t:
        #     break
    len_history_s = turn
    return paths

# @measure_time
def make_output(rx:int, ry:int,bit_now: bitarray, paths: list) -> str:
    global len_history_s
    
    # init_tree()は [0,1] [0,2] [0,3] ... [0,V-1] のようなスター型の木を返す
    tree = init_tree()

    print(V)

    for i in range(len(tree)):
        print(f"{tree[i][0]} {tree[i][1]}")

    # rootの初期位置
    print(f"{rx} {ry}")


    ds = [0 for _ in range(len(tree))]  # 全ての葉の回転方向
    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか

    picks = []

    turn = 0

    for path in paths:
        turn += 1
        # print(f"turn:{turn}")
        # print(f"path:{path}")
        # 乱択
        if path[0] == ".":
            dx = 0
            dy = 0
        elif path[0] == "R":
            dx = 0
            dy = 1
        elif path[0] == "D":
            dx = 1
            dy = 0
        elif path[0] == "L":
            dx = 0
            dy = -1
        else:
            dx = -1
            dy = 0
        rx += dx
        ry += dy
        pick = []
        pick.append(".")
        # 全ての頂点において行動を決定
        # print(f"len_tree:{len_tree}")
        # print(f"len(path):{len(path)}")
        for i in range(len_tree):
            if path[i+1] == "R":
                ds[i] = (ds[i] + 1) % 4
            elif path[i+1] == "L":
                ds[i] = (ds[i] + 3) % 4
            
            x = rx + DX[ds[i]] * (i%(ROOT_N)+1)
            y = ry + DY[ds[i]] * (i%(ROOT_N)+1)
            change = False
            if 0 <= x and x < N and 0 <= y and y < N:
                if (
                    (bit_now[x * N + y] == 1)
                    and (bit_t[x * N + y] == 0)
                    and (not holdings[i])
                ):
                    bit_now[x * N + y] = 0
                    holdings[i] = True
                    change = True
                elif (
                    (bit_now[x * N + y] == 0)
                    and (bit_t[x * N + y] == 1)
                    and (holdings[i])
                ):
                    bit_now[x * N + y] = 1
                    holdings[i] = False
                    change = True
            if change:
                pick.append("P")
            else:
                pick.append(".")
            # print(f"i:{i+1}")
            # print(f"x, y:{x}, {y}")
        picks.append(pick)
        # for i in range(N):
        #     print(bit_now[i * N:(i + 1) * N].to01())
        # print()
        if (bit_now == bit_t):
            len_history_s = i
            for i in range(len(pick), len(tree)+1):
                pick.append(".")
            break
    for i in range(len(picks)):
        print(("".join(paths[i])) + ("".join(picks[i])))

# print("--init--")

best_score = 10000000000
best_rx = 0
best_ry = 0

for _ in range(TOTAL_TURN):
    rx = random.randint(0, N-1)
    ry = random.randint(0, N-1)
    paths = create_init_paths(rx=rx,ry=ry,bit_now=bit_s.copy())
    score = eval_func(history_s[len_history_s - 1], len_history_s)
    if best_score > score:
        best_score = score
        best_paths = paths.copy()
        best_rx = rx
        best_ry = ry

# paths = create_init_paths(0, 0, bit_s.copy())

# paths = paths_yakinamasi()

# print("--yakinamasi--")
# print(eval_func(history_s[len_history_s - 1], len_history_s))

make_output(rx = best_rx, ry = best_ry, bit_now=bit_s.copy(), paths=best_paths.copy())
# make_output(bit_now=bit_s.copy(), paths=paths.copy())
# make_output(bit_now=bit_s.copy(), paths=init_paths.copy())
