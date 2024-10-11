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
t_final = 2.8

time_passed = time.perf_counter() - start_time
temp = t_start * math.pow((t_final / t_start), time_passed)

MAX_ITER = 100000
# MAX_RAND_ITER = 100

random.seed(42)
DX = [0, 1, 0, -1, 0]
DY = [1, 0, -1, 0, 0]
DIR = ["R", "D", "L", "U", "."]

# read input
N, M, V = map(int, input().split())
s = [list(map(int, list(input()))) for _ in range(N)]
t = [list(map(int, list(input()))) for _ in range(N)]

len_tree = min(N, V) - 1

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
history_xs = []
history_ys = []

# 全てのタイムステップにおける葉の持っているかどうか
history_holdings = []

# 全てのタイムステップにおける盤面
history_s = [bitarray("0" * N * N) for _ in range(MAX_ITER)]
len_history_s = 0

# @measure_time
def rotate(center_x: int, center_y: int, now_x: int, now_y: int, rot: int) -> tuple:
    # 時計回りに90度回転
    if rot == 1:
        return center_x + center_y - now_y, center_y - center_x + now_x
    # 反時計回りに90度回転
    elif rot == 2:
        return center_x - center_y + now_y, center_y + center_x - now_x
    else:
        return now_x, now_y


# @measure_time
def search_target_is_reachable(center_x: int, center_y: int, x: int, y: int, bit_now:bitarray) -> int:
    d = -1
    for i in range(3):
        x, y = rotate(center_x, center_y, x, y, i)
        if (x >= 0) and (x < N) and (y >= 0) and (y < N) and (bit_now[x * N + y] == 0) and (bit_t[x * N + y] == 1):
            d = i
    return d


# @measure_time
def search_source_is_reachable(
    center_x: int, center_y: int, x: int, y: int, bit_s: list
) -> int:
    d = -1
    for i in range(3):
        x, y = rotate(center_x, center_y, x, y, i)
        if (x >= 0) and (x < N) and (y >= 0) and (y < N) and bit_s[x * N + y] == 1:
            d = i
    return d


# @measure_time
def init_random_tree():
    tree = [
        [random.randint(0, i), random.randint(1, min(N, V))]
        for i in range(min(N, V) - 1)
    ]
    return tree


# @measure_time
def init_tree():
    tree = [[0, i] for i in range(1, min(N, V))]
    return tree


# @measure_time
def init_random_rx_ry():
    rx = random.randint(0, N - 1)
    ry = random.randint(0, N - 1)
    return rx, ry


# @measure_time
def num_diff_list(bit_now: bitarray) -> int:
    return (~bit_now & bit_t).count()


# @measure_time
def eval_func(a: bitarray, num) -> int:
    return 1000 * num_diff_list(a) + num


# @measure_time
def calc_score_paths(
    bit_now: bitarray,
    paths: list,
) -> str:
    rx = 0
    ry = 0
    # init_tree()は [0,1] [0,2] [0,3] ... [0,V-1] のようなスター型の木を返す
    tree = init_tree()

    xs = [0 for _ in range(len(tree))]  # 全ての葉のx座標
    ys = [tree[i][1] for i in range(len(tree))]  # 全ての葉のy座標
    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか

    score = 0
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
        score += 1
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
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
        if bit_now == bit_t:
            break

    return eval_func(bit_now, turn)


# @measure_time
def update_history_rx_history_ry(index: int, paths: list) -> None:
    global history_rx, history_ry, history_xs, history_ys
    for i in range(index, len_history_s):
        if paths[i][0] == "R":
            if history_rx[i - 1] + 1 < N:
                history_rx[i] = history_rx[i - 1] + 1
                for j in range(len_tree):
                    history_xs[i][j] = history_xs[i - 1][j] + 1
            else:
                paths[i][0] = "."
                for j in range(len_tree):
                    history_xs[i][j] = history_xs[i - 1][j]
        elif paths[i][0] == "L":
            if history_rx[i - 1] - 1 >= 0:
                history_rx[i] = history_rx[i - 1] - 1
                for j in range(len_tree):
                    history_xs[i][j] = history_xs[i - 1][j] - 1
            else:
                paths[i][0] = "."
                for j in range(len_tree):
                    history_xs[i][j] = history_xs[i - 1][j]
        elif paths[i][0] == "D":
            if history_ry[i - 1] + 1 < N:
                history_ry[i] = history_ry[i - 1] + 1
                for j in range(len_tree):
                    history_ys[i][j] = history_ys[i - 1][j] + 1
            else:
                paths[i][0] = "."
                for j in range(len_tree):
                    history_ys[i][j] = history_ys[i - 1][j]
        elif paths[i][0] == "U":
            if history_ry[i - 1] - 1 >= 0:
                history_ry[i] = history_ry[i - 1] - 1
                for j in range(len_tree):
                    history_ys[i][j] = history_ys[i - 1][j] - 1
            else:
                paths[i][0] = "."
                for j in range(len_tree):
                    history_ys[i][j] = history_ys[i - 1][j]


# @measure_time
# def update_history_xs_history_ys_all(index: int, paths: list) -> None:
#     global history_xs, history_ys
#     for i in range(index, len_history_s):
#         for j in range(1, V):
#             if paths[i][j] == "R":
#                 history_xs[i][j] += 1
#             elif paths[i][j] == "L":
#                 history_xs[i][j] -= 1
#             elif paths[i][j] == "D":
#                 history_ys[i][j] += 1
#             elif paths[i][j] == "U":
#                 history_ys[i][j] -= 1


def update_history_xs_history_ys_target_node(index: int, paths: list, num: int) -> None:
    global history_xs, history_ys, history_rx, history_ry
    for i in range(index, len_history_s):
        if paths[i][num] == "R":
            history_xs[i][num], history_ys[i][num] = rotate(history_rx[i-1], history_ry[i-1], history_xs[i-1][num], history_ys[i-1][num], 1)
        elif paths[i][num] == "L":
            history_xs[i][num], history_ys[i][num] = rotate(history_rx[i-1], history_ry[i-1], history_xs[i-1][num], history_ys[i-1][num], 2)
        else:
            history_xs[i][num] = history_xs[i - 1][num]


# @measure_time
def update_history_s_and_holdings(index: int, num:int) -> None:
    global history_xs, history_ys, history_s, history_holdings, len_history_s, bit_t
    bit_now = history_s[index].copy()
    for i in range(index, len_history_s):
        x = history_xs[i][num]
        y = history_ys[i][num]
        if 0 <= x and x < N and 0 <= y and y < N:
            if (
                (bit_now[x * N + y] == 1)
                and (bit_t[x * N + y] == 0)
                and (not history_holdings[i])
            ):
                bit_now[i][x * N + y] = 0
                history_holdings[i][num] = True
            elif (
                (bit_now[x * N + y] == 0)
                and (bit_t[x * N + y] == 1)
                and (history_holdings[i][num])
            ):
                history_s[i][x * N + y] = 1
                history_holdings[i][num] = False
        history_s[i] = bit_now & all_ones
        if (bit_now == bit_t):
            len_history_s = i
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

    # if operation == "insert":
    #     new_action = [
    #         random.choice([".", "L", "R", "D", "U"])
    #         if i == 0
    #         else random.choice([".", "L", "R"])
    #         for i in range(len_tree)
    #     ]
    #     index = random.randint(0, len(paths))
    #     paths.insert(index, new_action)
    # elif operation == "delete":
    #     if len(paths) <= 1:
    #         return paths
    #     index = random.randint(0, len(paths) - 1)
    #     paths.pop(index)
    # else:
    # if len(paths) <= 1:
    # return paths
    index = random.randint(1, len_history_s)
    act_index = random.randint(1, len_tree - 1)
    old_act = paths[index][act_index]
    
    if act_index == 0:
        new_action = change_root_action(old_act, history_rx[index-1], history_ry[index-1])
        paths[index][act_index] = new_action
        update_history_rx_history_ry(index, paths)
    else:
        new_action = change_action(old_act)
        paths[index][act_index] = new_action
        update_history_xs_history_ys_target_node(index, paths, act_index)

    update_history_s_and_holdings(index, act_index)

    return paths


# @measure_time
def paths_yakinamasi(paths: list) -> list:
    best_score = eval_func(history_s[len_history_s - 1], len_history_s)
    score = best_score
    best_paths = paths.copy()
    while time.perf_counter() - start_time < t_final:
        new_paths = change_paths(paths.copy())
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
def choice_root_dir(rx, ry, bit_now, r_goal_num:int, d_goal_num:int, l_goal_num:int, u_goal_num:int, r_num:int, d_num:int, l_num:int, u_num:int, catch_num:int, not_goal_num:int) -> int:
    choices = [0, 1, 2, 3]
    return random.choice(choices)
    # print(f"r_goal_num:{r_goal_num}")
    # print(f"d_goal_num:{d_goal_num}")
    # print(f"l_goal_num:{l_goal_num}")
    # print(f"u_goal_num:{u_goal_num}")
    # print(f"r_num:{r_num}")
    # print(f"d_num:{d_num}")
    # print(f"l_num:{l_num}")
    # print(f"u_num:{u_num}")
    # print(f"catch_num:{catch_num}")
    # print(f"not_goal_num:{not_goal_num}")
    # if (catch_num == 0):
    # 没
    # if catch_num == not_goal_num:
    #     # goto goal
    #     if (catch_num == 1) and (not_goal_num == 1):
    #         return random.choice(choices)
    #     elif (r_goal_num != 0) or (d_goal_num != 0) or (l_goal_num != 0) or (u_goal_num != 0):
    #         # return random.choices(choices, weights=[r_goal_num, d_goal_num, l_goal_num, u_goal_num], k=1)[0]
    #         max_val = max(r_goal_num, d_goal_num, l_goal_num, u_goal_num)
    #         if max_val == r_goal_num:
    #             return 0
    #         elif max_val == d_goal_num:
    #             return 1
    #         elif max_val == l_goal_num:
    #             return 2
    #         else:
    #             return 3
    #     else:
    #         return random.choice(choices)
    # elif (catch_num < (len_tree//2)):
    #     # goto catch
    #     if (r_num != 0) or (d_num != 0) or (l_num != 0) or (u_num != 0):
    #         # return random.choices(choices, weights=[r_num, d_num, l_num, u_num], k=1)[0]
    #         max_val = max(r_num, d_num, l_num, u_num)
    #         if max_val == r_num:
    #             return 0
    #         elif max_val == d_num:
    #             return 1
    #         elif max_val == l_num:
    #             return 2
    #         else:
    #             return 3
    #     else:
    #         return random.choice(choices)
    # elif (catch_num > (len_tree//2)):
    #     # goto goal
    #     if (r_goal_num != 0) or (d_goal_num != 0) or (l_goal_num != 0) or (u_goal_num != 0):
    #         # return random.choices(choices, weights=[r_goal_num, d_goal_num, l_goal_num, u_goal_num], k=1)[0]
    #         max_val = max(r_goal_num, d_goal_num, l_goal_num, u_goal_num)
    #         if max_val == r_goal_num:
    #             return 0
    #         elif max_val == d_goal_num:
    #             return 1
    #         elif max_val == l_goal_num:
    #             return 2
    #         else:
    #             return 3
    #     else:
    #         return random.choice(choices)
    # else:
    #     # goto catch or goal
    #     if (r_goal_num + r_num != 0) or (d_goal_num + d_num != 0) or (l_goal_num + l_num != 0) or (u_goal_num + u_num != 0):
    #         return random.choices(choices, weights=[r_goal_num + r_num, d_goal_num + d_num, l_goal_num + l_num, u_goal_num + u_num], k=1)[0]
    #     else:
    #         return random.choice(choices)


def count_ones_left_goal(x:int) -> int:
    count = 0
    for row in range(N):
        for col in range(x):
            if (bit_t[row * N + col] == 1) and (bit_s[row * N + col] == 0):
                count += 1
    return count

def count_ones_right_goal(x: int) -> int:
    count = 0
    for row in range(N):
        for col in range(x + 1, N):
            if (bit_t[row * N + col] == 1) and (bit_s[row * N + col] == 0):
                count += 1
    return count

def count_ones_up_goal(y:int) -> int:
    count = 0
    for row in range(y):
        for col in range(N):
            if (bit_t[row * N + col] == 1) and (bit_s[row * N + col] == 0):
                count += 1
    return count

def count_ones_down_goal(y:int) -> int:
    count = 0
    for row in range(y + 1, N):
        for col in range(N):
            if (bit_t[row * N + col] == 1) and (bit_s[row * N + col] == 0):
                count += 1
    return count

def count_ones_left_num(x:int) -> int:
    count = 0
    for row in range(N):
        for col in range(x):
            if (bit_t[row * N + col] == 0) and (bit_s[row * N + col] == 1):
                count += 1
    return count

def count_ones_right_num(x: int) -> int:
    count = 0
    for row in range(N):
        for col in range(x + 1, N):
            if (bit_t[row * N + col] == 0) and (bit_s[row * N + col] == 1):
                count += 1
    return count

def count_ones_up_num(y:int) -> int:
    count = 0
    for row in range(y):
        for col in range(N):
            if (bit_t[row * N + col] == 0) and (bit_s[row * N + col] == 1):
                count += 1
    return count

def count_ones_down_num(y:int) -> int:
    count = 0
    for row in range(y + 1, N):
        for col in range(N):
            if (bit_t[row * N + col] == 0) and (bit_s[row * N + col] == 1):
                count += 1
    return count

def update_right(y, r_goal_num:int, l_goal_num:int, bit_now:bitarray) -> tuple:
    r_count = 0
    now_count = 0
    for row in range(N):
        if (bit_t[row *N + (y+1)] == 1) and (bit_now[row * N + (y+1)] == 0):
            r_count += 1
        if (bit_t[row * N + y] == 1) and (bit_now[row * N + y] == 0):
            now_count += 1
    r_goal_num -= r_count
    l_goal_num += now_count
    return r_goal_num, l_goal_num

def update_right_num(y, r_num:int, l_num:int, bit_now:bitarray) -> tuple:
    r_count = 0
    now_count = 0
    for row in range(N):
        if (bit_now[row * N + (y+1)] == 1) and (bit_t[row * N + (y+1)] == 0):
            r_count += 1
        if (bit_now[row * N + y] == 1) and (bit_t[row * N + y] == 0):
            now_count += 1
    # print(f"r_count:{r_count}")
    # print(f"now_count:{now_count}")
    r_num -= r_count
    l_num += now_count
    return r_num, l_num

def update_down_num(x, d_num:int, u_num:int, bit_now:bitarray) -> tuple:
    d_count = 0
    now_count = 0
    for col in range(N):
        if (bit_now[(x+1) * N + col] == 1) and (bit_t[(x+1) * N + col] == 0):
            d_count += 1
        if (bit_now[x * N + col] == 1) and (bit_t[x * N + col] == 0):
            now_count += 1
    d_num -= d_count
    u_num += now_count
    return d_num, u_num

def update_left_num(y, r_num:int, l_num:int, bit_now:bitarray) -> tuple:
    l_count = 0
    now_count = 0
    for row in range(N):
        if (bit_now[row *N + (y-1)] == 1) and (bit_t[row * N + (y-1)] == 0):
            l_count += 1
        if (bit_now[row * N + y] == 1) and (bit_t[row * N + y] == 0):
            now_count += 1
    l_num -= l_count
    r_num += now_count
    return r_num, l_num

def update_up_num(x, d_num:int, u_num:int, bit_now:bitarray) -> tuple:
    u_count = 0
    now_count = 0
    for col in range(N):
        if (bit_now[(x-1) *N + col] == 1) and (bit_t[(x-1) *N + col] == 0):
            u_count += 1
        if (bit_now[x * N + col] == 1) and (bit_t[x * N + col] == 0):
            now_count += 1
    u_num -= u_count
    d_num += now_count
    return d_num, u_num

def update_left(y, r_goal_num:int, l_goal_num:int, bit_now:bitarray) -> tuple:
    l_count = 0
    now_count = 0
    # print(f"y:{y}")
    for row in range(N):
        if (bit_t[row *N + (y-1)] == 1) and (bit_now[row * N + (y-1)] == 0):
            l_count += 1
        if (bit_t[row * N + y] == 1) and (bit_now[row * N + y] == 0):
            now_count += 1
    # print(f"bit_t:{bit_t}")
    # print(f"bit_now:{bit_now}")
    # print(f"l_count:{l_count}")
    l_goal_num -= l_count
    r_goal_num += now_count
    return r_goal_num, l_goal_num

def update_down(x, d_goal_num:int, u_goal_num:int, bit_now:bitarray) -> tuple:
    d_count = 0
    now_count = 0
    for col in range(N):
        if (bit_t[(x+1) *N + col] == 1) and (bit_now[(x+1) *N + col] == 0):
            d_count += 1
        if (bit_t[x * N + col] == 1) and (bit_now[x * N + col] == 0):
            now_count += 1
    # print(f"x:{x}")
    # print(f"d_count:{d_count}") 
    # print("no")
    d_goal_num -= d_count
    u_goal_num += now_count
    return d_goal_num, u_goal_num

def update_up(x, d_goal_num:int, u_goal_num:int, bit_now:bitarray) -> tuple:
    u_count = 0
    now_count = 0
    for col in range(N):
        if (bit_t[(x-1) * N + col] == 1) and (bit_now[(x-1) * N + col] == 0):
            u_count += 1
        if (bit_t[x * N + col] == 1) and (bit_now[x * N + col] == 0):
            now_count += 1
    u_goal_num -= u_count
    d_goal_num += now_count
    return d_goal_num, u_goal_num

def set_position(self, x, y):
    if 0 <= x < self.m and 0 <= y < self.n:
        self.current_x = x
        self.current_y = y
        self.update_counts()  # 位置を変更したらカウントを更新

# 目的地の場所を把握する
def init_goal_num(rx: int, ry: int) -> tuple:
    r_goal_num = 0
    d_goal_num = 0
    l_goal_num = 0
    u_goal_num = 0
    r_goal_num = count_ones_right_goal(ry)
    d_goal_num = count_ones_down_goal(rx)
    l_goal_num = count_ones_left_goal(ry)
    u_goal_num = count_ones_up_goal(rx)
    return r_goal_num, d_goal_num, l_goal_num, u_goal_num

def init_num(rx:int,ry:int) -> tuple:
    r_num = 0
    d_num = 0
    l_num = 0
    u_num = 0
    r_num = count_ones_right_num(ry)
    d_num = count_ones_down_num(rx)
    l_num = count_ones_left_num(ry)
    u_num = count_ones_up_num(rx)
    return r_num, d_num, l_num, u_num

# @measure_time
def create_init_paths(rx: int, ry: int, bit_now: bitarray) -> list:
    global \
        history_rx, \
        history_ry, \
        history_xs, \
        history_ys, \
        history_holdings, \
        history_s, \
        len_history_s
    # design the tree
    tree = init_tree()
    # decide the initial position
    rx, ry = 0, 0
    
    not_goal_num = num_diff_list(bit_now)
    catch_num = 0
    r_goal_num = 0
    d_goal_num = 0
    l_goal_num = 0
    u_goal_num = 0
    r_num = 0
    d_num = 0
    l_num = 0
    u_num = 0
    r_goal_num, d_goal_num, l_goal_num, u_goal_num = init_goal_num(rx, ry)
    r_num, d_num, l_num, u_num = init_num(rx, ry)

    xs = [0 for _ in range(len(tree))]  # 全ての葉のx座標
    ys = [tree[i][1] for i in range(len(tree))]  # 全ての葉のy座標
    holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか

    history_rx[0] = rx
    history_ry[0] = ry
    history_xs.append(xs.copy())
    history_ys.append(ys.copy())
    history_holdings.append(holdings.copy())
    history_s[0] |= bit_now

    paths = []

    for turn in range(MAX_ITER):
        # print(f"turn数:{turn}")
        path = []
        # rootは多少賢く
        # DX = [0, 1, 0, -1, 0]
        # DY = [1, 0, -1, 0, 0]
        # DIR = ["R", "D", "L", "U", "."]
        # dir = random.randint(0, 4)
        dir = choice_root_dir(rx, ry, bit_now, r_goal_num, d_goal_num, l_goal_num, u_goal_num, r_num, d_num, l_num, u_num, catch_num, not_goal_num)
        # print(f"rx: {rx}, ry: {ry}")
        # print(f"catch_num: {catch_num}, not_goal_num: {not_goal_num}")
        # print(f"r_goal_num: {r_goal_num}, d_goal_num: {d_goal_num}, l_goal_num: {l_goal_num}, u_goal_num: {u_goal_num}")
        # print(f"r_num: {r_num}, d_num: {d_num}, l_num: {l_num}, u_num: {u_num}")
        # print(f"dir: {dir}")
        dx, dy = DX[dir], DY[dir]
        if 0 <= rx + dx < N and 0 <= ry + dy < N:
            if dir == 0:
                r_goal_num, l_goal_num = update_right(ry, r_goal_num, l_goal_num, bit_now)
                r_num, l_num = update_right_num(ry, r_num, l_num, bit_now)
            elif dir == 1:
                d_goal_num, u_goal_num = update_down(rx, d_goal_num, u_goal_num, bit_now)
                d_num, u_num = update_down_num(rx, d_num, u_num, bit_now)
            elif dir == 2:
                r_goal_num, l_goal_num = update_left(ry, r_goal_num, l_goal_num, bit_now)
                r_num, l_num = update_left_num(ry, r_num, l_num, bit_now)
            elif dir == 3:
                d_goal_num, u_goal_num = update_up(rx, d_goal_num, u_goal_num, bit_now)
                d_num, u_num = update_up_num(rx, d_num, u_num, bit_now)
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
                rot = search_target_is_reachable(rx, ry, x, y, bit_now)
            else:
                rot = search_source_is_reachable(rx, ry, x, y, bit_now)
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
        # if turn == 11:
        #     print(f"rx, ry: {rx}, {ry}")
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            # if turn == 11:
            #     print(f"x, y: {x}, {y}")
            if 0 <= x and x < N and 0 <= y and y < N:
                if (
                    (bit_now[x * N + y] == 1)
                    and (bit_t[x * N + y] == 0)
                    and (not holdings[i])
                ):
                    bit_now[x * N + y] = 0
                    holdings[i] = True
                    catch_num += 1
                    if rx < x:
                        d_num -= 1
                    if rx > x:
                        u_num -= 1
                    if ry < y:
                        r_num -= 1
                    if ry > y:
                        l_num -= 1
                elif (
                    (bit_now[x * N + y] == 0)
                    and (bit_t[x * N + y] == 1)
                    and (holdings[i])
                ):
                    bit_now[x * N + y] = 1
                    holdings[i] = False
                    not_goal_num -= 1
                    catch_num -= 1
                    if rx < x:
                        d_goal_num -= 1
                    if rx > x:
                        # print(f"rx, ry: {rx}, {ry}")
                        # print(f"x, y: {x}, {y}")
                        # print("yes")
                        u_goal_num -= 1
                    if ry < y:
                        r_goal_num -= 1
                    if ry > y:
                        l_goal_num -= 1
        # output the command
        paths.append(path)
        history_rx[turn] = rx
        history_ry[turn] = ry
        history_xs.append(xs.copy())
        history_ys.append(ys.copy())
        history_holdings.append(holdings.copy())
        history_s[turn] |= bit_now
        if bit_now == bit_t:
            break
    len_history_s = turn
    return paths


# @measure_time
def make_output(bit_now: bitarray, paths: list) -> str:
    global len_history_s
    rx = 0
    ry = 0
    # init_tree()は [0,1] [0,2] [0,3] ... [0,V-1] のようなスター型の木を返す
    tree = init_tree()

    print(min(N, V))

    for i in range(len(tree)):
        print(f"{tree[i][0]} {tree[i][1]}")

    # rootの初期位置
    print(f"{0} {0}")

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
        # 全ての頂点を移動
        for i in range(len(tree)):
            xs[i] += dx
            ys[i] += dy
        # 全ての頂点において行動を決定
        # print(f"xs[0] ys[0]:{xs[0]} {ys[0]}")
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            if path[i+1] == ".":
                rot = 0
            elif path[i+1] == "L":
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
        # pick.append(".")  # rootノードは何もしない
        # rootも行動
        pick.append(".")
        for i in range(len(tree)):
            x = xs[i]
            y = ys[i]
            change = False
            # if i == 1:
            #     print(f"x y:{x} {y}")
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
        picks.append(pick)
        if (bit_now == bit_t):
            len_history_s = i
            for i in range(len(pick), len(tree)+1):
                pick.append(".")
            break
    for i in range(len(picks)):
        print(("".join(paths[i])) + ("".join(picks[i])))


init_paths = create_init_paths(rx=0, ry=0, bit_now=bit_s.copy())

# print("--init--")
# print(calc_score_paths(bit_now=bit_s.copy(), bit_t=bit_t.copy(), paths=init_paths))

paths = paths_yakinamasi(paths=init_paths)

# print("--yakinamasi--")
# print(eval_func(history_s[len_history_s - 1], len_history_s))

make_output(bit_now=bit_s.copy(), paths=paths.copy())
# make_output(bit_now=bit_s.copy(), paths=init_paths.copy())
