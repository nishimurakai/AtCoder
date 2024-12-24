import random
import time
import math
import copy

N, T, sigma = map(int, input().split())

wh = [tuple(map(int, input().split())) for _ in range(N)]

### テスト用

# wh_act = [tuple(map(int, input().split())) for _ in range(N)]

# def output_with_act(prdb):
#     print(len(prdb))
#     for p, r, d, b in prdb:
#         print(p, r, d, b)
#     W_err, H_err = map(int, input().split())
#     W, H, x_list, y_list = calc_act_W_H(prdb)
#     return W + W_err, H + H_err, x_list, y_list
# def calc_act_W_H(prdb):
#     W, H, w, h = 0, 0, 0, 0
#     x_list = [(-1, -1) for _ in range(N)]
#     y_list = [(-1, -1) for _ in range(N)]
#     for i in range(len(prdb)):
#         block_num = prdb[i][0]
#         base_block_num = prdb[i][3]
#         if prdb[i][1] == 0:
#             w = wh_act[block_num][0]
#             h = wh_act[block_num][1]
#         else:
#             h = wh_act[block_num][0]
#             w = wh_act[block_num][1]
#         if prdb[i][2] == 'U':
#             if prdb[i][3] == -1:
#                 x_list[block_num] = (0, w)
#                 y_upper = calc_y_upper(x_list, x_list[block_num], y_list)
#                 y_list[block_num] = (y_upper, y_upper + h)
#             else:
#                 x_list[block_num] = (x_list[base_block_num][1], x_list[base_block_num][1] + w)
#                 y_upper = calc_y_upper(x_list, x_list[block_num], y_list)
#                 y_list[block_num] = (y_upper, y_upper + h)
#         else:
#             if prdb[i][3] == -1:
#                 y_list[block_num] = (0, h)
#                 x_upper = calc_x_upper(y_list, y_list[block_num], x_list)
#                 x_list[block_num] = (x_upper, x_upper + w)
#             else:
#                 y_list[block_num] = (y_list[base_block_num][1], y_list[base_block_num][1] + h)
#                 x_upper = calc_x_upper(y_list, y_list[block_num], x_list)
#                 x_list[block_num] = (x_upper, x_upper + w)
#         W = max(W, x_list[block_num][1])
#         H = max(H, y_list[block_num][1])
#     return W, H, x_list, y_list

# import matplotlib.pyplot as plt

# def plot_shapes(x_list, y_list, filename):
#     fig, ax = plt.subplots()
#     for i in range(len(x_list)):
#         x0, x1 = x_list[i]
#         y0, y1 = y_list[i]
#         rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=None, edgecolor='r')
#         ax.add_patch(rect)
#     ax.set_xlim(0, max(x[1] for x in x_list))
#     ax.set_ylim(0, max(y[1] for y in y_list))
#     ax.invert_yaxis()
#     ax.xaxis.tick_top()
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.savefig(filename)

###　テスト用

rng = random.Random(1234)

start_time = time.perf_counter()
end_time = 2.0

t_start = time.perf_counter()
t_final = end_time / float(T)

def output(prdb):
    print(len(prdb))
    for p, r, d, b in prdb:
        print(p, r, d, b)
    W, H = map(int, input().split())
    return W, H

def change_prdb_range(prdb, yaki_range):
    new_prdb = copy.deepcopy(prdb)
    i = rng.randint(0, N - 1)
    if yaki_range:
        i = rng.choice(yaki_range)
    else:
        i = rng.randint(0, N - 1)
    new_prdb[i] = (
        i,
        rng.randint(0, 1),
        ['U', 'L'][rng.randint(0, 1)],
        rng.randint(-1, i - 1),
    )
    return new_prdb

def yakinamashi_range(prdb, yaki_range):
    best_W, best_H, _, _ = calc_W_H(prdb)
    best_score = best_W + best_H
    best_prdb = prdb
    score = best_score
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < t_final:
        new_prdb = change_prdb_range(prdb, yaki_range)
        new_W, new_H, _, _ = calc_W_H(new_prdb)
        new_score = new_W + new_H
        time_passed = time.perf_counter() - start_time
        temp = start_time * math.pow((end_time / start_time), time_passed)
        if new_score < best_score:
            best_prdb = copy.deepcopy(new_prdb)
            best_W, best_H = new_W, new_H
            best_score = new_score
            prdb = new_prdb
            score = new_score
        elif random.random() < math.exp((score - new_score) / temp):
            prdb = new_prdb
            score = new_score
    return best_prdb, best_W, best_H

def calc_x_tyouhuku(x_list, x) -> list[int]:
    x_tyouhuku = []
    for i in range(len(x_list)):
        if (x_list[i][0] < x[1]) and (x_list[i][1] > x[0]):
            x_tyouhuku.append(i)
    return x_tyouhuku

def calc_y_tyouhuku(y_list, y) -> list[int]:
    y_tyouhuku = []
    for i in range(len(y_list)):
        if (y_list[i][0] < y[1]) and (y_list[i][1] > y[0]):
            y_tyouhuku.append(i)
    return y_tyouhuku

def calc_y_upper(x_list, x, y_list):
    y_upper = 0
    x_tyouhuku = calc_x_tyouhuku(x_list, x)
    if len(x_tyouhuku) == 0:
        return 0
    for i in x_tyouhuku:
        y_upper = max(y_list[i][1], y_upper)
    return y_upper

def calc_x_upper(y_list, y, x_list):
    x_upper = 0
    y_tyouhuku = calc_y_tyouhuku(y_list, y)
    if len(y_tyouhuku) == 0:
        return 0
    for i in y_tyouhuku:
        x_upper = max(x_list[i][1], x_upper)
    return x_upper

# prdbから想定W, Hを計算する
def calc_W_H(prdb):
    W, H, w, h = 0, 0, 0, 0
    x_list = [(-1, -1) for _ in range(N)]
    y_list = [(-1, -1) for _ in range(N)]
    for i in range(len(prdb)):
        block_num = prdb[i][0]
        base_block_num = prdb[i][3]
        if prdb[i][1] == 0:
            w = wh[block_num][0]
            h = wh[block_num][1]
        else:
            h = wh[block_num][0]
            w = wh[block_num][1]
        if prdb[i][2] == 'U':
            if prdb[i][3] == -1:
                x_list[block_num] = (0, w)
                y_upper = calc_y_upper(x_list, x_list[block_num], y_list)
                y_list[block_num] = (y_upper, y_upper + h)
            else:
                x_list[block_num] = (x_list[base_block_num][1], x_list[base_block_num][1] + w)
                y_upper = calc_y_upper(x_list, x_list[block_num], y_list)
                y_list[block_num] = (y_upper, y_upper + h)
        else:
            if prdb[i][3] == -1:
                y_list[block_num] = (0, h)
                x_upper = calc_x_upper(y_list, y_list[block_num], x_list)
                x_list[block_num] = (x_upper, x_upper + w)
            else:
                y_list[block_num] = (y_list[base_block_num][1], y_list[base_block_num][1] + h)
                x_upper = calc_x_upper(y_list, y_list[block_num], x_list)
                x_list[block_num] = (x_upper, x_upper + w)
        W = max(W, x_list[block_num][1])
        H = max(H, y_list[block_num][1])
    return W, H, x_list, y_list



prdb = []
for i in range(N):
    prdb.append((
        i,
        rng.randint(0, 1),
        ['U', 'L'][rng.randint(0, 1)],
        rng.randint(-1, i - 1),
    ))

for step in range(T):
    yaki_range = range(step*(N//T), (step+1)*(N//T))
    prdb, best_W, best_H = yakinamashi_range(prdb, yaki_range)
    W_obs, H_obs = output(prdb)

    # _, _, x_list, y_list = calc_W_H(prdb)
    # W_obs, H_obs, x_list_act, y_list_act = output_with_act(prdb)
    # plot_shapes(x_list_act, y_list_act, f"./output/result_{step}.png")
    # plot_shapes(x_list, y_list, f"./output/result_{step}_best.png")
