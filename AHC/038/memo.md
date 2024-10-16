# 考察
- 焼きなまし法
    - 制約を満たす範囲で遷移する
        - たこ焼きの持ち上げる持ち上げないを意識しない？（たこ焼きの有無が制約を満たすか満たさないかを決める）
        - そもそも持ち上げるとかを最後に回してスコアを取る？ <- 制約を満たさないで回すに等しいかも
        - 経路を修正した際，スコアが大きく損なわれるのはダメらしい
    - 新しい評価関数を作成して，焼きなます
        - 持ち上げるとかを最後に決定してスコアを出せばいけそう．

- モンテカルロ法
    - 1つ動作をした後に，評価を調べる．
        - 焼きなましと同じ評価方法を使おう

- 高速化したい

# 12: 標準的なSA
- 温度スケジュール: t = t_start * (t_final / t_start) ^ time_passed、time_passed は 0..1 の範囲

- 受理(最小値が良い場合): RNG() < exp((cur_result - new_result) / t)、RNG() は 0..1 を均一に返す

- 出発点としては、ほとんどの場合でこれが最も良いです。

- モンテカルロ法は行動が有限の時に有効かも <- 違うかも

- 経路変更に対して，高速に最後のsとpicksを求める
or
- 各葉ノードの座標のヒストリー，picksのヒストリーを保存する



# @measure_time
# def eval_initial_tree(tree: list, best_score: int = 100000) -> int:
#     _s = [bit_now[i][:] for i in range(N)]
#     _t = [bit_t[i][:] for i in range(N)]

#     # decide the initial position
#     rx, ry = 0, 0

#     xs = [0 for _ in range(len(tree))]  # 全ての葉のx座標
#     ys = [tree[i][1] for i in range(len(tree))]  # 全ての葉のy座標
#     holdings = [False for _ in range(len(tree))]  # たこ焼きを持っているかどうか

#     for turn in range(MAX_ITER):
#         # 乱択
#         d = random.randint(0, 4)
#         dx, dy = DX[d], DY[d]
#         if 0 <= rx + dx < N and 0 <= ry + dy < N:
#             rx += dx
#             ry += dy
#             # 全ての頂点を移動
#             for i in range(len(tree)):
#                 xs[i] += dx
#                 ys[i] += dy
#         # 全ての頂点において行動を決定
#         # 乱択
#         for i in range(len(tree)):
#             x = xs[i]
#             y = ys[i]
#             rot = -1
#             if holdings[i]:
#                 rot = search_target_is_reachable(rx, ry, x, y, _t)
#             if rot == -1 or not holdings[i]:
#                 rot = random.randint(0, 2)
#             # 回転の中心（今回は全てrootノードより，rx, ry）
#             rotate_center_x, rotate_center_y = rx, ry
#             # grab or release takoyaki
#             x, y = rotate(rotate_center_x, rotate_center_y, x, y, rot)
#             xs[i] = x
#             ys[i] = y

#         # 乱択
#         for i in range(len(tree)):
#             x = xs[i]
#             y = ys[i]
#             change = False
#             if 0 <= x and x < N and 0 <= y and y < N:
#                 if _s[x][y] == 1 and _t[x][y] == 0 and not holdings[i]:
#                     change = True
#                     _s[x][y] = 0
#                     holdings[i] = True
#                 elif _s[x][y] == 0 and _t[x][y] == 1 and holdings[i]:
#                     change = True
#                     _s[x][y] = 1
#                     holdings[i] = False
#         if _s == _t:
#             break
#         if turn > best_score:
#             break
#     return turn


# @measure_time
# def search_best_tree():
#     best_tree = []
#     best_score = 100000
#     for _ in range(MAX_RAND_ITER):
#         tree = init_random_tree()
#         score = eval_initial_tree(tree, best_score)
#         if score < best_score:
#             best_tree = tree
#             best_score = score
#     return best_tree

[RDLU]
ルートからの長さの順に
0が右, 1が下, 2が左, 3が上
xs[i] -> rx + DX[ds[i]]*(i+1)
ys[i] -> ry + DY[ds[i]]*(i+1)
ds = [0--3, 0--3, ...]
i番目の要素は，rx + DIR[ds[i]*(i+1)]
yも同様

DX = [0, 1, 0, -1, 0]
DY = [1, 0, -1, 0, 0]
DIR = ["R", "D", "L", "U", "."]
0, 1, 2, 3

左回転：
rx + DX[(ds + 3) % 4]
ry + DY[(ds + 3) % 4]
右回転
rx + DX[(ds + 1) % 4]
ry + DY[(ds + 1) % 4]
