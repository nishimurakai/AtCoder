import random

random.seed(42)
DX = [0, 1, 0, -1]
DY = [1, 0, -1, 0]
DIR = ['R', 'D', 'L', 'U']

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

# design the tree
tree = [[0, i] for i in range(1, min(N, V))]
print(len(tree)+1)
for p, L in tree:
    print(p, L)

# decide the initial position
rx, ry = 0, 0
print(rx, ry)

xs = [0 for _ in range(len(tree))] # 全ての葉のx座標
ys = [tree[i][1] for i in range(len(tree))] # 全ての葉のy座標
holdings = [False for _ in range(len(tree))] # たこ焼きを持っているかどうか

for turn in range(10000):
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
        S.append('.')
    # 全ての頂点において行動を決定
    # 乱択
    for i in range(len(tree)):
        x = xs[i]
        y = ys[i]
        rot = random.randint(0, 2)
        if rot == 0:
            S.append('.')
        elif rot == 1:
            S.append('L')
        else:
            S.append('R')
        # 回転の中心（今回は全てrootノードより，rx, ry）
        rotate_center_x, rotate_center_y = rx, ry
        # grab or release takoyaki
        x, y = rotate(rotate_center_x, rotate_center_y, x, y, rot)
        xs[i] = x
        ys[i] = y
    
    S.append('.') # vertex 0 (root) is not a leaf
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
            S.append('P')
        else:
            S.append('.')
    # output the command
    print(''.join(S))
    if s == t:
        break
