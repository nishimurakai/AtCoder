import sys
from collections import deque

Pos = tuple[int, int]
EMPTY = -1
DO_NOTHING = -1
STATION = 0
RAIL_HORIZONTAL = 1
RAIL_VERTICAL = 2
RAIL_LEFT_DOWN = 3
RAIL_LEFT_UP = 4
RAIL_RIGHT_UP = 5
RAIL_RIGHT_DOWN = 6
COST_STATION = 5000
COST_RAIL = 100

dist_list = [
    (-2, 0),
    (-1, 0),
    (-1, -1),
    (-1, 1),
    (0, -1),
    (0, 1),
    (0, -2),
    (0, 2),
    (1, 0),
    (1, -1),
    (1, 1),
    (2, 0),
]

# {(dr_prev, dr_next)): RailType}
rail_type_dict = {
    ((-1, 0), (1, 0)): RAIL_VERTICAL,
    ((1, 0), (-1, 0)): RAIL_VERTICAL,
    ((0, -1), (0, 1)): RAIL_HORIZONTAL,
    ((0, 1), (0, -1)): RAIL_HORIZONTAL,
    ((-1, 0), (0, 1)): RAIL_RIGHT_UP,
    ((0, 1), (-1, 0)): RAIL_RIGHT_UP,
    ((-1, 0), (0, -1)): RAIL_LEFT_UP,
    ((0, -1), (-1, 0)): RAIL_LEFT_UP,
    ((1, 0), (0, -1)): RAIL_LEFT_DOWN,
    ((0, -1), (1, 0)): RAIL_LEFT_DOWN,
    ((1, 0), (0, 1)): RAIL_RIGHT_DOWN,
    ((0, 1), (1, 0)): RAIL_RIGHT_DOWN,
}


class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.parents = [-1 for _ in range(n * n)]

    def _find_root(self, idx: int) -> int:
        if self.parents[idx] < 0:
            return idx
        self.parents[idx] = self._find_root(self.parents[idx])
        return self.parents[idx]

    def is_same(self, p: Pos, q: Pos) -> bool:
        p_idx = p[0] * self.n + p[1]
        q_idx = q[0] * self.n + q[1]
        return self._find_root(p_idx) == self._find_root(q_idx)

    def unite(self, p: Pos, q: Pos) -> None:
        p_idx = p[0] * self.n + p[1]
        q_idx = q[0] * self.n + q[1]
        p_root = self._find_root(p_idx)
        q_root = self._find_root(q_idx)

        if p_root != q_root:
            p_size = -self.parents[p_root]
            q_size = -self.parents[q_root]
            if p_size > q_size:
                p_root, q_root = q_root, p_root
            self.parents[q_root] += self.parents[p_root]
            self.parents[p_root] = q_root


def distance(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Action:
    def __init__(self, type: int, pos: Pos):
        self.type = type
        self.pos = pos

    def __str__(self):
        if self.type == DO_NOTHING:
            return "-1"
        else:
            return f"{self.type} {self.pos[0]} {self.pos[1]}"


class Result:
    def __init__(self, actions: list[Action], score: int):
        self.actions = actions
        self.score = score

    def __str__(self):
        return "\n".join(map(str, self.actions))


class Field:
    def __init__(self, N: int):
        self.N = N
        self.rail = [[EMPTY] * N for _ in range(N)]
        self.uf = UnionFind(N)

    def build(self, type: int, r: int, c: int) -> None:
        assert self.rail[r][c] != STATION
        if 1 <= type <= 6:
            assert self.rail[r][c] == EMPTY
        self.rail[r][c] = type

        # 隣接する区画と接続
        # 上
        if type in (STATION, RAIL_VERTICAL, RAIL_LEFT_UP, RAIL_RIGHT_UP):
            if r > 0 and self.rail[r - 1][c] in (
                STATION,
                RAIL_VERTICAL,
                RAIL_LEFT_DOWN,
                RAIL_RIGHT_DOWN,
            ):
                self.uf.unite((r, c), (r - 1, c))
        # 下
        if type in (STATION, RAIL_VERTICAL, RAIL_LEFT_DOWN, RAIL_RIGHT_DOWN):
            if r < self.N - 1 and self.rail[r + 1][c] in (
                STATION,
                RAIL_VERTICAL,
                RAIL_LEFT_UP,
                RAIL_RIGHT_UP,
            ):
                self.uf.unite((r, c), (r + 1, c))
        # 左
        if type in (STATION, RAIL_HORIZONTAL, RAIL_LEFT_DOWN, RAIL_LEFT_UP):
            if c > 0 and self.rail[r][c - 1] in (
                STATION,
                RAIL_HORIZONTAL,
                RAIL_RIGHT_DOWN,
                RAIL_RIGHT_UP,
            ):
                self.uf.unite((r, c), (r, c - 1))
        # 右
        if type in (STATION, RAIL_HORIZONTAL, RAIL_RIGHT_DOWN, RAIL_RIGHT_UP):
            if c < self.N - 1 and self.rail[r][c + 1] in (
                STATION,
                RAIL_HORIZONTAL,
                RAIL_LEFT_DOWN,
                RAIL_LEFT_UP,
            ):
                self.uf.unite((r, c), (r, c + 1))

    def is_connected(self, s: Pos, t: Pos) -> bool:
        assert distance(s, t) > 4  # 前提条件
        stations0 = self.collect_stations(s)
        stations1 = self.collect_stations(t)
        for station0 in stations0:
            for station1 in stations1:
                if self.uf.is_same(station0, station1):
                    return True
        return False

    def collect_stations(self, pos: Pos) -> list[Pos]:
        stations = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) > 2:
                    continue
                r = pos[0] + dr
                c = pos[1] + dc
                if 0 <= r < self.N and 0 <= c < self.N and self.rail[r][c] == STATION:
                    stations.append((r, c))
        return stations


class Solver:
    def __init__(
        self, N: int, M: int, K: int, T: int, home: list[Pos], workplace: list[Pos]
    ):
        self.N = N
        self.M = M
        self.K = K
        self.T = T
        self.home = home
        self.workplace = workplace
        self.distance = [distance(home[i], workplace[i]) for i in range(M)]

        self.field = Field(N)
        self.money = K
        self.actions = []
        self.non_station_person_list = [i for i in range(M)]

    def calc_income(self) -> int:
        income = 0
        for i in range(self.M):
            if self.field.is_connected(self.home[i], self.workplace[i]):
                income += distance(self.home[i], self.workplace[i])
        return income

    def build_rail(self, type: int, r: int, c: int) -> None:
        self.field.build(type, r, c)
        self.money -= COST_RAIL
        self.actions.append(Action(type, (r, c)))

    def build_station(self, r: int, c: int) -> None:
        self.field.build(STATION, r, c)
        self.money -= COST_STATION
        self.actions.append(Action(STATION, (r, c)))

    def build_nothing(self) -> None:
        self.actions.append(Action(DO_NOTHING, (0, 0)))

    def bfs(self, start: Pos, end: Pos) -> list[Pos]:
        directions: list[Pos, Pos] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        prev: dict[Pos, Pos] = {start: None}
        queue: deque = deque([start])
        current: Pos | None = None
        while queue:
            current = queue.popleft()
            if current == end:
                break
            r, c = current
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.N
                    and 0 <= nc < self.N
                    and self.field.rail[nr][nc] in [EMPTY, STATION]
                ):
                    next_pos = (nr, nc)
                    if next_pos not in prev:
                        prev[next_pos] = current
                        queue.append(next_pos)
        if current != end:
            return []
        route: list[Pos] = []
        cur: Pos = end
        while cur is not None:
            route.append(cur)
            cur = prev.get(cur)
        route.reverse()
        return route

    def solve(self) -> Result:
        expect_income_sum: dict[tuple[Pos, Pos], int] = {}
        person_idx = 0
        for person_idx in self.non_station_person_list:
            for ds in dist_list:
                for de in dist_list:
                    s = (
                        self.home[person_idx][0] + ds[0],
                        self.home[person_idx][1] + ds[1],
                    )
                    e = (
                        self.workplace[person_idx][0] + de[0],
                        self.workplace[person_idx][1] + de[1],
                    )
                    if (
                        0 <= s[0] < self.N
                        and 0 <= s[1] < self.N
                        and 0 <= e[0] < self.N
                        and 0 <= e[1] < self.N
                    ):
                        if (s, e) not in expect_income_sum:
                            expect_income_sum[(s, e)] = 0
                        expect_income_sum[(s, e)] += self.distance[person_idx]
        assert person_idx != self.M
        # Tターン行動する
        income = self.calc_income()
        while len(self.actions) < self.T:
            mod_thres = int(self.M * (100 / 1600))
            if self.money < COST_STATION * 2 or len(self.actions) % mod_thres != 0:
                # if self.money < COST_STATION * 2:
                self.build_nothing()
                self.money += income
                continue
            # 最大の収入を得られる駅を探す
            rail_count = (self.money - COST_STATION * 2) // COST_RAIL

            max_pair = max(expect_income_sum, key=expect_income_sum.get)

            start, end = max_pair

            # 線路を配置して駅を接続する
            route_list = self.bfs(start, end)

            if route_list == [] or len(route_list) > rail_count:
                self.build_nothing()
                self.money += income
                continue

            station_cnt = 0
            if self.field.rail[start[0]][start[1]] != STATION:
                station_cnt += 1
            if self.field.rail[end[0]][end[1]] != STATION:
                station_cnt += 1
            # 建築するメリットがあるかどうか
            if expect_income_sum[max_pair] * (
                self.T - len(self.actions)
            ) < COST_STATION * station_cnt + COST_RAIL * (len(route_list) - 2):
                self.build_nothing()
                self.money += income
                continue

            # 設置する駅によって，通勤可能な人を削除
            for person_idx in self.non_station_person_list:
                if (
                    distance(self.home[person_idx], start) <= 2
                    and distance(self.workplace[person_idx], end) <= 2
                ):
                    self.non_station_person_list.remove(person_idx)

            # 駅の配置
            if self.field.rail[start[0]][start[1]] != STATION:
                self.build_station(*start)
            if self.field.rail[end[0]][end[1]] != STATION:
                self.build_station(*end)

            # ルートに沿ってレールを敷く
            for i in range(1, len(route_list) - 1):
                now_r, now_c = route_list[i]
                if self.field.rail[now_r][now_c] == STATION:
                    continue
                assert self.field.rail[now_r][now_c] == EMPTY

                prev_pos = route_list[i - 1]
                next_pos = route_list[i + 1]

                dr_prev = prev_pos[0] - now_r
                dc_prev = prev_pos[1] - now_c
                dr_next = next_pos[0] - now_r
                dc_next = next_pos[1] - now_c

                rail_type = rail_type_dict[(dr_prev, dc_prev), (dr_next, dc_next)]

                self.build_rail(rail_type, now_r, now_c)
                self.money += income
            income = self.calc_income()

            expect_income_sum_old = expect_income_sum.copy()
            for s_dr, s_dc in dist_list:
                for e_dr, e_dc in dist_list:
                    s = (start[0] + s_dr, start[1] + s_dc)
                    e = (end[0] + e_dr, end[1] + e_dc)
                    if (
                        0 <= s[0] < self.N
                        and 0 <= s[1] < self.N
                        and 0 <= e[0] < self.N
                        and 0 <= e[1] < self.N
                    ):
                        if (s, e) not in expect_income_sum_old:
                            expect_income_sum_old[(s, e)] = 0
                        expect_income_sum_old[(s, e)] -= expect_income_sum_old[max_pair]
            expect_income_sum_old[max_pair] = -1000000000
            expect_income_sum = expect_income_sum_old

        return Result(self.actions, self.money)


def main():
    N, M, K, T = map(int, input().split())
    home = []
    workplace = []
    for _ in range(M):
        r0, c0, r1, c1 = map(int, input().split())
        home.append((r0, c0))
        workplace.append((r1, c1))

    solver = Solver(N, M, K, T, home, workplace)
    result = solver.solve()
    print(result)
    print(f"score={result.score}", file=sys.stderr)


if __name__ == "__main__":
    main()
