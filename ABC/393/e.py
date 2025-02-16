from collections import defaultdict
from math import sqrt

N, K = map(int, input().split())
A = list(map(int, input().split()))


def gcd_list(a: int) -> list[int]:
    res: list[int] = []
    for i in range(2, int(sqrt(a)) + 1):
        if a % i == 0:
            res.append(i)
            a //= i
            while a % i == 0:
                res.append(i)
                a //= i
    if a != 1:
        res.append(a)
    return res


a_gcd = [gcd_list(a) for a in A]

print(a_gcd)

gcd_dict: defaultdict[int, int] = defaultdict(int)
for a in a_gcd:
    for i in a:
        gcd_dict[i] += 1

for i in range(N):
    ans = 1
    for j in a_gcd[i]:
        if gcd_dict[j] >= K:
            ans *= j
    print(ans)
