N: int = int(input())
S: str = input()
S_int: list[int] = [int(x) for x in S]

S_zero_left: list[int] = [1 if S_int[0] == 0 else 0]
S_one_left: list[int] = []
if S_int[0] == 1:
    S_one_left.append(0)

for i in range(1, N):
    S_zero_left.append(S_zero_left[i - 1] + (1 if S_int[i] == 0 else 0))
    if S_int[i] == 1:
        S_one_left.append(S_zero_left[i])

mid = len(S_one_left) // 2

ans: int = 0

for i in range(len(S_one_left)):
    ans += abs(S_one_left[i] - S_one_left[mid])

print(ans)
