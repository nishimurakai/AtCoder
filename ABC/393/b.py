S:str = input()

a:list[int] = []
b:list[int] = []
c:list[int] = []

for i in range(len(S)):
    if S[i] == 'A':
        a.append(i)
    elif S[i] == 'B':
        b.append(i)
    elif S[i] == 'C':
        c.append(i)

ans:int = 0

for i in a:
    for j in b:
        for k in c:
            if (j - i == k - j) and (j - i >= 0):
                ans += 1

print(ans)