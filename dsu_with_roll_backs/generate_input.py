import random as rd

n, q = map(int, input().split())

vertices = [i for i in range(1, n+1)]

# note that q should be much greater than 10
# inserting some 10 random edges
print(n, q)
m = dict()
for i in range(10):
    while(1):
        u, v = tuple(rd.choices(vertices, k = 2))
        if u == v or m.get((u, v)) != None or m.get((v, u)) != None:
            continue
        print('+', u, v)
        m[(u, v)] = 1
        m[(v, u)] = 1
        break
for i in range(q - 10):
    op = rd.choice(['+', '?', '-'])
    
    if op == '+':
        while 1:
            u, v = tuple(rd.choices(vertices, k = 2))
            if u == v or m.get((u, v)) != None or m.get((v, u)) != None:
                continue
            print(op, u, v)
            m[(u, v)] = 1
            m[(v, u)] = 1
            break
    elif op == '-':
        if len(list(m.keys())) == 0:
            i -= 1
            continue
        u, v = rd.choices(list(m.keys()), k = 1)[0]    
        m.pop((u, v))
        m.pop((v, u))
        print(op, u, v)
    else:
        print(op)