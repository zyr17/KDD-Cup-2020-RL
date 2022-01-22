import main
ZKW_algo = main.ZKW_algo

edge = [
    [0,0,3],
    [0,1,2],
    [0,2,4],
    [1,0,2],
    [1,1,1],
    [1,2,3],
    [2,2,5]
]
n = 3
m = 3
print(ZKW_algo(n, m, edge))
'''
edge = [
    [0,1,2,5],
    [1,3,2,2],
    [0,2,2,6],
    [2,3,2,0],
    [3,4,3,10.1],
    [2,1,10,0.1]
]

z = ZKW(12, 5, 0, 4, 5)
for e in edge:
    z.addedge(*e)
print(z.head)
print(list(zip(range(100),z.e_n, z.e_v)))
z.costflow()
print(z.ans, z.cost)
'''
