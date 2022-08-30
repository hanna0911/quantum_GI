# 用来生成np.array矩阵
def generate_array():
    string = str(open('matrix_for_read.txt', 'r', encoding='utf-8').read())

    ans = '['
    for c in string:
        if c == '\n':
            ans = ans[:-2] + '],\n['
        else:
            ans = ans + c + ', '
    ans = ans + ']'

    print(ans[:-3] + ']')


# 生成图
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

A1 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 1, 0, 0, 0, 0, 1],
               [0, 0, 1, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 1, 0, 0, 1],
               [0, 0, 0, 0, 1, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 1, 0, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=complex)

A2 = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0],
               [1, 0, 1, 0, 0, 1, 1, 0, 0],
               [1, 1, 0, 0, 0, 0, 0, 1, 1],
               [1, 0, 0, 0, 1, 1, 0, 1, 0],
               [1, 0, 0, 1, 0, 0, 1, 0, 1],
               [0, 1, 0, 1, 0, 0, 1, 1, 0],
               [0, 1, 0, 0, 1, 1, 0, 0, 1],
               [0, 0, 1, 1, 0, 1, 0, 0, 1],
               [0, 0, 1, 0, 1, 0, 1, 1, 0]], dtype=complex)

def generate_graph():
    G1, G2 = nx.Graph(), nx.Graph()
    for i in range(len(A1)):
        for j in range(len(A1)):
            if A1[i][j] == 1:
                G1.add_edge(i, j)
            if A2[i][j] == 1:
                G2.add_edge(i, j)
    nx.draw(G1)
    plt.show()
    nx.draw(G2)
    plt.show()


generate_array()
# generate_graph()