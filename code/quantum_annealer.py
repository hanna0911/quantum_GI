import numpy as np
import math
import matplotlib.pyplot as plt
import time
import networkx as nx


sigma_i = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.identity(2)

# 目标图对应的邻接矩阵A1(9顶点)
'''
A1 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 1, 0, 0, 0, 0, 1],
               [0, 0, 1, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 1, 0, 0, 1],
               [0, 0, 0, 0, 1, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 1, 0, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=complex)

A2 = np.array([
        [0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 1, 0]
], dtype=complex)
'''

# 5结点SRG
'''
# 图1
A1 = np.array([[0, 1, 0, 0, 1],
               [1, 0, 1, 0, 0],
               [0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1],
               [1, 0, 0, 1, 0]], dtype = complex)

# 图2 同构
# A2 = np.array([[0, 0, 1, 1, 0],
#                [0, 0, 0, 1, 1],
#                [1, 0, 0, 0, 1],
#                [1, 1, 0, 0, 0],
#                [0, 1, 1, 0, 0]], dtype = complex)

# 图2 不同构
A2 = np.array([[0, 0, 1, 0, 1],
               [0, 0, 0, 1, 1],
               [1, 0, 0, 0, 1],
               [1, 1, 0, 0, 0],
               [0, 1, 1, 0, 0]], dtype = complex)
'''


# 16结点SRG
# dim=16, degree=6, lambda=2, mu=2
# TOTAL NUMBER OF STRONGLY REGULAR GRAPHS = 2
# http://www.maths.gla.ac.uk/~es/SRGs/16-6-2-2


A1 = np.array([
[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
[1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
[1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
[0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
[0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
[0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
[0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
[0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
], dtype = complex)

A2 = np.array([
[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
[1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
[1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
[0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
[0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
[0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
[0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
[0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
[0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
[0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
[0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0]
], dtype = complex)


# 8顶点非同构图
# A1 = np.array([
#     [0, 1, 0, 1, 0, 1, 0, 1],
#     [1, 0, 1, 0, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 1, 0, 1],
#     [1, 0, 1, 0, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 1, 0, 1],
#     [1, 0, 1, 0, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 1, 0, 1],
#     [1, 0, 1, 0, 1, 0, 1, 0]
# ], dtype = complex)

# A2 = np.array([
#     [0, 1, 0, 1, 0, 1, 1, 0],
#     [1, 0, 1, 1, 0, 0, 1, 0],
#     [0, 1, 0, 1, 1, 0, 0, 1],
#     [1, 1, 1, 0, 1, 0, 0, 0],
#     [0, 0, 1, 1, 0, 1, 0, 1],
#     [1, 0, 0, 0, 1, 0, 1, 1],
#     [1, 1, 0, 0, 0, 1, 0, 1],
#     [0, 0, 1, 0, 1, 1, 1, 0]
# ], dtype = complex)


# # 8顶点同构SRG
# A1 = np.array([
#     [0, 0, 0, 0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 1, 1, 0, 1],
#     [0, 0, 0, 0, 1, 0, 1, 1],
#     [0, 0, 0, 0, 0, 1, 1, 1],
#     [1, 1, 1, 0, 0, 0, 0, 0],
#     [1, 1, 0, 1, 0, 0, 0, 0],
#     [1, 0, 1, 1, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0, 0, 0, 0],
# ], dtype = complex)

# A2 = np.array([
#     [0, 1, 0, 1, 1, 0, 0, 0],
#     [1, 0, 1, 0, 0, 1, 0, 0],
#     [0, 1, 0, 1, 0, 0, 1, 0],
#     [1, 0, 1, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 1, 0, 1],
#     [0, 1, 0, 0, 1, 0, 1, 0],
#     [0, 0, 1, 0, 0, 1, 0, 1],
#     [0, 0, 0, 1, 1, 0, 1, 0],
# ], dtype = complex)


# # 10顶点SRG
# A1 = np.array([
#     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#     [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
#     [0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
#     [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
#     [0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
# ], dtype = complex)

# A2 = np.array([
#     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#     [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
#     [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
#     [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
#     [0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
# ], dtype = complex)


# # 6顶点非同构
# A1 = np.array([
#     [0, 1, 0, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0, 0],
#     [0, 0, 1, 0, 1, 0],
#     [0, 0, 0, 1, 0, 1],
#     [1, 0, 0, 0, 1, 0],
# ], dtype = complex)

# A2 = np.array([
#     [0, 0, 0, 0, 1, 1],
#     [0, 0, 1, 1, 0, 0],
#     [0, 1, 0, 1, 0, 0],
#     [0, 1, 1, 0, 0, 0],
#     [1, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 1, 0],
# ], dtype = complex)


# # 6顶点同构
# A1 = np.array([
#     [0, 1, 0, 0, 1, 1],
#     [1, 0, 1, 1, 0, 0],
#     [0, 1, 0, 1, 0, 1],
#     [0, 1, 1, 0, 1, 0],
#     [1, 0, 0, 1, 0, 1],
#     [1, 0, 1, 0, 1, 0],
# ], dtype = complex)

# A2 = np.array([
#     [0, 0, 0, 1, 1, 1],
#     [0, 0, 0, 1, 1, 1],
#     [0, 0, 0, 1, 1, 1],
#     [1, 1, 1, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0],
# ], dtype = complex)


# 显示二图
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

generate_graph()



def pauliZ_spin(k, N):  # 计算第k项为pauli-z，其他N-1项为I的张量积
    pauliZ_spin_matrix = sigma_z if k == 0 else I
    for i in range(N - 1):
        pauliZ_spin_matrix = np.kron(pauliZ_spin_matrix, sigma_z) if i + 1 == k \
            else np.kron(pauliZ_spin_matrix, I)
    return pauliZ_spin_matrix


def generate_H0(N):  # 容易制备和求解本征态的哈密顿量H0
    H0 = np.zeros(2 ** N)
    for i in range(N):
        tmp = sigma_x
        for j in range(i):
            tmp = np.kron(I, tmp)
        for j in range(N-i-1):
            tmp = np.kron(tmp, I)
        # print("tmp=", tmp)
        H0 = H0 + tmp
    H0 /= 2
    print("H0=", H0)
    return H0


def generate_HP(A):  # 目标哈密顿量（要求该哈密顿量的本征态）
    N =  len(A)
    HP = np.zeros(2 ** N)
    for i in range(N):
        for j in range(i+1, N):
            if A[i][j] == 1:
                tmp = sigma_z
                for k in range(i):
                    tmp = np.kron(I, tmp)
                for k in range(j - i - 1):
                    tmp = np.kron(tmp, I)
                tmp = np.kron(tmp, sigma_z)
                for k in range(N - j - 1):
                    tmp = np.kron(tmp, I)
                # print("tmp=", tmp)
                HP = HP + tmp
    print("HP=", HP)
    return HP


# 向量归一化
def uniform(state):
    s = 0
    for i in state:
        s += abs(i) ** 2
    for i in range(len(state)):
        state[i] /= np.sqrt(s)
    return state


def annealing_solver(steps, H0, H1):
    print(f'\nenter solver, total {steps} steps')
    
    t = 0
    eg_vector1 = np.abs(H0_eigen[1][0])
    eg_value1 = H0_eigen[0][0]
    energy1 = [eg_value1]
    Ht = H0   # 初始哈密顿量
    h = 6.62607015e-34   # 普朗克常量
    
    def expectation_value(A, psi):  # 求A在状态psi下的期望值
        norm_psi = psi / np.linalg.norm(psi)  # 单位化psi
        expectation  = np.inner(np.conj(norm_psi).T, np.matmul(A, norm_psi) )  # Calculate <norm_psi|A|norm_psi> 
        return expectation

    Q2 = []  # 无初始Q2
    def generate_Q2(psi):  # spin-glass order parameter
        # 计算 sigma { <pauli-z(i), pauli-z(j)> ^ 2 } (i != j)
        N = int(np.log2(len(H0)))
        sum = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                sum = sum + expectation_value((pauliZ_spin(i, N) * pauliZ_spin(j, N)), psi) ** 2
        
        sum = sum / (N * (N - 1))
        return np.sqrt(sum)

    Mx = []  # 无初始Mx
    
    for s in range(steps):
        start_loop = time.time()

        t += 1 / steps
        eg_vector_tmp1 = uniform(np.dot(Ht, eg_vector1) * (-1j) * (math.pi * 2 / steps) / h + eg_vector1)  # 当前状态
        Ht = (1 - t) * H0 + t * H1  # 哈密顿量

        eg_value1 = np.abs(eg_vector_tmp1[0]) * eg_value1 / np.abs(eg_vector1[0])  # E = <Ht>
        spin_glass = generate_Q2(eg_vector_tmp1) # 计算Q2
        x_magnetization = 2 * expectation_value(H0, eg_vector_tmp1) # 计算Mx

        eg_vector1 = eg_vector_tmp1  # 更新状态phi
        uniform(eg_vector1)

        energy1.append(eg_value1)
        Q2.append(spin_glass)
        Mx.append(x_magnetization)
        
        # print(f'step {s} finished in {time.time() - start_loop} s')

    # print(np.abs(uniform(eg_vector1)))
    # return energy1, uniform(eg_vector1)
    return energy1, Q2, Mx


start = time.time()
H0 = generate_H0(len(A1))
print(f'finished generating H0 in {time.time() - start} s')

start = time.time()
H0_eigen = np.linalg.eig(H0)  # 耗时间，只计算一次
print(f'finished generating eigen of H0 in {time.time() - start}\n')

start = time.time()
HP_1 = generate_HP(A1)
print(f'finished generating HP_1 in {time.time() - start} s\n')

start = time.time()
HP_2 = generate_HP(A2)
print(f'finished generating HP_2 in {time.time() - start} s\n')


start = time.time()
energy1, Q2_1, Mx_1 = annealing_solver(1000, H0, HP_1)
print(f'finished 1st annealing in {time.time() - start} s\n')

start = time.time()
energy2, Q2_2, Mx_2 = annealing_solver(1000, H0, HP_2)
print(f'finished 2nd annealing in {time.time() - start} s\n')

# print(np.linalg.eig(H1)[0])  # 验证特征值
# print(np.abs(np.linalg.eig(HP)[1][0]))  # 验证特征向量

plt.figure()
plt.plot(energy1, c = 'red')
plt.plot(energy2, c = 'blue')
plt.xlabel('adiabatic parameter steps')
plt.ylabel('average energy E')
plt.show()
print('energy1:', energy1[len(energy1) - 1])
print('energy2:', energy2[len(energy2) - 1])

# adiabatic_parameter_s = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
plt.plot(Q2_1, c = 'blue')
plt.plot(Q2_2, c = 'red')
plt.xlabel('adiabatic parameter steps')
plt.ylabel('spin-glass order parameter Q2')
plt.show()
print('Q2_1:', Q2_1[len(Q2_1) - 1])
print('Q2_2:', Q2_2[len(Q2_2) - 1])

plt.plot(Mx_1, c = 'blue')
plt.plot(Mx_2, c = 'red')
plt.xlabel('adiabatic parameter steps')
plt.ylabel('x-magnetization Mx')
plt.show()
print('Mx_1:', Q2_1[len(Mx_1) - 1])
print('Mx_2:', Q2_2[len(Mx_2) - 1])