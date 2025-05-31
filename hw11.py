import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.linalg import solve

# 條件設定
a, b = 0, 1
alpha, beta = 1, 2
h = 0.1
n = int((b - a) / h)
x_vals = np.linspace(a, b, n + 1)

# f(x, y, y') = -(x+1)y' + 2y + (1 - x^2)e^{-x}
def f(x, y, dy):
    return -(x + 1) * dy + 2 * y + (1 - x**2) * np.exp(-x)

# (a) Shooting Method
def shooting_rhs(x, Y):
    y, dy = Y
    return [dy, f(x, y, dy)]

def shooting_method():
    sol1 = solve_ivp(shooting_rhs, [a, b], [alpha, 0], t_eval=x_vals)
    y1 = sol1.y[0]
    sol2 = solve_ivp(shooting_rhs, [a, b], [0, 1], t_eval=x_vals)
    y2 = sol2.y[0]
    c = (beta - y1[-1]) / y2[-1]
    return y1 + c * y2

# (b) Finite Difference Method
def finite_difference():
    A = np.zeros((n - 1, n - 1))
    F = np.zeros(n - 1)

    for i in range(1, n):
        xi = a + i * h
        pi = -(xi + 1)
        qi = 2
        ri = (1 - xi**2) * np.exp(-xi)

        a_val = 1 - h * pi / 2
        b_val = -2 + h**2 * qi
        c_val = 1 + h * pi / 2

        if i > 1:
            A[i - 1, i - 2] = a_val
        A[i - 1, i - 1] = b_val
        if i < n - 1:
            A[i - 1, i] = c_val

        F[i - 1] = -h**2 * ri

    F[0] -= (1 - h * (x_vals[1] + 1) / 2) * alpha
    F[-1] -= (1 + h * (x_vals[-2] + 1) / 2) * beta

    Y_inner = solve(A, F)
    return np.concatenate(([alpha], Y_inner, [beta]))

# (c) Variation Approach
def phi(i, x):
    return np.sin(i * np.pi * x)

def dphi(i, x):
    return i * np.pi * np.cos(i * np.pi * x)

def variational_approach(M=10):
    A = np.zeros((M, M))
    b_vec = np.zeros(M)

    for i in range(1, M + 1):
        for j in range(1, M + 1):
            A[i - 1, j - 1] = quad(
                lambda x: dphi(i, x) * dphi(j, x) + 2 * phi(i, x) * phi(j, x),
                0, 1
            )[0]

        b_vec[i - 1] = quad(
            lambda x: ((1 - x**2) * np.exp(-x) - (x + 1) * (beta - alpha)) * phi(i, x),
            0, 1
        )[0]

    c = solve(A, b_vec)
    return np.array([alpha + (beta - alpha) * x + sum(c[i] * phi(i + 1, x) for i in range(M)) for x in x_vals])

# 執行各方法
y_shoot = shooting_method()
y_fd = finite_difference()
y_var = variational_approach()

# 輸出
print()
print(f"{'x':<5} | {'(a) Shooting':<14} | {'(b) Finite-Diff':<15} | {'(c) Variational':<14}")
print("-" * 58)
for i in range(n + 1):
    print(f"{x_vals[i]:5.2f} | {y_shoot[i]:^14.6f} | {y_fd[i]:^15.6f} | {y_var[i]:^14.6f}")
