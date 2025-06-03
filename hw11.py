import numpy as np
import math
from scipy.integrate import solve_ivp, quad
from scipy.linalg import solve
import matplotlib.pyplot as plt

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
        xi = x_vals[i]
        pi = -(xi + 1)
        qi = 2
        ri = (1 - xi**2) * np.exp(-xi)

        a_val = - (1 + h * pi / 2)
        b_val = 2 + h**2 * qi
        c_val = - (1 - h * pi / 2)

        if i > 1:
            A[i - 1, i - 2] = a_val
        A[i - 1, i - 1] = b_val
        if i < n - 1:
            A[i - 1, i] = c_val

        F[i - 1] = -h**2 * ri

    F[0] += (1 + h * (-(x_vals[1] + 1)) / 2) * alpha
    F[-1] += (1 - h * (-(x_vals[-2] + 1)) / 2) * beta
    Y_inner = solve(A, F)
    return np.concatenate(([alpha], Y_inner, [beta]))

# (c) Variation Approach
def phi(i, x):
    return np.sin(i * np.pi * x)

def dphi(i, x):
    return i * np.pi * np.cos(i * np.pi * x)

def y1(x):
    return 1 + x

def r(x):
    return (1 - x**2) * np.exp(-x)

def q(x):
    return 2

def F(x):
    return (1 - x*x) * math.exp(x * x / 2)

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h

def variational_approach(h=0.1, N=10):
    n_points = int((b - a) / h) + 1
    x_vals = [a + i * h for i in range(n_points)]
    A = np.zeros((N, N))
    B = np.zeros(N)

    for i in range(N):
        for j in range(N):
            def integrand_A(x):
                phi_i, phi_j = phi(i+1, x), phi(j+1, x)
                dphi_i, dphi_j = dphi(i+1, x), dphi(j+1, x)
                P = -math.exp(x * x / 2 + x)
                Q = -2 * math.exp(x * x / 2 + x)
                return P * dphi_i * dphi_j + Q * phi_i * phi_j
            
            A[i][j] = trapezoidal(integrand_A, a, b, n_points - 1)

        def integrand_B(x):
            return  F(x) * phi(i+1, x)

        B[i] = trapezoidal(integrand_B, a, b, n_points - 1)

    for i in range(N):
        for j in range(i + 1, N):
            factor = A[j][i] / A[i][i]
            for k in range(i, N):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]

    c = [0 for _ in range(N)]
    for i in range(N - 1, -1, -1):
        c[i] = B[i]
        for j in range(i + 1, N):
            c[i] -= A[i][j] * c[j]
        c[i] /= A[i][i]

    return [y1(x) + sum(c[i] * phi(i + 1, x) for i in range(N)) for x in x_vals]

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

plt.figure(figsize=(10, 6))
plt.grid(True)
plt.plot(x_vals, y_shoot, 'o-', label='(a)Shooting Method', color='red')
plt.plot(x_vals, y_fd, 'x--', label='(b)Finite-Difference', color='blue')
plt.plot(x_vals, y_var, '^-', label='(c)Variational Method', color='green')

plt.title('Comparison')
plt.xlabel('x')
plt.xticks(np.arange(0.1, 1.01, 0.1))
plt.ylabel('y(x)')
plt.legend()
plt.tight_layout()
plt.show()