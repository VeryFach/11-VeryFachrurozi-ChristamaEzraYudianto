import numpy as np
import matplotlib.pyplot as plt

N = 50
L = 2.0
dx = L / N
dt = 0.001
nu = 0.1

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

u = np.zeros((N, N))
v = np.zeros((N, N))

def solve_navier_stokes(u, v, steps=100):
    for _ in range(steps):
        u[1:-1, 1:-1] += dt * (
            - u[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
            - v[1:-1, 1:-1] * (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
            + nu * ((u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
                    (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2)
        )
        v[1:-1, 1:-1] += dt * (
            - u[1:-1, 1:-1] * (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)
            - v[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
            + nu * ((v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2 +
                    (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2)
        )
    return u, v


u, v = solve_navier_stokes(u, v, steps=200)

plt.quiver(X, Y, u, v)
plt.title("Simulasi Aliran Fluida (Navier-Stokes)")
plt.show()
