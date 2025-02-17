import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

L = 10
N = 100
dx = L / N
dt = 0.01
x = np.linspace(-L / 2, L / 2, N)

V = np.zeros(N)
V[0], V[-1] = 1e6, 1e6

alpha = 1j * dt / (2 * dx**2)
main_diag = (1 - 2 * alpha) * np.ones(N)
off_diag = alpha * np.ones(N - 1)

A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="csc")
B = diags([-off_diag, (1 + 2 * alpha) * np.ones(N), -off_diag], [-1, 0, 1], format="csc")

psi = np.exp(-x**2) * np.exp(1j * 5 * x)

for _ in range(100):
    psi = spsolve(B, A @ psi)

plt.plot(x, np.abs(psi)**2, label="Distribusi Probabilitas |ψ|²")
plt.legend()
plt.show()
