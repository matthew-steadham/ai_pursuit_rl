import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(trajs, arena_size=2000.0, capture_radius=20.0):
    """
    trajs: list of dict with keys 'self', 'target' -> arrays Nx2
    """
    plt.figure()
    ax = plt.gca()
    ax.set_xlim([-arena_size, arena_size])
    ax.set_ylim([-arena_size, arena_size])
    ax.set_aspect('equal', adjustable='box')
    for tr in trajs:
        s = tr['self']
        t = tr['target']
        ax.plot(s[:,0], s[:,1], label='interceptor')
        ax.plot(t[:,0], t[:,1], label='target', linestyle='--')
        ax.plot([s[0,0]], [s[0,1]], marker='o')
        ax.plot([t[0,0]], [t[0,1]], marker='x')
    circ = plt.Circle((0,0), capture_radius, fill=False, linestyle=':')
    ax.add_patch(circ)
    ax.legend()
    plt.title("2D Pursuit–Evasion Trajectories")
    plt.show()