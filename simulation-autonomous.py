#!/usr/bin/env python3
"""Terminal version of the research notebook; figures saved to figs/."""
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Run the model simulations and export figures.")
parser.add_argument("--fig-dir", default="figs", help="Output folder for figures")
args = parser.parse_args()
FIG_DIR = Path(args.fig_dir)
FIG_DIR.mkdir(parents=True, exist_ok=True)
_figure_index = 0

def _save_show(*args, **kwargs):
    global _figure_index
    for number in plt.get_fignums():
        fig = plt.figure(number)
        if not fig.axes:
            continue
        _figure_index += 1
        path = FIG_DIR / f"autonomous_{_figure_index:03d}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close("all")

plt.show = _save_show

# Notebook cell 4
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11

# Notebook cell 7
def system(v, emax, u, alpha, beta, gamma, delta):
    e, z = v
    interaction = e * max(1.0 - e / emax, 0.0) * z
    de = -alpha * e - beta * interaction + u
    dz = -gamma * z + delta * interaction
    return np.array([de, dz], dtype=float)

# Notebook cell 9
def simulate(e0, z0, emax, u, alpha, beta, gamma, delta, T, n):
    y0 = (e0, z0)
    t_eval = np.linspace(0.0, T, n)
    sol = solve_ivp(
        fun=lambda t, v : system(v, emax, u, alpha, beta, gamma, delta),
        t_span=(0.0, T),
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol

# Notebook cell 11
def interaction_on_e(t_eval, e):
    inter = []
    for i, t in enumerate(t_eval):
        e_value = e[i]
        inter.append(e_value * np.maximum(1.0 - e_value / emax, 0.0))
    return inter

# Notebook cell 12
def interaction_global(t_eval, e, z):
    inter = []
    for i, t in enumerate(t_eval):
        e_value = e[i]
        z_value = z[i]
        inter.append(e_value * np.maximum(1.0 - e_value / emax, 0.0) * z_value)
    return inter

# Notebook cell 14
def extinction_equilibria(u, alpha):
    e_star = u / alpha
    z_star = 0.0
    return (e_star, z_star)

def coexistence_equilibria(emax, u, alpha, beta, gamma, delta):
    disc = emax**2 - 4.0 * gamma * emax / delta
    
    if disc == 0:
        estar = emax / 2.0
        zstar = (-alpha * estar + u) * delta / (beta * gamma)
        return [
            (estar, zstar),
            (estar, zstar)
        ]
    elif disc > 0:
        estar1 = (emax - np.sqrt(disc)) / 2.0
        estar2 = (emax + np.sqrt(disc)) / 2.0
        zstar1 = (-alpha * estar1 + u) * delta / (beta * gamma)
        zstar2 = (-alpha * estar2 + u) * delta / (beta * gamma)
        return [
            (estar1, zstar1),
            (estar2, zstar2)
        ]
    
    return [(None, None), (None, None)]

# Notebook cell 18
T = 30 * 4 + 10
n = T * 5
t_eval = np.linspace(0.0, T, n)

# Notebook cell 20
alpha = 0.15
beta = 0.1
gamma = 0.07
delta = 0.06

# Notebook cell 22
emax = 11
u = 1.9

# Notebook cell 24
e0_ref = 2
z0_ref = 1

# Notebook cell 26
sol = simulate(e0_ref, z0_ref, emax, u, alpha, beta, gamma, delta, T, n)

# Notebook cell 29
plt.axhline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
plt.axhline(emax, linestyle="--", label=r"$e_{max}$")
plt.plot(sol.t, sol.y[0], label="e(t)", color='tab:blue')
plt.plot(sol.t, sol.y[1], label="z(t)", color='tab:green')
plt.xlabel("Time t")
plt.ylabel("e(t) and z(t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Notebook cell 30
plt.axhline(emax / 2, linestyle=":", label=r"$e_{max}/2$", color='tab:blue')
plt.axhline(emax, linestyle="--", label=r"$e_{max}$", color='tab:blue')
plt.plot(sol.t, sol.y[0], label="e(t)", color='tab:blue')
plt.xlabel("Time t")
plt.ylabel("e(t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Notebook cell 31
plt.plot(sol.t, sol.y[1], label="z(t)", color='tab:green')
plt.xlabel("Time t")
plt.ylabel("z(t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Notebook cell 33
plt.axvline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
plt.axvline(emax, linestyle="--", label=r"$e_{max}$")
plt.plot(sol.y[0], sol.y[1], linewidth=1.8, label="Phase (e, z)")
plt.scatter([e0_ref], [z0_ref], label="Initial condition", zorder=3)
plt.xlabel("e(t)")
plt.ylabel("z(t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Notebook cell 35
ex_eq = extinction_equilibria(u, alpha)
ex_eq

# Notebook cell 36
co_eqs = coexistence_equilibria(emax, u, alpha, beta, gamma, delta)
co_eqs

# Notebook cell 37
# plt.axvline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
# plt.axvline(emax, linestyle="--", label=r"$e_{max}$")

plt.plot(sol.y[0], sol.y[1], linewidth=1.8, label="Phase (e, z)")

plt.scatter([co_eqs[0][0]], [co_eqs[0][1]], marker='X', label="Interior equilibrium 1", zorder=3)
plt.scatter([co_eqs[1][0]], [co_eqs[1][1]], marker='X', label="Interior equilibrium 2", zorder=3)
plt.scatter([ex_eq[0]], [ex_eq[1]], marker='X', label="Boundary equilibrium", zorder=3)

plt.scatter([e0_ref], [z0_ref], label="Initial condition", zorder=3)

plt.xlabel("e(t)")
plt.ylabel("z(t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Notebook cell 39
g_interaction_values = interaction_global(t_eval, sol.y[0], sol.y[1])
interaction_values = interaction_on_e(t_eval, sol.y[0])

# Notebook cell 40
plt.axhline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
plt.axhline(emax, linestyle="--", label=r"$e_{max}$")
plt.plot(sol.t, sol.y[0], label="e(t)", color='tab:blue')
plt.plot(sol.t, sol.y[1], label="z(t)", color='tab:green')
plt.plot(t_eval, g_interaction_values, label='I(t, e, z)', color='gray')
# plt.plot(t_eval, interaction_values, label='I(t, e, 1)', color='gray', linestyle='--')
plt.xlabel("Time t")
plt.ylabel(r"$I(t, e, z), e(t), z(t)$")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

# Notebook cell 41
plt.axhline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
plt.axhline(emax, linestyle="--", label=r"$e_{max}$")
plt.plot(sol.t, sol.y[0], label="e(t)", color='tab:blue')
plt.plot(t_eval, g_interaction_values, label='I(t, e, z)', color='gray')
# plt.plot(t_eval, interaction_values, label='I(t, e, 1)', color='gray', linestyle='--')
plt.xlabel("Time t")
plt.ylabel(r"$I(t, e, z), e(t)$")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

# Notebook cell 42
plt.plot(sol.t, sol.y[1], label="z(t)", color='tab:green')
plt.plot(t_eval, g_interaction_values, label='I(t, e, z)', color='gray')
# plt.plot(t_eval, interaction_values, label='I(t, e, 1)', color='gray', linestyle='--')
plt.xlabel("Time t")
plt.ylabel(r"$I(t, e, z), z(t)$")
plt.title(r"Profil temporel de la composante biomasse et de l'interaction")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

# Notebook cell 43
ti_emax = np.argmax(sol.y[0] >= emax)
ti_emax_sur2 = np.argmax(sol.y[0] >= emax / 2)
print(f"Time t where e(t) reaches emax: {t_eval[ti_emax]}")
print(f"Time t where e(t) reaches emax/2: {t_eval[ti_emax_sur2]}")

# Notebook cell 44
zmax = np.max(sol.y[1])
ti_zmax = np.argmax(sol.y[1] == zmax)
print(f"Maximum of z(t): {zmax}")
print(f"Time t whiere z(t) reaches z_max : {t_eval[ti_zmax]}")

# Notebook cell 45
plt.axvline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
plt.axvline(emax, linestyle="--", label=r"$e_{max}$")
plt.plot(sol.y[0], g_interaction_values, label='I(t, e, z)', color='gray')
plt.plot(sol.y[0], interaction_values, label='I(t, e, 1)', color='gray', linestyle='--')
plt.xlabel("e(t)")
plt.ylabel(r"$I(t, e, z)$")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

# Notebook cell 46
ti_nulle_interaction = np.argmax(np.array(g_interaction_values) == 0)
print(f"Time i où I(t,e,z)=0: {ti_nulle_interaction}")
print(f"Time t où I(t,e,z)=0: {t_eval[ti_nulle_interaction]}")

# Notebook cell 47
print(f"Time TM2 : {t_eval[ti_nulle_interaction+10]}")

# Notebook cell 48
print(f"z(TM2) : {sol.y[1][ti_nulle_interaction+10]}")

# Notebook cell 51
initial_conditions = [
    (1, 0.5),
    (1, 1),
    (1, 1.5),
    (1, 2),
    (2, 0.5), 
    (2, 1), 
    (2, 1.5), 
    (2, 2), 
]

plt.figure(figsize=(9, 5.5))

for y0 in initial_conditions:
    s = simulate(y0[0], y0[1], emax, u, alpha, beta, gamma, delta, T, n)
    plt.plot(s.y[0], s.y[1], label=r"$(e_0, z_0)$="+str(y0))
    plt.scatter([y0[0]],[y0[1]], s=18)

plt.scatter([co_eqs[0][0]], [co_eqs[0][1]], marker='X', label="Interior equilibrium 1", zorder=3)
plt.scatter([co_eqs[1][0]], [co_eqs[1][1]], marker='X', label="Interior equilibrium 2", zorder=3)
plt.scatter([ex_eq[0]], [ex_eq[1]], marker='X', label="Boundadry equilibrium", zorder=3)
# plt.axvline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
# plt.axvline(emax, linestyle="--", label=r"$e_{max}$")

plt.xlabel("e(t)"); plt.ylabel("z(t)")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

# Notebook cell 53
def z0_sensibility(e0_ref, z0_values):
    for z0 in z0_values:
        s = simulate(e0_ref, z0, emax, u, alpha, beta, gamma, delta, T, n)
        plt.plot(s.y[0], s.y[1], label=r"$z_0$="+str(z0))
        plt.scatter([e0_ref],[z0], s=18)

    plt.scatter([co_eqs[0][0]], [co_eqs[0][1]], marker='X', label="Interior equilibrium 1", zorder=3)
    plt.scatter([co_eqs[1][0]], [co_eqs[1][1]], marker='X', label="Interior equilibrium 2", zorder=3)
    plt.scatter([ex_eq[0]], [ex_eq[1]], marker='X', label="Boundadry equilibrium", zorder=3)
    # plt.axvline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
    # plt.axvline(emax, linestyle="--", label=r"$e_{max}$")

    plt.xlabel("e(t)"); plt.ylabel("z(t)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

# Notebook cell 54
e0_ref = 2
# z0_values = [0.5, 0.75, 1, 1.5, 1.75, 2, 4]
# z0_values = [1, 1.125, 1.25, 1.375, 1.5]
z0_values = [1, 1.02, 1.025, 1.05, 1.1, 1.125]
z0_sensibility(e0_ref, z0_values)

# Notebook cell 55
e0_ref = 1
z0_values = [1, 1.125, 1.25, 1.375, 1.5]
z0_sensibility(e0_ref, z0_values)

# Notebook cell 56
e0_ref = 4
z0_values = [1, 1.125, 1.25, 1.375, 1.5]
z0_sensibility(e0_ref, z0_values)

# Notebook cell 58
def e0_sensibility(z0_ref, e0_values):
    for e0 in e0_values:
        s = simulate(e0, z0_ref, emax, u, alpha, beta, gamma, delta, T, n)
        plt.plot(s.y[0], s.y[1], label=r"$e_0$="+str(e0))
        plt.scatter([e0],[z0_ref], s=18)

    plt.scatter([co_eqs[0][0]], [co_eqs[0][1]], marker='X', label="Interior equilibrium 1", zorder=3)
    plt.scatter([co_eqs[1][0]], [co_eqs[1][1]], marker='X', label="Interior equilibrium 2", zorder=3)
    plt.scatter([ex_eq[0]], [ex_eq[1]], marker='X', label="Boundadry equilibrium", zorder=3)
    # plt.axvline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
    # plt.axvline(emax, linestyle="--", label=r"$e_{max}$")

    plt.xlabel("e(t)"); plt.ylabel("z(t)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

# Notebook cell 59
z0_ref = 1
e0_values = [0, 0.25, 0.5, 1, 2, 4]
e0_sensibility(z0_ref, e0_values)

# Notebook cell 60
z0_ref = 1.025 # 1.0249
e0_values = [0, 0.25, 0.5, 1, 2, 4]
e0_sensibility(z0_ref, e0_values)

# Notebook cell 61
z0_ref = 1.2
e0_values = [0, 0.25, 0.5, 1, 2, 4]
e0_sensibility(z0_ref, e0_values)

# Notebook cell 63
e0_critical = 0.5
z0_critical = 1.025
u_values = [0.5, 1, 1.5, 2, 4]   

results = []

for u in u_values:
    sol = simulate(e0_critical, z0_critical, emax, u, alpha, beta, gamma, delta, T, n)
    results.append(sol)

plt.axhline(emax / 2, linestyle=":", label=r"$e_{max}/2$")
plt.axhline(emax, linestyle="--", label=r"$e_{max}$")

for i, sol in enumerate(results):
    plt.plot(sol.t, sol.y[0], label=f"e(t) -- u={u_values[i]}")
plt.xlabel("Time t")
plt.ylabel(r"$e_{\max}$, $e$ and $I(t, e, 1)$")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

for i, sol in enumerate(results):
    plt.plot(sol.t, sol.y[1], label=f"z(t) -- u={u_values[i]}")
plt.xlabel("Time t")
plt.ylabel(r"$z$ and $I(t, e, z)$")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

# Save figures left open by cells without plt.show()
_save_show()
