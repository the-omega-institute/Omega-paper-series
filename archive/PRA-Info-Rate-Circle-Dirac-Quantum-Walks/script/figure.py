#!/usr/bin/env python3
"""
Generate figures for the Information Rate Circle paper.
Produces publication-quality plots for Physical Review A submission.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set publication quality defaults
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.usetex'] = False  # Set to True if you have LaTeX installed
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['lines.linewidth'] = 1.5

# Parameters
a = 1.0
dt = 1.0
c = a / dt
mu_dt_list = [0.0, 0.3, 0.8]  # Representing massless, light mass, heavy mass
labels = ['$\mu\Delta t=0$ (massless)', '$\mu\Delta t=0.3$', '$\mu\Delta t=0.8$']
colors = ['black', 'blue', 'red']
p = np.linspace(-np.pi/a, np.pi/a, 500)

# ============================================================
# Figure 1: Dispersion, Velocity Decomposition, and Entanglement
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

# Plot (a): Dispersion Relation
ax1 = axes[0]
for mu_dt, label, color in zip(mu_dt_list, labels, colors):
    if mu_dt == 0:
        # Massless case: Omega = |p*a|
        omega = np.abs(p * a)
    else:
        cos_omega = np.cos(mu_dt) * np.cos(p * a)
        cos_omega = np.clip(cos_omega, -1, 1)  # Numerical safety
        omega = np.arccos(cos_omega)
    ax1.plot(p * a / np.pi, omega, label=label, color=color)

ax1.set_title('(a) Dispersion Relation', fontsize=11, fontweight='bold')
ax1.set_xlabel('Momentum $pa/\pi$', fontsize=10)
ax1.set_ylabel('Energy $\Omega(p)$ [rad]', fontsize=10)
ax1.legend(fontsize=8, frameon=False)
ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_xlim([-1, 1])

# Plot (b): Velocity Partition (The Circle)
ax2 = axes[1]
mu_dt = 0.5  # Use intermediate mass for demonstration
cos_omega = np.cos(mu_dt) * np.cos(p * a)
cos_omega = np.clip(cos_omega, -1, 1)
sin_omega = np.sqrt(1 - cos_omega**2)
sin_omega[sin_omega < 1e-10] = 1e-10  # Avoid division by zero

# v_ext / c = n_3
v_ext = (np.cos(mu_dt) * np.sin(p * a)) / sin_omega
# v_int / c = sqrt(n1^2 + n2^2)
v_int = np.sin(mu_dt) / sin_omega

ax2.plot(p * a / np.pi, v_ext, 'b-', label='$v_{\mathrm{ext}}/c$', linewidth=2)
ax2.plot(p * a / np.pi, v_int, 'r--', label='$v_{\mathrm{int}}/c$', linewidth=2)
ax2.plot(p * a / np.pi, np.sqrt(v_ext**2 + v_int**2), 'k:', 
         label='$(v_{\mathrm{ext}}^2+v_{\mathrm{int}}^2)^{1/2}$', linewidth=1.5)
ax2.axhline(1.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

ax2.set_title(f'(b) Rate Partition ($\mu\Delta t={mu_dt}$)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Momentum $pa/\pi$', fontsize=10)
ax2.set_ylabel('Rate / $c$', fontsize=10)
ax2.legend(fontsize=8, frameon=False, loc='upper left')
ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_xlim([-1, 1])
ax2.set_ylim([0, 1.2])

# Plot (c): Entanglement Bound Proxy
ax3 = axes[2]
for mu_dt, label, color in zip(mu_dt_list, labels, colors):
    if mu_dt == 0:
        ax3.plot(p * a / np.pi, np.zeros_like(p), label=label, color=color)
        continue
    cos_omega = np.cos(mu_dt) * np.cos(p * a)
    cos_omega = np.clip(cos_omega, -1, 1)
    sin_omega = np.sqrt(1 - cos_omega**2)
    sin_omega[sin_omega < 1e-10] = 1e-10
    v_int = np.sin(mu_dt) / sin_omega
    # Entanglement bound is proportional to v_int^2 (from Proposition 1)
    ax3.plot(p * a / np.pi, v_int**2, label=label, color=color)

ax3.set_title('(c) Entanglement Bound $(\propto v_{\mathrm{int}}^2)$', fontsize=11, fontweight='bold')
ax3.set_xlabel('Momentum $pa/\pi$', fontsize=10)
ax3.set_ylabel('$v_{\mathrm{int}}^2/c^2$', fontsize=10)
ax3.legend(fontsize=8, frameon=False)
ax3.grid(alpha=0.3, linestyle='--', linewidth=0.5)
ax3.set_xlim([-1, 1])
ax3.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('figure_dispersion_and_rates.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figure_dispersion_and_rates.png', bbox_inches='tight', dpi=300)
print("✓ Generated: figure_dispersion_and_rates.pdf")
plt.close()

# ============================================================
# Figure 2: Geometric Visualization (Conceptual Sketch)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Plot (a): The Information Rate Circle
ax1 = axes[0]
theta = np.linspace(0, 2*np.pi, 200)
ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='$v_{\mathrm{ext}}^2 + v_{\mathrm{int}}^2 = c^2$')

# Example modes
modes = [
    {'p': 0, 'mu': 0.5, 'label': 'Rest frame\n$(p=0)$', 'color': 'blue'},
    {'p': 0.3*np.pi, 'mu': 0.5, 'label': 'Massive mode\n$(p\\neq 0)$', 'color': 'red'},
    {'p': 0.7*np.pi, 'mu': 0.0, 'label': 'Massless\n$(\\mu=0)$', 'color': 'black'},
]

for mode in modes:
    mu_dt = mode['mu']
    pa = mode['p']
    if mu_dt == 0:
        v_ext_val = np.sign(np.sin(pa)) if pa != 0 else 0
        v_int_val = 0
    else:
        cos_omega = np.cos(mu_dt) * np.cos(pa)
        sin_omega = np.sqrt(1 - cos_omega**2) if abs(cos_omega) < 1 else 1e-10
        v_ext_val = (np.cos(mu_dt) * np.sin(pa)) / sin_omega if sin_omega > 1e-10 else 0
        v_int_val = np.sin(mu_dt) / sin_omega if sin_omega > 1e-10 else 0
    
    ax1.arrow(0, 0, v_ext_val*0.95, v_int_val*0.95, head_width=0.05, 
              head_length=0.05, fc=mode['color'], ec=mode['color'], linewidth=2)
    ax1.plot(v_ext_val, v_int_val, 'o', color=mode['color'], markersize=8)
    ax1.text(v_ext_val*1.15, v_int_val*1.15, mode['label'], 
             fontsize=8, ha='center', color=mode['color'])

# Axes
ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
ax1.set_xlabel('External rate $v_{\mathrm{ext}}/c$', fontsize=11)
ax1.set_ylabel('Internal rate $v_{\mathrm{int}}/c$', fontsize=11)
ax1.set_title('(a) Information Rate Circle', fontsize=12, fontweight='bold')
ax1.set_aspect('equal')
ax1.set_xlim([-1.3, 1.3])
ax1.set_ylim([-0.2, 1.3])
ax1.grid(alpha=0.2, linestyle='--', linewidth=0.5)
ax1.legend(loc='upper right', fontsize=8, frameon=False)

# Plot (b): Bloch Sphere Decomposition
ax2 = axes[1]
# Draw Bloch sphere (circle in 2D projection)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = 0.8 * np.outer(np.cos(u), np.sin(v))
y_sphere = 0.8 * np.outer(np.sin(u), np.sin(v))
ax2.plot(x_sphere[0, :], y_sphere[0, :], 'k-', linewidth=1, alpha=0.3)
ax2.plot(x_sphere[:, 50], y_sphere[:, 50], 'k-', linewidth=1, alpha=0.3)

# Propagation direction d
ax2.arrow(0, 0, 0, 0.8, head_width=0.08, head_length=0.08, fc='green', ec='green', linewidth=2)
ax2.text(0.1, 0.85, r'$\hat{d}$ (propagation)', fontsize=9, color='green')

# Bloch axis n(p)
n_parallel = 0.4
n_perp = 0.7
ax2.arrow(0, 0, n_perp*0.95, n_parallel*0.95, head_width=0.08, head_length=0.08, 
          fc='purple', ec='purple', linewidth=2.5)
ax2.text(n_perp*1.1, n_parallel*1.1, r'$\hat{n}(p)$ (Bloch axis)', fontsize=9, color='purple')

# Decomposition
ax2.plot([n_perp, n_perp], [0, n_parallel], 'b--', linewidth=1.5, label=r'$\propto v_{\mathrm{ext}}$')
ax2.plot([0, n_perp], [0, 0], 'r--', linewidth=1.5, label=r'$\propto v_{\mathrm{int}}$')

ax2.set_xlabel('$\\hat{n} \\perp \\hat{d}$ component', fontsize=10)
ax2.set_ylabel('$\\hat{n} \\parallel \\hat{d}$ component', fontsize=10)
ax2.set_title('(b) Bloch Axis Decomposition', fontsize=12, fontweight='bold')
ax2.set_aspect('equal')
ax2.set_xlim([-1, 1])
ax2.set_ylim([-0.2, 1])
ax2.grid(alpha=0.2, linestyle='--', linewidth=0.5)
ax2.legend(loc='lower right', fontsize=8, frameon=False)

plt.tight_layout()
plt.savefig('figure_geometric_circle.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figure_geometric_circle.png', bbox_inches='tight', dpi=300)
print("✓ Generated: figure_geometric_circle.pdf")
plt.close()

print("\n✓ All figures generated successfully!")
print("  - figure_dispersion_and_rates.pdf (for paper)")
print("  - figure_geometric_circle.pdf (for paper)")
print("  - PNG versions also saved for preview")

