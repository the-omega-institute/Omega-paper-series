#!/usr/bin/env python3
"""
Generate figures for PRL paper on Fermi-Dirac statistics from QCA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch, Wedge
from matplotlib.path import Path
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set style for publication-quality figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

def figure1_feedback_loop():
    """
    Figure 1: Microscopic origin of mass and statistics
    Three panels: (a) QCA feedback loop, (b) impedance matching, (c) exchange braiding
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # Panel (a): QCA feedback loop
    ax = axes[0]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Feedback loop structure', fontsize=11, fontweight='bold')
    
    # Draw QCA lattice
    for i in range(5):
        for j in range(4):
            circle = Circle((i, j), 0.15, facecolor='lightgray', edgecolor='black', linewidth=1)
            ax.add_patch(circle)
    
    # Highlight feedback region
    feedback_nodes = [(2, 1), (3, 1), (3, 2), (2, 2)]
    for node in feedback_nodes:
        circle = Circle(node, 0.18, facecolor='red', edgecolor='darkred', linewidth=2, alpha=0.7)
        ax.add_patch(circle)
    
    # Draw feedback arrows
    arrow1 = FancyArrowPatch((2.15, 1), (2.85, 1), arrowstyle='->', lw=2.5, color='darkred', mutation_scale=20)
    arrow2 = FancyArrowPatch((3, 1.15), (3, 1.85), arrowstyle='->', lw=2.5, color='darkred', mutation_scale=20)
    arrow3 = FancyArrowPatch((2.85, 2), (2.15, 2), arrowstyle='->', lw=2.5, color='darkred', mutation_scale=20)
    arrow4 = FancyArrowPatch((2, 1.85), (2, 1.15), arrowstyle='->', lw=2.5, color='darkred', mutation_scale=20)
    for arrow in [arrow1, arrow2, arrow3, arrow4]:
        ax.add_patch(arrow)
    
    # Input/output labels
    ax.annotate('$\\psi_{\\mathrm{in}}$', xy=(1, 1.5), fontsize=11, color='blue', weight='bold')
    ax.arrow(1.2, 1.5, 0.6, 0, head_width=0.15, head_length=0.1, fc='blue', ec='blue')
    
    ax.annotate('$\\psi_{\\mathrm{refl}}$', xy=(3.7, 1.5), fontsize=11, color='green', weight='bold')
    ax.arrow(3.5, 1.5, -0.6, 0, head_width=0.15, head_length=0.1, fc='green', ec='green')
    
    # Panel (b): Impedance matching condition
    ax = axes[1]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) Fixed-point condition', fontsize=11, fontweight='bold')
    
    # Draw complex plane
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(1.3, 0.1, 'Re', fontsize=10, color='gray')
    ax.text(0.1, 1.3, 'Im', fontsize=10, color='gray')
    
    # Draw Möbius transformation trajectory
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.8
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    ax.plot(x_circle, y_circle, 'b-', linewidth=2, alpha=0.3, label='$\\Phi$ trajectory')
    
    # Fixed points
    fp1 = np.array([0.6, 0.4])
    fp2 = np.array([-0.6, -0.4])
    ax.plot(fp1[0], fp1[1], 'ro', markersize=12, label='$Z_+$ (stable)', zorder=5)
    ax.plot(fp2[0], fp2[1], 'kx', markersize=12, markeredgewidth=3, label='$Z_-$ (unstable)', zorder=5)
    
    # Draw discriminant sqrt(Delta)
    ax.annotate('', xy=fp1, xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    ax.text(0.3, 0.25, '$\\sqrt{\\Delta}$', fontsize=12, color='purple', weight='bold')
    
    # Equation
    ax.text(0, -1.3, '$Z_{\\mathrm{in}} = \\Phi(Z_{\\mathrm{in}})$', fontsize=11, 
            ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel (c): Exchange braiding
    ax = axes[2]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(c) Particle exchange', fontsize=11, fontweight='bold')
    
    # Initial state: two particles
    p1_init = np.array([0.5, 1.0])
    p2_init = np.array([2.5, 1.0])
    
    circle1 = Circle(p1_init, 0.25, facecolor='red', edgecolor='darkred', linewidth=2, alpha=0.7)
    circle2 = Circle(p2_init, 0.25, facecolor='blue', edgecolor='darkblue', linewidth=2, alpha=0.7)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.text(p1_init[0], p1_init[1]-0.5, 'Particle 1', ha='center', fontsize=9)
    ax.text(p2_init[0], p2_init[1]-0.5, 'Particle 2', ha='center', fontsize=9)
    
    # Exchange paths (braiding)
    t = np.linspace(0, 1, 50)
    path1_x = p1_init[0] + 2.0 * t
    path1_y = p1_init[1] + 0.8 * np.sin(np.pi * t)
    path2_x = p2_init[0] - 2.0 * t
    path2_y = p2_init[1] - 0.8 * np.sin(np.pi * t)
    
    ax.plot(path1_x, path1_y, 'r--', linewidth=2, alpha=0.6)
    ax.plot(path2_x, path2_y, 'b--', linewidth=2, alpha=0.6)
    
    # Crossing point annotation
    ax.plot(1.5, 1.0, 'ko', markersize=8, zorder=10)
    ax.annotate('Branch cut\ncrossing', xy=(1.5, 1.0), xytext=(1.5, 1.8),
                fontsize=9, ha='center', color='purple', weight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))
    
    # Phase accumulation
    ax.text(1.5, 0.2, 'Phase: $(-1)$', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('docs/submitted/PRL-deriving fermi/feedback_loop_diagram.pdf', 
                dpi=300, bbox_inches='tight')
    print("✓ Generated: feedback_loop_diagram.pdf")
    plt.close()


def figure2_spinor_mobius():
    """
    Figure 2: 4π periodicity and spinor emergence
    Three panels: (a) Delta trajectory, (b) Möbius strip, (c) double cover
    """
    fig = plt.figure(figsize=(12, 3.5))
    
    # Panel (a): Discriminant trajectory in complex plane
    ax1 = plt.subplot(131)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Re($\\Delta$)', fontsize=10)
    ax1.set_ylabel('Im($\\Delta$)', fontsize=10)
    ax1.set_title('(a) $\\Delta(\\phi)$ under rotation', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Trajectory for Delta(phi) = 4*sin^2(theta + phi/2)
    theta_0 = np.pi/4  # Example coin angle
    phi_range = np.linspace(0, 4*np.pi, 200)
    Delta_real = 4 * np.sin(theta_0 + phi_range/2)**2
    Delta_imag = np.zeros_like(Delta_real)  # Real for this example
    
    # Map to complex plane with winding
    Delta_complex = Delta_real * np.exp(1j * phi_range/2)
    
    # Plot trajectory
    colors = plt.cm.viridis(phi_range / (4*np.pi))
    for i in range(len(phi_range)-1):
        ax1.plot(Delta_complex.real[i:i+2], Delta_complex.imag[i:i+2], 
                color=colors[i], linewidth=2)
    
    # Mark key points
    ax1.plot(Delta_complex.real[0], Delta_complex.imag[0], 'go', markersize=10, label='$\\phi=0$')
    idx_2pi = len(phi_range)//2
    ax1.plot(Delta_complex.real[idx_2pi], Delta_complex.imag[idx_2pi], 'ro', 
            markersize=10, label='$\\phi=2\\pi$')
    ax1.plot(Delta_complex.real[-1], Delta_complex.imag[-1], 'bo', 
            markersize=10, label='$\\phi=4\\pi$')
    
    # Square root branch cut
    ax1.plot([0, 1.2], [0, 0], 'k--', linewidth=2, alpha=0.5)
    ax1.text(0.6, -0.15, 'Branch cut', fontsize=9, ha='center', color='black')
    
    ax1.legend(fontsize=8, loc='upper left')
    
    # Panel (b): Möbius strip visualization
    ax2 = plt.subplot(132, projection='3d')
    ax2.set_title('(b) Möbius strip topology', fontsize=11, fontweight='bold')
    
    # Create Möbius strip
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(-0.3, 0.3, 20)
    U, V = np.meshgrid(u, v)
    
    # Parametric equations for Möbius strip
    R = 1
    X = (R + V * np.cos(U/2)) * np.cos(U)
    Y = (R + V * np.cos(U/2)) * np.sin(U)
    Z = V * np.sin(U/2)
    
    # Plot surface
    surf = ax2.plot_surface(X, Y, Z, alpha=0.7, cmap='coolwarm', 
                            linewidth=0, antialiased=True)
    
    # Mark special points
    phi_points = [0, np.pi, 2*np.pi, 3*np.pi]
    for i, phi in enumerate(phi_points):
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        z = 0
        ax2.scatter([x], [y], [z], color='black', s=50, zorder=10)
        label = f'${i}\\pi$'
        ax2.text(x*1.2, y*1.2, z, label, fontsize=9, weight='bold')
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-0.5, 0.5)
    ax2.set_box_aspect([1, 1, 0.5])
    ax2.axis('off')
    
    # Panel (c): Double cover projection
    ax3 = plt.subplot(133)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('(c) $\\widetilde{Q}_N \\to Q_N$ projection', fontsize=11, fontweight='bold')
    
    # Upper hemisphere (SU(2) ~ S^3)
    theta = np.linspace(0, 2*np.pi, 100)
    x_upper = 0.8 * np.cos(theta)
    y_upper = 0.8 * np.sin(theta) + 0.5
    ax3.fill(x_upper, y_upper, color='lightblue', alpha=0.5, edgecolor='blue', linewidth=2)
    ax3.text(0, 0.5, '$\\chi$', fontsize=14, ha='center', va='center', weight='bold')
    
    # Lower hemisphere
    x_lower = 0.8 * np.cos(theta)
    y_lower = 0.8 * np.sin(theta) - 0.5
    ax3.fill(x_lower, y_lower, color='lightcoral', alpha=0.5, edgecolor='red', linewidth=2)
    ax3.text(0, -0.5, '$-\\chi$', fontsize=14, ha='center', va='center', weight='bold')
    
    # Projection arrow
    ax3.annotate('', xy=(1.2, 0), xytext=(0, 0.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
    ax3.annotate('', xy=(1.2, 0), xytext=(0, -0.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
    
    # Observable Z
    circle_obs = Circle((1.2, 0), 0.15, facecolor='yellow', edgecolor='black', linewidth=2)
    ax3.add_patch(circle_obs)
    ax3.text(1.2, -0.35, '$Z = \\chi^\\dagger \\sigma \\chi$', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Labels
    ax3.text(-0.7, 1.2, '$\\mathbb{Z}_2$ covering', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('docs/submitted/PRL-deriving fermi/spinor_mobius_strip.pdf', 
                dpi=300, bbox_inches='tight')
    print("✓ Generated: spinor_mobius_strip.pdf")
    plt.close()


def figure3_photonic_experiment():
    """
    Figure 3: Proposed photonic quantum walk experiment
    Three panels: (a) Waveguide setup, (b) Exchange protocol, (c) Phase measurement
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # Panel (a): Waveguide array schematic
    ax = axes[0]
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Waveguide array setup', fontsize=11, fontweight='bold')
    
    # Draw waveguides
    for y in [1, 2.5]:
        rect = FancyBboxPatch((0, y-0.15), 5, 0.3, boxstyle="round,pad=0.02",
                              edgecolor='blue', facecolor='lightblue', linewidth=2)
        ax.add_patch(rect)
    
    # Ring resonators (feedback loops)
    for x in [1.5, 3.5]:
        for y in [1, 2.5]:
            circle = Circle((x, y), 0.35, facecolor='none', edgecolor='red', linewidth=2.5)
            ax.add_patch(circle)
    
    ax.text(1.5, 0.3, 'Ring A', fontsize=9, ha='center', color='red', weight='bold')
    ax.text(3.5, 0.3, 'Ring B', fontsize=9, ha='center', color='red', weight='bold')
    
    # Input/output
    ax.arrow(-0.3, 1, 0.5, 0, head_width=0.15, head_length=0.15, fc='green', ec='green')
    ax.arrow(-0.3, 2.5, 0.5, 0, head_width=0.15, head_length=0.15, fc='green', ec='green')
    ax.text(-0.3, 1.5, 'Input', fontsize=10, rotation=90, ha='center', color='green', weight='bold')
    
    ax.arrow(5, 1, 0.3, 0, head_width=0.15, head_length=0.15, fc='orange', ec='orange')
    ax.arrow(5, 2.5, 0.3, 0, head_width=0.15, head_length=0.15, fc='orange', ec='orange')
    ax.text(5.3, 1.75, 'Output', fontsize=10, rotation=90, ha='center', color='orange', weight='bold')
    
    # Beam splitters
    for x in [2.5]:
        for y in [1, 2.5]:
            rect = Rectangle((x-0.1, y-0.3), 0.2, 0.6, facecolor='gray', 
                            edgecolor='black', linewidth=1.5, alpha=0.7)
            ax.add_patch(rect)
    
    # Panel (b): Exchange protocol
    ax = axes[1]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) Exchange braiding', fontsize=11, fontweight='bold')
    
    # Initial positions
    t_vals = np.array([0, 1, 2, 3])
    y_vals = [0.5, 1.5, 2.5]
    
    for i, t in enumerate(t_vals):
        # Particle A path
        if t == 0:
            ax.plot(t, y_vals[0], 'ro', markersize=15, label='Photon A')
        elif t < 3:
            y = y_vals[0] + (y_vals[2] - y_vals[0]) * (t/3)
            ax.plot(t, y, 'ro', markersize=15, alpha=0.7)
        else:
            ax.plot(t, y_vals[2], 'ro', markersize=15)
        
        # Particle B path
        if t == 0:
            ax.plot(t, y_vals[2], 'bs', markersize=15, label='Photon B')
        elif t < 3:
            y = y_vals[2] - (y_vals[2] - y_vals[0]) * (t/3)
            ax.plot(t, y, 'bs', markersize=15, alpha=0.7)
        else:
            ax.plot(t, y_vals[0], 'bs', markersize=15)
    
    # Connecting lines
    ax.plot([0, 3], [y_vals[0], y_vals[2]], 'r--', linewidth=2, alpha=0.5)
    ax.plot([0, 3], [y_vals[2], y_vals[0]], 'b--', linewidth=2, alpha=0.5)
    
    # Time labels
    for i, t in enumerate(t_vals):
        ax.text(t, -0.3, f'$t_{i}$', fontsize=10, ha='center')
    
    ax.legend(fontsize=9, loc='upper left')
    ax.text(1.5, 1.5, '×', fontsize=30, ha='center', va='center', color='purple')
    ax.text(1.5, 2.8, 'Crossing', fontsize=9, ha='center', color='purple', weight='bold')
    
    # Panel (c): Phase vs. detuning
    ax = axes[2]
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('Detuning $\\delta\\theta$ (rad)', fontsize=10)
    ax.set_ylabel('Exchange phase / $\\pi$', fontsize=10)
    ax.set_title('(c) Predicted phase shift', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Theoretical prediction
    delta_theta = np.linspace(0, 2, 100)
    F_critical = 0.5
    
    # Self-reference fidelity
    F = delta_theta  # Simplified: F increases with detuning
    
    # Phase: stays at pi while F < F_c, then drops
    phase = np.ones_like(delta_theta)
    mask = F > F_critical
    phase[mask] = 1.0 - 0.8 * (F[mask] - F_critical) / (2 - F_critical)
    phase = np.clip(phase, 0, 1)
    
    ax.plot(delta_theta, phase, 'b-', linewidth=3, label='Theory')
    ax.axvline(F_critical, color='red', linestyle='--', linewidth=2, label='$\\mathcal{F}_c$')
    ax.axhline(1.0, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Fermionic')
    ax.axhline(0.0, color='orange', linestyle=':', linewidth=2, alpha=0.5, label='Bosonic')
    
    # Annotations
    ax.text(0.2, 0.95, 'Impedance\nmatched', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(1.5, 0.3, 'Statistics\ntransmutation', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('docs/submitted/PRL-deriving fermi/photonic_experiment_schematic.pdf', 
                dpi=300, bbox_inches='tight')
    print("✓ Generated: photonic_experiment_schematic.pdf")
    plt.close()


if __name__ == '__main__':
    print("\n=== Generating Figures for PRL Paper ===\n")
    
    figure1_feedback_loop()
    figure2_spinor_mobius()
    figure3_photonic_experiment()
    
    print("\n✓ All figures generated successfully!")
    print("Files saved in: docs/submitted/PRL-deriving fermi/\n")


