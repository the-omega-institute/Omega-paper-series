"""
AdS4 Black Brane QNM and Unified Time Scale κ(ω) Computation
-------------------------------------------------------------
Numerical solver for scalar field quasinormal modes in planar AdS4-Schwarzschild
geometry and reconstruction of the unified time scale density κ(ω) from boundary
retarded Green's function.

This code accompanies the paper:
"Unified Physical Time Scale: Scattering Theory, Holographic Boundary Geometry,
and Dirac-QCA Continuum Limits"

Authors: Haobo Ma, Wenlin Zhang
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Geometry and Radial Equation ----------

def f(z):
    """
    Blackening factor for planar AdS4 black brane.
    Horizon at z=1, boundary at z=0.
    Units: AdS radius L=1, horizon radius r_h=1.
    """
    return 1.0 - z**3

def fp(z):
    """Derivative of blackening factor."""
    return -3.0 * z**2

def radial_rhs(z, y, omega):
    """
    Right-hand side of the radial ODE for massless scalar m=0, k=0:
    
    φ''(z) + [f'(z)/f(z) - 2/z] φ'(z) - [ω²/f(z)²] φ(z) = 0
    
    Parameters:
    -----------
    z : float
        Radial coordinate
    y : array [φ, φ']
        State vector (both complex)
    omega : complex
        Frequency
    
    Returns:
    --------
    dy/dz : array [φ', φ'']
    """
    phi, phip = y
    fz = f(z)
    fpz = fp(z)
    coeff = fpz/fz - 2.0/z
    phipp = -coeff*phip - (omega**2 / (fz**2))*phi
    return np.array([phip, phipp], dtype=complex)

# ---------- Integration with Ingoing Horizon BC ----------

def integrate_solution(omega, z_start=0.999, z_end=1e-4, n_steps=2000):
    """
    Integrate radial equation from near horizon to boundary
    with ingoing boundary condition at the horizon.
    
    Near z→1: φ(z) ~ (1-z)^(-iω/3) (ingoing wave)
    
    Parameters:
    -----------
    omega : complex
        Frequency
    z_start : float
        Starting point near horizon (default: 0.999)
    z_end : float
        Endpoint near boundary (default: 1e-4)
    n_steps : int
        Number of integration steps
    
    Returns:
    --------
    y : array [φ(z_end), φ'(z_end)]
        Solution at endpoint
    zs : array
        Grid points
    """
    z0 = z_start
    alpha = -1j * omega / 3.0  # Ingoing exponent
    phi0 = (1.0 - z0)**alpha
    phip0 = alpha * (1.0 - z0)**(alpha - 1.0) * (-1.0)
    
    y = np.array([phi0, phip0], dtype=complex)
    zs = np.linspace(z0, z_end, n_steps)
    dz = zs[1] - zs[0]
    
    # Fourth-order Runge-Kutta integration
    for z in zs[1:]:
        k1 = radial_rhs(z, y, omega)
        k2 = radial_rhs(z + 0.5*dz, y + 0.5*dz*k1, omega)
        k3 = radial_rhs(z + 0.5*dz, y + 0.5*dz*k2, omega)
        k4 = radial_rhs(z + dz, y + dz*k3, omega)
        y = y + (dz/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    return y, zs

# ---------- A(ω), B(ω) from Near-Boundary Expansion ----------

def compute_AB(omega, z_eps=1e-3):
    """
    Extract source A(ω) and response B(ω) from near-boundary expansion.
    
    Near z→0: φ(z) ≈ A(ω) + B(ω) z³
    
    AdS/CFT dictionary:
    - A(ω): source for boundary operator O
    - B(ω): response (VEV of O)
    - Retarded Green's function: G_R(ω) ∝ B(ω)/A(ω)
    
    Parameters:
    -----------
    omega : complex
        Frequency
    z_eps : float
        Evaluation point near boundary (default: 1e-3)
    
    Returns:
    --------
    A, B : complex
        Source and response coefficients
    """
    y, zs = integrate_solution(omega, z_start=0.999, z_end=z_eps, n_steps=2000)
    phi = y[0]
    phip = y[1]
    z = z_eps
    
    # From φ = A + B z³, φ' = 3B z²
    B = phip / (3.0 * z**2)
    A = phi - B * z**3
    
    return A, B

# ---------- QNM Finder: A(ω) = 0 with Ingoing BC ----------

def phi_boundary(omega, z_eps=1e-3):
    """
    Boundary value φ(z_eps) for QNM search.
    
    QNM condition: A(ω) = 0 (no source at boundary)
                   + ingoing at horizon
    
    This is equivalent to φ(z_eps) ≈ 0.
    """
    y, zs = integrate_solution(omega, z_start=0.999, z_end=z_eps, n_steps=2000)
    return y[0]

def refine_qnm_root(omega0, n_iter=7, delta=1e-3):
    """
    Complex Newton-Raphson method to solve φ_boundary(ω) = 0.
    
    Parameters:
    -----------
    omega0 : complex
        Initial guess for QNM frequency
    n_iter : int
        Number of Newton iterations
    delta : float
        Step size for numerical derivative
    
    Returns:
    --------
    omega : complex
        Refined QNM frequency
    residual : complex
        Final residual φ_boundary(ω)
    """
    omega = omega0
    for _ in range(n_iter):
        f = phi_boundary(omega)
        f_p = phi_boundary(omega + delta)
        f_m = phi_boundary(omega - delta)
        fp = (f_p - f_m) / (2.0 * delta)
        omega = omega - f / fp
    return omega, phi_boundary(omega)

# ---------- κ(ω) from Boundary Green's Function ----------

def kappa_from_phase(omega_grid):
    """
    Reconstruct unified time scale density κ(ω) from boundary Green's function.
    
    κ(ω) = (1/π) dδ(ω)/dω
    
    where δ(ω) = arg[G_R(ω)] and G_R(ω) ∝ B(ω)/A(ω).
    
    Parameters:
    -----------
    omega_grid : array
        Real frequency grid
    
    Returns:
    --------
    phases : array
        Unwrapped phase δ(ω)
    kappa_vals : array
        Time scale density κ(ω)
    """
    G_vals = []
    for w in omega_grid:
        A, B = compute_AB(w)
        G_vals.append(B / A)  # Retarded Green's function (up to normalization)
    
    G_vals = np.array(G_vals)
    phases = np.unwrap(np.angle(G_vals))
    dphi_domega = np.gradient(phases, omega_grid)
    kappa_vals = dphi_domega / np.pi
    
    return phases, kappa_vals

# ---------- Main Execution ----------

def main():
    """Main computation and plotting."""
    
    print("=" * 70)
    print(" AdS4 BLACK BRANE QNM AND κ(ω) COMPUTATION")
    print(" Paper: Unified Physical Time Scale (PRD Submission)")
    print("=" * 70)
    print()
    
    # Find fundamental QNM
    print("Finding fundamental scalar QNM (m=0, k=0)...")
    omega0 = 2.8 - 0.3j  # Initial guess
    omega_qnm, phi_qnm = refine_qnm_root(omega0, n_iter=6, delta=2e-3)
    
    print(f"Fundamental QNM: ω = {omega_qnm.real:.5f} - {-omega_qnm.imag:.5f}i")
    print(f"  (in units of r_h^-1 with AdS radius L=1)")
    print(f"Residual: |φ(ε)| = {abs(phi_qnm):.2e}")
    print()
    
    # Compute κ(ω) near QNM
    print("Computing κ(ω) from boundary Green's function...")
    omega_r = omega_qnm.real
    omega_min, omega_max = omega_r - 1.0, omega_r + 1.0
    omega_grid = np.linspace(omega_min, omega_max, 120)
    
    phases, kappa_vals = kappa_from_phase(omega_grid)
    
    print(f"Computed κ(ω) on [{omega_min:.2f}, {omega_max:.2f}]")
    print()
    
    # Plot Phase and κ(ω)
    print("Generating figures...")
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    
    axes[0].plot(omega_grid, phases, 'b-', linewidth=1.5)
    axes[0].axvline(omega_qnm.real, color='r', linestyle=':', alpha=0.6, 
                    label=f'QNM: $\\omega_0 = {omega_qnm.real:.3f} - {-omega_qnm.imag:.3f}i$')
    axes[0].set_ylabel(r'$\delta(\omega) = \arg G_R(\omega)$', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.5)
    
    axes[1].plot(omega_grid, kappa_vals, 'g-', linewidth=1.5)
    axes[1].axvline(omega_qnm.real, color='r', linestyle=':', alpha=0.6)
    axes[1].set_xlabel(r'$\omega$ (in units of $r_h^{-1}$)', fontsize=12)
    axes[1].set_ylabel(r'$\kappa(\omega) = \frac{1}{\pi}\frac{d\delta}{d\omega}$', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.5)
    
    fig.suptitle(r'Planar AdS$_4$ Black Brane: Phase and Unified Time Scale $\kappa(\omega)$', 
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig('ads4_kappa_phase.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('ads4_kappa_phase.png', dpi=300, bbox_inches='tight')
    print("  Saved: ads4_kappa_phase.pdf/png")
    
    # Plot QNM in complex plane
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.scatter([omega_qnm.real], [omega_qnm.imag], s=100, c='red', marker='x', 
                linewidths=2, label='Fundamental QNM')
    ax2.axhline(0.0, linestyle=':', color='gray', alpha=0.5)
    ax2.axvline(0.0, linestyle=':', color='gray', alpha=0.5)
    ax2.set_xlabel(r'$\mathrm{Re}\,\omega$', fontsize=12)
    ax2.set_ylabel(r'$\mathrm{Im}\,\omega$', fontsize=12)
    ax2.set_title(r'Fundamental Scalar QNM, Planar AdS$_4$ Black Brane', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(-1.0, 0.2)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    fig2.tight_layout()
    
    fig2.savefig('ads4_qnm_point.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig('ads4_qnm_point.png', dpi=300, bbox_inches='tight')
    print("  Saved: ads4_qnm_point.pdf/png")
    
    print()
    print("=" * 70)
    print(" COMPUTATION COMPLETE")
    print(" Results demonstrate holographic realization of κ(ω)")
    print("=" * 70)

if __name__ == "__main__":
    main()


