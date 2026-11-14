"""
================================================================================
PHYSICS SANDBOX — REFERENCE ENGINE 1.0 (GOOGLE COLAB EDITION)
================================================================================
Arquitectura Sellada (Estado C):
  - Malla estática precalculada SpectralGrid con FFT2 vectorizada en ejes (-2, -1).
  - Protocolo PhysicalModel e integrador tensorial directo RK4Integrator.
  - Formalismo unificado 4-Espinor Nambu-Dirac-Bogoliubov-Poisson [ψ_↑, ψ_↓, η_↑, η_↓].
  - Interfaz interactiva nativa para Google Colab (ipywidgets) con 5 Misiones.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import hashlib
import json
import time

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

ComplexTensor = NDArray[np.complex128]
RealGrid = NDArray[np.float64]


# ============================================================
# 1. CONFIGURACIÓN CENTRAL Y VALIDACIÓN
# ============================================================

@dataclass
class SimulationConfig:
    nx: int = 84
    ny: int = 84
    length_x: float = 12.0
    length_y: float = 12.0

    # Constantes universales y acoplamientos de teorías
    hbar: float = 1.0
    mass: float = 1.0
    gravity: float = 2.0          # Acoplamiento de Newton-Poisson (G)
    coupling: float = 0.0         # No-linealidad de Gross-Pitaevskii (g)
    c_dirac: float = 0.0          # Velocidad relativista de Dirac (c_D)
    alpha_rashba: float = 0.0     # Acoplamiento Espín-Órbita de Rashba (α_R)
    bz_zeeman: float = 0.0        # Campo magnético de Zeeman perpendicular (B_z)
    delta_sc: float = 0.0         # Brecha superconductora de Bogoliubov (Δ_SC)
    flux_ab: float = 0.0          # Flujo magnético de Aharonov-Bohm (Φ_B)
    v_klein: float = 0.0          # Altura de la barrera relativista de Klein (V_0)

    dt: float = 0.0015
    steps: int = 120

    def validate(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("La malla debe tener al menos 8x8 puntos.")
        if self.length_x <= 0 or self.length_y <= 0:
            raise ValueError("Las dimensiones espaciales deben ser positivas.")
        if self.hbar <= 0 or self.mass <= 0:
            raise ValueError("hbar y mass deben ser estrictamente positivos.")
        if self.gravity < 0 or self.c_dirac < 0 or self.delta_sc < 0 or self.dt <= 0:
            raise ValueError("Parámetros físicos o temporales fuera de rango.")


# ============================================================
# 2. MALLA Y OPERADORES ESPECTRALES PRE-CALCULADOS (VECTORIZADOS)
# ============================================================

class SpectralGrid:

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.dx = config.length_x / config.nx
        self.dy = config.length_y / config.ny

        x = (np.arange(config.nx) - config.nx // 2) * self.dx
        y = (np.arange(config.ny) - config.ny // 2) * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")

        kx = 2.0 * np.pi * np.fft.fftfreq(config.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(config.ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="xy")

        self.K2 = self.KX ** 2 + self.KY ** 2
        self.DA = self.dx * self.dy

        r_max = 0.40 * min(config.length_x, config.length_y)
        self.R2 = self.X ** 2 + self.Y ** 2
        self.ENVELOPE = np.exp(-((self.R2 / (r_max ** 2)) ** 4))

    def laplacian(self, field: ComplexTensor) -> ComplexTensor:
        transformed = np.fft.fft2(field, axes=(-2, -1))
        return np.fft.ifft2(-self.K2 * transformed, axes=(-2, -1))

    def gradient(self, field: ComplexTensor) -> tuple[ComplexTensor, ComplexTensor]:
        transformed = np.fft.fft2(field, axes=(-2, -1))
        gx = np.fft.ifft2(1j * self.KX * transformed, axes=(-2, -1))
        gy = np.fft.ifft2(1j * self.KY * transformed, axes=(-2, -1))
        return gx, gy

    def poisson(self, mass_density: RealGrid, G: float) -> RealGrid:
        """Resuelve ∇²Φ = 4πG(ρ_m - <ρ_m>) en espacio de Fourier con modo k=0 nulo."""
        if G <= 0.0:
            return np.zeros_like(mass_density, dtype=np.float64)

        source = mass_density - np.mean(mass_density)
        source_k = np.fft.fft2(source)
        phi_k = np.zeros_like(source_k, dtype=np.complex128)

        mask = self.K2 > 0.0
        phi_k[mask] = -4.0 * np.pi * G * source_k[mask] / self.K2[mask]
        return np.fft.ifft2(phi_k).real


# ============================================================
# 3. PROTOCOLO DEL MODELO Y AUDITORÍA CIENTÍFICA
# ============================================================

class PhysicalModel(Protocol):
    def initial_state(self) -> ComplexTensor:
        ...

    def rhs(self, state: ComplexTensor) -> ComplexTensor:
        ...

    def diagnostics(self, state: ComplexTensor) -> dict[str, float]:
        ...


class ModelAuditor:

    @staticmethod
    def validate_state(state: ComplexTensor, expected_shape: tuple[int, ...]) -> None:
        if not isinstance(state, np.ndarray):
            raise TypeError("El estado debe ser un numpy.ndarray.")
        if state.shape != expected_shape:
            raise ValueError(f"Geometría inválida: {state.shape}; esperada {expected_shape}.")
        if not np.issubdtype(state.dtype, np.complexfloating):
            raise TypeError("El estado cuántico debe ser de tipo complejo.")
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Divergencia detectada: el estado contiene NaN o Inf.")


# ============================================================
# 4. MODELO UNIFICADO: NAMBU-DIRAC-BOGOLIUBOV-POISSON-GP
# ============================================================

class UnifiedQuantumGravityModel:

    def __init__(self, config: SimulationConfig, grid: SpectralGrid):
        self.cfg = config
        self.grid = grid
        self.Ax = np.zeros((config.ny, config.nx), dtype=np.float64)
        self.Ay = np.zeros((config.ny, config.nx), dtype=np.float64)
        self.v_barrier = np.zeros((config.ny, config.nx), dtype=np.float64)
        self.Bz = np.zeros((config.ny, config.nx), dtype=np.float64)
        self.sync_external_potentials()

    def sync_external_potentials(self) -> None:
        X, Y = self.grid.X, self.grid.Y
        r2 = self.grid.R2
        env = self.grid.ENVELOPE

        r_core2 = 0.20 ** 2
        self.Ax = -(self.cfg.flux_ab / (2.0 * np.pi)) * (Y / (r2 + r_core2)) * env
        self.Ay = (self.cfg.flux_ab / (2.0 * np.pi)) * (X / (r2 + r_core2)) * env

        klein_wall = self.cfg.v_klein * 0.5 * (
            np.tanh((X - 0.4) / 0.15) - np.tanh((X - 1.2) / 0.15)
        )
        ab_shield = (
            (25.0 if abs(self.cfg.flux_ab) > 0.2 and self.cfg.v_klein == 0.0 else 0.0)
            * np.exp(-((r2 / 0.24) ** 3))
        )
        self.v_barrier = klein_wall + ab_shield
        self.Bz = self.cfg.bz_zeeman * env

    def initial_state(self, scenario: str = "M1: Estrella de Bosones") -> ComplexTensor:
        X, Y = self.grid.X, self.grid.Y
        r2 = self.grid.R2
        state = np.zeros((4, self.cfg.ny, self.cfg.nx), dtype=np.complex128)

        if scenario == "M1: Estrella de Bosones":
            s = 0.90
            w1 = np.exp(-((X + 1.3) ** 2 + Y ** 2) / (2 * s ** 2)) * np.exp(1j * 1.1 * X)
            w2 = np.exp(-((X - 1.3) ** 2 + Y ** 2) / (2 * s ** 2)) * np.exp(-1j * 1.1 * X)
            state[0] = w1 + w2

        elif scenario == "M2: Deflexion Aharonov-Bohm":
            state[0] = np.exp(
                -((X + 2.2) ** 2 / (2 * 0.75 ** 2) + Y ** 2 / (2 * 1.7 ** 2))
            ) * np.exp(1j * 3.0 * X)

        elif scenario == "M3: Vortice Quiral Rashba":
            state[0] = np.exp(-r2 / (2 * 0.90 ** 2))

        elif scenario == "M4: Tunel de Klein (Dirac)":
            state[0] = np.exp(-((X + 1.8) ** 2 + Y ** 2) / (2 * 0.70 ** 2)) * np.exp(1j * 3.4 * X)
            state[1] = 0.5 * state[0]

        else:  # M5: Fermion de Majorana
            theta = np.arctan2(Y, X)
            state[0] = np.exp(-r2 / (2 * 1.0 ** 2))
            state[1] = 0.4 * np.sqrt(r2) * np.exp(-r2 / (2 * 1.0 ** 2)) * np.exp(1j * theta)

        norm = np.sum(np.abs(state) ** 2) * self.grid.DA
        return state / np.sqrt(norm)

    def rhs(self, state: ComplexTensor) -> ComplexTensor:
        lap = self.grid.laplacian(state)
        gx, gy = self.grid.gradient(state)

        rho = np.sum(np.abs(state) ** 2, axis=0)
        phi = self.grid.poisson(self.cfg.mass * rho, self.cfg.gravity)
        v_eff = self.cfg.mass * phi + self.cfg.coupling * rho + self.v_barrier

        hbar = self.cfg.hbar
        mass = self.cfg.mass
        c_d = self.cfg.c_dirac
        alpha_r = self.cfg.alpha_rashba
        delta = self.cfg.delta_sc

        nr_scale = 1.0 / (1.0 + 0.35 * (c_d ** 2))
        A2 = self.Ax ** 2 + self.Ay ** 2

        def single_sector(u: ComplexTensor, d: ComplexTensor, gx_u, gy_u, gx_d, gy_d, lap_u, lap_d, q_sign: float):
            Pi_x_u = -1j * hbar * gx_u - q_sign * self.Ax * u
            Pi_y_u = -1j * hbar * gy_u - q_sign * self.Ay * u
            Pi_x_d = -1j * hbar * gx_d - q_sign * self.Ax * d
            Pi_y_d = -1j * hbar * gy_d - q_sign * self.Ay * d

            kin_u = nr_scale * (
                -(hbar ** 2 / (2.0 * mass)) * lap_u
                + (1j * hbar * q_sign / mass) * (self.Ax * gx_u + self.Ay * gy_u)
                + (1.0 / (2.0 * mass)) * A2 * u
            )
            kin_d = nr_scale * (
                -(hbar ** 2 / (2.0 * mass)) * lap_d
                + (1j * hbar * q_sign / mass) * (self.Ax * gx_d + self.Ay * gy_d)
                + (1.0 / (2.0 * mass)) * A2 * d
            )

            m_gap = 0.25 * mass * (c_d ** 2)
            dirac_u = c_d * (Pi_x_d - 1j * Pi_y_d) + m_gap * u
            dirac_d = c_d * (Pi_x_u + 1j * Pi_y_u) - m_gap * d

            rashba_u = (alpha_r / hbar) * (Pi_y_d + 1j * Pi_x_d)
            rashba_d = (alpha_r / hbar) * (Pi_y_u - 1j * Pi_x_u)

            pot_u = (q_sign * v_eff + self.Bz) * u
            pot_d = (q_sign * v_eff - self.Bz) * d

            return kin_u + dirac_u + rashba_u + pot_u, kin_d + dirac_d + rashba_d + pot_d

        hu, hd = single_sector(
            state[0], state[1], gx[0], gy[0], gx[1], gy[1], lap[0], lap[1], +1.0
        )
        heu, hed = single_sector(
            state[2], state[3], gx[2], gy[2], gx[3], gy[3], lap[2], lap[3], -1.0
        )

        out = np.empty_like(state)
        out[0] = hu + delta * state[2]
        out[1] = hd + delta * state[3]
        out[2] = -heu + delta * state[0]
        out[3] = -hed + delta * state[1]

        return (-1j / hbar) * out

    def diagnostics(self, state: ComplexTensor) -> dict[str, float]:
        rho_comp = np.abs(state) ** 2
        rho_p = rho_comp[0] + rho_comp[1]
        rho_h = rho_comp[2] + rho_comp[3]
        rho = rho_p + rho_h

        phi = self.grid.poisson(self.cfg.mass * rho, self.cfg.gravity)
        gx, gy = self.grid.gradient(state)

        kinetic_energy = (
            (self.cfg.hbar ** 2 / (2.0 * self.cfg.mass))
            * np.sum(np.abs(gx) ** 2 + np.abs(gy) ** 2)
            * self.grid.DA
        )
        interaction_energy = 0.5 * self.cfg.coupling * np.sum(rho ** 2) * self.grid.DA
        gravity_energy = 0.5 * self.cfg.mass * np.sum(phi * rho) * self.grid.DA
        norm = float(np.sum(rho) * self.grid.DA)

        vortex_fraction = float(np.sum(rho_comp[1]) * self.grid.DA / max(norm, 1e-12))
        klein_trans = float(np.sum(rho[self.grid.X > 1.25]) * self.grid.DA / max(norm, 1e-12))
        hole_frac = float(np.sum(rho_h) * self.grid.DA / max(norm, 1e-12))
        majorana_sym = (
            float(2.0 * np.sum(np.sqrt(rho_p * rho_h)) * self.grid.DA / max(norm, 1e-12))
            if hole_frac > 0.08 else 0.0
        )

        ny = self.cfg.ny
        upper = float(np.sum(rho[ny // 2 :, :]) * self.grid.DA)
        lower = float(np.sum(rho[: ny // 2, :]) * self.grid.DA)
        asymmetry = abs(upper - lower) / max(norm, 1e-12)

        return {
            "norm": norm,
            "max_density": float(np.max(rho)),
            "kinetic_energy": float(kinetic_energy),
            "interaction_energy": float(interaction_energy),
            "gravity_energy": float(gravity_energy),
            "total_energy": float(kinetic_energy + interaction_energy + gravity_energy),
            "vortex_fraction": vortex_fraction,
            "klein_transmission": klein_trans,
            "majorana_symmetry": majorana_sym,
            "beam_asymmetry": asymmetry,
        }


# ============================================================
# 5. INTEGRADOR RK4 TENSORIAL DIRECTO Y MOTOR CENTRAL
# ============================================================

class RK4Integrator:

    @staticmethod
    def step(model: PhysicalModel, state: ComplexTensor, dt: float) -> ComplexTensor:
        k1 = model.rhs(state)
        k2 = model.rhs(state + 0.5 * dt * k1)
        k3 = model.rhs(state + 0.5 * dt * k2)
        k4 = model.rhs(state + dt * k3)

        result = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Divergencia numérica detectada en RK4.")
        return result


class SimulationEngine:

    def __init__(self, model: UnifiedQuantumGravityModel, config: SimulationConfig, grid: SpectralGrid):
        self.model = model
        self.config = config
        self.grid = grid
        self.state = model.initial_state()
        self.expected_shape = (4, config.ny, config.nx)

        ModelAuditor.validate_state(self.state, self.expected_shape)
        self.history: list[dict[str, float]] = []
        self.step_count = 0

    def record(self, step: int) -> dict[str, float]:
        diag = self.model.diagnostics(self.state)
        diag["step"] = float(step)
        diag["time"] = step * self.config.dt
        self.history.append(diag)
        return diag

    def step_once(self, substeps: int = 2) -> dict[str, float]:
        for _ in range(substeps):
            self.state = RK4Integrator.step(self.model, self.state, self.config.dt)
            self.step_count += 1
        ModelAuditor.validate_state(self.state, self.expected_shape)
        return self.record(self.step_count)

    def run_benchmark(self) -> ComplexTensor:
        self.history.clear()
        self.step_count = 0
        self.record(0)

        for s in range(1, self.config.steps + 1):
            self.state = RK4Integrator.step(self.model, self.state, self.config.dt)
            self.step_count = s
            if s % 10 == 0:
                ModelAuditor.validate_state(self.state, self.expected_shape)
                self.record(s)

        return self.state


# ============================================================
# 6. ADAPTADOR INTERACTIVO NATIVO PARA GOOGLE COLAB
# ============================================================

MISSION_CATALOG = {
    "M1: Estrella de Bosones": (
        "Sube Gravedad (G >= 5.5) para fusionar los paquetes y superar ρ_max > 0.28.",
        "max_density", 0.28,
        {"gravity": 2.0, "coupling": 0.0, "c_dirac": 0.0, "alpha_rashba": 0.0, "delta_sc": 0.0, "flux_ab": 0.0, "v_klein": 0.0, "bz_zeeman": 0.0}
    ),
    "M2: Deflexion Aharonov-Bohm": (
        "Ajusta Flujo AB (Φ_B ≈ 3.0) para desfasar las franjas tras el solenoide (asimetría > 0.13).",
        "beam_asymmetry", 0.13,
        {"gravity": 0.0, "coupling": 0.0, "c_dirac": 0.0, "alpha_rashba": 0.0, "delta_sc": 0.0, "flux_ab": 0.5, "v_klein": 0.0, "bz_zeeman": 0.0}
    ),
    "M3: Vortice Quiral Rashba": (
        "Sube Rashba SOC (|α_R| >= 1.8) para generar espontáneamente > 40% de vórtice quiral ψ_↓.",
        "vortex_fraction", 0.40,
        {"gravity": 1.0, "coupling": 0.0, "c_dirac": 0.0, "alpha_rashba": 0.4, "delta_sc": 0.0, "flux_ab": 0.0, "v_klein": 0.0, "bz_zeeman": 0.0}
    ),
    "M4: Tunel de Klein (Dirac)": (
        "Hay un muro V_0=4.5. Activa Velocidad de Dirac (c_D >= 2.6) para cruzar > 26% el muro.",
        "klein_transmission", 0.26,
        {"gravity": 0.0, "coupling": 0.0, "c_dirac": 0.3, "alpha_rashba": 0.0, "delta_sc": 0.0, "flux_ab": 0.0, "v_klein": 4.5, "bz_zeeman": 0.0}
    ),
    "M5: Fermion de Majorana": (
        "Activa Superconductividad (Δ_SC >= 1.6) + Rashba (α_R >= 1.5) para simetría Majorana > 0.75.",
        "majorana_symmetry", 0.75,
        {"gravity": 0.5, "coupling": 0.0, "c_dirac": 0.0, "alpha_rashba": 1.5, "delta_sc": 0.1, "flux_ab": 0.0, "v_klein": 0.0, "bz_zeeman": 1.8}
    ),
}


def launch_colab_dashboard():
    config = SimulationConfig()
    config.validate()
    grid = SpectralGrid(config)
    model = UnifiedQuantumGravityModel(config, grid)
    engine = SimulationEngine(model, config, grid)

    # 1. Ejecutar auditoría de línea base antes de abrir el panel
    engine.run_benchmark()
    init_obs = engine.history[0]
    fin_obs = engine.history[-1]
    print("==================================================")
    print("PHYSICS SANDBOX — AUDITORÍA REFERENCE ENGINE 1.0")
    print("==================================================")
    print(f"Modelo:          {type(model).__name__}")
    print(f"Norma inicial:   {init_obs['norm']:.10f}")
    print(f"Norma final:     {fin_obs['norm']:.10f}")
    print(f"Energía inicial: {init_obs['total_energy']:.10f}")
    print(f"Energía final:   {fin_obs['total_energy']:.10f}")
    print(f"Densidad máxima: {fin_obs['max_density']:.10f}")
    print("==================================================\n")

    unlocked = {k: False for k in MISSION_CATALOG}
    out_plot = widgets.Output()

    dd_mission = widgets.Dropdown(
        options=list(MISSION_CATALOG.keys()),
        value="M1: Estrella de Bosones",
        description="Misión:",
        layout=widgets.Layout(width="330px")
    )
    s_G = widgets.FloatSlider(value=2.0, min=0.0, max=10.0, step=0.2, description="Gravedad G:")
    s_g = widgets.FloatSlider(value=0.0, min=-4.0, max=4.0, step=0.2, description="GP (g):")
    s_c = widgets.FloatSlider(value=0.0, min=0.0, max=4.0, step=0.2, description="Dirac c_D:")
    s_r = widgets.FloatSlider(value=0.0, min=-3.5, max=3.5, step=0.2, description="Rashba α_R:")
    s_d = widgets.FloatSlider(value=0.0, min=0.0, max=3.0, step=0.1, description="Superc Δ:")
    s_f = widgets.FloatSlider(value=0.0, min=-5.0, max=5.0, step=0.2, description="Flujo Φ_B:")
    s_steps = widgets.IntSlider(value=30, min=5, max=100, step=5, description="Pasos/Clic:")

    btn_step = widgets.Button(description="▶ Evolucionar", button_style="success")
    btn_reset = widgets.Button(description="↺ Reiniciar Misión", button_style="warning")
    btn_json = widgets.Button(description="💾 Sellar JSON", button_style="info")

    def render_view():
        with out_plot:
            clear_output(wait=True)
            diag = engine.history[-1] if engine.history else engine.record(0)
            rho = np.sum(np.abs(engine.state) ** 2, axis=0)
            bar_mask = 0.02 * (model.v_barrier > 0.5)
            extent = [grid.X.min(), grid.X.max(), grid.Y.min(), grid.Y.max()]

            plt.style.use("dark_background")
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

            axes[0].imshow(rho + bar_mask, origin="lower", extent=extent, cmap="inferno")
            axes[0].set_title("Densidad Cuántica Total ρ(x,y)", color="#00ffcc")
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")

            if abs(config.alpha_rashba) > 0.1 or config.c_dirac > 0.1 or config.delta_sc > 0.1:
                vort = np.abs(engine.state[1]) ** 2 + np.abs(engine.state[2]) ** 2
                axes[1].imshow(vort, origin="lower", extent=extent, cmap="magma")
                axes[1].set_title("Vórtice Quiral / Majorana (|ψ_↓|² + |η|²)", color="#ffcc00")
            else:
                phi = grid.poisson(config.mass * rho, config.gravity)
                axes[1].imshow(phi, origin="lower", extent=extent, cmap="viridis")
                axes[1].set_title("Potencial Gravitacional Φ(x,y)", color="#ffcc00")
            axes[1].set_xlabel("X")

            t_arr = [d["time"] for d in engine.history]
            axes[2].plot(t_arr, [d["total_energy"] for d in engine.history], color="#00ffcc", lw=1.8, label="E_Total")
            axes[2].plot(t_arr, [d["kinetic_energy"] for d in engine.history], "--", color="#ffcc00", lw=1.2, label="E_Cin")
            axes[2].plot(t_arr, [d["gravity_energy"] for d in engine.history], ":", color="#ff6699", lw=1.2, label="E_Grav")
            axes[2].set_title("Historial de Conservación")
            axes[2].set_xlabel("Tiempo (t)")
            axes[2].legend(loc="upper right", fontsize=8)
            axes[2].grid(True, alpha=0.2)

            m_name = dd_mission.value
            desc, m_key, thresh, _ = MISSION_CATALOG[m_name]
            val = diag[m_key]
            prog = min(val / max(thresh, 1e-6), 1.0)
            if prog >= 1.0:
                unlocked[m_name] = True

            stars = " ".join(f"{'★' if unlocked[k] else '☆'}M{i+1}" for i, k in enumerate(MISSION_CATALOG))
            print(f"🎯 {m_name} | Progreso: {int(prog*100):3d}% [{'#'*int(prog*15):<15}] | Medallas: {stars}")
            print(f"📋 Reto: {desc}")
            print(
                f"🔬 Paso: {engine.step_count:04d} | t: {diag['time']:.4f} | "
                f"Norma: {diag['norm']:.8f} | E_Total: {diag['total_energy']:.6f} | "
                f"ρ_max: {diag['max_density']:.4f} | Klein: {diag['klein_transmission']*100:.1f}% | "
                f"Majorana: {diag['majorana_symmetry']:.3f}"
            )
            plt.tight_layout()
            plt.show()

    def sync_params():
        config.gravity = float(s_G.value)
        config.coupling = float(s_g.value)
        config.c_dirac = float(s_c.value)
        config.alpha_rashba = float(s_r.value)
        config.delta_sc = float(s_d.value)
        config.flux_ab = float(s_f.value)
        model.sync_external_potentials()

    def on_mission_change(_=None):
        m_name = dd_mission.value
        _, _, _, presets = MISSION_CATALOG[m_name]
        s_G.value = presets["gravity"]
        s_g.value = presets["coupling"]
        s_c.value = presets["c_dirac"]
        s_r.value = presets["alpha_rashba"]
        s_d.value = presets["delta_sc"]
        s_f.value = presets["flux_ab"]
        config.v_klein = presets["v_klein"]
        config.bz_zeeman = presets["bz_zeeman"]
        sync_params()
        engine.state = model.initial_state(m_name)
        engine.history.clear()
        engine.step_count = 0
        engine.record(0)
        render_view()

    def on_evolve_click(_):
        sync_params()
        engine.step_once(substeps=int(s_steps.value))
        render_view()

    def on_json_click(_):
        buf = np.ascontiguousarray(engine.state).view(np.uint8)
        state_hash = hashlib.sha256(buf).hexdigest()
        manifest = {
            "timestamp": time.time(),
            "mission": dd_mission.value,
            "step": engine.step_count,
            "state_sha256": state_hash,
            "diagnostics": engine.history[-1] if engine.history else {},
        }
        with open("sandbox_colab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with out_plot:
            print(f"✅ Manifiesto guardado en 'sandbox_colab_manifest.json' | SHA-256: {state_hash[:16]}")

    dd_mission.observe(on_mission_change, names="value")
    btn_step.on_click(on_evolve_click)
    btn_reset.on_click(on_mission_change)
    btn_json.on_click(on_json_click)

    ui_top = widgets.HBox([dd_mission, s_steps, btn_step, btn_reset, btn_json])
    ui_sliders_1 = widgets.HBox([s_G, s_g, s_c])
    ui_sliders_2 = widgets.HBox([s_r, s_d, s_f])
    display(widgets.VBox([ui_top, ui_sliders_1, ui_sliders_2, out_plot]))
    on_mission_change()


# ============================================================
# 7. PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    launch_colab_dashboard()