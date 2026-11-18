#!/usr/bin/env python3
import os
import warnings
import csv
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from datetime import datetime
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import math
import time
import tensorflow as tf
import soapy
from keras import layers, Model
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.signal import CZT
from numpy.fft import fft2, ifft2

np.random.seed(42)
for _g in tf.config.list_physical_devices('GPU'):
    tf. config.experimental.set_memory_growth(_g, True)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)

from phase_cnn import (
    FFPhaseCNN, DatasetStore, WarmupCosine, build_variable_groups, update_ratio_medians,
    twin_invariant_loss, direct_phase_loss, predicted_angles, twin_permutation,
    twin_floor_deg, twin_aware_compare, REF_FRINGE_PX, fringe_geometry, center_crop,
    read_run_config, load_trained_model, predict_outer_phases,
)

class timer_class:
    def __init__(self):
        self.start_time = time.monotonic()
    def elapsed(self):
        return time.monotonic() - self.start_time
    def reset(self):
        self.start_time = time.monotonic()

Timer = timer_class()

class ArrayDesign:
    def __init__(self, side: int, shape: str):
        self.arrayvector = None
        self.arrayconstruct(side, shape)

    def arrayconstruct(self, sidelen=2, shape="hex"):
        positions_list = []
        if shape == "hex":
            total_rows = 2 * sidelen - 1
            for row in range(total_rows):
                if row < sidelen:
                    n_in_row = sidelen + row
                else:
                    n_in_row = 3 * sidelen - 2 - row
                x = (row - (sidelen - 1)) * math.sqrt(3) / 2
                for col in range(n_in_row):
                    y = col - (n_in_row - 1) / 2.0
                    positions_list.append([x, y])
        elif shape == "square":
            for row in range(sidelen):
                x = row - (sidelen - 1) / 2.0
                for col in range(sidelen):
                    y = col - (sidelen - 1) / 2.0
                    positions_list.append([x, y])
        else:
            raise ValueError("Unsupported shape. Use 'hex' or 'square'.")

        positions_array = np.array(positions_list, dtype=np.float64)
        self.arrayvector = np.column_stack((positions_array, np.zeros(positions_array.shape[0])))

    def find_center_index(self):
        """Index of the array element closest to (0,0) — the reference beam."""
        dists = np.linalg.norm(self.arrayvector[:, :2], axis=1)
        return int(np.argmin(dists))

class TurbulenceModeler:
    def __init__(self, wavelength: float = 1064e-9, 
    outer_scale: float = np.inf, grid_size: int = 1024, screen_physical_size: float = 2.0, grnd_speed = 5.0, r500 = 0.05):
        self.r0_check = False
        self.phase_catalog = None
        self.n_screens = 16
        self.r500 = r500
        self.wavelen = wavelength
        self.outer_scale = outer_scale
        self.grid_size = grid_size
        self.pixel_scale = screen_physical_size / grid_size
        self.grnd_speed = grnd_speed
        self.v_rms = self.rms_velocity(0.0, 20.0)
        self.b_val = optimize.brentq(lambda B: self.hv_fried_param(0, 20000, b_val = B) - self.r500, 0, 1e4)

    def rms_velocity(self, h1_km: float, h2_km: float):
        h = np.linspace(h1_km, h2_km, 2001)
        V = self.grnd_speed + 30 * np.exp(-((h - 9.4) / 4.8) ** 2)
        return np.sqrt(np.trapz(y=V ** 2, x=h) / (h2_km - h1_km))
    
    def cn2(self, h_km, v_rms, b_val = None):
        b = self.b_val if b_val is None else b_val
        return (5.94e-23 * h_km ** 10 * (v_rms / 27) ** 2 * np.exp(-h_km)
               + self.b_val * 2.7e-16 * np.exp(-2 * h_km / 3))

    def hv_fried_param(self, height1, height2, wavelen = 500e-9, b_val=None):
        h_km = np.linspace(height1 / 1000, height2 / 1000, num=100)
        cn2 = self.cn2(h_km, self.v_rms, b_val)
        integral = np.trapz(y=cn2, x=h_km * 1000)          # dh in metres
        k_wavenum = 2 * np.pi / wavelen
        return (0.423 * k_wavenum ** 2 * integral) ** (-3 / 5)
    
    def layer_r0(self, h1, h2):
        return self.hv_fried_param(h1, h2, self.b_val, self.wavelen)

    def phase_screen_gen(self, height1, height2):
        r0 = self.layer_r0(height1, height2)
        phase_screens = soapy.atmosphere.makePhaseScreens(
            nScrns=1,
            r0=r0,
            N=self.grid_size,
            pxlScale=self.pixel_scale,
            L0=self.outer_scale,
            l0=0.01,
            returnScrns=True,
            SH=True
        )

        return phase_screens[0], r0

    def all_screen_gen(self, gen_height = 20000):
        r0_total = 0
        nx = self.grid_size
        self.phase_catalog = np.zeros((self.n_screens, nx, nx))
        screen_heights = np.linspace(0, gen_height, num = (self.n_screens+1))
        self.v_rms = self.rms_velocity(0.0, gen_height / 1000)
        for i in range(self.n_screens):
            phase_screen, r0 = self.phase_screen_gen(height1 = screen_heights[i], 
            height2 = screen_heights[i+1])
            r0_total += r0 ** (-5/3)
            self.phase_catalog[i] = phase_screen
        if self.r0_check is False:
            print(r0_total ** (-3/5))
            self.r0_check = True

class BeamPerturbator:
    def __init__(self, ssl, targ_pos: list, array_des: ArrayDesign, grid_length: np.float64 = 1024, Ideal = True, r500 = 0.05):
        self.waist = self.atmo_height = self.apert_radi = self.wavelength = self.spatial_samp_length = self.n_screens = np.float64(0)
        self.chan_amps = self.turb_model = self.base_fields = None
        self.phase_screens = self.apply_turb = self.h_layers = None
        self.target_position = targ_pos
        self.rel_array = array_des
        self.center_idx = self.rel_array.find_center_index()
        self.grid_length = grid_length
        self.chan_rel_phases = np.zeros(self.rel_array.arrayvector.shape[0])
        self.set_beam_elements(ideal = Ideal, r500 = r500, ssl = ssl)

    def set_beam_elements(self,
                          clip_ratio: np.float64 = 0.6,
                          apert_radi: np.float64 = 5e-2,
                          wavelength: np.float64 = 1064e-9,
                          init_amp: np.complex128 = 10,
                          ssl: np.float64 = 4e-4,
                          ideal: bool = True,
                          gaussian: bool = True,
                          r500 = 0.05):
        self.chan_amps = np.full(self.rel_array.arrayvector.shape[0], init_amp)
        self.r500 = r500
        self.atmo_height = 20000
        self.base_fields = []
        self.waist = apert_radi * clip_ratio
        self.apert_radi = apert_radi
        self.wavelength = wavelength
        self.spatial_samp_length = ssl
        self.gauss_power = gaussian

        #Element spacing
        self.element_space = 2 * self.apert_radi
        self.element_positions = self.rel_array.arrayvector * self.element_space

        #initializing turbulence module ONLY if non-ideal prop
        self.phase_screens = None
        self.h_layers = []
        self.apply_turb = not ideal
        if not ideal:
            self.turb_model = TurbulenceModeler(wavelength = wavelength, grid_size = self.grid_length,
                                                screen_physical_size=self.grid_length * self.spatial_samp_length,
                                                r500 = r500)
            self.n_screens = self.turb_model.n_screens
            self.generate_turbulence()
            self.generate_transfer()
        
    def generate_turbulence(self):
        self.turb_model.all_screen_gen()
        self.phase_screens = self.turb_model.phase_catalog.copy()

    def generate_transfer(self):
        dim = self.grid_length
        fx = np.fft.fftfreq(dim, d=self.spatial_samp_length)
        fy = np.fft.fftfreq(dim, d=self.spatial_samp_length)
        FX, FY = np.meshgrid(fx, fy)
        k = 2*np.pi/self.wavelength
        screen_heights = np.linspace(0, min(self.target_position[2], self.atmo_height), num=self.n_screens+1)
        self.h_layers = []
        z_current = 0.0
        for h in screen_heights[1:]:
            dz = h - z_current
            H = np.exp(1j * k * dz) * np.exp(-1j * np.pi * self.wavelength * dz * (FX ** 2 + FY ** 2))
            self.h_layers.append(H)
            z_current = h

    def input_e_field_base(self):
        self.base_fields = []
        # relevant constants
        k = 2 * np.pi / self.wavelength
        ssl = self.spatial_samp_length
        # aperture grid creation
        pos_apert = self.element_positions
        num_apertures = pos_apert.shape[0]
        # output grid creation
        dim_y, dim_x = [self.grid_length, self.grid_length]
        x = (np.arange(dim_x) - dim_x / 2) * ssl
        y = (np.arange(dim_y) - dim_y / 2) * ssl
        X, Y = np.meshgrid(x, y)
        x_f, y_f, z_f = self.target_position
        x_n = pos_apert[:, 0]
        y_n = pos_apert[:, 1]
        # vectorized steering phases
        dx = x_f - x_n
        dy = y_f - y_n
        R_target = np.sqrt(dx ** 2 + dy ** 2 + z_f ** 2)
        sin_theta_x = dx / R_target
        sin_theta_y = dy / R_target
        piston = -k * (dx ** 2 + dy ** 2) / (R_target + z_f)
        # summing over contributions from each aperture
        for n in range(num_apertures):
            X_local = X - x_n[n]
            Y_local = Y - y_n[n]
            r = np.sqrt(X_local ** 2 + Y_local ** 2)
            aperture = (r <= self.apert_radi).astype(np.float64)
            if self.gauss_power:
                amp_prof = aperture * np.exp(-r ** 2 / self.waist ** 2)
            else:
                amp_prof = aperture
            phi_steer = k * (sin_theta_x[n] * X_local + sin_theta_y[n] * Y_local)
            total_phase = phi_steer + piston[n]
            self.base_fields.append(amp_prof * self.chan_amps[n] * np.exp(1j * total_phase))
        return x, y

def optimal_phases_from_M(M, n_iter, center_idx, tol=1e-6):
    """Exactly the optimizer in BeamPerturbator.compute_optimal_phases,
    applied to a precomputed bucket matrix M (n_elem x n_elem)."""
    n_elem = M.shape[0]
    eigvals, eigvecs = np.linalg.eigh(M)
    c = np.exp(1j * np.angle(eigvecs[:, -1]))
    c_ones = np.ones(n_elem, dtype=np.complex128)
    power_ones = np.real(c_ones.conj() @ M @ c_ones)
    prev_power = -np.inf
    power = np.real(c.conj() @ M @ c)
    n_used = 0
    for i in range(n_iter):
        c = np.exp(1j * np.angle(M @ c))
        power = np.real(c.conj() @ M @ c)
        n_used = i + 1
        if power - prev_power < tol * abs(power):
            break
        prev_power = power
    if power_ones > power:
        c = c_ones
        power = power_ones
    c = c * np.exp(-1j * np.angle(c[center_idx]))
    return np.angle(c), power, n_used
 
diversity_delta = np.pi / 2
def diversity_phases(element_positions, delta = diversity_delta):
    pos = np.asarray(element_positions)[:,:2]
    on_axis = (np.abs(pos[:,0]) < 1e-9) & (np.abs(pos[:,1]) > 1e-9)
    ph = np.zeros(len(pos), np.float32)
    ph[on_axis] = delta
    return ph

def slab_wind_speeds(sim, n_slabs):
    """Cn^2-weighted 5/3-moment wind speed of each generation slab [m/s].
    Uses the same Cn^2 expression as TurbulenceModeler.hv_fried_param and the
    same wind profile as TurbulenceModeler.rms_velocity."""
    tm = sim.turb_model
    S_total = sim.n_screens
    gen_h = np.linspace(0, sim.atmo_height, S_total + 1) / 1000          # km
    v_rms = tm.rms_velocity(0.0, sim.atmo_height / 1000)
    v = []
    for i in range(n_slabs):
        h = np.linspace(gen_h[i], gen_h[i + 1], 2001)
        cns = tm.cn2(h, v_rms)
        V = tm.grnd_speed + 30 * np.exp(-((h - 9.4) / 4.8) ** 2)
        v.append((np.trapz(cns * V ** (5 / 3), h) / np.trapz(cns, h)) ** (3 / 5))
    return np.array(v)

class GPUDataGenerator:
    """GPU port of: TurbulenceModeler screens (aotools ft_sh_phase_screen,
    same PSD / subharmonics / scaling), BeamPerturbator.fresnel_propagation
    (turbulent branch) and fresnel_czt (CZT written as two matmuls).
 
    Uses linearity: the 7 element far fields are propagated once; the
    zero-phase image is |sum F_n|^2 and the optimized image is
    |sum e^{i phi_n} F_n|^2 -- identical to re-propagating the summed field."""
 
    def __init__(self, sim, seed=0, l0=0.01):
        if not sim.apply_turb:
            raise ValueError("sim must be built with Ideal=False")
        if not sim.base_fields:
            sim.input_e_field_base()
        tm = sim.turb_model
        N = int(sim.grid_length)
        ssl = float(sim.spatial_samp_length)
        if abs(tm.pixel_scale - ssl) > 1e-9:
            raise ValueError(f"screen pixel {tm.pixel_scale} != field pixel {ssl}: "
                             "screen_physical_size fix missing")
        self.sim, self.N, self.L = sim, N, len(sim.base_fields)
        self.center_idx = sim.center_idx
        x_f, y_f, z = sim.target_position
        k = 2 * np.pi / sim.wavelength
 
        # ---- layer bookkeeping: same as fresnel_propagation ----
        S_total = sim.n_screens
        self.S = min(S_total, int(round(z / 500)))
        z_atmo = min(z, sim.atmo_height)
        self.z_rem = z - np.linspace(0, z_atmo, self.S + 1)[-1]
        if self.z_rem <= 0:
            raise ValueError("target inside the atmosphere is not supported here")
        self.H = tf.constant(np.stack(sim.h_layers[:self.S]).astype(np.complex64))
 
        # ---- per-slab r0: same as all_screen_gen / phase_screen_gen ----
        gen_h = np.linspace(0, sim.atmo_height, S_total + 1)
        tm.v_rms = tm.rms_velocity(0.0, sim.atmo_height / 1000)
        r0 = np.array([tm.layer_r0(gen_h[i], gen_h[i + 1])
                       for i in range(S_total)])[:self.S]
        self.r0 = r0
        self.r0_scale = tf.constant((r0 ** (-5 / 6)).astype(np.float32))   # sqrt(r0^-5/3)
 
        # ---- screen PSDs (aotools ft_phase_screen / ft_sh_phase_screen) ----
        delta, L0 = tm.pixel_scale, tm.outer_scale
        fm = 5.92 / l0 / (2 * np.pi)
        f0 = 0.0 if np.isinf(L0) else 1.0 / L0
        del_f = 1.0 / (N * delta)
        fx = np.arange(-N / 2, N / 2) * del_f
        FX, FY = np.meshgrid(fx, fx)
        f = np.hypot(FX, FY)
        with np.errstate(divide='ignore'):
            psd = 0.023 * np.exp(-(f / fm) ** 2) / ((f ** 2 + f0 ** 2) ** (11 / 6))
        psd[N // 2, N // 2] = 0
        self.amp_hi = tf.constant((np.sqrt(psd) * del_f).astype(np.float32))
        coords = np.arange(-N / 2, N / 2) * delta
        D = N * delta
        U, amp_lo = [], []
        for p in range(1, 4):
            dfp = 1 / (3 ** p * D)
            f3 = np.arange(-1, 2) * dfp
            F3x, F3y = np.meshgrid(f3, f3)
            f3r = np.hypot(F3x, F3y)
            with np.errstate(divide='ignore'):
                psd3 = 0.023 * np.exp(-(f3r / fm) ** 2) / ((f3r ** 2 + f0 ** 2) ** (11 / 6))
            psd3[1, 1] = 0
            amp_lo.append(np.sqrt(psd3) * dfp)
            U.append(np.exp(1j * 2 * np.pi * f3[:, None] * coords[None, :]))   # [3, N]
        self.amp_lo = tf.constant(np.stack(amp_lo).astype(np.float32))        # [3,3,3]
        self.U = tf.constant(np.stack(U).astype(np.complex64))                 # [3,3,N]
 
        # ---- fresnel_czt as matmuls ----
        x1 = (np.arange(N) - N / 2) * ssl
        X1, Y1 = np.meshgrid(x1, x1)
        Q1 = np.exp(1j * k * (X1 ** 2 + Y1 ** 2) / (2 * self.z_rem))
        ep = sim.element_positions
        d_array = (ep[:, 0].max() - ep[:, 0].min()) + 2 * sim.apert_radi
        out_window = 20 * sim.wavelength * self.z_rem / d_array
        x2 = (np.arange(N) - N // 2) * (out_window / N)
        fk = x2 / (sim.wavelength * self.z_rem)
        fk = fk[0] + np.arange(N) * (fk[1] - fk[0])          # scipy CZT's grid
        Mczt = np.exp(-2j * np.pi * fk[:, None] * (np.arange(N) * ssl)[None, :])
        X2, Y2 = np.meshgrid(x2, x2)
        Q2 = np.exp(1j * k * (X2 ** 2 + Y2 ** 2) / (2 * self.z_rem))
        pref = np.exp(1j * k * self.z_rem) / (1j * sim.wavelength * self.z_rem)
        self.base = tf.constant(np.stack(sim.base_fields).astype(np.complex64))
        self.Q1 = tf.constant(Q1.astype(np.complex64))
        self.Mczt = tf.constant(Mczt.astype(np.complex64))
        self.P = tf.constant((pref * Q2 * ssl ** 2).astype(np.complex64))
        self.x_out = x2
 
        # ---- bucket mask: same as compute_optimal_phases ----
        cx = np.argmin(np.abs(x2 - x_f))
        cy = np.argmin(np.abs(x2 - y_f))
        Yg, Xg = np.mgrid[0:N, 0:N]
        self.mask = np.sqrt((Xg - cx) ** 2 + (Yg - cy) ** 2) <= N / 40
        self.mask_idx = tf.constant(np.flatnonzero(self.mask), dtype=tf.int64)
 
        self.rng = tf.random.Generator.from_seed(seed)
 
    # ------------------------------------------------------------------
    @tf.function
    def generate_screens(self, B, r0_scale):
        """[B, S, N, N] float32 phase screens in radians. r0_scale[n] = r0_n^(-5/6)."""
        N = self.N
        out = []
        for n in range(self.S):
            g = self.rng.normal([B, 2, N, N])
            cn = tf.complex(g[:, 0], g[:, 1]) * tf.cast(self.amp_hi * r0_scale[n], tf.complex64)
            hi = tf.math.real(tf.signal.ifftshift(
                tf.signal.ifft2d(tf.signal.fftshift(cn, axes=(-2, -1))), axes=(-2, -1))) * float(N * N)
            lo = tf.zeros([B, N, N], tf.complex64)
            for p in range(3):
                g3 = self.rng.normal([B, 2, 3, 3])
                c3 = tf.complex(g3[:, 0], g3[:, 1]) * tf.cast(self.amp_lo[p] * r0_scale[n], tf.complex64)
                lo += tf.einsum('ir,bij,jc->brc', self.U[p], c3, self.U[p])
            lo = tf.math.real(lo)
            lo -= tf.reduce_mean(lo, axis=[1, 2], keepdims=True)
            out.append(hi + lo)
        return tf.stack(out, axis=1)
 
    @tf.function
    def far_fields(self, screens):
        """screens [B, S, N, N] -> element far fields [B, L, N, N] and bucket matrices [B, L, L]."""
        B = tf.shape(screens)[0]
        f = tf.tile(self.base[None], [B, 1, 1, 1])
        for n in range(self.S):
            f = tf.signal.ifft2d(tf.signal.fft2d(f) * self.H[n])
            s = screens[:, n][:, None]
            f = f * tf.complex(tf.cos(s), tf.sin(s))
        f = f * self.Q1
        f = tf.einsum('blrn,kn->blrk', f, self.Mczt)          # CZT along x (axis 1 in numpy)
        f = tf.einsum('jr,blrk->bljk', self.Mczt, f)          # CZT along y (axis 0 in numpy)
        F = f * self.P
        E = tf.gather(tf.reshape(F, [B, self.L, -1]), self.mask_idx, axis=2)
        M = tf.einsum('bip,bjp->bij', tf.math.conj(E), E)
        return F, M
 
    @tf.function
    def images(self, F, phases):
        """phases [B, L] float32 -> |sum_n e^{i phi_n} F_n|^2, [B, N, N]."""
        ph = tf.complex(tf.cos(phases), tf.sin(phases))
        return tf.abs(tf.einsum('bl,bljk->bjk', ph, F)) ** 2
    
    def set_wind(self, directions='aligned', seed=0):
        """Wind velocity per layer, [S, 2] m/s as (vx, vy).
        directions: 'aligned' = every layer blows along +x; 'random' = an
        independent uniform azimuth per layer; or an array of S angles (rad).
        Returns tau0 from 6.88 sum_n (v_n tau / r0_n)^(5/3) = 1."""
        speed = slab_wind_speeds(self.sim, self.S)
        if isinstance(directions, str) and directions == 'aligned':
            ang = np.zeros(self.S)
        elif isinstance(directions, str) and directions == 'random':
            ang = np.random.default_rng(seed).uniform(0, 2 * np.pi, self.S)
        else:
            ang = np.asarray(directions, dtype=np.float64)
        self.wind_speed, self.wind_angle = speed, ang
        self.wind = np.stack([speed * np.cos(ang), speed * np.sin(ang)], axis=1)
        dz = min(self.sim.target_position[2], self.sim.atmo_height) / len(self.sim.phase_screens)
        self.layer_heights = (np.arange(self.S) + 1) * dz   # screen n sits after n+1 propagation steps
        N, delta = self.N, self.sim.turb_model.pixel_scale
        fx = np.arange(-N / 2, N / 2) / (N * delta)
        FX, FY = np.meshgrid(fx, fx)                         # FX varies along columns (x)
        self.FXg = tf.constant(FX.astype(np.float32))
        self.FYg = tf.constant(FY.astype(np.float32))
        D = N * delta
        self.F3 = tf.constant(np.stack([np.arange(-1, 2) / (3 ** p * D) for p in range(1, 4)]
                                       ).astype(np.float32))  # [3 levels, 3]
        self.screen_period = D
        return (6.88 * np.sum((speed / self.r0) ** (5 / 3))) ** (-3 / 5)
 
    def max_time_s(self):
        """Longest sequence before the fastest layer's FFT part repeats."""
        return self.screen_period / self.wind_speed.max()
 
    def draw_screen_coeffs(self, B):
        """Random coefficients for B realizations, drawn in the same order as
        generate_screens. Returns hi [B, S, N, N] and lo [B, S, 3, 3, 3] complex64."""
        N = self.N
        hi, lo = [], []
        for n in range(self.S):
            g = self.rng.normal([B, 2, N, N])
            hi.append(tf.complex(g[:, 0], g[:, 1]) * tf.cast(self.amp_hi * self.r0_scale[n], tf.complex64))
            levels = []
            for p in range(3):
                g3 = self.rng.normal([B, 2, 3, 3])
                levels.append(tf.complex(g3[:, 0], g3[:, 1])
                              * tf.cast(self.amp_lo[p] * self.r0_scale[n], tf.complex64))
            lo.append(tf.stack(levels, axis=1))
        return tf.stack(hi, axis=1), tf.stack(lo, axis=1)
 
    @tf.function
    def screens_from_coeffs(self, hi_c, lo_c, shift, remove_mean=True):
        """Screens [B, S, N, N] (rad) with layer n translated by shift[n] = (dx, dy) metres.
        shift = wind * t gives frozen flow at time t. remove_mean=True matches
        generate_screens; the mean is a piston common to all elements and never
        changes the images or bucket power. Use False only for structure-function
        checks."""
        N = self.N
        two_pi = 2 * np.pi
        shift = tf.cast(shift, tf.float32)
        out = []
        for n in range(self.S):
            dx, dy = shift[n, 0], shift[n, 1]
            ramp = tf.exp(tf.complex(tf.zeros_like(self.FXg), -two_pi * (self.FXg * dx + self.FYg * dy)))
            cn = hi_c[:, n] * ramp
            hi = tf.math.real(tf.signal.ifftshift(
                tf.signal.ifft2d(tf.signal.fftshift(cn, axes=(-2, -1))), axes=(-2, -1))) * float(N * N)
            lo = tf.zeros(tf.shape(hi), tf.complex64)
            for p in range(3):
                f3 = self.F3[p]
                ph = -two_pi * (f3[:, None] * dy + f3[None, :] * dx)          # [i (y freq), j (x freq)]
                c3 = lo_c[:, n, p] * tf.exp(tf.complex(tf.zeros_like(ph), ph))
                lo += tf.einsum('ir,bij,jc->brc', self.U[p], c3, self.U[p])
            lo = tf.math.real(lo)
            if remove_mean:
                lo -= tf.reduce_mean(lo, axis=[1, 2], keepdims=True)
            out.append(hi + lo)
        return tf.stack(out, axis=1)
 
    # ------------------------------------------------------------------
    def run_batch(self, B, n_iter=100, screens=None, div_phases=None):
        """Returns list of per-realization dicts matching the CPU generator's outputs."""
        if screens is None:
            screens = self.generate_screens(tf.constant(B), self.r0_scale)
        F, M = self.far_fields(screens)
        M = M.numpy().astype(np.complex128)
        zeros = tf.zeros([B, self.L], tf.float32)
        init_imgs = self.images(F, zeros).numpy()
        div_imgs = None
        if div_phases is not None:
            div = tf.tile(tf.constant(np.asarray(div_phases,np.float32)[None]), [B,1])
            div_imgs = self.images(F, div).numpy()
        results, phases = [], np.zeros((B, self.L), np.float32)
        for b in range(B):
            Mb = 0.5 * (M[b] + M[b].conj().T)                 # remove float32 asymmetry
            best, power, n_used = optimal_phases_from_M(Mb, n_iter, self.center_idx)
            phases[b] = best
            results.append({'best_params': best, 'bucket_power': power, 'iterations': n_used})
        opt_imgs = self.images(F, tf.constant(phases)).numpy()
        for b in range(B):
            results[b]['init_image'] = init_imgs[b]
            results[b]['opt_image'] = opt_imgs[b]
            results[b]['init_intensity'] = float(np.mean(init_imgs[b][self.mask]))
            results[b]['best_intensity'] = float(np.mean(opt_imgs[b][self.mask]))
            if div_imgs is not None:
                results[b]['div_image'] = div_imgs[b]
        return results, screens
 
def preprocess_images(imgs, norm_scale):
    """Same transform as make_dataset's preprocess. imgs [n, N, N] (one image) or
    [n, N, N, C] (zero-phase + diversity); norm_scale has one value per channel."""
    imgs = np.asarray(imgs, dtype=np.float32)
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    ns = np.atleast_1d(np.asarray(norm_scale, dtype=np.float32))
    return np.log1p(np.clip(imgs, 0.0, ns) / ns * 1000.0)

def classify(name):
    n = name.lower()
    if 'gamma' in n or 'beta' in n:
        return 'norm_scale_shift'
    if 'kernel' in n:
        return 'conv_or_dense_weight'
    if 'bias' in n:
        return 'bias'
    return 'other'

def circular_phase_loss(y_true, y_pred):
    """y_true: [B, L] target phases in radians.
       y_pred: [B, L, 2] predicted unit vectors (cos, sin)."""
    target = tf.stack([tf.cos(y_true), tf.sin(y_true)], axis=-1)       # [B, L, 2]
    cos_d = tf.reduce_sum(target * y_pred, axis=-1)                    # [B, L] = cos(error)
    return tf.reduce_mean(1.0 - cos_d)

def interp_traces(t_coarse, X, t):
    """X [K, ...] sampled at uniform times t_coarse [K] -> X(t) by linear interpolation."""
    dt = t_coarse[1] - t_coarse[0]
    s = min(max((t - t_coarse[0]) / dt, 0.0), len(t_coarse) - 1 - 1e-9)
    k = int(s)
    w = s - k
    return (1 - w) * X[k] + w * X[k + 1]
 
 
def ou_step(x, dt, tau, sigma, rng):
    """One exact step of an Ornstein-Uhlenbeck process: rms sigma, correlation time tau."""
    if sigma == 0:
        return x
    a = np.exp(-dt / tau)
    return x * a + sigma * np.sqrt(1 - a * a) * rng.standard_normal(x.shape)
 
 
def bucket_power_batch(M, phases):
    """M [B, L, L] complex, phases [..., B, L] (rad) -> Re(c^H M c) [..., B]."""
    c = np.exp(1j * phases)
    return np.real(np.einsum('...bi,bij,...bj->...b', c.conj(), M, c))
 
 
def camera_frames(clean, kappa, sigma_read, rng):
    """clean [b, H, W, C] simulated intensity -> noisy frame in simulation units:
    Poisson(kappa * I) + Normal(0, sigma_read), divided back by kappa. Same camera
    model as photon_budget (1x1 binning, no background)."""
    counts = rng.poisson(kappa * clean.astype(np.float64)).astype(np.float64)
    if sigma_read > 0:
        counts += rng.normal(0.0, sigma_read, counts.shape)
    return (counts / kappa).astype(np.float32)

def simulate_spgd(trace, cfg, rng):
    t_c, M_c, Popt_c, v_c = trace['t'], trace['M'], trace['P_opt'], trace['v']
    outer = trace['outer']
    B, L = M_c.shape[1], M_c.shape[2]
    T = t_c[-1]
    gains = np.repeat(np.asarray(cfg['gains'], float), len(cfg['dithers']))
    dithers = np.tile(np.asarray(cfg['dithers'], float), len(cfg['gains']))
    G = len(gains)
    gam, dlt = gains[:, None, None], dithers[:, None, None]
    t_meas = 0.5 / cfg['f_iter']                       # two measurements per iteration
    n_half = int(T / t_meas)
    alpha = 1.0 if np.isinf(cfg['f_act']) else 1 - np.exp(-2 * np.pi * cfg['f_act'] * t_meas)
    R = cfg['act_range']
    tau_psi = 1 / (2 * np.pi * cfg['f_psi']) if cfg['f_psi'] > 0 else np.inf
    eps_rx = 1 + cfg['sigma_cal'] * rng.standard_normal((B, L))
    eps_tx = 1 + cfg['sigma_cal'] * rng.standard_normal((B, L))
    eps_rx[:, [i for i in range(L) if i not in outer]] = 1
    eps_tx[:, [i for i in range(L) if i not in outer]] = 1
    psi = cfg['sigma_psi'] * rng.standard_normal((B, L)) if cfg['sigma_psi'] > 0 else np.zeros((B, L))
    T_ret_c = np.sum(np.abs(v_c) ** 2, axis=2)          # [K, B]
    k_d = int(cfg['delay_iter'])
    scale = cfg['phi_rate'] * t_meas / cfg['J_ref']     # mean photons per unit metric per sample
 
    def full(x):                                        # [..., B, L-1] outer -> [..., B, L]
        out = np.zeros(x.shape[:-1] + (L,))
        out[..., outer] = x
        return out
 
    def measure(J):
        n = rng.poisson(np.maximum(J, 0) * scale).astype(float)
        if cfg['sigma_det'] > 0:
            n += rng.normal(0.0, cfg['sigma_det'], n.shape)
        return n
 
    u = np.zeros((G, B, L - 1))                         # SPGD's commanded mean phases
    a = np.zeros((G, B, L - 1))                         # what the actuators actually apply
    s = np.ones((G, B, L - 1))
    J_plus = np.zeros((G, B))
    hist = []                                           # (J+, J-, s) per iteration, for the delay
    ratio_spgd = np.zeros((G, n_half))
    ratio_nn = np.zeros(n_half)
    ratio_zero = np.zeros(n_half)
    steady_spgd = np.zeros((G, B))
    steady_nn = np.zeros(B)
    n_steady = 0
    nn = trace.get('nn')
    for h in range(n_half):
        t = h * t_meas
        M = interp_traces(t_c, M_c, t)
        P_opt = interp_traces(t_c, Popt_c, t)
        psi = ou_step(psi, t_meas, tau_psi, cfg['sigma_psi'], rng)
        if h % 2 == 0:
            s = rng.choice([-1.0, 1.0], size=(G, B, L - 1))
            cmd = u + dlt * s
        else:
            cmd = u - dlt * s
        a = a + alpha * (cmd - a)
 
        # ---- the metric SPGD measures ----
        if cfg['mode'] == 'beacon':
            J = bucket_power_batch(M, eps_rx * full(a))
            tx = eps_tx * full(u) + psi                 # transmit copies the undithered setting
        else:
            tx = eps_tx * full(a) + psi                 # the transmit itself carries the dither
            v = interp_traces(t_c, v_c, t)
            t_ret = t + cfg['t_rt']
            T_ret = interp_traces(t_c, T_ret_c, t_ret) if t_ret <= T else np.zeros(B)
            J = np.abs(np.einsum('gbl,bl->gb', np.exp(1j * tx), v)) ** 2 * T_ret
        Jm = measure(J)
        if h % 2 == 0:
            J_plus = Jm
        else:
            hist.append((J_plus, Jm, s))
            m = len(hist) - 1 - k_d                     # measurement that has arrived by now
            if m >= 0:
                Jp, Jn, s_old = hist[m]
                tot = Jp + Jn
                g = np.where(tot > 0, (Jp - Jn) / np.where(tot > 0, tot, 1.0), 0.0)
                u = u + gam * g[..., None] * s_old
                if np.isfinite(R):                      # actuator stroke reset by 2 pi
                    u = np.where(u > R, u - 2 * np.pi, np.where(u < -R, u + 2 * np.pi, u))
                hist[m] = None                          # used; free it
 
        # ---- what the transmit actually delivers ----
        P_tx = bucket_power_batch(M, tx)                # [G, B]
        ratio_spgd[:, h] = (P_tx / P_opt).mean(1)
        ratio_zero[h] = (bucket_power_batch(M, psi) / P_opt).mean()
        if nn is not None:
            f_idx = int(np.floor((t - trace['nn_latency']) / trace['t_frame'] + 1e-9))
            ph_nn = full(nn[min(f_idx, len(nn) - 1)]) if f_idx >= 0 else np.zeros((B, L))
            P_nn = bucket_power_batch(M, eps_tx * ph_nn + psi)
            ratio_nn[h] = (P_nn / P_opt).mean()
        if t >= cfg['t_burn']:
            steady_spgd += P_tx / P_opt
            if nn is not None:
                steady_nn += P_nn / P_opt
            n_steady += 1
    n_steady = max(n_steady, 1)
    return {'t': np.arange(n_half) * t_meas, 'gains': gains, 'dithers': dithers,
            'ratio_spgd': ratio_spgd, 'ratio_nn': ratio_nn if nn is not None else None,
            'ratio_zero': ratio_zero, 'steady_spgd': steady_spgd / n_steady,
            'steady_nn': steady_nn / n_steady if nn is not None else None}

###  Code Beginning
if __name__ == "__main__":
    print(f"Time to import: {Timer.elapsed()}")
    choice = input("""What test are you running?
    -h5     -train data gpu     -training 
    -twin_check   -val_outliers -time_eval
    -nn_bucket_eval     -ckpt_check -photon_budget -spgd_eval\n""")

#looking at testing data
if choice == "h5":
    example_num = int(input("What example do you want to view? \n"))
    filename = "data/testtrainingdata.h5"
    h5_path = Path(filename)
    catalog_path = h5_path.with_suffix('.catalog.csv')
    with h5py.File(h5_path, 'r') as f:
        image = f['images'][example_num]
        opt_image = f['opt_images'][example_num]
        label = f['labels'][example_num]
    catalog = pd.read_csv(catalog_path)
    row = catalog[catalog['index'] == example_num].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    im1 = ax1.imshow(image, cmap='plasma', origin='lower')
    ax1.set_title(f"Example {example_num} — Pre-Optimization")
    plt.colorbar(im1, ax=ax1, label='Intensity')

    im2 = ax2.imshow(opt_image, cmap='plasma', origin='lower')
    ax2.set_title(f"Example {example_num} — Post-Optimization")
    plt.colorbar(im2, ax=ax2, label='Intensity')

    print(f"index:          {example_num}")
    print(f"label (params): {list(map(lambda x: f'{x:.2e}', label))}")
    print(f"r500:     {row['r500']}")
    print(f"init_intensity: {row['init_intensity']:.2e}")
    print(f"best_intensity: {row['best_intensity']:.2e}")
    print(f"iterations:     {row['iterations']}")
    print(f"target:         ({row['target_x']}, {row['target_y']})")
    plt.tight_layout()
    plt.savefig(f'outputs/example_{example_num}_unlog.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

#training CNN
if choice == "training":
    # ---- run identity: CHANGE FOR EVERY NEW RUN ---------------------------
    RUN_LABEL = 'training_div_v1'
    LOG_PATH  = f'outputs/train_logs/{RUN_LABEL}.csv'
    CKPT_DIR  = f'outputs/ckpts/{RUN_LABEL}'
 
    # ---- config ------------------------------------------------------------
    EPOCHS       = int(input("How many epochs of training will you run?\n"))
    BATCH_SIZE   = 32
    MAIN_LR      = 2.0e-3 
    NORM_LR      = 3.0e-5
    WARMUP_STEPS = 100
    MIN_LR_FRAC  = 0.05
    PATIENCE     = None          # epochs without val improvement before stopping; None disables
    CKPT_EVERY   = 10          # epochs between periodic snapshots
    PRINT_EVERY  = 0          # steps within an epoch; 0 disables
 
    array = ArrayDesign(2, 'hex')
    center_idx = array.find_center_index()
    outer_positions = np.delete(array.arrayvector, center_idx, axis=0)
    L = outer_positions.shape[0] 
    twin_perm = tf.constant(twin_permutation(outer_positions), tf.int32)

    # ---- data --------------------------------------------------------------
    DATA_PATH = 'data/divtrainingdata.h5'
    CROP = 512
    train_ds, val_ds, grid_size, norm_scale = DatasetStore.make_dataset(
        DATA_PATH, batch_size=BATCH_SIZE, crop=CROP
    )
    norm_scale = np.atleast_1d(norm_scale)
    C_IN = len(norm_scale)
    USE_TWIN_LOSS = (C_IN == 1)
    with h5py.File(DATA_PATH, 'r') as f:
        stored_N = f['images'].shape[1]
    SCALE = max(1, int(round(fringe_geometry(array, grid_N=stored_N, verbose=False)['fringe_nearest_px']/REF_FRINGE_PX)))

    steps_per_epoch = int(train_ds.cardinality().numpy())
    val_labels = np.concatenate([lab.numpy() for _, lab in val_ds], axis=0)
    n_val = val_labels.shape[0]
    if steps_per_epoch == 0 or n_val == 0:
        raise ValueError(f"Not enough data: {steps_per_epoch} train batches, "
                         f"{n_val} val examples. Generate more realizations.")
 
    # "Do nothing" reference: error if every correction phase were predicted as 0.
    # The model must beat this, not just the 90° random baseline, to be useful.
    wrapped = np.arctan2(np.sin(val_labels), np.cos(val_labels))
    zero_baseline_deg = float(np.degrees(np.mean(np.abs(wrapped))))
    print(f"twin floor for standard MAE: "
          f"{twin_floor_deg(val_labels, twin_perm.numpy()):.2f}°")

    print(f"train: {steps_per_epoch} batches x {BATCH_SIZE} = "
          f"{steps_per_epoch * BATCH_SIZE} examples/epoch | val: {n_val} examples")
    print(f"reference points: random phases = 90.0°, "
          f"zero-correction = {zero_baseline_deg:.2f}°")
 
    # ---- model + optimizers -----------------------------------------------
    model = FFPhaseCNN(laser_count=L, element_positions=outer_positions, scale = SCALE)
    print(f"input {grid_size}x{grid_size}x{C_IN} | loss: {'twin-invariant' if USE_TWIN_LOSS else 'direct'} |"
      f"SCALE = {SCALE}", "| zernike stride", model.zernike_conv.stride,
      "| gabor stride", model.gabor_conv.stride,
      "| i_downsample first-conv stride", model.i_downsample.layers[1].strides)
    model.build(input_shape=(1, grid_size, grid_size, C_IN))
    norm_vars, other_vars, all_vars, all_names, n_norm = build_variable_groups(model)
 
    total_steps = EPOCHS * steps_per_epoch
    main_lr = WarmupCosine(MAIN_LR, total_steps, WARMUP_STEPS, min_lr_frac=MIN_LR_FRAC)
    norm_lr = WarmupCosine(NORM_LR, total_steps, WARMUP_STEPS, min_lr_frac=MIN_LR_FRAC)
    optimizer_main = tf.keras.optimizers.Adam(
        learning_rate=main_lr, clipnorm=0.5, beta_1=0.9, beta_2=0.99, epsilon=1e-8
    )
    optimizer_norm = tf.keras.optimizers.Adam(
        learning_rate=norm_lr, clipnorm=0.2, beta_1=0.9, beta_2=0.995, epsilon=1e-8
    )
    def phase_loss(labels, pred, reduce=True):
        return (twin_invariant_loss(labels, pred, twin_perm, reduce=reduce)) if USE_TWIN_LOSS else direct_phase_loss(labels, pred, reduce=reduce)
        
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            pred = model(images, training=True)
            loss, eff = phase_loss(labels, pred)
        grads = tape.gradient(loss, all_vars)
        optimizer_main.apply_gradients(zip(grads[n_norm:], other_vars))
        optimizer_norm.apply_gradients(zip(grads[:n_norm], norm_vars))
        ang = predicted_angles(pred)
        err_deg = tf.abs(tf.atan2(tf.sin(eff - ang), tf.cos(eff - ang))) * (180.0 / np.pi)
        grad_max = tf.reduce_max(tf.stack([tf.norm(g) for g in grads]))
        return loss, grad_max, tf.reduce_mean(err_deg)
 
    @tf.function
    def val_step(images, labels):
        """Returns SUMS so the ragged last val batch is weighted correctly."""
        pred = model(images, training=False)
        l_min, eff = phase_loss(labels, pred, reduce=False)
        ang = predicted_angles(pred)
        err = tf.abs(tf.atan2(tf.sin(eff-ang), tf.cos(eff-ang))) * (180.0 / np.pi)
        _, eff_tw = twin_invariant_loss(labels, pred, twin_perm, reduce=False)
        err_tw = tf.abs(tf.atan2(tf.sin(eff_tw-ang), tf.cos(eff_tw-ang))) * (180 / np.pi)
        return tf.reduce_sum(l_min) * L, tf.reduce_sum(err, axis=0), tf.reduce_max(err), tf.reduce_sum(err_tw)
 
    def run_validation():
        loss_sum, per_laser_sum, worst, tw_sum = 0.0, np.zeros(L), 0.0, 0.0
        for images, labels in val_ds:
            ls, pls, mx, tws = val_step(images, labels)
            loss_sum += float(ls); per_laser_sum += pls.numpy()
            worst = max(worst, float(mx)); tw_sum += float(tws)
        per_laser = per_laser_sum / n_val
        return loss_sum / (n_val * L), float(per_laser.mean()), worst, per_laser, tw_sum / (n_val * L)
 
    # ---- logging -----------------------------------------------------------
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
 
    import json
    with open(LOG_PATH.replace('.csv', '_config.json'), 'w') as f:
        json.dump({
            'run_label': RUN_LABEL, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'main_lr_peak': MAIN_LR, 'norm_lr_peak': NORM_LR,
            'warmup_steps': WARMUP_STEPS, 'min_lr_frac': MIN_LR_FRAC,
            'schedule': 'linear warmup -> cosine decay',
            'loss': '1 - cos(d)', 'output': 'unit vector (cos, sin)',
            'decoder_stride': 1, 'film_early': False,
            'pre_laser_downsample': 'before tile',
            'steps_per_epoch': steps_per_epoch, 'n_val': n_val,
            'zero_correction_baseline_deg': zero_baseline_deg,
            # REQUIRED to run the trained model on any new data later:
            'norm_scale': [float(v) for v in norm_scale],
            'crop': CROP, 'input_channels': C_IN, 'scale': SCALE, 'data_path': DATA_PATH,
            'twin_invariant_loss': USE_TWIN_LOSS,
        }, f, indent=2)
 
    csv_file = open(LOG_PATH, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ['epoch', 'elapsed_s', 'main_lr', 'norm_lr',
         'train_loss', 'train_mae_deg', 'val_loss', 'val_mae_deg', 'val_max_err_deg',
         'best_val_mae_deg', 'grad_norm_max', 'conv_update_ratio', 'norm_update_ratio']
        + [f'val_mae_laser_{k}' for k in range(L)]
    )
 
    best_val_mae = float('inf')
    epochs_since_best = 0
    t_start = Timer.elapsed()
    mid = steps_per_epoch // 2          # measure update ratios mid-epoch, past warmup
    print(f"{RUN_LABEL} begin - {EPOCHS} epochs, {total_steps} steps")
 
    for epoch in range(EPOCHS):
        ep_start = Timer.elapsed()
        losses = maes = gmax = tf.constant(0.0)
        n_step = 0
        conv_r = norm_r = float('nan')
 
        for i, (images, labels) in enumerate(train_ds):
            if i == mid:
                before = [tf.identity(v) for v in all_vars]
            loss, grad_max, mae = train_step(images, labels)
            if i == mid:
                conv_r, norm_r = update_ratio_medians(all_vars, before, n_norm)
            losses += loss
            maes += mae
            gmax = tf.maximum(gmax, grad_max)
            n_step += 1
            if PRINT_EVERY and i % PRINT_EVERY == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch}: "
                      f"loss={float(loss):.4f}  mae={float(mae):.2f}°", flush=True)
 
        train_loss = float(losses) / n_step
        train_mae = float(maes) / n_step
        grad_max_ep = float(gmax)
 
        val_loss, val_mae, val_max, val_per_laser = run_validation()
 
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_since_best = 0
            model.save_weights(f'{CKPT_DIR}/best.weights.h5')
        else:
            epochs_since_best += 1
        if epoch > 0 and epoch % CKPT_EVERY == 0:
            model.save_weights(f'{CKPT_DIR}/epoch_{epoch:04d}.weights.h5')
 
        lr_m = float(main_lr(optimizer_main.iterations))
        lr_n = float(norm_lr(optimizer_norm.iterations))
        elapsed = Timer.elapsed() - t_start
 
        csv_writer.writerow(
            [epoch, round(elapsed, 1), lr_m, lr_n,
             train_loss, train_mae, val_loss, val_mae, val_max,
             best_val_mae, grad_max_ep, conv_r, norm_r]
            + [float(v) for v in val_per_laser]
        )
        csv_file.flush()
 
        per_laser_str = ' '.join(f'{v:.1f}' for v in val_per_laser)
        print(f"epoch {epoch}: train_mae={train_mae:.2f}°  val_mae={val_mae:.2f}°  "
              f"(best {best_val_mae:.2f}°)  val_worst={val_max:.1f}°  "
              f"per-laser=[{per_laser_str}]  lr={lr_m:.1e}  "
              f"upd_conv={conv_r:.1e}  "
              f"{Timer.elapsed() - ep_start:.0f}s/epoch  t={elapsed/3600:.2f}h",
              flush=True)
 
        if PATIENCE is not None and epochs_since_best >= PATIENCE:
            print(f"early stop: val_mae has not improved for {PATIENCE} epochs")
            break
 
    csv_file.close()
    print(f"done. best val mae: {best_val_mae:.2f}° "
          f"(zero-correction baseline {zero_baseline_deg:.2f}°) "
          f"-> {CKPT_DIR}/best.weights.h5")
    
if choice == "train data gpu":
    grid_data_size = int(input("How long should the grid be?\n"))
    grid_ssl = int(input("How large should sampling be?(0.1mm)\n"))
    real_count = int(input("How many realizations do you want to generate?\n"))
    batch = int(input("Realizations per GPU batch? (4 is a safe start)\n"))
    seed = int(input("Random seed? (use a new one for each extra chunk of data)\n"))
    r500 = 0.05
    out_path = 'data/divtrainingdata.h5'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    array = ArrayDesign(2, 'hex')
    dist = 384000000
    target = [0, 0, dist]
    sim = BeamPerturbator(ssl=grid_ssl * 1e-4, targ_pos=target, array_des=array,
                          Ideal=False, grid_length=grid_data_size)
    sim.input_e_field_base()
    gen = GPUDataGenerator(sim, seed=seed)
    writer = DatasetStore(out_path, image_shape=(grid_data_size, grid_data_size),
                          num_channels=array.arrayvector.shape[0] - 1, flush_count=16,
                          with_diversity=True)
    div_ph = diversity_phases(sim.element_positions)
    print(f"diversity image: +{np.degrees(diversity_delta):.0f} deg on elements {np.flatnonzero(div_ph).tolist()}")
    print(f"r0 per slab (cm): {np.round(gen.r0 * 100, 1)}")
    print(f"total r0: {(gen.r0 ** (-5/3)).sum() ** (-3/5) * 100:.1f} cm | "
          f"existing examples in file: {writer._n}")
    done = 0
    t0 = Timer.elapsed()
    while done < real_count:
        B = min(batch, real_count - done)
        results, _ = gen.run_batch(B, div_phases=div_ph)
        for r in results:
            outer_params = np.delete(r['best_params'], gen.center_idx)
            writer.add_example(r['init_image'].astype('float32'), r['opt_image'].astype('float32'),
                               outer_params.astype('float32'), div_image=r['div_image'].astype('float32'), r500=r500, div_delta=float(diversity_delta),
                               init_intensity=r['init_intensity'], best_intensity=r['best_intensity'],
                               iterations=r['iterations'], target_x=target[0], target_y=target[1])
        done += B
        el = Timer.elapsed() - t0
        print(f"{done}/{real_count} realizations | {el / done:.2f} s each | "
              f"ETA {(real_count - done) * el / done / 60:.1f} min", flush=True)
    writer.close()

if choice == "val_outliers":
    run_label = input("Run label to analyse (e.g. training_twin_v4)?\n").strip()
    K = int(input("How many worst examples to plot? (20 recommended)\n") or 20)
    H5 = Path(read_run_config(run_label).get('data_path', 'data/testtrainingdata.h5'))
    VAL_SPLIT = 0.1                                           # make_dataset default
    wrap = lambda v: np.angle(np.exp(1j * v))
 
    array = ArrayDesign(2, 'hex')
    cat = pd.read_csv(H5.with_suffix('.catalog.csv'))
    with h5py.File(H5, 'r') as f:
        grid_N = f['images'].shape[1]
    model, norm_scale, outer_positions, center_idx, crop = load_trained_model(run_label, grid_N, array)
    perm = twin_permutation(outer_positions)
 
    # identical split to make_dataset
    n_total = len(cat)
    indices = np.arange(n_total)
    np.random.default_rng(42).shuffle(indices)
    val_idx = np.sort(indices[:int(n_total * VAL_SPLIT)])
    print(f"{n_total} examples, {len(val_idx)} validation")
 
    # predictions + per-sample image diagnostics, read in chunks
    preds, labels, diag = [], [], []
    yy, xx = np.mgrid[:grid_N, :grid_N]
    c0 = grid_N // 2                                          # far-field zero / target pixel
    r = np.hypot(xx - c0, yy - c0)
    bucket = r <= grid_N / 40                                 # same bucket as the optimizer
    core = r <= 256 * grid_N / 1024                           # holds ~99% of the power
    with h5py.File(H5, 'r') as f:
        for s in range(0, len(val_idx), 32):
            idx = val_idx[s:s + 32]
            imgs = f['images'][idx].astype(np.float32)
            labels.append(f['labels'][idx].astype(np.float64))
            net_in = (np.stack([imgs, f['div_images'][idx].astype(np.float32)], axis=-1)
                      if len(norm_scale) == 2 else imgs)
            preds.append(predict_outer_phases(model, net_in, norm_scale, crop))
            for I in imgs:
                tot = I.sum()
                cy, cx = (I * yy).sum() / tot, (I * xx).sum() / tot
                diag.append({'centroid_offset_px': float(np.hypot(cx - c0, cy - c0)),
                             'bucket_power_frac': float(I[bucket].sum() / tot),
                             'outside_core_frac': float(1 - I[core].sum() / tot),
                             'peak_over_mean_bucket': float(I.max() / I[bucket].mean()),
                             'image_contrast': float(I.std() / I.mean())})
            print(f"  {min(s + 32, len(val_idx))}/{len(val_idx)}", end='\r', flush=True)
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    cmp_ = twin_aware_compare(preds, labels, perm)
    if len(norm_scale) == 2:      # diversity model: scored against the true labels, no twin branch
        err_direct = np.degrees(np.abs(wrap(labels - preds)))
        print(f"\nval MAE (direct, should match the log's best): {err_direct.mean():.2f}°   "
              f"twin-aware: {cmp_['err_deg'].mean():.2f}°")
        cmp_['err_deg'], cmp_['eff'] = err_direct, labels
    else:
        print(f"\nval MAE (twin-aware, should match the log's best): {cmp_['err_deg'].mean():.2f}°")
    mae = cmp_['err_deg'].mean(1)
 
    # assemble per-sample table
    df = pd.DataFrame(diag)
    df.insert(0, 'h5_index', val_idx)
    df['mae_deg'] = mae
    df['max_laser_err_deg'] = cmp_['err_deg'].max(1)
    df['n_lasers_over_60deg'] = (cmp_['err_deg'] > 60).sum(1)
    df['twin_branch_used'] = cmp_['use_twin']
    df['branch_gap'] = cmp_['branch_gap']                      # ~0 => twin choice is a coin flip
    df['label_to_twin_deg'] = np.degrees(np.abs(wrap(labels + labels[:, perm]))).mean(1)
    df['label_spread_deg'] = np.degrees(np.abs(wrap(labels))).mean(1)
    sub = cat.set_index('index').loc[val_idx]
    df['bucket_gain'] = (sub['best_intensity'] / sub['init_intensity']).values
    df['iterations'] = sub['iterations'].values
    for k in range(labels.shape[1]):
        df[f'true_{k}_deg'] = np.degrees(cmp_['eff'][:, k])
        df[f'pred_{k}_deg'] = np.degrees(preds[:, k])
    df = df.sort_values('mae_deg', ascending=False).reset_index(drop=True)
    out_dir = Path(f'outputs/analysis/{run_label}')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'val_errors_ranked.csv', index=False)
 
    # summaries
    q = np.percentile(mae, [50, 75, 90, 95, 99])
    print("per-sample MAE percentiles 50/75/90/95/99: " + " / ".join(f"{v:.1f}°" for v in q))
    for thr in (30, 60, 90):
        print(f"  samples with MAE > {thr}°: {(mae > thr).mean() * 100:.1f}%")
    worst = df['mae_deg'] >= np.percentile(mae, 95)
    print(f"  share of total error carried by the worst 5%: "
          f"{df.loc[worst, 'mae_deg'].sum() / df['mae_deg'].sum() * 100:.1f}%")
    print(f"  MAE without the worst 5%: {df.loc[~worst, 'mae_deg'].mean():.2f}°")
    print(f"  worst 5%: errors spread over {df.loc[worst, 'n_lasers_over_60deg'].mean():.1f} "
          f"lasers on average (6 = whole sample wrong, 1 = one laser wrong)")
 
    print("\nwhat distinguishes the worst 5%   [median: worst 5% | rest]   Spearman rho with MAE")
    cols = ['label_to_twin_deg', 'branch_gap', 'bucket_gain', 'label_spread_deg',
            'centroid_offset_px', 'bucket_power_frac', 'outside_core_frac',
            'peak_over_mean_bucket', 'image_contrast', 'iterations']
    for c in cols:
        rho = df[c].rank().corr(df['mae_deg'].rank())
        print(f"  {c:24s} {df.loc[worst, c].median():10.4g} | {df.loc[~worst, c].median():10.4g}   rho {rho:+.2f}")
    print(f"\nranked table: {out_dir / 'val_errors_ranked.csv'}")
 
    # plot the worst K: centre crop, log scale, with true vs predicted phases
    from matplotlib.colors import LogNorm
    K = min(K, len(df))
    ncol = 5
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4.8 * nrow))
    h = 256 * grid_N // 1024
    with h5py.File(H5, 'r') as f:
        for j, ax in enumerate(np.atleast_1d(axes).ravel()):
            ax.axis('off')
            if j >= K:
                continue
            row = df.iloc[j]
            I = f['images'][int(row['h5_index'])][c0 - h:c0 + h, c0 - h:c0 + h]
            ax.imshow(I, cmap='plasma', origin='lower', norm=LogNorm(vmin=I.max() * 1e-4, vmax=I.max()))
            t = ' '.join(f"{row[f'true_{k}_deg']:.0f}" for k in range(labels.shape[1]))
            p = ' '.join(f"{row[f'pred_{k}_deg']:.0f}" for k in range(labels.shape[1]))
            ax.set_title(f"#{int(row['h5_index'])}  MAE {row['mae_deg']:.0f}°  gain {row['bucket_gain']:.2f}\n"
                         f"true {t}\npred {p}", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f'worst_{K}.png', dpi=120)
    plt.close(fig)
    print(f"worst-{K} figure: {out_dir / f'worst_{K}.png'}")

if choice == "nn_bucket_eval":
    run_label = input("Run label to evaluate (e.g. training_twin_v4)?\n").strip()
    grid_N = int(input("Grid size used for the training data? (1024)\n") or 1024)
    grid_ssl = int(input("Sampling used for the training data (0.1mm)? (20)\n") or 20)
    r500 = float(input("r500 used for the training data? (0.05m)\n") or 0.05)
    n_real = int(input("How many fresh realizations? (500 recommended)\n") or 500)
    batch = int(input("Realizations per GPU batch? (4)\n") or 4)
    seed = int(input("Random seed (use one never used for training data)?\n"))
 
    array = ArrayDesign(2, 'hex')
    dist = 384000000
    sim = BeamPerturbator(ssl=grid_ssl * 1e-4, targ_pos=[0, 0, dist], array_des=array,
                          r500=r500, Ideal=False, grid_length=grid_N)
    sim.input_e_field_base()
    gen = GPUDataGenerator(sim, seed=seed)
    model, norm_scale, outer_positions, center_idx, crop = load_trained_model(run_label, grid_N, array)
    perm = twin_permutation(outer_positions)
    outer = np.delete(np.arange(len(sim.base_fields)), center_idx)
    print(f"total r0 = {(gen.r0 ** (-5 / 3)).sum() ** (-3 / 5) * 100:.1f} cm")
 
    def full_phases(outer_ph):
        ph = np.zeros(len(sim.base_fields))
        ph[outer] = outer_ph
        return ph
 
    def power(M, ph):
        c = np.exp(1j * ph)
        return float(np.real(c.conj() @ M @ c))
 
    rows = []
    done = 0
    t0 = Timer.elapsed()
    while done < n_real:
        B = min(batch, n_real - done)
        screens = gen.generate_screens(tf.constant(B), gen.r0_scale)
        F, M = gen.far_fields(screens)
        M = M.numpy().astype(np.complex128)
        init_imgs = gen.images(F, tf.zeros([B, gen.L], tf.float32)).numpy()
        if len(norm_scale) == 2:      # diversity model: also record the diversity image
            div = tf.tile(tf.constant(diversity_phases(sim.element_positions)[None]), [B, 1])
            net_in = np.stack([init_imgs, gen.images(F, div).numpy()], axis=-1)
        else:
            net_in = init_imgs
        pred = predict_outer_phases(model, net_in, norm_scale, crop)
        for b in range(B):
            Mb = 0.5 * (M[b] + M[b].conj().T)
            opt, P_opt, _ = optimal_phases_from_M(Mb, 100, center_idx)
            p_nn = full_phases(pred[b])
            p_tw = full_phases(-pred[b][perm])                # the twin of the prediction
            P_nn, P_tw = power(Mb, p_nn), power(Mb, p_tw)
            rows.append({'P_zero': power(Mb, np.zeros(len(p_nn))), 'P_opt': P_opt,
                         'P_nn': P_nn, 'P_twin': P_tw, 'P_nn_resolved': max(P_nn, P_tw),
                         'label': np.delete(opt, center_idx), 'pred': pred[b]})
        done += B
        el = Timer.elapsed() - t0
        print(f"  {done}/{n_real}  ({el / done:.2f} s each)", end='\r', flush=True)
 
    P = {k: np.array([r[k] for r in rows]) for k in ('P_zero', 'P_opt', 'P_nn', 'P_twin', 'P_nn_resolved')}
    labels = np.array([r['label'] for r in rows])
    preds = np.array([r['pred'] for r in rows])
    cmp_ = twin_aware_compare(preds, labels, perm)
    mae = cmp_['err_deg'].mean(1)
    wrap = lambda v: np.angle(np.exp(1j * v))
    mae_direct = np.degrees(np.abs(wrap(labels - preds))).mean(1)
 
    def show(name, ratio):
        q = np.percentile(ratio, [5, 25, 50, 75, 95])
        print(f"  {name:38s} mean {ratio.mean():.3f}  median {q[2]:.3f}   5–95%: {q[0]:.3f}–{q[4]:.3f}")
 
    print(f"\n{n_real} fresh realizations   (MAE on them: direct {mae_direct.mean():.2f}°, "
          f"twin-aware {mae.mean():.2f}°)")
    print("bucket power relative to the optimizer's optimum:")
    show("no correction (all phases 0)", P['P_zero'] / P['P_opt'])
    show("NN prediction as output", P['P_nn'] / P['P_opt'])
    show("NN + one twin-check measurement", P['P_nn_resolved'] / P['P_opt'])
    print(f"  NN beats no-correction (twin-checked): {(P['P_nn_resolved'] > P['P_zero']).mean() * 100:.1f}% of realizations")
    print(f"  twin of the prediction was the better one: {(P['P_twin'] > P['P_nn']).mean() * 100:.1f}%")
    gain_opt = P['P_opt'] / P['P_zero']
    gain_nn = P['P_nn_resolved'] / P['P_zero']
    print(f"  gain over no correction: optimizer {np.median(gain_opt):.2f}x, NN {np.median(gain_nn):.2f}x (medians)")
    print(f"  share of the optimizer's improvement captured by the NN (median): "
          f"{np.median((P['P_nn_resolved'] - P['P_zero']) / np.maximum(P['P_opt'] - P['P_zero'], 1e-30)) * 100:.1f}%")
 
    c_mean = 1 - cmp_['loss'].mean()                          # <cos err>, no Gaussian assumption
    est = (7 + 12 * c_mean + 30 * c_mean ** 2) / 49
    print(f"  analytic estimate from these errors ((7 + 12<cos> + 30<cos>^2)/49): {est:.3f}")
 
    out_dir = Path(f'outputs/analysis/{run_label}')
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({**{k: v for k, v in P.items()}, 'mae_deg': mae,
                  'ratio_nn_resolved': P['P_nn_resolved'] / P['P_opt']}
                 ).to_csv(out_dir / f'bucket_eval_seed{seed}.csv', index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(P['P_nn_resolved'] / P['P_opt'], bins=40, color='C0')
    ax[0].set_xlabel('NN bucket power / optimal (twin-checked)')
    ax[0].set_ylabel('realizations')
    ax[1].scatter(mae, P['P_nn_resolved'] / P['P_opt'], s=6)
    ax[1].set_xlabel('twin-aware MAE (deg)')
    ax[1].set_ylabel('NN bucket power / optimal')
    plt.tight_layout()
    plt.savefig(out_dir / f'bucket_eval_seed{seed}.png', dpi=120)
    plt.close(fig)
    print(f"per-realization table and plot in {out_dir}")

if choice == "ckpt_check":
    import json, datetime
    run_label = input("Run label (e.g. training_twin_v4)?\n").strip()
    n_check = int(input("Validation samples to evaluate per checkpoint? (200)\n") or 200)
    extra = input("Epoch checkpoints to also test, comma-separated (e.g. 172,448), or blank:\n").strip()
    cfg = json.load(open(f'outputs/train_logs/{run_label}_config.json'))
    H5 = Path(cfg.get('data_path', 'data/testtrainingdata.h5'))
    ck_dir = Path(f'outputs/ckpts/{run_label}')
    cfg_p = Path(f'outputs/train_logs/{run_label}_config.json')
    log_p = Path(f'outputs/train_logs/{run_label}.csv')
    ts = lambda p: datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S') \
        if p.exists() else 'MISSING'
 
    print("---- (1) files ----")
    for p in (ck_dir / 'best.weights.h5', cfg_p, log_p, H5, H5.with_suffix('.catalog.csv')):
        print(f"  {str(p):55s} modified {ts(p)}")
    print(f"  config: epochs {cfg.get('epochs')}, n_val {cfg.get('n_val')}, "
          f"steps/epoch {cfg.get('steps_per_epoch')}, norm_scale {cfg.get('norm_scale')}")
    if log_p.exists():
        log = pd.read_csv(log_p)
        b = log.loc[log['val_mae_deg'].idxmin()]
        print(f"  CSV log: {len(log)} epochs, best val {b['val_mae_deg']:.2f}° at epoch {int(b['epoch'])}")
    epoch_files = sorted(ck_dir.glob('epoch_*.weights.h5'))
    print(f"  periodic checkpoints on disk: {len(epoch_files)}"
          + (f" ({epoch_files[0].name} … {epoch_files[-1].name})" if epoch_files else ""))
 
    print("---- (2) norm_scale ----")
    cat = pd.read_csv(H5.with_suffix('.catalog.csv'))
    n_total = len(cat)
    idx = np.arange(n_total)
    np.random.default_rng(42).shuffle(idx)
    n_val = int(n_total * 0.1)
    val_idx, train_idx = np.sort(idx[:n_val]), idx[n_val:]
    with h5py.File(H5, 'r') as f:
        grid_N = f['images'].shape[1]
        s_idx = np.sort(np.random.default_rng(42).choice(train_idx, min(100, len(train_idx)), replace=False))
        crop_cfg = cfg.get('crop', 512)
        crop_cfg = None if (crop_cfg is None or crop_cfg >= grid_N) else int(crop_cfg)
        chans = ['images'] + (['div_images'] if len(np.atleast_1d(cfg['norm_scale'])) == 2 else [])
        ns_now = [float(np.percentile(center_crop(f[c][s_idx], crop_cfg), 99.999)) for c in chans]
        sel = val_idx[:n_check]
        imgs = np.stack([f[c][sel].astype(np.float32) for c in chans], axis=-1)
        labels = f['labels'][sel].astype(np.float64)
    print(f"  (images cropped to {crop_cfg or grid_N} px, as make_dataset does)")
    print(f"  recomputed from the current file: {np.round(ns_now, 12).tolist()} | saved in config: "
          f"{np.round(np.atleast_1d(cfg['norm_scale']), 12).tolist()} | "
          f"n_val now {n_val} vs config {cfg.get('n_val')}")
 
    print("---- (3) checkpoints on the same validation samples ----")
    array = ArrayDesign(2, 'hex')
    model, norm_scale, outer_positions, center_idx, crop = load_trained_model(run_label, grid_N, array)
    perm = twin_permutation(outer_positions)
    def val_mae(m):
        pred = predict_outer_phases(m, imgs, norm_scale, crop)
        if len(norm_scale) == 2:                                  # diversity model: direct error
            return np.degrees(np.abs(np.angle(np.exp(1j * (labels - pred))))).mean()
        return twin_aware_compare(pred, labels, perm)['err_deg'].mean()
    print(f"  best.weights.h5: {val_mae(model):.2f}°  (on {len(sel)} val samples)")
    for e in [int(x) for x in extra.split(',') if x.strip()]:
        p = ck_dir / f'epoch_{e:04d}.weights.h5'
        if p.exists():
            model.load_weights(str(p))
            print(f"  {p.name}: {val_mae(model):.2f}°   (modified {ts(p)})")
        else:
            print(f"  {p.name}: not found")
    print("If best.weights.h5 is far above the log's best but an epoch checkpoint near the best epoch\n"
          "matches it, best.weights.h5 was overwritten by a later run. If every checkpoint is bad, the\n"
          "model code changed since training (layer order / sizes) - compare with the training version.")

if choice == "photon_budget":
    run_label = input("Run label to evaluate (e.g. training_div_v1)?\n").strip()
    grid_N = int(input("Grid size used for the training data? (1024)\n") or 1024)
    grid_ssl = int(input("Sampling used for the training data (0.1mm)? (20)\n") or 20)
    r500 = float(input("r500 used for the training data? (0.05m)\n") or 0.05)
    n_real = int(input("How many fresh realizations? (300 recommended)\n") or 300)
    batch = int(input("Realizations per GPU batch? (4)\n") or 4)
    seed = int(input("Random seed (use one never used for training data)?\n"))
    levels_in = input("Detected photons per frame, total over both arms, comma-separated\n"
                      "(blank = 1e3,3e3,1e4,3e4,1e5,3e5,1e6,3e6):\n").strip()
    sigma_read = float(input("Read noise per pixel in photoelectrons? (0 = photon noise only; "
                             "sCMOS ~1-2, InGaAs ~30-100)\n") or 0)
    bkg = float(input("Background photons per pixel per frame? (0)\n") or 0)
    t_frame = float(input("Frame (exposure) time in ms for the beacon-power estimate? (0.2)\n") or 0.2) * 1e-3
    eta = float(input("Total throughput: atmosphere x optics x detector efficiency? (0.3)\n") or 0.3)
    div_urad = float(input("Beacon full-angle divergence (1/e^2) in urad? (100)\n") or 100)
    levels = np.array([float(v) for v in levels_in.split(',')] if levels_in
                      else [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6])
 
    array = ArrayDesign(2, 'hex')
    dist = 384000000
    sim = BeamPerturbator(ssl=grid_ssl * 1e-4, targ_pos=[0, 0, dist], array_des=array,
                          r500=r500, Ideal=False, grid_length=grid_N)
    sim.input_e_field_base()
    gen = GPUDataGenerator(sim, seed=seed)
    model, norm_scale, outer_positions, center_idx, crop = load_trained_model(run_label, grid_N, array)
    perm = twin_permutation(outer_positions)
    outer = np.delete(np.arange(len(sim.base_fields)), center_idx)
    n_arms = len(norm_scale)
    in_N = grid_N if crop is None else crop
    print(f"total r0 = {(gen.r0 ** (-5 / 3)).sum() ** (-3 / 5) * 100:.1f} cm | camera region "
          f"{in_N}x{in_N} px per arm, {n_arms} arm(s) | read noise {sigma_read} e- | background {bkg} ph/px")
 
    def full_phases(outer_ph):
        ph = np.zeros(len(sim.base_fields))
        ph[outer] = outer_ph
        return ph
 
    def power(M, ph):
        c = np.exp(1j * ph)
        return float(np.real(c.conj() @ M @ c))
 
    # ---- 1) simulate once: noise-free network-input crops, bucket matrices, optimum ----
    clean, Ms, P_opt, P_zero, labels = [], [], [], [], []
    div_ph = diversity_phases(sim.element_positions) if n_arms == 2 else None
    done, t0 = 0, Timer.elapsed()
    while done < n_real:
        B = min(batch, n_real - done)
        screens = gen.generate_screens(tf.constant(B), gen.r0_scale)
        F, M = gen.far_fields(screens)
        M = M.numpy().astype(np.complex128)
        imgs = gen.images(F, tf.zeros([B, gen.L], tf.float32)).numpy()
        if n_arms == 2:
            div = gen.images(F, tf.tile(tf.constant(div_ph[None]), [B, 1])).numpy()
            imgs = np.stack([imgs, div], axis=-1)
        else:
            imgs = imgs[..., None]
        clean.append(center_crop(imgs, crop).astype(np.float32))   # only the read-out region
        for b in range(B):
            Mb = 0.5 * (M[b] + M[b].conj().T)
            opt, Po, _ = optimal_phases_from_M(Mb, 100, center_idx)
            Ms.append(Mb)
            P_opt.append(Po)
            P_zero.append(power(Mb, np.zeros(len(sim.base_fields))))
            labels.append(np.delete(opt, center_idx))
        done += B
        print(f"  simulated {done}/{n_real}  ({(Timer.elapsed() - t0) / done:.2f} s each)", end='\r', flush=True)
    clean = np.concatenate(clean)                               # [n, in_N, in_N, n_arms]
    P_opt, P_zero, labels = np.array(P_opt), np.array(P_zero), np.array(labels)
    arm_sum = clean[..., 0].sum(axis=(1, 2))                    # zero-phase arm, per frame
    print(f"\nphotons per arm vary between frames by {arm_sum.std() / arm_sum.mean() * 100:.1f}% (rms) "
          f"from scintillation and beam wander")
 
    # beacon power at the Moon that delivers N_tot detected photons per frame
    A_coll = len(sim.base_fields) * np.pi * sim.apert_radi ** 2
    E_ph = 6.62607015e-34 * 2.99792458e8 / sim.wavelength
    w = dist * div_urad * 1e-6 / 2
 
    def beacon_power(N_tot):
        irradiance = N_tot * E_ph / (t_frame * eta * A_coll)    # W/m^2 needed at the array
        return irradiance * np.pi * w ** 2 / 2
 
    # ---- 2) sweep photon levels on the same realizations ----
    rng = np.random.default_rng(seed + 1)                       # noise draws, separate from screens
    wrap = lambda v: np.angle(np.exp(1j * v))
 
    def evaluate(net_in):
        pred = predict_outer_phases(model, net_in, norm_scale, crop=None)   # already cropped
        P_nn = np.array([power(Ms[i], full_phases(pred[i])) for i in range(len(pred))])
        P_tw = np.array([power(Ms[i], full_phases(-pred[i][perm])) for i in range(len(pred))])
        mae_direct = np.degrees(np.abs(wrap(labels - pred))).mean(1)
        mae_twin = twin_aware_compare(pred, labels, perm)['err_deg'].mean(1)
        return {'mae_direct': mae_direct.mean(), 'mae_twin_aware': mae_twin.mean(),
                'ratio_nn_mean': (P_nn / P_opt).mean(), 'ratio_nn_median': np.median(P_nn / P_opt),
                'ratio_twin_checked_mean': (np.maximum(P_nn, P_tw) / P_opt).mean(),
                'beats_no_correction_pct': (P_nn > P_zero).mean() * 100}
 
    rows = [{'photons_total': np.inf, 'photons_per_arm': np.inf, 'peak_px_photons_median': np.inf,
             **evaluate(clean)}]
    print(f"\n{'photons/frame':>13s} {'peak px':>9s} {'MAE direct':>11s} {'twin-aware':>11s} "
          f"{'P/P_opt mean':>13s} {'median':>7s} {'twin-checked':>13s} {'beats P0':>9s} {'beacon':>10s}")
 
    def show(r):
        N_tot = r['photons_total']
        beacon = f"{beacon_power(N_tot) * 1e3:.3g} mW" if np.isfinite(N_tot) else "-"
        print(f"{N_tot:13.3g} {r['peak_px_photons_median']:9.3g} {r['mae_direct']:10.2f}° "
              f"{r['mae_twin_aware']:10.2f}° {r['ratio_nn_mean']:13.3f} {r['ratio_nn_median']:7.3f} "
              f"{r['ratio_twin_checked_mean']:13.3f} {r['beats_no_correction_pct']:8.1f}% {beacon:>10s}")
    show(rows[0])
    for N_tot in levels:
        kappa = (N_tot / n_arms) / arm_sum.mean()               # photons per unit simulated intensity
        noisy = np.empty_like(clean)
        for s in range(0, len(clean), 32):                      # chunked to bound memory
            n_exp = kappa * clean[s:s + 32].astype(np.float64)
            counts = rng.poisson(n_exp + bkg).astype(np.float64) - bkg
            if sigma_read > 0:
                counts += rng.normal(0.0, sigma_read, counts.shape)
            noisy[s:s + 32] = (counts / kappa).astype(np.float32)
        r = {'photons_total': N_tot, 'photons_per_arm': N_tot / n_arms,
             'peak_px_photons_median': float(np.median(kappa * clean[..., 0].max(axis=(1, 2)))),
             **evaluate(noisy)}
        rows.append(r)
        show(r)
    df = pd.DataFrame(rows)
 
    df['beacon_power_W'] = [beacon_power(n) if np.isfinite(n) else np.nan for n in df['photons_total']]
 
    ref = df.iloc[0]['ratio_nn_mean']
    print(f"\nnoise-free: bucket power {ref:.3f} of optimum | no correction {(P_zero / P_opt).mean():.3f}")
    print(f"beacon column: power the beacon must emit, for {t_frame * 1e3:.2f} ms frames, throughput {eta}, "
          f"{div_urad:.0f} urad divergence (spot {2 * w / 1e3:.1f} km at the Moon), "
          f"collecting area {A_coll * 1e4:.0f} cm^2")
    fin = df.iloc[1:].sort_values('photons_total')
    for tol in (0.01, 0.05):
        # lowest tested level from which every higher level also stays within tol of noise-free
        good = (fin['ratio_nn_mean'] >= ref - tol).values
        k = len(good)
        while k > 0 and good[k - 1]:
            k -= 1
        if k < len(good):
            r = fin.iloc[k]
            print(f"  within {tol:.2f} of noise-free bucket power from {r['photons_total']:.3g} photons/frame "
                  f"(on the tested grid) -> beacon ~{r['beacon_power_W'] * 1e3:.3g} mW")
        else:
            print(f"  the highest tested level is not within {tol:.2f} of noise-free; test higher photon counts")
 
    # ---- 3) save ----
    out_dir = Path(f'outputs/analysis/{run_label}')
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'seed{seed}_read{sigma_read:g}_bkg{bkg:g}'
    df.to_csv(out_dir / f'photon_budget_{tag}.csv', index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(fin['photons_total'], fin['mae_direct'], 'o-', label='direct')
    if n_arms == 1:
        ax[0].semilogx(fin['photons_total'], fin['mae_twin_aware'], 's-', label='twin-aware')
    ax[0].axhline(df.iloc[0]['mae_direct' if n_arms == 2 else 'mae_twin_aware'], ls='--', c='gray',
                  label='noise-free')
    ax[0].set_xlabel('detected photons per frame (all arms)')
    ax[0].set_ylabel('validation MAE (deg)')
    ax[0].legend()
    ax[1].semilogx(fin['photons_total'], fin['ratio_nn_mean'], 'o-', label='NN output')
    ax[1].axhline(ref, ls='--', c='gray', label='noise-free')
    ax[1].axhline((P_zero / P_opt).mean(), ls=':', c='k', label='no correction')
    ax[1].set_xlabel('detected photons per frame (all arms)')
    ax[1].set_ylabel('mean bucket power / optimum')
    ax[1].legend()
    fig.suptitle(f'{run_label}: read noise {sigma_read:g} e-, background {bkg:g} ph/px, {n_real} realizations')
    plt.tight_layout()
    plt.savefig(out_dir / f'photon_budget_{tag}.png', dpi=120)
    plt.close(fig)
    print(f"table and plot: {out_dir / f'photon_budget_{tag}.csv'} (.png)")

if choice == "time_eval":
    grid_N = int(input("Grid size used for the training data? (1024)\n") or 1024)
    grid_ssl = int(input("Sampling used for the training data (0.1mm)? (20)\n") or 20)
    r500 = float(input("r500 used for the training data? (0.05m)\n") or 0.05)
    wind_dir = input("Wind directions: aligned / random? (aligned)\n").strip() or 'aligned'
    n_seq = int(input("How many independent sequences? (100)\n") or 100)
    batch = int(input("Sequences per GPU batch? (4)\n") or 4)
    seed = int(input("Random seed (use one never used for training data)?\n"))
    delays_in = input("Delays in ms, comma-separated (blank = 0,0.1,0.25,0.5,1,2,4,8):\n").strip()
    run_label = input("Run label for the network (blank = optimizer phases only):\n").strip()
    delays = np.array([float(v) for v in delays_in.split(',')] if delays_in
                      else [0, 0.1, 0.25, 0.5, 1, 2, 4, 8]) * 1e-3
 
    array = ArrayDesign(2, 'hex')
    dist = 384000000
    sim = BeamPerturbator(ssl=grid_ssl * 1e-4, targ_pos=[0, 0, dist], array_des=array,
                          r500=r500, Ideal=False, grid_length=grid_N)
    sim.input_e_field_base()
    gen = GPUDataGenerator(sim, seed=seed)
    tau0_th = gen.set_wind(wind_dir, seed=seed)
    r0_tot = (gen.r0 ** (-5 / 3)).sum() ** (-3 / 5)
    print(f"total r0 = {r0_tot * 100:.1f} cm | tau0 (Kolmogorov, 6.88 sum (v_n/r0_n)^5/3) = {tau0_th * 1e3:.2f} ms | "
          f"sequences valid up to {gen.max_time_s() * 1e3:.0f} ms")
    print("layer  height  wind   dir    r0")
    for n in range(gen.S):
        print(f"  {n:2d}  {gen.layer_heights[n] / 1e3:5.2f} km {gen.wind_speed[n]:5.1f} m/s "
              f"{np.degrees(gen.wind_angle[n]):4.0f}°  {gen.r0[n] * 100:7.1f} cm")
    if delays.max() > gen.max_time_s():
        print("WARNING: longest delay exceeds the screen period; the turbulence repeats")
    wind = gen.wind.astype(np.float32)
    shift_at = lambda t: tf.constant((wind * t).astype(np.float32))
 
### section 1:
    taus = np.array([0.25, 0.5, 1, 2, 4, 8, 16]) * 1e-3
    taus = taus[taus < gen.max_time_s()]
    a_hi = (gen.amp_hi.numpy().astype(np.float64)) ** 2                    # r0 = 1
    a_lo = (gen.amp_lo.numpy().astype(np.float64)) ** 2                    # [3, 3, 3]
    FXn, FYn, F3n = gen.FXg.numpy().astype(np.float64), gen.FYg.numpy().astype(np.float64), \
        gen.F3.numpy().astype(np.float64)
 
    def d_unit(dx, dy):           # structure function of an r0 = 1 screen at separation (dx, dy)
        d = 2 * np.sum(a_hi * (1 - np.cos(2 * np.pi * (FXn * dx + FYn * dy))))
        for p in range(3):
            ph = 2 * np.pi * (F3n[p][:, None] * dy + F3n[p][None, :] * dx)
            d += 2 * np.sum(a_lo[p] * (1 - np.cos(ph)))
        return d
    D_exp = np.array([sum(gen.r0[n] ** (-5 / 3) * d_unit(*(gen.wind[n] * t)) for n in range(gen.S))
                      for t in taus])
    D_kol = 6.88 * np.array([np.sum((gen.wind_speed * t / gen.r0) ** (5 / 3)) for t in taus])
 
    sl = slice(None, None, max(1, grid_N // 128))          # every k-th pixel (screens are periodic)
    D_sum, n_D = np.zeros(len(taus)), 0
    n_sf = min(n_seq, 32)
    for s in range(0, n_sf, batch):
        B = min(batch, n_sf - s)
        hi_c, lo_c = gen.draw_screen_coeffs(B)
        p0 = tf.reduce_sum(gen.screens_from_coeffs(hi_c, lo_c, shift_at(0.0), False), axis=1).numpy()[:, sl, sl]
        for i, tau in enumerate(taus):
            pt = tf.reduce_sum(gen.screens_from_coeffs(hi_c, lo_c, shift_at(tau), False), axis=1).numpy()[:, sl, sl]
            D_sum[i] += ((pt - p0) ** 2).sum()
        n_D += p0.size
    D_meas = D_sum / n_D
    print(f"\ntemporal structure function of the summed screen phase ({n_sf} sequences):")
    print("   tau     measured  expected(screens)  ratio    Kolmogorov  expected/Kolmogorov")
    for t, dm, de, dk in zip(taus, D_meas, D_exp, D_kol):
        print(f"  {t * 1e3:5.2f} ms  {dm:8.3f}  {de:12.3f}     {dm / de:6.3f}   {dk:9.3f}   {de / dk:8.3f}")
 
    def crossing(D):              # log-log interpolation to D = 1 rad^2
        k = np.searchsorted(D, 1.0)
        if 0 < k < len(D):
            return float(np.exp(np.interp(0.0, np.log(D[k - 1:k + 1]), np.log(taus[k - 1:k + 1]))))
        return np.nan
    print(f"tau0: Kolmogorov formula {tau0_th * 1e3:.2f} ms | expected for these screens "
          f"{crossing(D_exp) * 1e3:.2f} ms | measured {crossing(D_meas) * 1e3:.2f} ms "
          f"(nan = outside the tested taus)")
 
    # ---- Part 2: servo lag ----
    use_nn = bool(run_label)
    if use_nn:
        model, norm_scale, outer_positions, center_idx, crop = load_trained_model(run_label, grid_N, array)
        div_ph = diversity_phases(sim.element_positions) if len(norm_scale) == 2 else None
    outer = np.delete(np.arange(gen.L), gen.center_idx)
 
    def power(M, ph):
        c = np.exp(1j * ph)
        return float(np.real(c.conj() @ M @ c))
 
    def full(outer_ph):
        ph = np.zeros(gen.L)
        ph[outer] = outer_ph
        return ph
 
    R = {key: [[] for _ in delays] for key in ('zero', 'opt0', 'nn0')}
    done, t0 = 0, Timer.elapsed()
    while done < n_seq:
        B = min(batch, n_seq - done)
        hi_c, lo_c = gen.draw_screen_coeffs(B)
        F0, M0 = gen.far_fields(gen.screens_from_coeffs(hi_c, lo_c, shift_at(0.0)))
        M0 = M0.numpy().astype(np.complex128)
        opt0 = []
        for b in range(B):
            ph, _, _ = optimal_phases_from_M(0.5 * (M0[b] + M0[b].conj().T), 100, gen.center_idx)
            opt0.append(ph)
        if use_nn:
            img = gen.images(F0, tf.zeros([B, gen.L], tf.float32)).numpy()
            if div_ph is not None:
                img = np.stack([img, gen.images(F0, tf.tile(tf.constant(div_ph[None]), [B, 1])).numpy()], -1)
            nn0 = predict_outer_phases(model, img, norm_scale, crop)
        for i, d in enumerate(delays):
            if d == 0:
                Md = M0
            else:
                _, Md = gen.far_fields(gen.screens_from_coeffs(hi_c, lo_c, shift_at(d)))
                Md = Md.numpy().astype(np.complex128)
            for b in range(B):
                Mb = 0.5 * (Md[b] + Md[b].conj().T)
                _, P_opt, _ = optimal_phases_from_M(Mb, 100, gen.center_idx)
                R['zero'][i].append(power(Mb, np.zeros(gen.L)) / P_opt)
                R['opt0'][i].append(power(Mb, opt0[b]) / P_opt)
                if use_nn:
                    R['nn0'][i].append(power(Mb, full(nn0[b])) / P_opt)
        done += B
        print(f"  {done}/{n_seq} sequences ({(Timer.elapsed() - t0) / done:.2f} s each)", end='\r', flush=True)
 
    print(f"\n\nbucket power / optimum at time d, phases computed from the frame at t = 0 "
          f"({n_seq} sequences, tau0 = {tau0_th * 1e3:.2f} ms):")
    print(f"  {'delay':>8s} {'d/tau0':>7s} {'no corr':>8s} {'optimizer(0)':>13s}" + (f" {'NN(0)':>8s}" if use_nn else ""))
    rows = []
    for i, d in enumerate(delays):
        r = {'delay_ms': d * 1e3, 'd_over_tau0': d / tau0_th, 'no_correction': np.mean(R['zero'][i]),
             'optimizer_t0': np.mean(R['opt0'][i])}
        if use_nn:
            r['nn_t0'] = np.mean(R['nn0'][i])
        rows.append(r)
        print(f"  {d * 1e3:6.2f}ms {d / tau0_th:7.2f} {r['no_correction']:8.3f} {r['optimizer_t0']:13.3f}"
              + (f" {r['nn_t0']:8.3f}" if use_nn else ""))
    df = pd.DataFrame(rows)
    out_dir = Path('outputs/analysis/time_eval')
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"seed{seed}_{wind_dir}" + (f"_{run_label}" if use_nn else "")
    df.to_csv(out_dir / f'servo_lag_{tag}.csv', index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].loglog(taus * 1e3, D_meas, 'o-', label='measured')
    ax[0].loglog(taus * 1e3, D_exp, '--', label='expected for these screens')
    ax[0].loglog(taus * 1e3, D_kol, ':', label='Kolmogorov 6.88 Σ (v_n τ / r0_n)^{5/3}')
    ax[0].axhline(1, c='gray', lw=0.8)
    ax[0].set_xlabel('τ (ms)')
    ax[0].set_ylabel('D_φ(τ)  (rad²)')
    ax[0].legend()
    ax[1].plot(df['delay_ms'], df['optimizer_t0'], 'o-', label='optimizer phases from t=0')
    if use_nn:
        ax[1].plot(df['delay_ms'], df['nn_t0'], 's-', label='network from t=0 image')
    ax[1].plot(df['delay_ms'], df['no_correction'], ':', c='k', label='no correction')
    ax[1].axvline(tau0_th * 1e3, c='gray', lw=0.8, ls='--')
    ax[1].set_xlabel('delay before the correction is applied (ms)')
    ax[1].set_ylabel('bucket power / optimum at that time')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'servo_lag_{tag}.png', dpi=120)
    plt.close(fig)
    print(f"table and plot: {out_dir / f'servo_lag_{tag}.csv'} (.png)")

if choice == "spgd_eval":
    grid_N = int(input("Grid size used for the training data? (1024)\n") or 1024)
    grid_ssl = int(input("Sampling used for the training data (0.1mm)? (20)\n") or 20)
    r500 = float(input("r500 used for the training data? (0.05m)\n") or 0.05)
    wind_dir = input("Wind directions: aligned / random? (aligned)\n").strip() or 'aligned'
    n_seq = int(input("How many independent sequences? (32)\n") or 32)
    batch = int(input("Sequences per GPU batch? (4)\n") or 4)
    seed = int(input("Random seed (use one never used for training data)?\n"))
    T_ms = float(input("Sequence length in ms (blank = 90% of the screen period):\n") or 0)
    dtc_ms = float(input("Propagation (coarse) time step in ms? (0.1)\n") or 0.1)
    mode = input("SPGD metric: beacon (receive-side actuators) / target (retroreflector return)? "
                 "(beacon)\n").strip() or 'beacon'
    f_iter = float(input("SPGD iteration rate in kHz (two measurements per iteration)? (100)\n") or 100) * 1e3
    gains = [float(v) for v in (input("SPGD gains, comma-separated (blank = 0.5,1,2,4):\n").strip()
                                or "0.5,1,2,4").split(',')]
    dithers = [float(v) for v in (input("Dither amplitudes in rad, comma-separated "
                                        "(blank = 0.1,0.2,0.3):\n").strip() or "0.1,0.2,0.3").split(',')]
    lat_us = float(input("Detector + electronics latency per measurement in us? (0)\n") or 0)
    dist = 384000000
    t_rt = 2 * dist / 2.99792458e8
    if mode == 'target':
        rt_in = input(f"Round-trip delay in ms (blank = 2z/c = {t_rt * 1e3:.0f} ms):\n").strip()
        t_rt = float(rt_in) * 1e-3 if rt_in else t_rt
    else:
        t_rt = 0.0
    phi_rate = float(input("Detected photons per second at the SPGD detector when corrected? "
                           "(1e9)\n") or 1e9)
    sigma_det = float(input("SPGD detector noise per sample in photoelectrons? (0)\n") or 0)
    f_act_in = input("Actuator bandwidth in kHz (blank = ideal):\n").strip()
    f_act = float(f_act_in) * 1e3 if f_act_in else np.inf
    rng_in = input("Actuator range in rad before a 2 pi reset (blank = unlimited):\n").strip()
    act_range = float(rng_in) if rng_in else np.inf
    sigma_psi = float(input("Transmit-channel phase noise rms in rad? (0)\n") or 0)
    f_psi = float(input("Transmit-channel phase noise bandwidth in Hz? (1000)\n") or 1000)
    sigma_cal = float(input("Per-channel phase calibration error, rms fraction (e.g. 0.02)? (0)\n") or 0)
    t_burn = float(input("Start of the steady-state average in ms? (5)\n") or 5) * 1e-3
    run_label = input("Run label for the network comparison (blank = skip):\n").strip()
    if run_label:
        t_frame = float(input("Camera frame period in ms? (0.2)\n") or 0.2) * 1e-3
        nn_lat = float(input("Network latency (frame -> correction applied) in ms? (0.2)\n") or 0.2) * 1e-3
        sigma_cam = float(input("Camera read noise per pixel in photoelectrons? (1)\n") or 1)
 
    array = ArrayDesign(2, 'hex')
    sim = BeamPerturbator(ssl=grid_ssl * 1e-4, targ_pos=[0, 0, dist], array_des=array,
                          r500=r500, Ideal=False, grid_length=grid_N)
    sim.input_e_field_base()
    gen = GPUDataGenerator(sim, seed=seed)
    tau0 = gen.set_wind(wind_dir, seed=seed)
    T = 0.9 * gen.max_time_s() if T_ms <= 0 else T_ms * 1e-3
    if T > gen.max_time_s():
        raise ValueError(f"{T * 1e3:.0f} ms is longer than the screen period "
                         f"{gen.max_time_s() * 1e3:.0f} ms; the turbulence would repeat")
    dtc = dtc_ms * 1e-3
    K = int(round(T / dtc)) + 1
    t_coarse = np.arange(K) * dtc
    T = t_coarse[-1]
    wind = gen.wind.astype(np.float32)
    shift_at = lambda t: tf.constant((wind * t).astype(np.float32))
    L = gen.L
    outer = np.delete(np.arange(L), gen.center_idx)
    cx = int(np.argmin(np.abs(gen.x_out - sim.target_position[0])))   # same pixel as the bucket centre
    cy = int(np.argmin(np.abs(gen.x_out - sim.target_position[1])))
    delay_iter = int(round((t_rt + lat_us * 1e-6) * f_iter))
    print(f"total r0 {(gen.r0 ** (-5 / 3)).sum() ** (-3 / 5) * 100:.1f} cm | tau0 (Kolmogorov) "
          f"{tau0 * 1e3:.2f} ms | {n_seq} sequences of {T * 1e3:.1f} ms | propagation every "
          f"{dtc * 1e3:.2f} ms ({K} steps per sequence)")
    print(f"SPGD: {mode} metric, {f_iter / 1e3:.0f} kHz, measurement delay {delay_iter} iterations "
          f"({delay_iter / f_iter * 1e3:.3f} ms), {len(gains) * len(dithers)} gain/dither settings")
    if mode == 'target' and delay_iter * 2 >= T / (0.5 / f_iter):
        print("WARNING: the round trip is longer than the sequence; no measurement ever reaches "
              "the controller, so SPGD cannot correct anything here")
 
    use_nn = bool(run_label)
    if use_nn:
        model, norm_scale, outer_positions, center_idx_nn, crop = load_trained_model(run_label, grid_N, array)
        n_arms = len(norm_scale)
        div_ph = diversity_phases(sim.element_positions) if n_arms == 2 else None
        frame_every = int(round(t_frame / dtc))
        if abs(frame_every * dtc - t_frame) > 1e-9 or frame_every < 1:
            raise ValueError("the camera frame period must be a whole number of coarse time steps")
 
    def power(Mb, ph):
        c = np.exp(1j * ph)
        return float(np.real(c.conj() @ Mb @ c))
 
    rng = np.random.default_rng(seed + 7)
    sums = None
    steady_rows, n_done, t0 = [], 0, Timer.elapsed()
    interp_checked = False
    while n_done < n_seq:
        B = min(batch, n_seq - n_done)
        hi_c, lo_c = gen.draw_screen_coeffs(B)
        M_c = np.zeros((K, B, L, L), np.complex128)
        v_c = np.zeros((K, B, L), np.complex128)
        Popt_c = np.zeros((K, B))
        opt_ph0 = np.zeros((B, L))
        P_window = np.zeros(B)
        nn_ph, kappa = [], None
        for k in range(K):
            F, M = gen.far_fields(gen.screens_from_coeffs(hi_c, lo_c, shift_at(t_coarse[k])))
            Mk = M.numpy().astype(np.complex128)
            Mk = 0.5 * (Mk + np.conj(np.transpose(Mk, (0, 2, 1))))
            M_c[k] = Mk
            v_c[k] = F[:, :, cy, cx].numpy()
            for b in range(B):
                ph, Popt_c[k, b], _ = optimal_phases_from_M(Mk[b], 100, gen.center_idx)
                if k == 0:
                    opt_ph0[b] = ph
            if k == 0:
                P_window = tf.reduce_sum(gen.images(F, tf.zeros([B, L], tf.float32)), axis=[1, 2]).numpy()
            if use_nn and k % frame_every == 0:
                img = gen.images(F, tf.zeros([B, L], tf.float32)).numpy()
                img = (np.stack([img, gen.images(F, tf.tile(tf.constant(div_ph[None]), [B, 1])).numpy()], -1)
                       if n_arms == 2 else img[..., None])
                img = center_crop(img, crop)
                if kappa is None:   # photons per unit intensity, fixed from the first frame
                    bucket_frac = Popt_c[0].mean() / P_window.mean()
                    N_frame = phi_rate / bucket_frac * t_frame          # same beacon, whole focal plane
                    kappa = (N_frame / n_arms) / img[..., 0].sum(axis=(1, 2)).mean()
                nn_ph.append(predict_outer_phases(model, camera_frames(img, kappa, sigma_cam, rng),
                                                  norm_scale, crop=None))
            print(f"  sequences {n_done + 1}-{n_done + B}: propagation step {k + 1}/{K}", end='\r', flush=True)
 
        # ---- check the linear interpolation between propagation steps (first batch) ----
        if not interp_checked:
            errs = []
            for k in np.linspace(0, K - 2, min(5, K - 1)).astype(int):
                tm = t_coarse[k] + 0.5 * dtc
                _, Md = gen.far_fields(gen.screens_from_coeffs(hi_c, lo_c, shift_at(tm)))
                Md = Md.numpy().astype(np.complex128)
                Mi = interp_traces(t_coarse, M_c, tm)
                for b in range(B):
                    for ph in (opt_ph0[b], rng.uniform(-np.pi, np.pi, L)):
                        Pd = power(0.5 * (Md[b] + Md[b].conj().T), ph)
                        errs.append(abs(power(Mi[b], ph) - Pd) / Pd)
            errs = np.array(errs)
            print(f"\ninterpolation check at mid-steps: bucket-power error mean {errs.mean() * 100:.3f}%, "
                  f"max {errs.max() * 100:.3f}%  (reduce the coarse step if this is not << 1%)")
            interp_checked = True
 
        J_ref = (Popt_c[0].mean() if mode == 'beacon' else
                 np.mean([abs(np.exp(1j * opt_ph0[b]) @ v_c[0, b]) ** 2 * np.sum(abs(v_c[0, b]) ** 2)
                          for b in range(B)]))
        trace = {'t': t_coarse, 'M': M_c, 'P_opt': Popt_c, 'v': v_c, 'outer': outer,
                 'nn': np.array(nn_ph) if use_nn else None,
                 't_frame': t_frame if use_nn else None, 'nn_latency': nn_lat if use_nn else None}
        cfg = {'mode': mode, 'gains': gains, 'dithers': dithers, 'f_iter': f_iter,
               'delay_iter': delay_iter, 't_rt': t_rt, 'f_act': f_act, 'act_range': act_range,
               'sigma_cal': sigma_cal, 'sigma_psi': sigma_psi, 'f_psi': f_psi,
               'phi_rate': phi_rate, 'sigma_det': sigma_det, 'J_ref': J_ref, 't_burn': t_burn}
        res = simulate_spgd(trace, cfg, rng)
        if sums is None:
            sums = {key: np.zeros_like(res[key]) for key in ('ratio_spgd', 'ratio_zero')}
            sums['ratio_nn'] = np.zeros_like(res['ratio_zero'])
            steady_s, steady_n = [], []
        for key in ('ratio_spgd', 'ratio_zero'):
            sums[key] += res[key] * B
        if use_nn:
            sums['ratio_nn'] += res['ratio_nn'] * B
            steady_n.append(res['steady_nn'])
        steady_s.append(res['steady_spgd'])
        n_done += B
        print(f"\n  {n_done}/{n_seq} sequences done ({(Timer.elapsed() - t0) / n_done:.1f} s each)")
 
    t_f = res['t']
    curves = {key: v / n_seq for key, v in sums.items()}
    steady_s = np.concatenate(steady_s, axis=1)                     # [G, n_seq]
    win = lambda c, a, b: c[..., (t_f >= a) & (t_f < b)].mean(-1)
    bucket_frac = Popt_c[0].mean() / P_window.mean()
    print(f"\nphoton scale: {phi_rate:.3g} photons/s at the SPGD detector when corrected = "
          f"{phi_rate * 0.5 / f_iter:.3g} per measurement; the same beacon gives "
          f"{phi_rate / bucket_frac * 2e-4:.3g} photons per 0.2 ms camera frame (photon_budget units)")
    print(f"bucket power / optimum, mean over {n_seq} sequences "
          f"(steady state = t >= {t_burn * 1e3:.1f} ms, ± = sequence-to-sequence std error):")
    print(f"  {'gain':>6s} {'dither':>7s} {'0-1 ms':>8s} {'1-5 ms':>8s} {'steady':>8s}")
    rows = []
    for g_i in range(len(res['gains'])):
        st = steady_s[g_i]
        r = {'gain': res['gains'][g_i], 'dither_rad': res['dithers'][g_i],
             'first_1ms': win(curves['ratio_spgd'][g_i], 0, 1e-3),
             '1_to_5ms': win(curves['ratio_spgd'][g_i], 1e-3, 5e-3),
             'steady': st.mean(), 'steady_se': st.std() / np.sqrt(len(st))}
        rows.append(r)
        print(f"  {r['gain']:6.2f} {r['dither_rad']:7.2f} {r['first_1ms']:8.3f} {r['1_to_5ms']:8.3f} "
              f"{r['steady']:8.3f} ± {r['steady_se']:.3f}")
    best = int(np.argmax([r['steady'] for r in rows]))
    print(f"best SPGD setting: gain {rows[best]['gain']:g}, dither {rows[best]['dither_rad']:g} rad "
          f"-> {rows[best]['steady']:.3f}")
    print(f"no correction (with the channel noise): {win(curves['ratio_zero'], t_burn, np.inf):.3f}")
    if use_nn:
        st = np.concatenate(steady_n)
        print(f"network ({t_frame * 1e3:.2f} ms frames, {nn_lat * 1e3:.2f} ms latency, "
              f"{sigma_cam:g} e- read noise): {st.mean():.3f} ± {st.std() / np.sqrt(len(st)):.3f}")
 
    out_dir = Path('outputs/analysis/spgd_eval')
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"seed{seed}_{mode}_{f_iter / 1e3:.0f}kHz" + (f"_{run_label}" if use_nn else "")
    pd.DataFrame(rows).to_csv(out_dir / f'spgd_settings_{tag}.csv', index=False)
    stride = max(1, len(t_f) // 4000)
    ts = {'t_ms': t_f[::stride] * 1e3, 'spgd_best': curves['ratio_spgd'][best][::stride],
          'no_correction': curves['ratio_zero'][::stride]}
    if use_nn:
        ts['network'] = curves['ratio_nn'][::stride]
    pd.DataFrame(ts).to_csv(out_dir / f'spgd_timeseries_{tag}.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts['t_ms'], ts['spgd_best'], label=f"SPGD (gain {rows[best]['gain']:g}, "
                                                  f"dither {rows[best]['dither_rad']:g} rad)")
    if use_nn:
        ax.plot(ts['t_ms'], ts['network'], label='network')
    ax.plot(ts['t_ms'], ts['no_correction'], ':', c='k', label='no correction')
    ax.axvline(t_burn * 1e3, c='gray', lw=0.8, ls='--')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('bucket power / optimum (mean over sequences)')
    ax.set_title(f'{mode} metric, {f_iter / 1e3:.0f} kHz, tau0 {tau0 * 1e3:.1f} ms')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'spgd_timeseries_{tag}.png', dpi=120)
    plt.close(fig)
    print(f"tables and plot: {out_dir}/spgd_*_{tag}.*")

if choice == "headroom":
    r500s = [0.1,0.15,0.2]