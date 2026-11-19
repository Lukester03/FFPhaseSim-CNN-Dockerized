import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import math
import numpy as np
import tensorflow as tf
from scipy import optimize
import soapy

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
               + b * 2.7e-16 * np.exp(-2 * h_km / 3))

    def hv_fried_param(self, height1, height2, wavelen = 500e-9, b_val=None):
        h_km = np.linspace(height1 / 1000, height2 / 1000, num=100)
        cn2 = self.cn2(h_km, self.v_rms, b_val)
        integral = np.trapz(y=cn2, x=h_km * 1000)          # dh in metres
        k_wavenum = 2 * np.pi / wavelen
        return (0.423 * k_wavenum ** 2 * integral) ** (-3 / 5)
    
    def layer_r0(self, h1, h2):
        return self.hv_fried_param(h1, h2, self.wavelen, self.b_val)

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

def setup_tf(seed=42):
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(seed)