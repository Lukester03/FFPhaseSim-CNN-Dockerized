#!/usr/bin/env python3
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

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
from scipy.signal import CZT
from numpy.fft import fft2, ifft2

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

class TurbulenceModeler:
    def __init__(self, wavelength: float = 1064e-9, 
    outer_scale: float = np.inf, grid_size: int = 1024, screen_physical_size: float = 2.0, grnd_speed = 5.0, fried_mult = 1.0):
        self.r0_check = False
        self.phase_catalog = None
        self.fried_mult = fried_mult
        self.wavelen = wavelength
        self.outer_scale = outer_scale
        self.grid_size = grid_size
        self.pixel_scale = screen_physical_size / grid_size
        self.grnd_speed = grnd_speed

    def rms_velocity(self, height1: float, height2: float, heights: np.ndarray):
        square_v = (self.grnd_speed + 30 * np.exp(-((heights-9.4)/(4.8))**2))**2
        v_integral = np.trapz(y = square_v, x = heights)
        v_rms = (1 / ((height2 - height1)) * v_integral * 1000)**0.5
        return v_rms

    def hv_fried_param(self, height1, height2, b_value):
        h1_km = height1 / 1000
        h2_km = height2 / 1000
        heights = np.linspace(h1_km, h2_km, num = 30)
        v_rms = self.rms_velocity(h1_km, h2_km, heights)
        struct_fxn = (5.94e-23 * heights ** 10 * (v_rms / 27) ** 2 * np.exp(-heights) + b_value * 2.7e-16 * np.exp(-2 * heights / 3))
        struct_fxn_integral = np.trapz(y = struct_fxn, x = heights * 1000)
        k_wavenum = self.wavelen ** -1 * 2 * np.pi
        r0 = (0.423 * 0.5 * struct_fxn_integral * k_wavenum ** 2) ** (-3/5) * self.fried_mult
        return r0

    def phase_screen_gen(self, height1, height2, b_val = None):
        if b_val == None:
            b_val = 1
        r0 = self.hv_fried_param(height1, height2, b_val)
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

    def all_screen_gen(self, b_val = None, gen_height = 20000):
        r0_total = 0
        screen_count = 32
        nx = self.grid_size
        self.phase_catalog = np.zeros((screen_count, nx, nx))
        screen_heights = np.linspace(0, gen_height, num = (screen_count+1))
        for i in range(screen_count):
            phase_screen, r0 = self.phase_screen_gen(b_val = b_val, height1 = screen_heights[i], 
            height2 = screen_heights[i+1])
            r0_total += r0 ** (-5/3)
            self.phase_catalog[i] = phase_screen
        if self.r0_check is False:
            print(r0_total ** (-3/5))
            self.r0_check = True

class BeamPerturbator:
    def __init__(self, ssl, targ_pos: list, array_des: ArrayDesign, grid_length: np.float64 = 1024, Ideal = True, fried_mult = 1.0):
        self.waist = self.apert_radi = self.wavelength = self.spatial_samp_length = np.float64(0)
        self.chan_amps = self.turb_model = self.base_fields = None
        self.phase_screens = self.apply_turb = self.h_layers = None
        self.target_position = targ_pos
        self.rel_array = array_des
        self.grid_length = grid_length
        self.set_rel_phases()
        self.chan_rel_phases = np.zeros(self.rel_array.arrayvector.shape[0])
        self.set_beam_elements(ideal = Ideal, fried_mult = fried_mult, ssl = ssl)

    def set_beam_elements(self,
                          clip_ratio: np.float64 = 0.6,
                          apert_radi: np.float64 = 5e-2,
                          wavelength: np.float64 = 1064e-9,
                          init_amp: np.complex128 = 10,
                          ssl: np.float64 = 4e-4,
                          ideal: bool = True,
                          gaussian: bool = True,
                          fried_mult = 1.0):
        self.chan_amps = np.full(self.rel_array.arrayvector.shape[0], init_amp)
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
                                                fried_mult=fried_mult)
            self.generate_turbulence()
            self.generate_transfer()
        
    def generate_turbulence(self, b_val = 1):
        self.turb_model.all_screen_gen(b_val = b_val)
        self.phase_screens = self.turb_model.phase_catalog.copy()

    def generate_transfer(self):
        dim = self.grid_length
        fx = np.fft.fftfreq(dim, d=self.spatial_samp_length)
        fy = np.fft.fftfreq(dim, d=self.spatial_samp_length)
        FX, FY = np.meshgrid(fx, fy)
        k = 2*np.pi/self.wavelength
        screen_heights = np.linspace(0, min(self.target_position[2], 20000), num=len(self.phase_screens)+1)
        self.h_layers = []
        z_current = 0.0
        for h in screen_heights[1:]:
            dz = h - z_current
            H = np.exp(1j * k * dz) * np.exp(-1j * np.pi * self.wavelength * dz * (FX ** 2 + FY ** 2))
            self.h_layers.append(H)
            z_current = h

    def set_rel_phases(self,
                       rel_phases=None):
        if rel_phases is not None:
            self.chan_rel_phases = rel_phases
        else:
            self.chan_rel_phases = np.zeros(self.rel_array.arrayvector.shape[0])

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
        piston = -k * (R_target - z_f)
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

    def input_e_quick(self):
        input_e = np.zeros_like(self.base_fields[0], dtype=np.complex128)
        for n, base in enumerate(self.base_fields):
            input_e += base * np.exp(1j * self.chan_rel_phases[n])
        return input_e

    def fresnel_czt(self, input_field, z: np.float64, k):
        dim_y, dim_x = input_field.shape
        ssl = self.spatial_samp_length
        x1 = (np.arange(dim_x) - dim_x / 2) * ssl
        y1 = (np.arange(dim_y) - dim_y / 2) * ssl
        X1, Y1 = np.meshgrid(x1,y1)
        Q1 = np.exp((1j * k * (X1 ** 2 + Y1 ** 2)) / (2 * z))
        array_extent = (self.element_positions[:,0].max() - self.element_positions[:,0].min())
        d_array = array_extent + 2 * self.apert_radi
        diff_limited_spot = self.wavelength * z / d_array
        out_window = 20 * diff_limited_spot
        E_mod = input_field * Q1
        out_size = self.grid_length
        out_center = [0,0]
        dx_out = out_window / out_size
        x2 = out_center[0] + (np.arange(out_size) - out_size // 2) * dx_out
        y2 = out_center[1] + (np.arange(out_size) - out_size // 2) * dx_out
        def make_czt(f_axis, n_in):
            df, f0 = f_axis[1] - f_axis[0], f_axis[0]
            W = np.exp(-1j * 2 * np.pi * df * ssl)
            A = np.exp(1j * 2 * np.pi * f0 * ssl)
            return CZT(n_in, out_size, W, A)
        tmp = make_czt(x2 / (self.wavelength * z), dim_x)(E_mod, axis=1)
        tmp = make_czt(y2 / (self.wavelength * z), dim_y)(tmp, axis=0)
        prefactor = np.exp(1j * k * z) / (1j * self.wavelength * z)
        X2, Y2 = np.meshgrid(x2, y2)
        Q2 = np.exp(1j * k * (X2 ** 2 + Y2 ** 2) / (2 * z))
        output_field = prefactor * Q2 * tmp * ssl ** 2
        return output_field, x2, y2

    def fresnel_propagation(self, input_field, z: np.float64):
        dim_y, dim_x = input_field.shape
        ssl = self.spatial_samp_length
        k = 2 * np.pi / self.wavelength
        if self.apply_turb:
            z_atmo = min(z, 20000)
            screen_num = min(len(self.phase_screens), int(round(z/500)))
            field = input_field.copy()
            screen_heights = np.linspace(0, z_atmo, num = screen_num + 1)
            z_current = 0.0
            for n in range(screen_num):
                z_current = screen_heights[n+1]
                field = ifft2(fft2(field) * self.h_layers[n])
                field *= np.exp(1j * self.phase_screens[n])
            dist_remainder = z-z_current
            if dist_remainder > 0:
                return self.fresnel_czt(input_field = field, z = dist_remainder, k = k)
            else:
                x_out = (np.arange(dim_x) - dim_x / 2) * ssl
                y_out = (np.arange(dim_y) - dim_y / 2) * ssl
                return field, x_out, y_out
        return self.fresnel_czt(input_field = input_field, z = z, k = k)

    def compute_optimal_phases(self, n_iter, tol=1e-6, return_m = False):
        n_elem = len(self.base_fields)
        z = self.target_position[2]
        bucket_radius = self.grid_length / 40
        bucket_rows = []
        mask = None
        for n in range(n_elem):
            field_out, x_out, y_out = self.fresnel_propagation(self.base_fields[n], z)
            if mask is None:
                cx = np.argmin(np.abs(x_out))
                cy = np.argmin(np.abs(y_out))
                Y, X = np.mgrid[0:field_out.shape[0], 0:field_out.shape[1]]
                r = np.sqrt((X-cx) ** 2 + (Y-cy) ** 2)
                mask = r <= bucket_radius
            bucket_rows.append(field_out[mask])
        E = np.stack(bucket_rows, axis=0)
        M = E.conj() @ E.T
        #print("sample M[0,0]:", M[0,0], "trace:", np.trace(M).real)

        eigvals, eigvecs = np.linalg.eigh(M)
        c = np.exp(1j * np.angle(eigvecs[:, -1]))

        # baseline candidate — guarantees the result can never be worse than doing nothing
        c_ones = np.ones(n_elem, dtype=np.complex128)
        power_ones = np.real(c_ones.conj() @ M @ c_ones)

        prev_power = -np.inf
        power = np.real(c.conj() @ M @ c)
        n_used = 0
        for i in range(n_iter):
            c = np.exp(1j * np.angle(M @ c))
            power = np.real(c.conj() @ M @ c)
            n_used = i + 1
            if power - prev_power < tol * abs(power):     # relative tolerance
                break
            prev_power = power

        ub = n_elem * eigvals[-1]
        gap = (ub - power) / ub
        top_eig_fraction = eigvals[-1] / np.trace(M).real
        print(gap, top_eig_fraction)

        if power_ones > power:            # safety net — should basically never trigger if math is right
            c = c_ones
            power = power_ones

        optimal_phases = np.angle(c)
        if return_m:
            return optimal_phases, power, n_used, mask, M
        return optimal_phases, power, n_used, mask

class ZernikeFilterBank:
    def __init__(self, kernel_size=21, num_modes=36):
        self.kernel_size = kernel_size
        self.num_modes = num_modes
        # generate grid
        center = (kernel_size - 1) / 2
        y, x = np.ogrid[:kernel_size, :kernel_size]
        y = (y - center) / center
        x = (x - center) / center

        self.rho = np.sqrt(x ** 2 + y ** 2)
        self.theta = np.arctan2(y, x)
        self.mask = self.rho <= 1.0

    def radial_polynomial(self, n, m):
        R = np.zeros_like(self.rho)
        for k in range((n - abs(m)) // 2 + 1):
            coeff = ((-1) ** k * math.factorial(n - k) /
                     (math.factorial(k) *
                      math.factorial((n + abs(m)) // 2 - k) *
                      math.factorial((n - abs(m)) // 2 - k)))
            R += coeff * self.rho ** (n - 2 * k)
        return R

    def generate_filter_bank(self):
        filters = np.zeros((self.kernel_size, self.kernel_size, 1, self.num_modes))

        for J in range(self.num_modes):
            n = int(np.ceil((np.sqrt(9 + 8*J) - 3) / 2))
            m = 2*J - n * (n + 2)

            if m >= 0:
                Z = self.radial_polynomial(n, abs(m)) * np.cos(m * self.theta)
            else:
                Z = self.radial_polynomial(n, abs(m)) * np.sin(abs(m) * self.theta)

            Z[~self.mask] = 0
            if np.any(self.mask):
                Z[self.mask] /= np.std(Z[self.mask]) + 1e-8

            filters[..., 0, J] = Z * self.mask
        return filters

class GaborFilterBank(layers.Layer):
    def __init__(self,
                 kernel_size,
                 num_orientations,
                 wavelengths,
                 gamma,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv_kernel = None
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.gamma = gamma
        self.phase_pairs = self.include_dc_balance = True
        if wavelengths == None:
            self.wavelengths = [3.0, 5.0, 7.0, 10.0, 14.0]
        else:
            self.wavelengths = wavelengths
        self.sigma = [lam * 0.56 for lam in self.wavelengths]
        self.kernel_bank, self.param_list = self._build_bank()
        self.num_filters = len(self.param_list)

    def _build_bank(self):
        kernels = []
        param_list = []

        center = self.kernel_size // 2
        x, y = np.meshgrid(
            np.arange(-center, center + 1),
            np.arange(-center, center + 1)
        )
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        phases = [0.0, np.pi / 2.0] if self.phase_pairs else [0.0]

        for lam, sigma in zip(self.wavelengths, self.sigma):
            for theta in np.linspace(0, np.pi, self.num_orientations, endpoint=False):
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                x_rot = x * cos_t + y * sin_t
                y_rot = -x * sin_t + y * cos_t

                gaussian = np.exp(-0.5 * (
                        (x_rot ** 2) / (sigma ** 2) +
                        (y_rot ** 2) / ((sigma * self.gamma) ** 2)
                ))

                for phi in phases:
                    kernel = gaussian * np.cos(2 * np.pi * x_rot / lam + phi)

                    if self.include_dc_balance:
                        kernel -= kernel.mean()

                    norm = np.sqrt(np.sum(kernel ** 2))
                    if norm > 1e-8:
                        kernel /= norm
                    kernels.append(kernel)

                    phase_name = "cos" if phi == 0 else "sin"
                    param_list.append({
                        'wavelength': lam,
                        'sigma': sigma,
                        'orientation_rad': round(theta, 4),
                        'phase': phase_name,
                        'phase_rad': phi
                    })
        kernel_tensor = np.stack(kernels, axis=-1)
        kernel_tensor = kernel_tensor[..., np.newaxis, :]

        return tf.constant(kernel_tensor, dtype=tf.float64), param_list

    def build(self, input_grid):
        in_channels = input_grid[-1]
        full_kernel = tf.tile(self.kernel_bank, [1, 1, in_channels, 1])
        self.conv_kernel = self.add_weight(
            name="gabor_kernels",
            shape=full_kernel.shape,
            initializer=tf.constant_initializer(full_kernel.numpy()),
            trainable=False
        )
        super().build(input_grid)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        outputs = tf.nn.depthwise_conv2d(
            inputs,
            self.conv_kernel,
            strides=[1, 1, 1, 1],
            padding='VALID',
            data_format='NHWC'
        )
        return outputs

    def get_filter_info(self, index:int):
        return self.param_list[index]

class ZernikeConvLayer(layers.Layer):

    def __init__(self, kernel_size=21, num_modes=36, **kwargs):
        super().__init__(**kwargs)
        self.zernike_kernel = None
        self.kernel_size = kernel_size
        self.num_modes = num_modes

    def build(self, input_grid):
        zernike_gen = ZernikeFilterBank(self.kernel_size, self.num_modes)
        kernel_np = zernike_gen.generate_filter_bank()
        # store Zernike polynomial filter as untrainable weight to ensure consistency
        self.zernike_kernel = self.add_weight(
            name='zern_kern',
            shape=(self.kernel_size, self.kernel_size, 1, self.num_modes),
            initializer=tf.constant_initializer(kernel_np),
            trainable=False
        )
        super().build(input_grid)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        pad_size = self.kernel_size // 2
        
        # Reflect padding preserves edge fringe information
        padded = tf.pad(
            inputs,
            [[0, 0],
             [pad_size, pad_size],
             [pad_size, pad_size],
             [0, 0]],
            mode='REFLECT'
        )

        output = tf.nn.conv2d(
            padded,
            self.zernike_kernel,
            strides=[1, 1, 1, 1],
            padding='VALID',
            )
        return output


    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'num_modes': self.num_modes
        })
        return config

class GaborConvLayer(layers.Layer):
    
    def __init__(self,
                 kernel_size=15,
                 num_orientations=8,
                 wavelengths=None,
                 gamma=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Store parameters for serialization
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.wavelengths = wavelengths
        self.gamma = gamma
        
        # Build the filter bank
        self.filter_bank = GaborFilterBank(
            kernel_size=kernel_size,
            num_orientations=num_orientations,
            wavelengths=wavelengths,
            gamma=gamma
        )
        
        self.num_filters = self.filter_bank.num_filters
    
    def build(self, input_grid):
        """Initialize the convolution weight tensor"""
        in_channels = input_grid[-1]
        # Get kernels from filter bank: [H, W, num_filters]
        kernels = self.filter_bank.kernel_bank
        
        # Expand for conv2d: [H, W, in_channels, out_channels]
        kernels = tf.tile(kernels, [1, 1, in_channels, 1])  # [H, W, C, F]
        
        # Store as non-trainable weight
        self.conv_kernel = self.add_weight(
            name="gabor_kernels",
            shape=kernels.shape,
            initializer=tf.constant_initializer(kernels.numpy()),
            trainable=False
        )
        
        super().build(input_grid)
    
    def call(self, inputs):
        """Apply Gabor filter bank convolution with reflect padding"""
        pad_size = self.kernel_size // 2
        
        # Reflect padding preserves edge fringe information
        padded = tf.pad(
            inputs,
            [[0, 0],
             [pad_size, pad_size],
             [pad_size, pad_size],
             [0, 0]],
            mode='REFLECT'
        )
        
        # Standard convolution
        output = tf.nn.conv2d(
            padded,
            self.conv_kernel,
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        
        return output
    
    def get_filter_info(self, index: int):
        """Get parameters for a specific filter"""
        return self.filter_bank.get_filter_info(index)
    
    def get_config(self):
        """Serialization for model saving"""
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'num_orientations': self.num_orientations,
            'wavelengths': self.wavelengths,
            'gamma': self.gamma,
        })
        return config

class FiLMConditioner(layers.Layer):
    def __init__(self, num_features, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            layers.Dense(32,activation='relu'),
            layers.Dense(2 * num_features),
        ])
    def call(self, features, position):
        gb = self.mlp(position)
        gamma, beta = tf.split(gb, 2, axis=-1)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return features * (1 + gamma) + beta

class FFPhaseCNN(Model):
    def __init__(self, grid_width = 256, mode_count=36, 
        gabor_kernel_size = 15, gabor_orientations = 8, gabor_lmbda = None, laser_count = 7,
        element_positions = None, processor_channels = [64, 64, 32], **kwargs):
        super().__init__(**kwargs)
        self.num_modes = mode_count
        self.gabor_kern_size = gabor_kernel_size
        self.gabor_orientations = gabor_orientations
        self.gabor_lmbda = gabor_lmbda
        self.num_lasers = laser_count
        if element_positions == None:
            element_positions = [[-0.8660254, -0.5, 0.],
                                 [-0.8660254, 0.5, 0.],
                                 [ 0., -1., 0.],
                                 [ 0., 0., 0.],
                                 [ 0., 1., 0.],
                                 [ 0.8660254, -0.5, 0.],
                                 [ 0.8660254, 0.5, 0.]]
        self.element_positions = np.asarray(element_positions)

        # Zernike Basis for reconstructing phase
        self.zernike_basis = ZernikeFilterBank(grid_width, self.num_modes).generate_filter_bank()
        
        # Non-trainable filter layers:
        
        self.zernike_conv = ZernikeConvLayer(kernel_size = 21, num_modes = mode_count, name = "zernike_features")
        self.gabor_conv = GaborConvLayer(kernel_size = gabor_kernel_size, 
        num_orientations = gabor_orientations, wavelengths = gabor_lmbda, gamma = 0.5, name = "gabor_features")

        # Processor Channel Counts:

        self.Zernike_chan = int(processor_channels[0])
        self.Gabor_chan = int(processor_channels[1])
        self.Raw_chan = int(processor_channels[2])
        
        # Processors
        self.zernike_processor = tf.keras.Sequential([
            layers.BatchNormalization(name='zernike_bn'),
            layers.Conv2D(self.Zernike_chan, 1, activation='relu', name='zernike_compress'),
            layers.Conv2D(self.Zernike_chan, 3, padding='SAME', activation='relu', 
                         name='zernike_process1'),
            layers.Conv2D(self.Zernike_chan, 3, padding='SAME', activation='relu', 
                         name='zernike_process2'),
        ], name='zernike_processor')
        
        self.gabor_processor = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(name='gabor_bn'),
            layers.Conv2D(self.Gabor_chan, 1, activation='relu', name='gabor_compress'),
            layers.Conv2D(self.Gabor_chan, 3, padding='SAME', activation='relu', 
                         name='gabor_process1'),
            layers.Conv2D(self.Gabor_chan, 3, padding='SAME', activation='relu', 
                         name='gabor_process2'),
        ], name='gabor_processor')
        
        self.intensity_processor = tf.keras.Sequential([
            layers.Conv2D(self.Raw_chan, 7, padding='SAME', activation='relu', 
                         name='intensity_conv1'),
            layers.BatchNormalization(name='intensity_bn1'),
            layers.Conv2D(self.Raw_chan, 5, padding='SAME', activation='relu', 
                         name='intensity_conv2'),
        ], name='intensity_processor')
        
        #Laser attention to separate lasers in image
        self.laser_attention = tf.keras.Sequential([
            layers.Conv2D(64, 1, activation='relu'),
            layers.Conv2D(self.num_lasers, 1, activation='sigmoid'),
        ], name='laser_attention')
        
        #decodes each laser's values
        self.shared_decoder = tf.keras.Sequential([
            layers.Conv2D(64, 3, padding='SAME', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='SAME', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(16, 3, padding='SAME', activation='relu'),
            layers.Conv2D(2, 3, padding='SAME'),
        ], name='shared_decoder')

        # coefficients
        self.coeff_head = tf.keras.Sequential([
            layers.Dense (64, activation='relu'),
            layers.Dense(self.num_modes),
        ], name = 'coeff_head')

        self.film = FiLMConditioner(sum(processor_channels))
        self.global_pool = layers.GlobalAveragePooling2D()

    def reconstruct_phase(self, coeffs, zernike_basis, aperture_mask):
        phase = tf.einsum('bm,hwlm->bhw', coeffs, zernike_basis)
        return phase[..., None] * aperture_mask
    
    def call(self, inputs, training = False):
        zernike_raw = self.zernike_conv(inputs)  # [B, H, W, 36]
        gabor_raw = self.gabor_conv(inputs)      # [B, H, W, 80]

        # Features processing
        z_features = self.zernike_processor(zernike_raw, training=training)
        g_features = self.gabor_processor(gabor_raw, training=training)
        i_features = self.intensity_processor(inputs, training=training)

        # Concatenate all features
        combined = tf.concat([z_features, g_features, i_features], axis=-1)
        
        # Implement attention per laser
        attention_maps = self.laser_attention(combined)
        
        # Phase decoder
        phases = []
        for i in range(self.num_lasers):
            laser_mask = attention_maps[..., i:i+1]
            laser_features = combined * laser_mask
            pos = tf.constant([[self.element_positions[i,0], self.element_positions[i,1]]], dtype=tf.float32)
            pos = tf.tile(pos, [tf.shape(inputs)[0], 1])
            conditioned = self.film(laser_features, pos)
            pooled = self.global_pool(conditioned)
            coeffs = self.coeff_head(pooled)
            phase_components = self.shared_decoder(conditioned, training=training)
            phase_sin = phase_components[..., 0:1]
            phase_cos = phase_components[..., 1:2]
            phase_analytic = self.reconstruct_phase(coeffs, self.zernike_basis, laser_mask)
            phase_residual = tf.atan2(phase_sin, phase_cos)
            phase = phase_analytic + phase_residual
            phases.append(phase)
        
        return phases, attention_maps

class DatasetStore:
    def __init__(self, h5_path, image_shape, num_channels, flush_count = 16):
        self.h5_path = Path(h5_path)
        self.catalog_path = self.h5_path.with_suffix('.catalog.csv')
        self.image_shape = image_shape
        self.num_channels = num_channels
        self.flush_count = flush_count
        self._catalog_rows = []
        self._n = 0

        self.f = h5py.File(self.h5_path, 'a')
        if 'images' not in self.f:
            self.f.create_dataset('images', shape=(0, *image_shape), maxshape=(None, *image_shape),
                                   dtype='float32', chunks=(1, *image_shape), compression='gzip')
            self.f.create_dataset('opt_images', shape=(0, *image_shape), maxshape=(None, *image_shape),
                                   dtype='float32', chunks=(1, *image_shape), compression='gzip')
            self.f.create_dataset('labels', shape=(0, num_channels), maxshape=(None, num_channels),
                                   dtype='float32')
        self._n = self.f['images'].shape[0]

    def add_example(self, image, opt_image, label_rad, **metadata):
        n = self._n
        for ds_name, arr in [('images', image), ('opt_images', opt_image), ('labels', label_rad)]:
            ds = self.f[ds_name]
            ds.resize(n + 1, axis=0)
            ds[n] = arr
        row = {'index':n, **metadata}
        self._catalog_rows.append(row)
        self._n += 1
        if self._n % self.flush_count == 0:
            self.flush()

    def flush(self):
        self.f.flush()
        pd.DataFrame(self._catalog_rows).to_csv(
            self.catalog_path, mode='a', header = not self.catalog_path.exists(), index=False
        )
        self._catalog_rows = []

    def close(self):
        self.flush()
        self.f.close()

# Testing elements
def linearity_test(sim):
    z = sim.target_position[2]
    sim.set_rel_phases()  # zero correction
    combined_out, xo, yo = sim.fresnel_propagation(sim.input_e_quick(), z)

    summed = np.zeros_like(combined_out)
    for base in sim.base_fields:
        out_n, _, _ = sim.fresnel_propagation(base, z)
        summed += out_n

    diff = combined_out - summed
    print("max |combined|:  ", np.max(np.abs(combined_out)))
    print("max |diff|:      ", np.max(np.abs(diff)))
    print("relative error:  ", np.max(np.abs(diff)) / np.max(np.abs(combined_out)))

def sdp_bound(M):
    import cvxpy as cp
    n = M.shape[0]
    scale = 1.0 / np.max(np.abs(M))
    M_scaled = M * scale

    X = cp.Variable((n, n), hermitian=True)
    constraints = [X >> 0] + [X[i, i] == 1 for i in range(n)]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(M_scaled @ X))), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(f"WARNING: solver status = {prob.status}")

    return prob.value / scale

def multi_restart_check(M, n_restarts=20, n_iter=100, tol=1e-6):
    n = M.shape[0]
    best_power = -np.inf
    powers = []
    for _ in range(n_restarts):
        c = np.exp(1j * np.random.uniform(0, 2 * np.pi, n))
        prev = -np.inf
        for _ in range(n_iter):
            c = np.exp(1j * np.angle(M @ c))
            power = np.real(c.conj() @ M @ c)
            if power - prev < tol * abs(power):
                break
            prev = power
        powers.append(power)
        best_power = max(best_power, power)
    return best_power, powers

def run_optimality_diagnostics(M, power_method_result):
    n = M.shape[0]
    eigvals = np.linalg.eigvalsh(M)
    ub_loose = n * eigvals[-1]
    top_eig_fraction = eigvals[-1] / np.trace(M).real
    best_restart, all_restarts = multi_restart_check(M)
    ub_tight = sdp_bound(M)

    print(f"power-method result:          {power_method_result:.6e}")
    print(f"best of {len(all_restarts)} restarts:          {best_restart:.6e}")
    print(f"loose bound (n*lambda_max):   {ub_loose:.6e}")
    print(f"tight bound (SDP):            {ub_tight:.6e}")
    print(f"top_eig_fraction:             {top_eig_fraction:.4f}")
    print(f"loose gap:                    {(ub_loose - power_method_result) / ub_loose:.4f}")
    print(f"tight gap:                    {(ub_tight - power_method_result) / ub_tight:.4f}")

###  Code Beginning
if __name__ == "__main__":
    print(f"Time to import: {Timer.elapsed()}")
    choice = input("What test are you running?\n -h5 \n -train \n -optsim \n -cnn \n -diagnostics \n")

#looking at testing data
if choice == "h5":
    example_num = int(input("What example do you want to view? \n"))
    filename = "testtrainingdata.h5"
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
    print(f"fried_mult:     {row['fried_mult']}")
    print(f"init_intensity: {row['init_intensity']:.2e}")
    print(f"best_intensity: {row['best_intensity']:.2e}")
    print(f"iterations:     {row['iterations']}")
    print(f"target:         ({row['target_x']}, {row['target_y']})")
    plt.tight_layout()
    plt.savefig(f'outputs/example_{example_num}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

#testing data generation
if choice == "train":
    grid_data_size = int(input("How long should the grid be?\n"))
    grid_ssl = int(input("How large should sampling be?(0.1mm)\n"))
    real_count = int(input("How many realizations do you want to generate?\n"))
    writer = DatasetStore('testtrainingdata.h5', image_shape = (grid_data_size,grid_data_size), 
                          num_channels = 7, flush_count=16)
    dist = 384000000
    target = [0, 0, dist]
    array = ArrayDesign(2, 'hex')
    sim = BeamPerturbator(ssl = grid_ssl*1e-4, targ_pos = target, array_des = array, 
                          fried_mult = 16, Ideal=False, grid_length = grid_data_size)
    x, y = sim.input_e_field_base()

    for real in range(real_count):
        if real != 0:
            sim.generate_turbulence()
        #linearity_test(sim)
        sim.set_rel_phases()
        input = sim.input_e_quick()
        init_e_image, x_def, y_def = sim.fresnel_propagation(input, dist)
        init_image = np.abs(init_e_image) ** 2

        best_params, bucket_power, n_iters, mask = sim.compute_optimal_phases(100)

        sim.set_rel_phases(best_params)
        opt_input = sim.input_e_quick()
        opt_e_image, _, _ = sim.fresnel_propagation(opt_input, dist)
        recomputed_power = np.sum(np.abs(opt_e_image[mask]) ** 2)
        #print("power from M:", bucket_power, " power from direct repropagation:", recomputed_power)
        opt_image = np.abs(opt_e_image)**2

        best_intensity = np.average(opt_image[mask])
        init_intensity = np.average(init_image[mask])

        writer.add_example(init_image.astype('float32'), opt_image.astype('float32'), 
                           best_params.astype('float32'), fried_mult = 64, 
                           init_intensity = init_intensity, best_intensity = best_intensity, 
                           iterations = n_iters, target_x = target[0], 
                           target_y = target[1])
        print(f"Realization #{real+1} generated. Time elapsed: {Timer.elapsed():.2f}")
    writer.close()

# diagnostics:
if choice == "diagnostics":
    grid_data_size = int(input("Grid size?\n"))
    grid_ssl = int(input("Sampling (0.1mm)?\n"))
    n_samples = int(input("How many realizations to check?\n"))
    dist = 384000000
    target = [0, 0, dist]
    array = ArrayDesign(2, 'hex')
    sim = BeamPerturbator(ssl=grid_ssl * 1e-4, targ_pos=target, array_des=array,
                          fried_mult=16, Ideal=False, grid_length=grid_data_size)
    x, y = sim.input_e_field_base()

    for i in range(n_samples):
        if i != 0:
            sim.generate_turbulence()
        optimal_phases, power, n_iters, mask, M = sim.compute_optimal_phases(40, return_m=True)
        print(f"\n--- Sample {i} ---")
        run_optimality_diagnostics(M, power)

# optical simulator test 
if choice == "optsim":
    dist = np.float64(384000000)
    target = [0, 0, dist]
    john_array = ArrayDesign(2, 'hex')
    john_perturbation = BeamPerturbator(ssl = 16e-4, targ_pos = target, array_des = john_array, 
                                        Ideal = True, fried_mult = 64.0, grid_length = 256)
    x_in, y_in = john_perturbation.input_e_field_base()
    john_perturbation.set_rel_phases()
    e_field = john_perturbation.input_e_quick()
    e_field_ff, x_out, y_out = john_perturbation.fresnel_propagation(e_field, dist)
    I_field = np.abs(e_field) ** 2
    I_field_ff = np.abs(e_field_ff) ** 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    peak_idx = np.unravel_index(np.argmax(I_field_ff), I_field_ff.shape)
    print(f"target = ({target[0]}, {target[1]}) m at z = {dist/1e3:.0f} km")
    print(f"far-field peak at x={x_out[peak_idx[1]]:.3f} m, y={y_out[peak_idx[0]]:.3f} m")
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
    im1 = ax1.imshow(I_field, cmap='plasma', origin='lower',
                      extent=[x_in.min(), x_in.max(), y_in.min(), y_in.max()])
    ax1.set_title('Intensity at Apertures')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    plt.colorbar(im1, ax=ax1)
 
    im2 = ax2.imshow(I_field_ff, cmap='plasma', origin='lower',
                      extent=[x_out.min(), x_out.max(), y_out.min(), y_out.max()])
    ax2.plot(target[0], target[1], '+', color='cyan', markersize=12, mew=2)
    ax2.set_title('Intensity at Far Field')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    plt.colorbar(im2, ax=ax2)
 
    plt.tight_layout()
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)  # FIX: no longer a hardcoded /workspace/ path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{out_dir}/intensity_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

#CNN test
if choice == "cnn":
    cnn = FFPhaseCNN()
    dummy_input = tf.random.normal((1, 256, 256, 1))
    _ = cnn(dummy_input)
    cnn.summary()