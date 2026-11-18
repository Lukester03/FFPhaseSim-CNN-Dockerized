import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import math
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model

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
            self.wavelengths = [5.0, 10.0]
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

    def __init__(self, kernel_size=21, num_modes=36, stride=4, **kwargs):
        super().__init__(**kwargs)
        self.zernike_kernel = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_modes = num_modes

    def build(self, input_grid):
        zernike_gen = ZernikeFilterBank(self.kernel_size, self.num_modes)
        kernel_np = zernike_gen.generate_filter_bank()
        in_channels = input_grid[-1]
        kernel_np = np.tile(kernel_np, [1, 1, in_channels, 1])
        # store Zernike polynomial filter as untrainable weight to ensure consistency
        self.zernike_kernel = self.add_weight(
            name='zern_kern',
            shape=(self.kernel_size, self.kernel_size, in_channels, self.num_modes),
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
            strides=[1, self.stride, self.stride, 1],
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
                 num_orientations=4,
                 wavelengths=None,
                 gamma=0.5,
                 stride=4,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Store parameters for serialization
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.wavelengths = wavelengths
        self.stride = stride
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
            strides=[1, self.stride, self.stride, 1],
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
    def __init__(self, num_features, pos_dim, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            layers.Dense(64, activation='leaky_relu', kernel_initializer='he_normal'),
            layers.Dense(64, activation='leaky_relu', kernel_initializer='he_normal'),
            layers.Dense(2 * num_features,
                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
        ])
        self.pos_dim = pos_dim

    def build(self, input_shape):
        self.mlp.build((None, self.pos_dim))
        super().build(input_shape)

    def call(self, features, position_encoded):
        gb = self.mlp(position_encoded)
        gamma, beta = tf.split(gb, 2, axis=-1)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return features * (1 + gamma) + beta

class FFPhaseCNN(Model):
    def __init__(self, mode_count = 16, 
        gabor_kernel_size = 9, gabor_orientations = 8, gabor_lmbda = None, laser_count = 7,
        element_positions = None, raw_channels = 16, stride = 4, scale = 4, **kwargs):
        super().__init__(**kwargs)
        self.num_modes = mode_count
        self.gabor_kern_size = gabor_kernel_size
        self.gabor_orientations = gabor_orientations
        self.gabor_lmbda = gabor_lmbda
        self.num_lasers = laser_count
        if element_positions is None:
            element_positions = [[-0.8660254, -0.5, 0.],
                                 [-0.8660254, 0.5, 0.],
                                 [ 0., -1., 0.],
                                 [ 0., 0., 0.],
                                 [ 0., 1., 0.],
                                 [ 0.8660254, -0.5, 0.],
                                 [ 0.8660254, 0.5, 0.]]
        self.element_positions = np.asarray(element_positions, dtype=np.float32)

        self.scale = int(scale)
        odd = lambda k: int(round(k)) + (1- int(round(k)) % 2)
        zern_kernel = odd(11 * self.scale)
        gabor_kernel = odd(gabor_kernel_size * self.scale)
        gabor_waves = [w * self.scale for w in (gabor_lmbda if gabor_lmbda is not None else [5.0, 10.0])]
        in_stride = stride * self.scale

        self.zernike_conv = ZernikeConvLayer(kernel_size = zern_kernel, num_modes = mode_count, stride = in_stride, name = "zernike_features")
        self.gabor_conv = GaborConvLayer(kernel_size = gabor_kernel, 
        num_orientations = gabor_orientations, wavelengths = gabor_waves, stride=in_stride, gamma = 0.5, name = "gabor_features")

        # Processor Channel Counts:
        self.Zernike_chan = mode_count
        self.Gabor_chan = gabor_orientations * 4
        self.Raw_chan = int(raw_channels)

        #early downsample to improve model speed
        self.i_downsample = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(16, odd(1.5 * (in_stride // 2)), strides = in_stride // 2, padding='SAME', activation='leaky_relu', kernel_initializer = 'he_normal'),  # 128×128
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, strides=2, padding='SAME', activation='leaky_relu', kernel_initializer = 'he_normal'),  # 64×64
        ], name='downsample_i')
        
        # Processors
        self.zernike_processor = tf.keras.Sequential([
            layers.BatchNormalization(name='zernike_bn'),
            layers.Conv2D(self.Zernike_chan, 1, kernel_initializer = 'he_normal', activation='leaky_relu', name='zernike_compress'),
            layers.BatchNormalization(),
            layers.Conv2D(self.Zernike_chan, 3, kernel_initializer = 'he_normal', padding='SAME', activation='leaky_relu', 
                         name='zernike_process1'),
            layers.BatchNormalization(),
            layers.Conv2D(self.Zernike_chan, 3, padding='SAME', activation='leaky_relu', 
                         name='zernike_process2'),
        ], name='zernike_processor')
        
        self.gabor_processor = tf.keras.Sequential([
            layers.BatchNormalization(name='gabor_bn'),
            layers.Conv2D(self.Gabor_chan, 1, activation='leaky_relu', kernel_initializer='he_normal', name='gabor_compress'),
            layers.BatchNormalization(),
            layers.Conv2D(self.Gabor_chan, 3, padding='SAME', activation='leaky_relu', kernel_initializer = 'he_normal', 
                         name='gabor_process1'),
            layers.BatchNormalization(),
            layers.Conv2D(self.Gabor_chan, 3, padding='SAME', activation='leaky_relu', kernel_initializer = 'he_normal', 
                         name='gabor_process2'),
        ], name='gabor_processor')
        
        self.intensity_processor = tf.keras.Sequential([
            layers.Conv2D(self.Raw_chan, 7, padding='SAME', activation='leaky_relu', kernel_initializer = 'he_normal', 
                         name='intensity_conv1'),
            layers.BatchNormalization(name='intensity_bn1'),
            layers.Conv2D(self.Raw_chan, 5, padding='SAME', activation='leaky_relu', kernel_initializer = 'he_normal', 
                         name='intensity_conv2'),
        ], name='intensity_processor')

        # extra prelaser downsample
        self.pre_laser_downsample = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(self.Zernike_chan + self.Gabor_chan + self.Raw_chan, 3, strides = 2, padding = 'SAME', activation='leaky_relu', kernel_initializer = 'he_normal')
        ], name = 'pre_laser_downsample')
        
        #decodes each laser's values
        self.shared_decoder = tf.keras.Sequential([
            layers.Conv2D(64, 3, padding='SAME', activation='leaky_relu', kernel_initializer='he_normal'),
            layers.GroupNormalization(groups = 8, axis = -1),
            layers.Conv2D(32, 3, padding='SAME', activation='leaky_relu', kernel_initializer='he_normal'),
            layers.GroupNormalization(groups = 8, axis = -1),
            layers.Conv2D(16, 3, padding='SAME', activation='leaky_relu', kernel_initializer='he_normal'),
            layers.Conv2D(2, 3, padding='SAME'),
        ], name='shared_decoder')

        # coefficients
        self.coeff_head = tf.keras.Sequential([
            layers.Dense(64, activation='leaky_relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dense(self.num_modes, activation='tanh', bias_initializer=tf.keras.initializers.RandomUniform(-0.3,0.3)),
            layers.Lambda(lambda x: x * np.pi)
        ], name = 'coeff_head')

        pos_dim = 2 + 4 * 6  # matches sinusoidal_position_encoding defaults
        self.film = FiLMConditioner(self.Zernike_chan + self.Gabor_chan + self.Raw_chan, pos_dim, name="film_late")
        self.global_pool = layers.GlobalAveragePooling2D()

    # sinusoidal pos encoding
    @staticmethod
    def sinusoidal_position_encoding(positions, num_bands=6, max_freq=8.0):
        """positions: [N, 2] raw (x, y) -> [N, 2 + 4*num_bands] encoded."""
        freqs = tf.constant(
            np.geomspace(1.0, max_freq, num_bands), dtype=tf.float32
        )  # [num_bands]
        pos = tf.cast(positions, tf.float32)  # [N, 2]
        # [N, 2, num_bands]
        scaled = pos[:, :, None] * freqs[None, None, :] * np.pi
        sin_feats = tf.sin(scaled)
        cos_feats = tf.cos(scaled)
        encoded = tf.concat([
            pos,
            tf.reshape(sin_feats, [tf.shape(pos)[0], -1]),
            tf.reshape(cos_feats, [tf.shape(pos)[0], -1]),
        ], axis=-1)  # [N, 2 + 4*num_bands]
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            'mode_count': self.num_modes,
            'gabor_kernel_size': self.gabor_kern_size,
            'gabor_orientations': self.gabor_orientations,
            'gabor_lmbda': self.gabor_lmbda,
            'laser_count': self.num_lasers,
            'element_positions': self.element_positions.tolist(),
            'scale': self.scale,
        })
        return config

    def call(self, inputs, training = False):
        B = tf.shape(inputs)[0]
        H, W, C_in = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        x = tf.reshape(tf.transpose(inputs, [0, 3, 1, 2]), [-1, H, W, 1])

        down_inputs = self.i_downsample(x,training = training)
        zernike_down = self.zernike_conv(x)
        gabor_down = self.gabor_conv(x)

        z_features = self.zernike_processor(zernike_down, training = training)
        g_features = self.gabor_processor(gabor_down, training = training)
        i_features = self.intensity_processor(down_inputs, training = training)

        feats = tf.concat([z_features, g_features, i_features], axis=-1)
        h, w, Fc = feats.shape[1], feats.shape[2], feats.shape[3]
        feats = tf.reshape(feats, [B, C_in, h, w, Fc])
        combined = tf.reshape(tf.transpose(feats, [0, 2, 3, 1, 4]), [B, h, w, C_in * Fc])
        combined = self.pre_laser_downsample(combined, training = training)

        L = self.num_lasers
        pos_all = tf.constant(self.element_positions[:, :2], dtype=tf.float32)
        pos_encoded = self.sinusoidal_position_encoding(pos_all)          # [L, pos_dim]
        pos_encoded_tiled = tf.tile(pos_encoded[None, ...], [B, 1, 1])
        pos_encoded_flat = tf.reshape(pos_encoded_tiled, [B * L, -1])  # [B*L, pos_dim]

        Hc, Wc, Cc = combined.shape[1], combined.shape[2], combined.shape[3]
        combined_tiled_full = tf.repeat(combined[:, None, ...], L, axis=1)
        combined_tiled_full = tf.reshape(combined_tiled_full, [B * L, Hc, Wc, Cc])

        conditioned = self.film(combined_tiled_full, pos_encoded_flat)
        pooled = self.global_pool(conditioned)
        coeffs = self.coeff_head(pooled, training = training)
        piston_coeff = coeffs[:, 0]

        phase_components = self.shared_decoder(conditioned, training=training)
        vec = tf.reduce_mean(phase_components, axis=[1, 2])

        cp, sp = tf.cos(piston_coeff), tf.sin(piston_coeff)
        rot = tf.stack([vec[:, 0] * cp - vec[:, 1] * sp,
                        vec[:, 0] * sp + vec[:, 1] * cp], axis=-1)
        rot = tf.math.l2_normalize(rot, axis=-1, epsilon=1e-6)
        return tf.reshape(rot, [B, L, 2])

    def build(self, input_shape):
        dummy = tf.zeros(input_shape)
        self.call(dummy, training=False)
        super().build(input_shape)

class DatasetStore:
    def __init__(self, h5_path, image_shape, num_channels, flush_count = 8, initial_size = 100, with_diversity = False):
        self.h5_path = Path(h5_path)
        self.catalog_path = self.h5_path.with_suffix('.catalog.csv')
        self.image_shape = image_shape
        self.num_channels = num_channels
        self.flush_count = flush_count
        self._catalog_rows = []
        self._n = 0
        self._allocated = 0

        self.f = h5py.File(self.h5_path, 'a')
        if 'images' not in self.f:
            self._allocated = initial_size
            self.f.create_dataset('images', shape=(0, *image_shape), maxshape=(None, *image_shape),
                                   dtype='float32', chunks=(1, *image_shape), compression=None)
            self.f.create_dataset('opt_images', shape=(0, *image_shape), maxshape=(None, *image_shape),
                                   dtype='float32', chunks=(1, *image_shape), compression=None)
            if with_diversity:
                self.f.create_dataset('div_images', shape = (0, *image_shape), maxshape=(None, *image_shape),
                                      dtype = 'float32', chunks = (1, *image_shape), compression = None)
            self.f.create_dataset('labels', shape=(0, num_channels), maxshape=(None, num_channels),
                                   dtype='float32')
        if with_diversity and 'div_images' not in self.f:
            raise ValueError(f"{self.h5_path} already exists without 'div_images'; write div data to new file")
        self._ds_names = [n for n in ('images', 'opt_images', 'div_images', 'labels') if n in self.f]
        self._allocated = self.f['images'].shape[0]
        self._n = self._count_actual_examples()

    def _count_actual_examples(self):
        if self.catalog_path.exists():
            catalog = pd.read_csv(self.catalog_path)
            return len(catalog)
        return 0

    def _ensure_capacity(self, needed):
        current_size = self.f['images'].shape[0]
        if needed > self._allocated:
            new_size = max(needed, self._allocated * 2 if current_size > 0 else self._allocated)
            for ds_name in self._ds_names:
                self.f[ds_name].resize(new_size, axis=0)
            self._allocated = new_size

    def add_example(self, image, opt_image, label_rad, div_image = None,  **metadata):
        if ('div_images' in self._ds_names) != (div_image is not None):
            raise ValueError('pass div_image exactly when the file stores div_images')
        self._ensure_capacity(self._n+1)
        n = self._n
        self.f['images'][n] = image
        self.f['opt_images'][n] = opt_image
        if div_image is not None:
            self.f['div_images'][n] = div_image
        self.f['labels'][n] = label_rad

        row = {'index':n, **metadata}
        self._catalog_rows.append(row)
        self._n += 1

        if self._n % self.flush_count == 0:
            self.flush()
    
    @staticmethod
    def make_dataset(h5_path, batch_size=8, shuffle=True, val_split=0.1,
                     intensity_clip_percentile=99.999, crop=512, pool=1, diversity=None):
        h5_path = Path(h5_path)
 
        def reduce(imgs):
            """[n, G, G] -> [n, crop/pool, crop/pool], centre crop then block mean."""
            imgs = np.asarray(imgs, dtype=np.float32)
            G = imgs.shape[1]
            if crop is not None and crop < G:
                c0 = (G - crop) // 2
                imgs = imgs[:, c0:c0 + crop, c0:c0 + crop]
            if pool > 1:
                n, h, w = imgs.shape
                imgs = imgs[:, :h // pool * pool, :w // pool * pool]
                imgs = imgs.reshape(n, h // pool, pool, w // pool, pool).mean(axis=(2, 4))
            return imgs
 
        with h5py.File(h5_path, 'r') as f:
            n_total = f['images'].shape[0]
            num_channels = f['labels'].shape[1]
            if h5_path.with_suffix('.catalog.csv').exists():
                n_total = len(pd.read_csv(h5_path.with_suffix('.catalog.csv')))
 
            indices = np.arange(n_total)
            if shuffle:
                np.random.default_rng(42).shuffle(indices)
            n_val = int(n_total * val_split)
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            use_div = ('div_images' in f) if diversity is None else bool(diversity)
            channels = ['images'] +(['div_images'] if use_div else [])
            n_samples = min(100, len(train_idx))
            sample_idx = np.sort(np.random.default_rng(42).choice(train_idx, n_samples, replace=False))
            sample = np.stack([reduce(f[c][sample_idx]) for c in channels], axis = -1)
            norm_scale = np.percentile(sample.reshape(-1, len(channels)), intensity_clip_percentile, axis=0).astype(np.float32)
            grid_size = sample.shape[1]
        del sample
        ns = tf.constant(norm_scale, tf.float32)

        def preprocess(image, label):
            image = tf.clip_by_value(image, 0.0, ns) / ns
            return tf.math.log1p(image * 1000.0), label
 
        def build(idx_array, training):
            idx_sorted = np.sort(idx_array)
            with h5py.File(h5_path, 'r') as f:
                # read in chunks so the full 1024^2 images never sit in RAM all at once
                all_images = np.concatenate([
                    np.stack([reduce(f[c][idx_sorted[s:s + 64]]) for c in channels], axis = -1)
                    for s in range(0, len(idx_sorted), 64)])
                all_labels = f['labels'][idx_sorted].astype(np.float32)
            ds = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
            ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            if training:
                ds = ds.shuffle(len(idx_sorted), seed=42, reshuffle_each_iteration=True)
                ds = ds.map(symmetry_augment, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(batch_size, drop_remainder=training)   # keep every validation example
            return ds.prefetch(tf.data.AUTOTUNE)
 
        return build(train_idx, True), build(val_idx, False), grid_size, norm_scale

    def flush(self):
        self.f.flush()
        pd.DataFrame(self._catalog_rows).to_csv(
            self.catalog_path, mode='a', header = not self.catalog_path.exists(), index=False
        )
        self._catalog_rows = []

    def close(self):
        self.flush()
        for ds_name in self._ds_names:
            self.f[ds_name].resize(self._n, axis=0)
        self.f.close()


def center_crop(imgs, crop):
    """[n, G, G] -> central [n, crop, crop]; crop=None or >= G returns imgs unchanged.
    Must match make_dataset's crop (c0 = (G - crop) // 2)."""
    G = imgs.shape[1]
    if crop is None or crop >= G:
        return imgs
    c0 = (G - crop) // 2
    return imgs[:, c0:c0 + crop, c0:c0 + crop, ...]

def predict_outer_phases(model, imgs, norm_scale, crop=None, batch=8):
    """[n, N, N] or [n, N, N, C] raw intensity -> [n, L] predicted outer phases (rad).
    crop must be the crop the model was trained with (from load_trained_model)."""
    out = []
    for s in range(0, len(imgs), batch):
        x = tf.constant(preprocess_images(center_crop(imgs[s:s + batch], crop), norm_scale))
        out.append(predicted_angles(model(x, training=False)).numpy())
    return np.concatenate(out)
 
 
def twin_aware_compare(pred, labels, perm):
    """Same branch rule as twin_invariant_loss. Returns per-sample dict of arrays."""
    wrap = lambda v: np.angle(np.exp(1j * v))
    tw = -labels[:, perm]
    l_orig = (1 - np.cos(labels - pred)).mean(1)
    l_twin = (1 - np.cos(tw - pred)).mean(1)
    use_twin = l_twin < l_orig
    eff = np.where(use_twin[:, None], tw, labels)
    err = wrap(eff - pred)                                   # signed, rad, [n, L]
    return {'err_deg': np.degrees(np.abs(err)), 'signed_err': err, 'use_twin': use_twin,
            'loss': np.minimum(l_orig, l_twin), 'branch_gap': np.abs(l_orig - l_twin),
            'eff': eff}

REF_FRINGE_PX = 35

def fringe_geometry(array, apert_radi=0.05, clip_ratio=0.6, grid_N=1024, verbose=True):
    pitch = 2 * apert_radi
    pos = np.asarray(array.arrayvector)[:, :2] * pitch
    d_array = (pos[:, 0].max() - pos[:, 0].min()) + 2 * apert_radi
    px = lambda X: X * d_array * grid_N / 20.0
    b = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    b = b[b > 1e-9]
    g = {'fringe_nearest_px': px(1 / b.min()),                # coarsest fringes (neighbouring elements)
         'fringe_finest_px': px(1 / b.max()),                 # finest fringes (longest baseline)
         'envelope_radius_px': px(1 / (np.pi * clip_ratio * apert_radi)),
         'airy_zero_px': px(1.22 / (2 * apert_radi))}
    if verbose:
        print("far-field geometry (output px): " + ", ".join(f"{k} {v:.1f}" for k, v in g.items()))
    return g

def read_run_config(run_label):
    import json
    with open(f'outputs/train_logs/{run_label}_config.json') as f:
        return json.load(f) 
 
def load_trained_model(run_label, grid_N, array):
    """Rebuild the model exactly as the training branch does and load best weights.
    grid_N is the size of the STORED images; the model is built for the size it was
    trained on (the crop, if make_dataset cropped).
    Returns (model, norm_scale, outer_positions, center_idx, crop)."""
    cfg = read_run_config(run_label)
    norm_scale = np.atleast_1d(np.asarray(cfg['norm_scale'], dtype=np.float32))   # one per input image
    C_in = len(norm_scale)
    # make_dataset's default is crop=512, pool=1; record 'crop' in the config to override
    crop = cfg.get('crop', 512)
    crop = None if (crop is None or crop >= grid_N) else int(crop)
    in_N = grid_N if crop is None else crop
    center_idx = array.find_center_index()
    outer_positions = np.delete(array.arrayvector, center_idx, axis=0)
    geom = fringe_geometry(array, grid_N=grid_N, verbose=False)
    scale = max(1, int(round(geom['fringe_nearest_px'] / REF_FRINGE_PX)))
    model = FFPhaseCNN(laser_count=len(outer_positions), element_positions=outer_positions, scale=scale)
    model.build(input_shape=(1, in_N, in_N, C_in))
    model.load_weights(f'outputs/ckpts/{run_label}/best.weights.h5')
    print(f"loaded {run_label}: scale {scale}, stored images {grid_N}x{grid_N}, "
          f"network input {in_N}x{in_N}x{C_in} ({'centre crop' if crop else 'full image'}"
          f"{', + diversity image' if C_in == 2 else ''}), norm_scale {np.round(norm_scale, 12).tolist()}")
    return model, norm_scale, outer_positions, center_idx, crop
 
 
def preprocess_images(imgs, norm_scale):
    """Same transform as make_dataset's preprocess. imgs [n, N, N] (one image) or
    [n, N, N, C] (zero-phase + diversity); norm_scale has one value per channel."""
    imgs = np.asarray(imgs, dtype=np.float32)
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    ns = np.atleast_1d(np.asarray(norm_scale, dtype=np.float32))
    return np.log1p(np.clip(imgs, 0.0, ns) / ns * 1000.0)
 
 
def predict_outer_phases(model, imgs, norm_scale, crop=None, batch=8):
    """[n, N, N] or [n, N, N, C] raw intensity -> [n, L] predicted outer phases (rad).
    crop must be the crop the model was trained with (from load_trained_model)."""
    out = []
    for s in range(0, len(imgs), batch):
        x = tf.constant(preprocess_images(center_crop(imgs[s:s + batch], crop), norm_scale))
        out.append(predicted_angles(model(x, training=False)).numpy())
    return np.concatenate(out)
 
 
def twin_aware_compare(pred, labels, perm):
    """Same branch rule as twin_invariant_loss. Returns per-sample dict of arrays."""
    wrap = lambda v: np.angle(np.exp(1j * v))
    tw = -labels[:, perm]
    l_orig = (1 - np.cos(labels - pred)).mean(1)
    l_twin = (1 - np.cos(tw - pred)).mean(1)
    use_twin = l_twin < l_orig
    eff = np.where(use_twin[:, None], tw, labels)
    err = wrap(eff - pred)                                   # signed, rad, [n, L]
    return {'err_deg': np.degrees(np.abs(err)), 'signed_err': err, 'use_twin': use_twin,
            'loss': np.minimum(l_orig, l_twin), 'branch_gap': np.abs(l_orig - l_twin),
            'eff': eff}

class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup from init_lr to peak_lr, then cosine decay to
    peak_lr * min_lr_frac at total_steps."""
 
    def __init__(self, peak_lr, total_steps, warmup_steps=100,
                 init_lr=1e-6, min_lr_frac=0.02):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.init_lr = float(init_lr)
        self.min_lr_frac = float(min_lr_frac)
 
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        w = float(max(self.warmup_steps, 1))
        min_lr = self.peak_lr * self.min_lr_frac
        warm = self.init_lr + (self.peak_lr - self.init_lr) * (step / w)
        decay_len = float(max(self.total_steps - self.warmup_steps, 1))
        prog = tf.clip_by_value((step - w) / decay_len, 0.0, 1.0)
        cosine = min_lr + 0.5 * (self.peak_lr - min_lr) * (1.0 + tf.cos(np.pi * prog))
        return tf.where(step < w, warm, cosine)
 
    def get_config(self):
        return {'peak_lr': self.peak_lr, 'total_steps': self.total_steps,
                'warmup_steps': self.warmup_steps, 'init_lr': self.init_lr,
                'min_lr_frac': self.min_lr_frac}

def update_ratio_medians(all_vars, before, n_norm):
    """True ||dw||/||w|| for multiplicative weights only (kernels, gammas).
    Betas and biases are skipped: their natural scale is ~0, so the ratio is undefined."""
    r_main, r_norm = [], []
    for i, (v, b) in enumerate(zip(all_vars, before)):
        if 'beta' in v.name or 'bias' in v.name:
            continue
        r = float(tf.norm(v - b)) / (float(tf.norm(b)) + 1e-12)
        (r_norm if i < n_norm else r_main).append(r)
    return float(np.median(r_main)), float(np.median(r_norm))

def build_variable_groups(model):
    """Returns (norm_vars, other_vars, all_vars, all_names, n_norm)."""
    gain_layers = [model.film.mlp.layers[-1]]
    gain_var_ids = set()
    gain_vars = []
    for layer in gain_layers:
        for v in layer.trainable_variables:
            gain_vars.append(v)
            gain_var_ids.add(id(v))
 
    norm_vars, other_vars = [], []
    for v in model.trainable_variables:
        if id(v) in gain_var_ids:
            continue                      # folded in below
        if classify(v.name) == 'norm_scale_shift':
            norm_vars.append(v)
        else:
            other_vars.append(v)
    norm_vars = norm_vars + gain_vars
 
    all_vars = norm_vars + other_vars
    all_names = [v.name for v in all_vars]
    return norm_vars, other_vars, all_vars, all_names, len(norm_vars)

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

def predicted_angles(y_pred):
    """[B, L, 2] unit vectors -> [B, L] angles in radians."""
    return tf.atan2(y_pred[..., 1], y_pred[..., 0])

def twin_permutation(positions):
    """perm[i] = index of the element at -position[i]. Works on the full array
    (arrayvector) or on outer_positions. Raises if the array isn't centrosymmetric."""
    p = np.asarray(positions)[:, :2]
    d = np.linalg.norm(p[:, None, :] + p[None, :, :], axis=-1)
    perm = np.argmin(d, axis=1)
    if d[np.arange(len(p)), perm].max() > 1e-6:
        raise ValueError("array is not centrosymmetric; no exact twin")
    return perm

AUG_PERMS = np.array([[0, 1, 2, 3, 4, 5],
                      [4, 5, 2, 3, 0, 1],
                      [1, 0, 3, 2, 5, 4],
                      [5, 4, 3, 2, 1, 0]], np.int32)

def symmetry_augment(image, label):
    """image [H, W, 1] (after preprocess), label [6] radians, outer-element order."""
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    fx = tf.logical_or(tf.equal(k, 1), tf.equal(k, 3))
    fy = tf.logical_or(tf.equal(k, 2), tf.equal(k, 3))
    image = tf.cond(fx, lambda: tf.roll(tf.reverse(image, [1]), 1, axis=1), lambda: image)
    image = tf.cond(fy, lambda: tf.roll(tf.reverse(image, [0]), 1, axis=0), lambda: image)
    return image, tf.gather(label, tf.gather(AUG_PERMS, k))
 
def twin_invariant_loss(y_true, y_pred, perm, reduce=True):
    """1 - cos loss scored against whichever of {phi, twin(phi)} the prediction is
    closer to, per sample. twin(phi)_i = -phi_perm(i). Center reference stays 0.
    y_true: [B, L] radians, y_pred: [B, L, 2] unit vectors.
    Returns (loss, effective_labels) where effective_labels is the chosen branch."""
    tw = -tf.gather(y_true, perm, axis=1)
 
    def per_sample(t):
        target = tf.stack([tf.cos(t), tf.sin(t)], axis=-1)
        return tf.reduce_mean(1.0 - tf.reduce_sum(target * y_pred, axis=-1), axis=1)  # [B]
 
    l_orig, l_twin = per_sample(y_true), per_sample(tw)
    use_twin = l_twin < l_orig
    l_min = tf.minimum(l_orig, l_twin)
    eff = tf.where(use_twin[:, None], tw, y_true)
    return (tf.reduce_mean(l_min) if reduce else l_min), eff
 
def direct_phase_loss(y_true, y_pred, reduce=True):
    target = tf.stack([tf.cos(y_true), tf.sin(y_true)], axis = -1)
    l = tf.reduce_mean(1.0 - tf.reduce_sum(target * y_pred, axis=-1), axis=1)
    return (tf.reduce_mean(l) if reduce else l), y_true

def twin_floor_deg(labels, perm):
    """Best achievable *standard* MAE (deg) for any image-only predictor if the
    image cannot distinguish phi from twin(phi): predict the circular midpoint."""
    tw = -labels[:, perm]
    mid = np.angle(np.exp(1j * labels) + np.exp(1j * tw))
    err = np.abs(np.angle(np.exp(1j * (labels - mid))))
    return float(np.degrees(err).mean())