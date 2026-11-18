### unused CPU path:
    # def input_e_quick(self):
    #     input_e = np.zeros_like(self.base_fields[0], dtype=np.complex128)
    #     for n, base in enumerate(self.base_fields):
    #         input_e += base * np.exp(1j * self.chan_rel_phases[n])
    #     return input_e
    # def set_rel_phases(self,
    #                 rel_phases=None):
    #     if rel_phases is not None:
    #         self.chan_rel_phases = rel_phases
    #     else:
    #         self.chan_rel_phases = np.zeros(self.rel_array.arrayvector.shape[0])
    # def fresnel_czt(self, input_field, z: np.float64, k):
    #     dim_y, dim_x = input_field.shape
    #     ssl = self.spatial_samp_length
    #     x1 = (np.arange(dim_x) - dim_x / 2) * ssl
    #     y1 = (np.arange(dim_y) - dim_y / 2) * ssl
    #     X1, Y1 = np.meshgrid(x1,y1)
    #     Q1 = np.exp((1j * k * (X1 ** 2 + Y1 ** 2)) / (2 * z))
    #     array_extent = (self.element_positions[:,0].max() - self.element_positions[:,0].min())
    #     d_array = array_extent + 2 * self.apert_radi
    #     diff_limited_spot = self.wavelength * z / d_array
    #     out_window = 20 * diff_limited_spot
    #     E_mod = input_field * Q1
    #     out_size = self.grid_length
    #     out_center = [0,0]
    #     dx_out = out_window / out_size
    #     x2 = out_center[0] + (np.arange(out_size) - out_size // 2) * dx_out
    #     y2 = out_center[1] + (np.arange(out_size) - out_size // 2) * dx_out
    #     def make_czt(f_axis, n_in):
    #         df, f0 = f_axis[1] - f_axis[0], f_axis[0]
    #         W = np.exp(-1j * 2 * np.pi * df * ssl)
    #         A = np.exp(1j * 2 * np.pi * f0 * ssl)
    #         return CZT(n_in, out_size, W, A)
    #     tmp = make_czt(x2 / (self.wavelength * z), dim_x)(E_mod, axis=1)
    #     tmp = make_czt(y2 / (self.wavelength * z), dim_y)(tmp, axis=0)
    #     prefactor = np.exp(1j * k * z) / (1j * self.wavelength * z)
    #     X2, Y2 = np.meshgrid(x2, y2)
    #     Q2 = np.exp(1j * k * (X2 ** 2 + Y2 ** 2) / (2 * z))
    #     output_field = prefactor * Q2 * tmp * ssl ** 2
    #     return output_field, x2, y2

    # def fresnel_propagation(self, input_field, z: np.float64):
    #     dim_y, dim_x = input_field.shape
    #     ssl = self.spatial_samp_length
    #     k = 2 * np.pi / self.wavelength
    #     if self.apply_turb:
    #         z_atmo = min(z, self.atmo_height)
    #         screen_num = min(len(self.phase_screens), int(round(z/500)))
    #         field = input_field.copy()
    #         screen_heights = np.linspace(0, z_atmo, num = screen_num + 1)
    #         z_current = 0.0
    #         for n in range(screen_num):
    #             z_current = screen_heights[n+1]
    #             field = ifft2(fft2(field) * self.h_layers[n])
    #             field *= np.exp(1j * self.phase_screens[n])
    #         dist_remainder = z-z_current
    #         if dist_remainder > 0:
    #             return self.fresnel_czt(input_field = field, z = dist_remainder, k = k)
    #         else:
    #             x_out = (np.arange(dim_x) - dim_x / 2) * ssl
    #             y_out = (np.arange(dim_y) - dim_y / 2) * ssl
    #             return field, x_out, y_out
    #     return self.fresnel_czt(input_field = input_field, z = z, k = k)

    # def compute_optimal_phases(self, n_iter, tol=1e-6, return_m = False):
    #     n_elem = len(self.base_fields)
    #     x_f, y_f, z = self.target_position
    #     bucket_radius = self.grid_length / 40
    #     bucket_rows = []
    #     mask = None
    #     for n in range(n_elem):
    #         field_out, x_out, y_out = self.fresnel_propagation(self.base_fields[n], z)
    #         if mask is None:
    #             cx = np.argmin(np.abs(x_out - x_f))
    #             cy = np.argmin(np.abs(y_out - y_f))
    #             Y, X = np.mgrid[0:field_out.shape[0], 0:field_out.shape[1]]
    #             r = np.sqrt((X-cx) ** 2 + (Y-cy) ** 2)
    #             mask = r <= bucket_radius
    #         bucket_rows.append(field_out[mask])
    #     E = np.stack(bucket_rows, axis=0)
    #     M = E.conj() @ E.T

    #     eigvals, eigvecs = np.linalg.eigh(M)
    #     c = np.exp(1j * np.angle(eigvecs[:, -1]))

    #     # baseline candidate — guarantees the result can never be worse than doing nothing
    #     c_ones = np.ones(n_elem, dtype=np.complex128)
    #     power_ones = np.real(c_ones.conj() @ M @ c_ones)

    #     prev_power = -np.inf
    #     power = np.real(c.conj() @ M @ c)
    #     n_used = 0
    #     for i in range(n_iter):
    #         c = np.exp(1j * np.angle(M @ c))
    #         power = np.real(c.conj() @ M @ c)
    #         n_used = i + 1
    #         if power - prev_power < tol * abs(power):     # relative tolerance
    #             break
    #         prev_power = power

    #     ub = n_elem * eigvals[-1]
    #     gap = (ub - power) / ub
    #     top_eig_fraction = eigvals[-1] / np.trace(M).real

    #     if power_ones > power:            # safety net — should basically never trigger if math is right
    #         c = c_ones
    #         power = power_ones

    #     c = c * np.exp(-1j * np.angle(c[self.center_idx]))

    #     optimal_phases = np.angle(c)
    #     if return_m:
    #         return optimal_phases, power, n_used, mask, M
    #     return optimal_phases, power, n_used, mask

# if choice == "optsim":
#     dist = np.float64(384000000)
#     target = [0, 0, dist]
#     john_array = ArrayDesign(2, 'square')
#     john_perturbation = BeamPerturbator(ssl = 16e-4, targ_pos = target, array_des = john_array, 
#                                         Ideal = True, grid_length = 256)
#     x_in, y_in = john_perturbation.input_e_field_base()
#     john_perturbation.set_rel_phases()
#     e_field = john_perturbation.input_e_quick()
#     e_field_ff, x_out, y_out = john_perturbation.fresnel_propagation(e_field, dist)
#     I_field = np.abs(e_field) ** 2
#     I_field_ff = np.abs(e_field_ff) ** 2
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     peak_idx = np.unravel_index(np.argmax(I_field_ff), I_field_ff.shape)
#     print(f"target = ({target[0]}, {target[1]}) m at z = {dist/1e3:.0f} km")
#     print(f"far-field peak at x={x_out[peak_idx[1]]:.3f} m, y={y_out[peak_idx[0]]:.3f} m")
 
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
#     im1 = ax1.imshow(I_field, cmap='plasma', origin='lower',
#                       extent=[x_in.min(), x_in.max(), y_in.min(), y_in.max()], norm=LogNorm(vmin=I_field.max()*1e-4, vmax=I_field.max()))
#     ax1.set_title('Intensity at Apertures')
#     ax1.set_xlabel('x [m]')
#     ax1.set_ylabel('y [m]')
#     plt.colorbar(im1, ax=ax1)
 
#     im2 = ax2.imshow(I_field_ff, cmap='plasma', origin='lower',
#                       extent=[x_out.min(), x_out.max(), y_out.min(), y_out.max()], norm=LogNorm(vmin=I_field_ff.max()*1e-4, vmax=I_field_ff.max()))
#     ax2.plot(target[0], target[1], '+', color='cyan', markersize=12, mew=2)
#     ax2.set_title('Intensity at Far Field')
#     ax2.set_xlabel('x [m]')
#     ax2.set_ylabel('y [m]')
#     plt.colorbar(im2, ax=ax2)
 
#     plt.tight_layout()
#     out_dir = 'outputs'
#     os.makedirs(out_dir, exist_ok=True)  # FIX: no longer a hardcoded /workspace/ path
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(f'{out_dir}/intensity_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
#     plt.close(fig)

# ----------------------------------------------------------------------------
# Branch: check the GPU pipeline against the CPU simulator before trusting it
# ----------------------------------------------------------------------------
# if choice == "gpu_validate":
#     grid_data_size = int(input("Grid size?\n"))
#     grid_ssl = int(input("Sampling (0.1mm)?\n"))
#     array = ArrayDesign(2, 'hex')
#     dist = 384000000
#     sim = BeamPerturbator(ssl=grid_ssl * 1e-4, targ_pos=[0, 0, dist], array_des=array,
#                           r500=0.2, Ideal=False, grid_length=grid_data_size)
#     sim.input_e_field_base()
#     gen = GPUDataGenerator(sim, seed=123)
#     N, d = grid_data_size, sim.turb_model.pixel_scale
 
#     # (1) screen statistics vs soapy, at the strongest slab's r0
#     r0_test = float(gen.r0.min())
#     seps = [1, 2, 4, 8, 16, 32, 64, 128]
#     def struct_fn(scrs):    # per-screen structure function, [n_screens, n_seps]
#         return np.array([[np.mean((s[:, j:] - s[:, :-j]) ** 2) for j in seps] for s in scrs])
#     n_s = 64                # 12 is too few: one screen's strength varies ~14% even at 1 px
#     soapy_s = [soapy.atmosphere.makePhaseScreens(1, r0_test, N, d, np.inf, 0.01, True, SH=True)[0]
#                for _ in range(n_s)]
#     flat_r0 = tf.constant(np.full(gen.S, r0_test ** (-5 / 6), np.float32))
#     gpu_s = np.concatenate([gen.generate_screens(tf.constant(1), flat_r0).numpy()[0]
#                             for _ in range(-(-n_s // gen.S))])[:n_s]
#     Ds, Dg = struct_fn(soapy_s), struct_fn(gpu_s)
#     theory = 6.88 * (np.array(seps) * d / r0_test) ** (5 / 3)
#     print(f"r0 = {r0_test*100:.1f} cm, {n_s} screens each")
#     print("separation(px)  D_soapy        D_gpu          gpu/soapy  z-score  soapy/Kolmogorov")
#     for k, (j, t) in enumerate(zip(seps, theory)):
#         ma, sa = Ds[:, k].mean(), Ds[:, k].std() / np.sqrt(n_s)
#         mb, sb = Dg[:, k].mean(), Dg[:, k].std() / np.sqrt(n_s)
#         print(f"{j:8d}      {ma:.4f}±{sa:.4f}  {mb:.4f}±{sb:.4f}   {mb / ma:6.3f}   {(mb - ma) / np.hypot(sa, sb):+5.2f}"
#               f"     {ma / t:6.2f}")
#     print("|z| < 2 at every separation -> statistically indistinguishable from soapy.")
 
#     # (2) propagation + labels vs the CPU code on identical screens
#     results, screens = gen.run_batch(1)
#     sim.phase_screens = screens.numpy()[0].astype(np.float64)
#     best_cpu, _, n_cpu, mask = sim.compute_optimal_phases(100)
#     sim.set_rel_phases()
#     E0, _, _ = sim.fresnel_propagation(sim.input_e_quick(), dist)
#     sim.set_rel_phases(best_cpu)
#     E1, _, _ = sim.fresnel_propagation(sim.input_e_quick(), dist)
#     sim.set_rel_phases()
#     I0, I1 = np.abs(E0) ** 2, np.abs(E1) ** 2
#     r = results[0]
#     rel = lambda A, B_: np.linalg.norm(A - B_) / np.linalg.norm(B_)
#     dphi = np.degrees(np.abs(np.angle(np.exp(1j * (r['best_params'] - best_cpu)))))
#     print(f"init image rel diff GPU vs CPU: {rel(r['init_image'], I0):.1e}")
#     print(f"opt  image rel diff GPU vs CPU: {rel(r['opt_image'], I1):.1e}")
#     print(f"label max diff: {dphi.max():.3f} deg   iterations GPU/CPU: {r['iterations']}/{n_cpu}")
#     print(f"bucket mask identical: {np.array_equal(mask, gen.mask)}")