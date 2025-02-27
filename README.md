# pyH2O2DeconNeb_KineticsModeling
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Define custom colors (hex codes)
# ---------------------------
RED       = "#8B0000"    # For experimental endpoints
BLUE      = "#0000FF"    # For refined GAN curve plot
YELLOW    = "#FFC300"    # For fitted exponential curve
GREEN     = "#DAF7A6"    # For fitted linear curve (dashed)
BLACK     = "#000000"    # For theoretical curve (legend)
FGreen    = "#228B22"    # For theoretical curve (legend)
INDIGO    = "#4B0082"    # For experimental endpoints (legend and shadow)
VIOLET    = "#9400D3"    # For synthetic data markers
ORANGE    = "#FF5733"    # (Not used here but available)

# ---------------------------
# 1) PIECEWISE EXPONENTIAL MODEL (for fitting)
# ---------------------------
def piecewise_exponential(t, N0, k, tau):
    # Returns N0 for t < tau; otherwise, returns N0 * exp(-k*(t-tau))
    return np.where(t < tau, N0, N0 * np.exp(-k * (t - tau)))

# ---------------------------
# 2) SHAPE FUNCTION WITH CURVATURE AND DELAY (for generating synthetic data)
# ---------------------------
def shape_function_with_delay(t, N0, N48, c, tau, T=48.0):
    """
    Generates an interpolated trajectory with a fixed delay:
      - For t < tau, returns N0.
      - For t >= tau, maps t to x in [0,1] and applies:
            F_c(x) = (N0 + (N48 - N0)*x) + c * x*(1 - x).
    Endpoints are forced to match experimental values.
    """
    result = np.full_like(t, N0, dtype=float)
    mask = t >= tau
    x = (t[mask] - tau) / (T - tau)
    line_part = N0 + (N48 - N0) * x
    curve_part = c * x * (1 - x)
    result[mask] = line_part + curve_part
    result[0] = N0
    result[-1] = N48
    return result

# ---------------------------
# 3) EXPERIMENTAL ENDPOINTS (original values and their uncertainties)
# ---------------------------
# Experimental values: 0.7443 ± 0.3734 (initial) and 0.3196 ± 0.0428 (final)
N0_exp      = 0.7443
N48_exp     = 0.3196
std_N0_exp  = 0.3734
std_N48_exp = 0.0428

# Experimental endpoints times:
t_real = np.array([0.0, 48.0])

# Normalization: use the maximum value (here, N0_exp) for scaling
norm_factor   = np.max([N0_exp, N48_exp])
N0_norm       = N0_exp / norm_factor
N48_norm      = N48_exp / norm_factor
N_real        = np.array([N0_exp, N48_exp]) / norm_factor

# Normalized uncertainties:
std_N0_norm = std_N0_exp / norm_factor
std_N48_norm = std_N48_exp / norm_factor

# For synthetic data, use a percentage of the larger normalized uncertainty
std_noise = 0.075 * max(std_N0_norm, std_N48_norm)

# ---------------------------
# 4) DEFINE 3 SCENARIOS (Convex, Linear, Concave)
# ---------------------------
diff = N0_norm - N48_norm
c_convex  = -1.3 * abs(diff)
c_linear  = 0.0
c_concave = 1.3 * abs(diff)

scenarios = {
    "Convex": c_convex,
    "Linear": c_linear,
    "Concave": c_concave
}

# ---------------------------
# 5) GENERATE SYNTHETIC DATA USING THE SHAPE FUNCTION WITH DELAY
# ---------------------------
n_point = 100
t_synth = np.linspace(0, 48, n_point)
np.random.seed(12345)

scenario_data = {}
for name, c_value in scenarios.items():
    N_teo = shape_function_with_delay(t_synth, N0_norm, N48_norm, c_value, tau=10.0, T=48.0)
    N_noise = N_teo.copy()
    for idx in range(1, len(N_noise) - 1):
        N_noise[idx] += np.random.normal(0, std_noise)
        N_noise[idx] = np.clip(N_noise[idx], N48_norm + 1e-6, N0_norm - 1e-6)
    N_noise[0] = N0_norm
    N_noise[-1] = N48_norm
    combined_max = max(np.max(N_noise), np.max(N_real))
    N_teo = N_teo / combined_max
    N_noise = N_noise / combined_max
    scenario_data[name] = {
        "t": t_synth,
        "N_teo": N_teo,
        "N_noise": N_noise,
        "c": c_value,
        "tau": 10.0
    }

# ---------------------------
# 6) WRAPPER FUNCTION FOR FITTING THE PIECEWISE EXPONENTIAL MODEL
# ---------------------------
def piecewise_fit(t, k, tau):
    return piecewise_exponential(t, N0_norm, k, tau)

# ---------------------------
# 7) PLOT ORIGINAL SYNTHETIC DATA WITH FITS (for each scenario)
# ---------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, (name, info) in enumerate(scenario_data.items()):
    t_vals  = info["t"]
    N_teo   = info["N_teo"].copy()  # Ensure endpoints are correct
    N_noise = info["N_noise"]
    c_val   = info["c"]
    
    popt, pcov = curve_fit(piecewise_fit, t_vals, N_noise, p0=[0.015, 10.0])
    k_fit, tau_fit = popt
    N_fit = piecewise_exponential(t_vals, N0_norm, k_fit, tau_fit)
    
    residuals = N_noise - N_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((N_noise - np.mean(N_noise))**2)
    r2exp = 1 - (ss_res / ss_tot)
    
    coeffs = np.polyfit(t_vals, N_noise, 1)
    slope, intercept = coeffs  # β₁ and β₀
    poly_linear = np.poly1d(coeffs)
    N_linear_fit = poly_linear(t_vals)
    residuals_linear = N_noise - N_linear_fit
    ss_res_linear = np.sum(residuals_linear**2)
    r2linear = 1 - (ss_res_linear / ss_tot)
    
    # Linear interpolation of experimental data
    exp_interp = np.linspace(N_real[0], N_real[1], len(t_vals))
    exp_err = np.linspace(std_N0_norm, std_N48_norm, len(t_vals))
    
    # << ADD: Larger markers for experimental start and end points >>
    axs[i].scatter(t_real, N_real, color=INDIGO, s=100, zorder=10, label="Exp. Data")

    # Experimental uncertainty shadow
    axs[i].fill_between(t_vals, exp_interp - exp_err, exp_interp + exp_err,
                        color=INDIGO, alpha=0.1, label="Exp. Uncertainty")
    
    # Least-squares fit using experimental data (two points; perfect fit yields R² = 1.0)
    coeffs_exp = np.polyfit(t_real, N_real, 1)
    slope_exp, intercept_exp = coeffs_exp
    exp_fit = slope_exp * t_vals + intercept_exp
    axs[i].plot(t_vals, exp_fit, color=INDIGO, lw=2, linestyle="--",
                label=f"LS-Fit: β₁={slope_exp:.4f}, R²=1.00")
     
    # Plot synthetic data
    axs[i].scatter(t_vals, N_noise, color=VIOLET, label="Synthetic Data")
    
    # Group 2: Nonlinear fit and theoretical curve
    axs[i].plot(t_vals, N_fit, color=ORANGE, lw=2,
                label=f"NL-Fit: k={k_fit:.3f}, τ={tau_fit:.1f}, R²={r2exp:.3f}")
    axs[i].plot(t_vals, N_linear_fit, color=YELLOW, linestyle="--", lw=2,
                label=f"L-Fit: β₁={slope:.4f}, R²={r2linear:.3f}")
    N_teo[0] = N0_norm
    N_teo[-1] = N48_norm
    # Uncomment below to plot the theoretical curve:
    # axs[i].plot(t_vals, N_teo, color=FGreen, lw=2, alpha=0.7, label=f"Theoretical (c={c_val:.3f})")
    
    # Create legend and assign colors to each entry according to its label
    legend1 = axs[i].legend(loc="upper right")
    for text in legend1.get_texts():
        label = text.get_text()
        if label.startswith("Exp. Data"):
            text.set_color(INDIGO)
        elif label.startswith("Exp. Uncertainty"):
            text.set_color(INDIGO)
        elif label.startswith("LS-Fit"):
            text.set_color(INDIGO)
        elif label.startswith("Synthetic Data"):
            text.set_color(VIOLET)
        elif label.startswith("NL-Fit:"):
            text.set_color(ORANGE)
        elif label.startswith("L-Fit:"):
            text.set_color(YELLOW)
    axs[i].add_artist(legend1)
    
    axs[i].set_title(name)
    axs[i].set_xlabel("Time (hours)")
    axs[i].set_ylabel("Load (N)" if i == 0 else "")
    axs[i].grid(False)
plt.tight_layout()
plt.savefig("plot_original_synthetic_data_P2.png", dpi=2000)
plt.show()

# ---------------------------
# 8) APPLY A MINIMAL GAN TO OPTIMIZE THE SYNTHETIC DATA FOR EACH SCENARIO
# ---------------------------
def build_generator(noise_dim=10, output_dim=n_point):
    model = Sequential([
        Dense(32, input_dim=noise_dim),
        LeakyReLU(alpha=0.2),
        Dense(64),
        LeakyReLU(alpha=0.2),
        Dense(output_dim, activation='linear')
    ])
    return model

def build_discriminator(input_dim=n_point):
    model = Sequential([
        Dense(64, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(32),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

noise_dim = 10
epochs_gan = 1000  # Adjust as needed
batch_size = 16
alpha_noise = 0.05   # Coefficient to adjust noise dispersion

gan_refined = {}

for scenario_name, info in scenario_data.items():
    base_curve = info["N_noise"].reshape(1, -1)
    train_samples = np.tile(base_curve, (100, 1))
    train_samples += np.random.normal(0, 0.01, train_samples.shape)
    
    generator = build_generator(noise_dim, output_dim=n_point)
    discriminator = build_discriminator(input_dim=n_point)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    
    discriminator.trainable = False
    gan_input = Input(shape=(noise_dim,))
    generated_curve = generator(gan_input)
    gan_output = discriminator(generated_curve)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    
    for epoch in range(epochs_gan):
        idx = np.random.randint(0, train_samples.shape[0], batch_size)
        real_samples = train_samples[idx]
        
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_samples = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    noise = np.random.normal(0, 1, (1, noise_dim))
    refined_curve = generator.predict(noise)[0]
    refined_curve = refined_curve * alpha_noise + info["N_noise"] * (1 - alpha_noise)
    epsilon = 1e-6
    refined_curve = np.clip(refined_curve, N48_norm + epsilon, N0_norm - epsilon)
    refined_curve[0] = N0_norm
    refined_curve[-1] = N48_norm
    combined_max_refined = max(np.max(refined_curve), np.max(N_real))
    refined_curve = refined_curve / combined_max_refined
    gan_refined[scenario_name] = refined_curve

# ---------------------------
# 9) PLOT GAN-REFINED DATA WITH FITS FOR EACH SCENARIO
# ---------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, (name, refined_data) in enumerate(gan_refined.items()):
    t_vals = t_synth
    popt, pcov = curve_fit(piecewise_fit, t_vals, refined_data, p0=[0.015, 10.0])
    k_fit, tau_fit = popt
    N_fit = piecewise_exponential(t_vals, N0_norm, k_fit, tau_fit)
    
    coeffs = np.polyfit(t_vals, refined_data, 1)
    slope, intercept = coeffs
    poly_linear = np.poly1d(coeffs)
    N_linear_fit = poly_linear(t_vals)
    
    # Calculate R² for the exponential fit
    residuals_exp = refined_data - N_fit
    ss_res_exp = np.sum(residuals_exp**2)
    ss_tot = np.sum((refined_data - np.mean(refined_data))**2)
    r2_exp = 1 - (ss_res_exp / ss_tot)
    
    # Calculate R² for the linear fit
    residuals_lin = refined_data - N_linear_fit
    ss_res_lin = np.sum(residuals_lin**2)
    r2_lin = 1 - (ss_res_lin / ss_tot)
    
    # Linear interpolation for experimental shadow
    exp_interp = np.linspace(N_real[0], N_real[1], len(t_vals))
    exp_err = np.linspace(std_N0_norm, std_N48_norm, len(t_vals))
    
    # << ADD: Larger markers for experimental start and end points >>
    axs[i].scatter(t_real, N_real, color=INDIGO, s=100, zorder=10, label="Exp. Data")

    axs[i].fill_between(t_vals, exp_interp - exp_err, exp_interp + exp_err,
                        color=INDIGO, alpha=0.1, label="Exp. Uncertainty")
    
    coeffs_exp = np.polyfit(t_real, N_real, 1)
    slope_exp, intercept_exp = coeffs_exp
    exp_fit = slope_exp * t_vals + intercept_exp
    axs[i].plot(t_vals, exp_fit, color=INDIGO, lw=2, linestyle="--",
                label=f"LS-Fit: β₁={slope_exp:.4f}, R²=1.00")

    axs[i].scatter(t_vals, refined_data, color=VIOLET, label="Synthetic Data (GAN)")
    
    # Include R² in the fit labels
    axs[i].plot(t_vals, N_fit, color=ORANGE, lw=2,
                label=f"NL-Fit: k={k_fit:.3f}, τ={tau_fit:.1f}, R²={r2_exp:.3f}")
    axs[i].plot(t_vals, N_linear_fit, color=YELLOW, linestyle="--", lw=2,
                label=f"L-Fit: β₁={slope:.4f}, R²={r2_lin:.3f}")
    
    N_theo = shape_function_with_delay(t_vals, N0_norm, N48_norm, scenarios[name], tau=10.0, T=48.0)
    N_theo[0] = N0_norm
    N_theo[-1] = N48_norm
    # Uncomment the line below to include the theoretical curve:
    # axs[i].plot(t_vals, N_theo, color=FGreen, lw=2, alpha=0.7, label=f"Theoretical (c={scenarios[name]:.3f})")
    
    legend1 = axs[i].legend(loc="upper right")
    for text in legend1.get_texts():
        label = text.get_text()
        if label.startswith("Exp. Data"):
            text.set_color(INDIGO)
        elif label.startswith("Exp. Uncertainty"):
            text.set_color(INDIGO)
        elif label.startswith("LS-Fit:"):
            text.set_color(INDIGO)
        elif label.startswith("Synthetic Data (GAN)"):
            text.set_color(VIOLET)
        elif label.startswith("NL-Fit:"):
            text.set_color(ORANGE)
        elif label.startswith("L-Fit:"):
            text.set_color(YELLOW)
    axs[i].add_artist(legend1)
    
    axs[i].set_title(name)
    axs[i].set_xlabel("Time (hours)")
    axs[i].set_ylabel("Load (N)" if i == 0 else "")
    axs[i].grid(False)

plt.tight_layout()
plt.savefig("plot_synthetic_data_GAN_P2.png", dpi=2000)
plt.show()

# ---------------------------
# 10) PRINT FIT RESULTS FOR ORIGINAL SCENARIOS TO CONSOLE
# ---------------------------
for name, info in scenario_data.items():
    t_vals  = info["t"]
    N_noise = info["N_noise"]
    popt, pcov = curve_fit(piecewise_fit, t_vals, N_noise, p0=[0.015, 10.0])
    k_fit, tau_fit = popt
    N_fit = piecewise_exponential(t_vals, N0_norm, k_fit, tau_fit)
    residuals = N_noise - N_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((N_noise - np.mean(N_noise))**2)
    r2_exp = 1 - (ss_res / ss_tot)
    
    coeffs = np.polyfit(t_vals, N_noise, 1)
    slope, intercept = coeffs
    poly_linear = np.poly1d(coeffs)
    N_linear_fit = poly_linear(t_vals)
    residuals_linear = N_noise - N_linear_fit
    ss_res_linear = np.sum(residuals_linear**2)
    r2_linear = 1 - (ss_res_linear / ss_tot)
    
    print(f"=== SCENARIO {name} (Original Synthetic Data) ===")
    print(f"  Curvature (c)      = {info['c']:.4f}")
    print(f"  Fitted k           = {k_fit:.4f}")
    print(f"  Fitted τ           = {tau_fit:.4f}")
    print(f"  R² (Exponential)   = {r2_exp:.4f}")
    print(f"  Linear Fit: β₁ = {slope:.4f}, β₀ = {intercept:.4f}, R² = {r2_linear:.4f}")
    print("  Covariance matrix:\n", pcov)
    print("------------------------------------------")

# ---------------------------
# 11) PRINT FIT RESULTS FOR GAN-REFINED DATA TO CONSOLE
# ---------------------------
for name, refined in gan_refined.items():
    t_vals = t_synth
    popt, pcov = curve_fit(piecewise_fit, t_vals, refined, p0=[0.015, 10.0])
    k_fit, tau_fit = popt
    N_fit = piecewise_exponential(t_vals, N0_norm, k_fit, tau_fit)
    residuals = refined - N_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((refined - np.mean(refined))**2)
    r2_exp = 1 - (ss_res / ss_tot)
    
    coeffs = np.polyfit(t_vals, refined, 1)
    slope, intercept = coeffs
    poly_linear = np.poly1d(coeffs)
    N_linear_fit = poly_linear(t_vals)
    residuals_linear = refined - N_linear_fit
    ss_res_linear = np.sum(residuals_linear**2)
    r2_linear = 1 - (ss_res_linear / ss_tot)
    
    print(f"=== SCENARIO {name} (GAN-Refined Synthetic Data) ===")
    print(f"  Fitted k           = {k_fit:.4f}")
    print(f"  Fitted τ           = {tau_fit:.4f}")
    print(f"  R² (Exponential)   = {r2_exp:.4f}")
    print(f"  Linear Fit: β₁ = {slope:.4f}, β₀ = {intercept:.4f}, R² = {r2_linear:.4f}")
    print("  Covariance matrix:\n", pcov)
    print("------------------------------------------")
    
#%%
