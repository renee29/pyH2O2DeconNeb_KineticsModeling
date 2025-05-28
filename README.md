# pyH2O2DeconNeb_KineticsModeling: Advanced Kinetic Modeling and GAN-Enhanced Synthetic Data Generation

## Overview

`pyH2O2DeconNeb_KineticsModeling.py` is a Python script designed for the advanced analysis of kinetic data, particularly relevant to processes like hydrogen peroxide decomposition or similar time-dependent phenomena observed in microfluidic or electrochemical systems. The script focuses on generating realistic synthetic datasets under various kinetic profiles (convex, linear, concave) anchored by experimental endpoints. It further employs a Generative Adversarial Network (GAN) to refine these synthetic datasets, aiming to produce data that, while synthetic, closely mimics potential experimental trajectories. The script includes functionalities for fitting these datasets with both non-linear (piecewise exponential decay) and linear models, providing robust statistical evaluation ($R^2$) and parameter estimation.

## Core Functionality & Scientific Context

The primary scientific objective is to explore plausible kinetic pathways between sparsely measured experimental data points. This is achieved by:

1.  **Defining Theoretical Kinetic Scenarios:** Establishing distinct qualitative behaviors (convex, linear, concave decay) for the process under investigation between initial ($N_0$) and final ($N_{48}$) observed states.
2.  **Generating Synthetic Data:** Creating time-series data that adheres to these scenarios, incorporating a delay parameter ($\tau$) and stochastic noise to simulate experimental conditions.
3.  **Kinetic Model Fitting:**
    *   **Piecewise Exponential Decay:** A non-linear model representing an initial lag phase followed by first-order decay, $N(t) = N_0 \cdot e^{-k(t-\tau)}$ for $t \ge \tau$.
    *   **Linear Decay:** A baseline linear model for comparison.
4.  **GAN-Based Data Refinement:** Utilizing a Generative Adversarial Network to learn from the characteristics of the initially generated synthetic data and produce refined versions that are potentially more representative of complex underlying processes.
5.  **Comparative Analysis:** Evaluating the fit parameters and goodness-of-fit ($R^2$) for both the initial and GAN-refined synthetic datasets across all scenarios.

This approach is particularly useful in situations where extensive experimental data collection is costly or challenging, allowing researchers to computationally explore the implications of different kinetic assumptions.

## Mathematical Models Implemented

### 1. Piecewise Exponential Decay Model (for fitting)

The concentration $N(t)$ at time $t$ is modeled as:
$N(t) = \begin{cases} N_0 & \text{if } t < \tau \\ N_0 \cdot e^{-k(t-\tau)} & \text{if } t \ge \tau \end{cases}$
where:
-   $N_0$: Initial concentration (or load).
-   $k$: Rate constant.
-   $\tau$: Delay time or lag phase duration.

### 2. Shape Function with Curvature and Delay (for synthetic data generation)

To generate synthetic data between experimental endpoints $N_0$ (at $t=0$) and $N_{48}$ (at $t=T=48$), considering a delay $\tau$:
For $t < \tau$, $N(t) = N_0$.
For $t \ge \tau$, let $x = \frac{t-\tau}{T-\tau}$. The normalized trajectory $F_c(x)$ is given by:
$F_c(x) = (N_0 + (N_{48} - N_0)x) + c \cdot x(1-x)$
where:
-   $c$: Curvature parameter.
    -   $c < 0$: Convex decay.
    -   $c = 0$: Linear decay.
    -   $c > 0$: Concave decay.
-   The function ensures that $N(0) = N_0$ and $N(T) = N_{48}$.

## Key Steps in the Script

1.  **Setup:**
    *   Import necessary libraries: `numpy`, `matplotlib`, `scipy.optimize`, `tensorflow`.
    *   Define custom color codes for consistent plotting.

2.  **Model Definitions:**
    *   `piecewise_exponential(t, N0, k, tau)`: Implements the piecewise exponential decay model.
    *   `shape_function_with_delay(t, N0, N48, c, tau, T)`: Implements the synthetic data generation function.

3.  **Experimental Data Input & Normalization:**
    *   Defines experimental endpoints ($N_0^{exp}$, $N_{48}^{exp}$) and their uncertainties.
    *   Normalizes these values using the maximum experimental value as the `norm_factor`.
    *   Calculates normalized uncertainties and a standard deviation for noise addition to synthetic data.

4.  **Scenario Definition:**
    *   Defines three kinetic scenarios: "Convex", "Linear", and "Concave" by setting appropriate values for the curvature parameter `c`.

5.  **Synthetic Data Generation:**
    *   For each scenario, generates a theoretical curve using `shape_function_with_delay`.
    *   Adds Gaussian noise to this curve (excluding endpoints) to create "noisy" synthetic data.
    *   Ensures data points are clipped within the normalized experimental bounds.
    *   Stores time points, theoretical curves, and noisy synthetic data.

6.  **Fitting Procedures:**
    *   `piecewise_fit(t, k, tau)`: A wrapper for `piecewise_exponential` used with `scipy.optimize.curve_fit`.
    *   Fits both the piecewise exponential model and a simple linear model to the noisy synthetic data for each scenario.
    *   Calculates $R^2$ values for both fits.

7.  **Visualization (Initial Synthetic Data + Fits):**
    *   Generates a 3-panel plot (`plot_original_synthetic_data_P2.png`):
        *   Each panel corresponds to a scenario (Convex, Linear, Concave).
        *   Displays:
            *   Experimental endpoints and uncertainty.
            *   A linear fit to the experimental endpoints.
            *   The noisy synthetic data points.
            *   The fitted piecewise exponential curve.
            *   The fitted linear curve.
        *   Legends include fit parameters ($k, \tau, \beta_1$) and $R^2$ values.

8.  **Generative Adversarial Network (GAN) Implementation:**
    *   **Generator (`build_generator`):** A simple feedforward neural network with LeakyReLU activations, transforming a noise vector into a synthetic data curve.
    *   **Discriminator (`build_discriminator`):** A feedforward neural network that classifies input curves as "real" (from the initial synthetic dataset) or "fake" (generated by the GAN).
    *   **Training Loop:** For each scenario:
        *   The GAN is trained for a specified number of `epochs_gan`.
        *   The discriminator learns to distinguish between initial synthetic samples and GAN-generated samples.
        *   The generator learns to produce samples that fool the discriminator.

9.  **Application of GAN to Refine Synthetic Data:**
    *   After training, the generator produces a refined data curve for each scenario.
    *   This refined curve is a weighted average of the GAN output and the original noisy synthetic data, controlled by `alpha_noise`.
    *   Endpoints are re-enforced, and values are clipped.

10. **Visualization (GAN-Refined Data + Fits):**
    *   Generates a second 3-panel plot (`plot_synthetic_data_GAN_P2.png`) similar to the first, but using the GAN-refined synthetic data.
    *   This allows for a direct comparison of model fits before and after GAN refinement.

11. **Output of Fit Parameters:**
    *   Prints detailed fit results (fitted $k, \tau, \beta_0, \beta_1$, $R^2$ values, and covariance matrices) to the console for both the original synthetic data and the GAN-refined data for each scenario.

## Dependencies

The script requires the following Python libraries:

*   **NumPy:** For numerical operations (`pip install numpy`)
*   **Matplotlib:** For plotting (`pip install matplotlib`)
*   **SciPy:** For curve fitting (`pip install scipy`)
*   **TensorFlow:** For GAN implementation (`pip install tensorflow`)
    *   The script prints the TensorFlow version used (e.g., `2.x.x`).

It is recommended to use a virtual environment to manage these dependencies.

## Usage

1.  **Ensure all dependencies are installed.**
2.  **Place the script `pyH2O2DeconNeb_KineticsModeling.py` in your desired directory.**
3.  **Run the script from the command line:**
    ```bash
    python pyH2O2DeconNeb_KineticsModeling.py
    ```

## Output

The script will produce:

1.  **Console Output:**
    *   TensorFlow version.
    *   Detailed fit parameters ($k, \tau, \beta_1, \beta_0$), $R^2$ values, and covariance matrices for:
        *   Each scenario (Convex, Linear, Concave) using the original noisy synthetic data.
        *   Each scenario (Convex, Linear, Concave) using the GAN-refined synthetic data.

2.  **Image Files (saved in the script's directory):**
    *   `plot_original_synthetic_data_P2.png`: A multi-panel figure showing the original synthetic data, experimental bounds, and the corresponding model fits (piecewise exponential and linear) for each scenario.
    *   `plot_synthetic_data_GAN_P2.png`: A multi-panel figure showing the GAN-refined synthetic data, experimental bounds, and the corresponding model fits for each scenario.

## Customization

Several parameters within the script can be adjusted to explore different conditions or refine the models:

*   **Experimental Data (`N0_exp`, `N48_exp`, `std_N0_exp`, `std_N48_exp`):** Modify these to reflect different experimental measurements.
*   **Synthetic Data Generation:**
    *   `n_point`: Number of points in the synthetic time series.
    *   `tau` (in `shape_function_with_delay` call): The delay time for synthetic data generation (default is 10.0 hours).
    *   `c_convex`, `c_concave`: Curvature parameters for non-linear scenarios.
    *   `std_noise`: Controls the amount of noise added to synthetic data, based on experimental uncertainties.
*   **Curve Fitting:**
    *   `p0` (in `curve_fit`): Initial guesses for $k$ and $\tau$ for the piecewise exponential fit.
*   **GAN Parameters:**
    *   `noise_dim`: Dimensionality of the noise vector input to the generator.
    *   `epochs_gan`: Number of training epochs for the GAN.
    *   `batch_size`: Batch size for GAN training.
    *   `alpha_noise`: Weighting factor for combining GAN output with original noisy data during refinement.
    *   Network architecture (layers, neurons) in `build_generator` and `build_discriminator`.

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Citation

If you use this script or the methodologies presented in your research, I kindly request that you cite it as follows:

Fabregas, R. (2025). Kinetic Modeling with GAN-Enhanced Synthetic Data Generation (Version V1.0.1). Zenodo. https://doi.org/10.5281/zenodo.15536388

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15536388.svg)](https://doi.org/10.5281/zenodo.15536388)

@software{fabregas_2025_15536388,
  author       = {Fabregas, Rene},
  title        = {{Kinetic Modeling with GAN-Enhanced Synthetic Data Generation}},
  month        = oct,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {V1.0.1},
  doi          = {10.5281/zenodo.15536388},
  url          = {https://doi.org/10.5281/zenodo.15536388}
}

