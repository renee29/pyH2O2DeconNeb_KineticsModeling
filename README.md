# pyH2O2DeconNeb_KineticsModeling
Welcome to pyH2O2DeconNeb_KineticsModeling, a Python-based toolkit designed to simulate, analyze, and refine kinetic data related to hydrogen peroxide (H₂O₂) decontamination and nebulization processes. This repository integrates numerical modeling with a minimal Generative Adversarial Network (GAN) to generate synthetic datasets and improve their fidelity through GAN-based refinement. The goal is to foster reproducibility, encourage further research, and provide a flexible environment for exploring various kinetic scenarios.

Key Features.
Piecewise Exponential Modeling
Implements a piecewise exponential function to capture kinetic behaviors under different conditions.

Shape Function with Delay.
Simulates complex trajectories by introducing a curvature parameter and a delay time, allowing for convex, linear, and concave scenarios.

Experimental Uncertainty: Incorporates random noise within user-defined bounds to emulate realistic experimental conditions.

Curve Fitting: Utilizes both nonlinear (exponential) and linear least-squares methods to estimate parameters (e.g., rate constants, breakpoints) and to quantify the goodness-of-fit (R²).

Minimal GAN Refinement: This method leverages a compact GAN architecture (generator + discriminator) to refine synthetic data, enhancing the match to experimental endpoints and reducing artifacts.

Data Visualization.: Provides convenient plotting functions to illustrate raw synthetic data, fitted curves, and GAN-refined trajectories side-by-side for easy comparison.

Usage Notes.

Parameter Adjustments: epochs_gan, batch_size, and alpha_noise to control training duration, batch size, and the level of synthetic noise. The script includes comments that guide you through each function and its role in the simulation pipeline.

Data Fitting: Nonlinear fitting (curve_fit) is applied to estimate the exponential decay rate k and the breakpoint τ.
Linear fitting is a baseline comparison to illustrate the performance difference between exponential and simple linear models.

Plotting: Plots for each scenario (Convex, linear, Concave) are generated, showcasing both the original synthetic data and the refined data after GAN processing. Experimental endpoints and confidence intervals are highlighted to visualize uncertainty.

Contributing: We welcome contributions that expand the functionality of this toolkit or adapt it to new kinetic contexts. Please open an issue or submit a pull request with your suggested changes. Before making significant modifications, please discuss your proposal with the maintainers.

Acknowledgments: This project was developed as part of ongoing research at the University of Granada, focusing on H₂O₂ decontamination and nebulization dynamics. We are grateful for the support of our colleagues and funding agencies, who have made this work possible.
