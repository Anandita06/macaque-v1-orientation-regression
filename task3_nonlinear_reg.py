from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit


project_dir = Path(__file__).parent
recording_path = (
    project_dir
    / "dataset"
    / "dataset"
    / "dataset1"
    / "106r001p26_spikes.mat"
)
selected_v1_neuron = 98
stimulus_duration_seconds = 1.28
polynomial_degrees = (1, 3, 4, 7)

# MATLAB stores spikes as trials x milliseconds x neurons.
with h5py.File(recording_path, "r") as recording:
    v1_spikes = recording[recording["spikes"][0, 0]][()].transpose(2, 1, 0)
    stimulus_codes = recording["stim"][()].reshape(-1).astype(int)

# Session 106r001p26 stimulus IDs and their orientations.
degrees_by_code = {
    1: 0,
    2: 22.5,
    3: 45,
    4: 67.5,
    5: 90,
    6: 112.5,
    7: 135,
    8: 157.5,
}

# Calculate the selected neuron's mean firing rate for each orientation.
trial_spike_counts = v1_spikes[selected_v1_neuron - 1].sum(axis=0)
trial_firing_rates = trial_spike_counts / stimulus_duration_seconds
stimulus_codes_present = np.unique(stimulus_codes)
orientation_degrees = np.array(
    [degrees_by_code[code] for code in stimulus_codes_present],
    dtype=float,
)
mean_firing_rates = np.array(
    [
        trial_firing_rates[stimulus_codes == code].mean()
        for code in stimulus_codes_present
    ]
)
orientation_order = np.argsort(orientation_degrees)
orientation_degrees = orientation_degrees[orientation_order]
mean_firing_rates = mean_firing_rates[orientation_order]


def gaussian_tuning(
    theta: np.ndarray, baseline: float, amplitude: float, mu: float, sigma: float
) -> np.ndarray:
    return baseline + amplitude * np.exp(-((theta - mu) ** 2) / (2 * sigma**2))


# Try multiple initial preferred orientations and widths to reduce sensitivity
# to the nonlinear optimizer's starting point.
lower_bounds = (0.0, 0.0, 0.0, 1.0)
upper_bounds = (np.inf, np.inf, 180.0, 360.0)
initial_guesses = [
    (float(mean_firing_rates.min()), float(mean_firing_rates.max() - mean_firing_rates.min()), mu, sigma)
    for mu in np.linspace(0.0, 180.0, 9)
    for sigma in (15.0, 30.0, 60.0, 120.0)
]

gaussian_fits = []
for initial_guess in initial_guesses:
    try:
        parameters, _ = curve_fit(
            gaussian_tuning,
            orientation_degrees,
            mean_firing_rates,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000,
        )
        residuals = mean_firing_rates - gaussian_tuning(
            orientation_degrees, *parameters
        )
        gaussian_fits.append((float(np.sum(residuals**2)), parameters))
    except RuntimeError:
        continue

if not gaussian_fits:
    raise RuntimeError("Gaussian curve fitting failed for all initial guesses.")

gaussian_sse, gaussian_parameters = min(gaussian_fits, key=lambda fit: fit[0])
gaussian_mse = gaussian_sse / mean_firing_rates.size
baseline, amplitude, preferred_orientation, tuning_width = gaussian_parameters

# Refit Task 2 polynomial models on the same eight means for a directly
# comparable in-sample MSE table.
polynomial_models = {
    degree: Polynomial.fit(orientation_degrees, mean_firing_rates, degree)
    for degree in polynomial_degrees
}
polynomial_mse = {
    degree: float(
        np.mean(
            (polynomial_models[degree](orientation_degrees) - mean_firing_rates)
            ** 2
        )
    )
    for degree in polynomial_degrees
}

print(f"Task 3: Gaussian nonlinear regression for V1 neuron {selected_v1_neuron}")
print("Fitted Gaussian parameters:")
print(f"  Baseline B: {baseline:.4f} spikes/second")
print(f"  Amplitude A: {amplitude:.4f} spikes/second")
print(f"  Preferred orientation mu: {preferred_orientation:.4f} degrees")
print(f"  Tuning width sigma: {tuning_width:.4f} degrees")
print(f"Gaussian training MSE: {gaussian_mse:.6f} (spikes/second)^2")
print("Training MSE comparison on the same eight orientation means:")
for degree in polynomial_degrees:
    print(f"  Polynomial degree {degree}: {polynomial_mse[degree]:.6f}")
print(f"  Gaussian: {gaussian_mse:.6f}")

best_polynomial_degree = min(polynomial_mse, key=polynomial_mse.get)
lowest_mse_model = (
    f"degree-{best_polynomial_degree} polynomial"
    if polynomial_mse[best_polynomial_degree] < gaussian_mse
    else "Gaussian"
)
print("\nTask 3 questions")
print(
    f"1. The {lowest_mse_model} has the lowest training MSE among the "
    "four polynomials and the Gaussian. Degree 7 interpolates the eight "
    "means, so its near-zero training MSE does not establish best "
    "generalization."
)
print(
    f"2. The Gaussian estimated preferred orientation is "
    f"mu={preferred_orientation:.2f} degrees."
)
print(
    f"3. Sigma={tuning_width:.2f} degrees describes the width of the fitted "
    "Gaussian around its preferred orientation. A larger sigma means a "
    "broader tuning curve; a smaller sigma means a narrower one."
)
print(
    "4. All four Gaussian parameters have biological interpretations: "
    f"B={baseline:.2f} spikes/second is baseline response, "
    f"A={amplitude:.2f} spikes/second is response amplitude, "
    f"mu={preferred_orientation:.2f} degrees is preferred orientation, "
    f"and sigma={tuning_width:.2f} degrees is tuning width."
)
print(
    "5. Compared with a polynomial, the Gaussian parameters have direct "
    "tuning interpretations and the model uses a compact, smooth shape. "
    "However, this ordinary Gaussian does not account for circularity and "
    "may fit poorly near the 0/180-degree boundary."
)

# Plot experimental means and the fitted Gaussian function.
plot_angles = np.linspace(orientation_degrees.min(), orientation_degrees.max(), 500)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    orientation_degrees,
    mean_firing_rates,
    color="black",
    zorder=5,
    label="Experimental means",
)
ax.plot(
    plot_angles,
    gaussian_tuning(plot_angles, *gaussian_parameters),
    color="tab:red",
    label=f"Gaussian fit (MSE={gaussian_mse:.3f})",
)
ax.set_title(f"Gaussian tuning fit for V1 neuron {selected_v1_neuron}")
ax.set_xlabel("Stimulus orientation (degrees)")
ax.set_ylabel("Mean firing rate (spikes/second)")
ax.set_xticks(orientation_degrees)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

figure_path = project_dir / "task3_gaussian_fit_neuron98.png"
fig.savefig(figure_path, dpi=150)
plt.close(fig)
print(f"Saved Gaussian-fit figure: {figure_path}")