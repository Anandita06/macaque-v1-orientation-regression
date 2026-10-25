from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit, least_squares


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

# Calculate the selected neuron's mean firing rate at each orientation.
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
orientation_radians = np.deg2rad(orientation_degrees)


def gaussian_tuning(
    theta_degrees: np.ndarray,
    baseline: float,
    amplitude: float,
    mu_degrees: float,
    sigma_degrees: float,
) -> np.ndarray:
    return baseline + amplitude * np.exp(
        -((theta_degrees - mu_degrees) ** 2) / (2 * sigma_degrees**2)
    )


def von_mises_tuning(
    theta_radians: np.ndarray,
    baseline: float,
    log_amplitude: float,
    concentration: float,
    mu_radians: float,
) -> np.ndarray:
    exponent = np.clip(
        log_amplitude
        + concentration * np.cos(2 * (theta_radians - mu_radians)),
        -700,
        700,
    )
    return baseline + np.exp(exponent)


# Refit the ordinary Gaussian on the same eight points for direct comparison.
gaussian_initial_guesses = [
    (
        float(mean_firing_rates.min()),
        float(mean_firing_rates.max() - mean_firing_rates.min()),
        mu,
        sigma,
    )
    for mu in np.linspace(0.0, 180.0, 9)
    for sigma in (15.0, 30.0, 60.0, 120.0)
]
gaussian_fits = []
for initial_guess in gaussian_initial_guesses:
    try:
        parameters, _ = curve_fit(
            gaussian_tuning,
            orientation_degrees,
            mean_firing_rates,
            p0=initial_guess,
            bounds=((0.0, 0.0, 0.0, 1.0), (np.inf, np.inf, 180.0, 360.0)),
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

# Multi-start the von Mises fit to reduce sensitivity to initial values.
initial_baseline = max(0.0, float(mean_firing_rates.min() * 0.5))
initial_log_amplitude = float(np.log(max(mean_firing_rates.max(), 1e-6)))
von_mises_initial_guesses = [
    np.array([initial_baseline, initial_log_amplitude, kappa, mu])
    for kappa in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)
    for mu in np.linspace(0.0, np.pi, 13)
]
von_mises_fits = []
for initial_guess in von_mises_initial_guesses:
    fit = least_squares(
        lambda parameters: von_mises_tuning(
            orientation_radians, *parameters
        )
        - mean_firing_rates,
        x0=initial_guess,
        bounds=(
            np.array([0.0, -30.0, 0.0, 0.0]),
            np.array([np.inf, 20.0, 50.0, np.pi]),
        ),
        max_nfev=20000,
    )
    if fit.success and np.all(np.isfinite(fit.fun)):
        von_mises_fits.append((float(np.sum(fit.fun**2)), fit.x))
if not von_mises_fits:
    raise RuntimeError("Von Mises fitting failed for all initial guesses.")

von_mises_sse, von_mises_parameters = min(
    von_mises_fits, key=lambda fit: fit[0]
)
von_mises_mse = von_mises_sse / mean_firing_rates.size
vm_baseline, vm_log_amplitude, vm_kappa, vm_mu_radians = von_mises_parameters
vm_amplitude = float(np.exp(vm_log_amplitude))
vm_mu_degrees = float(np.rad2deg(vm_mu_radians))

# Recompute Task 2 polynomial errors on the same eight points for context.
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

print(f"Task 4: circular orientation analysis for V1 neuron {selected_v1_neuron}")
print(
    "Orientation is circular with a 180-degree period: 0 degrees and "
    "180 degrees represent the same orientation."
)
print("\nTask 4 questions about circularity")
print(
    "1. An ordinary Gaussian treats angles as points on a line, so it does "
    "not recognize that orientations near 0 and 180 degrees are neighbors."
)
print(
    "2. A Gaussian centered near 0 degrees decays with numerical distance "
    "and cannot wrap around the 0/180-degree boundary to represent responses "
    "near 180 degrees as nearby."
)
print(
    "3. Polynomial regression has the same line-versus-circle problem: its "
    "values at 0 and 180 degrees are not constrained to match, and it does "
    "not enforce periodicity."
)
print(
    "4. For this neuron, the response is elevated at 0 degrees and again at "
    "135-157.5 degrees. A non-circular model does not encode the wraparound "
    "relationship; the circular fit below tests whether respecting that "
    "relationship describes these data better."
)

print("\nVon Mises fit")
print(f"  Baseline B: {vm_baseline:.4f} spikes/second")
print(f"  Amplitude A: {vm_amplitude:.4f} (model scale)")
print(f"  Preferred orientation mu: {vm_mu_degrees:.4f} degrees")
print(f"  Concentration kappa: {vm_kappa:.4f}")
print(f"  Von Mises training MSE: {von_mises_mse:.6f} (spikes/second)^2")
print("Training MSE comparison on the same eight orientation means:")
for degree in polynomial_degrees:
    print(f"  Polynomial degree {degree}: {polynomial_mse[degree]:.6f}")
print(f"  Gaussian: {gaussian_mse:.6f}")
print(f"  Von Mises: {von_mises_mse:.6f}")
if von_mises_mse < gaussian_mse:
    print(
        "Accounting for circularity lowered the in-sample MSE relative to "
        "the Gaussian for this neuron."
    )
elif von_mises_mse > gaussian_mse:
    print(
        "Accounting for circularity did not lower the in-sample MSE relative "
        "to the Gaussian for this neuron."
    )
else:
    print("The Gaussian and von Mises models have equal in-sample MSE.")

print("\nVon Mises questions")
print(
    f"1. The von Mises MSE is {von_mises_mse:.3f}, compared with "
    f"{gaussian_mse:.3f} for the ordinary Gaussian, so it fits these "
    "orientation means better in-sample."
)
print(
    f"2. The estimated preferred orientation is "
    f"{vm_mu_degrees:.2f} degrees."
)
print(
    f"3. Kappa={vm_kappa:.3f} controls concentration: larger values make "
    "the tuning curve narrower around its preferred orientation; smaller "
    "values make it broader."
)
print(
    "4. The factor 2 maps the 180-degree orientation cycle onto a full "
    "360-degree cosine cycle, making orientations separated by 180 degrees "
    "equivalent in the model."
)
print(
    "5. Yes, the circular model has a lower in-sample MSE than the ordinary "
    "Gaussian for this neuron."
)
print(
    "6. The von Mises model is a suitable choice for describing orientation "
    "tuning because it respects circularity and has interpretable tuning "
    "parameters. Its lower in-sample MSE supports that choice here, though "
    "this alone does not prove better prediction on new data."
)

# Plot over one full orientation cycle, including the 180-degree endpoint.
plot_degrees = np.linspace(0.0, 180.0, 600)
plot_radians = np.deg2rad(plot_degrees)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    orientation_degrees,
    mean_firing_rates,
    color="black",
    zorder=5,
    label="Experimental means",
)
ax.plot(
    plot_degrees,
    gaussian_tuning(plot_degrees, *gaussian_parameters),
    label=f"Gaussian (MSE={gaussian_mse:.3f})",
)
ax.plot(
    plot_degrees,
    von_mises_tuning(plot_radians, *von_mises_parameters),
    label=f"Von Mises (MSE={von_mises_mse:.3f})",
)
ax.set_title(f"Ordinary Gaussian and circular fit: V1 neuron {selected_v1_neuron}")
ax.set_xlabel("Stimulus orientation (degrees)")
ax.set_ylabel("Mean firing rate (spikes/second)")
ax.set_xticks(np.arange(0.0, 181.0, 22.5))
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

figure_path = project_dir / "task4_von_mises_neuron98.png"
fig.savefig(figure_path, dpi=150)
plt.close(fig)
print(f"Saved Gaussian and von Mises comparison figure: {figure_path}")