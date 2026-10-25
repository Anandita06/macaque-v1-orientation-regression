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

degrees_by_code = {
    1: 0.0,
    2: 22.5,
    3: 45.0,
    4: 67.5,
    5: 90.0,
    6: 112.5,
    7: 135.0,
    8: 157.5,
}

# MATLAB stores spikes as trials x milliseconds x neurons.
with h5py.File(recording_path, "r") as recording:
    v1_spikes = recording[recording["spikes"][0, 0]][()].transpose(2, 1, 0)
    stimulus_codes = recording["stim"][()].reshape(-1).astype(int)

unknown_codes = set(np.unique(stimulus_codes)) - degrees_by_code.keys()
if unknown_codes:
    raise ValueError(f"Unrecognized stimulus codes: {sorted(unknown_codes)}")
if v1_spikes.shape[2] != stimulus_codes.size:
    raise ValueError("Spike trial count does not match the stimulus-code count.")

orientation_codes = np.unique(stimulus_codes)
orientation_degrees = np.array(
    [degrees_by_code[code] for code in orientation_codes],
    dtype=float,
)
orientation_order = np.argsort(orientation_degrees)
orientation_codes = orientation_codes[orientation_order]
orientation_degrees = orientation_degrees[orientation_order]

trial_firing_rates = (
    v1_spikes.sum(axis=1) / stimulus_duration_seconds
)
mean_rates = np.array(
    [
        trial_firing_rates[:, stimulus_codes == code].mean(axis=1)
        for code in orientation_codes
    ]
).T
selected_v1_index = selected_v1_neuron - 1
if selected_v1_index < 0 or selected_v1_index >= v1_spikes.shape[0]:
    raise ValueError(f"V1 neuron {selected_v1_neuron} is not in this recording.")

# Rank neurons by pooled within-orientation trial variability, excluding the
# already-selected neuron. This identifies two reproducibly noisier examples.
within_orientation_variances = np.zeros(v1_spikes.shape[0])
for neuron_index in range(v1_spikes.shape[0]):
    residuals = np.concatenate(
        [
            trial_firing_rates[
                neuron_index, stimulus_codes == code
            ]
            - mean_rates[neuron_index, orientation_index]
            for orientation_index, code in enumerate(orientation_codes)
        ]
    )
    within_orientation_variances[neuron_index] = np.mean(residuals**2)
noise_order = np.argsort(within_orientation_variances)[::-1]
noisier_neuron_indices = [
    index for index in noise_order if index != selected_v1_index
][:2]
comparison_neuron_indices = [selected_v1_index, *noisier_neuron_indices]


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


def fit_gaussian(
    fit_angles: np.ndarray, fit_rates: np.ndarray
) -> np.ndarray:
    initial_guesses = [
        (
            float(max(0.0, fit_rates.min())),
            float(max(1e-6, fit_rates.max() - fit_rates.min())),
            mu,
            sigma,
        )
        for mu in np.linspace(0.0, 180.0, 7)
        for sigma in (20.0, 60.0, 120.0)
    ]
    fits = []
    for initial_guess in initial_guesses:
        parameters, _ = curve_fit(
            gaussian_tuning,
            fit_angles,
            fit_rates,
            p0=initial_guess,
            bounds=((0.0, 0.0, 0.0, 1.0), (np.inf, np.inf, 180.0, 360.0)),
            maxfev=20000,
        )
        residuals = fit_rates - gaussian_tuning(
            fit_angles, *parameters
        )
        fits.append((float(np.sum(residuals**2)), parameters))
    return min(fits, key=lambda fit: fit[0])[1]


def fit_von_mises(fit_angles: np.ndarray, fit_rates: np.ndarray) -> np.ndarray:
    angles_radians = np.deg2rad(fit_angles)
    initial_baseline = max(0.0, float(fit_rates.min() * 0.5))
    initial_log_amplitude = float(np.log(max(fit_rates.max(), 1e-6)))
    initial_guesses = [
        np.array([initial_baseline, initial_log_amplitude, kappa, mu])
        for kappa in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)
        for mu in np.linspace(0.0, np.pi, 13)
    ]
    fits = []
    for initial_guess in initial_guesses:
        fit = least_squares(
            lambda parameters: von_mises_tuning(
                angles_radians, *parameters
            )
            - fit_rates,
            x0=initial_guess,
            bounds=(
                np.array([0.0, -30.0, 0.0, 0.0]),
                np.array([np.inf, 20.0, 50.0, np.pi]),
            ),
            max_nfev=20000,
        )
        if fit.success and np.all(np.isfinite(fit.fun)):
            fits.append((float(np.sum(fit.fun**2)), fit.x))
    if not fits:
        raise RuntimeError("Von Mises curve fitting failed for all initial guesses.")
    return min(fits, key=lambda fit: fit[0])[1]


def predict_model(
    model_name: str,
    fit_angles: np.ndarray,
    fit_rates: np.ndarray,
    predict_angles: np.ndarray,
) -> np.ndarray:
    if model_name.startswith("Polynomial degree "):
        degree = int(model_name.rsplit(" ", 1)[1])
        model = Polynomial.fit(fit_angles, fit_rates, degree)
        return model(predict_angles)
    if model_name == "Gaussian":
        parameters = fit_gaussian(fit_angles, fit_rates)
        return gaussian_tuning(predict_angles, *parameters)
    raise ValueError(f"Unsupported model: {model_name}")


# Full-data MSEs for all six models are reported for the selected neuron.
selected_mean_rates = mean_rates[selected_v1_index]
polynomial_models = {
    degree: Polynomial.fit(
        orientation_degrees, selected_mean_rates, degree
    )
    for degree in polynomial_degrees
}
model_summary = []
for degree in polynomial_degrees:
    predictions = polynomial_models[degree](orientation_degrees)
    model_summary.append(
        (
            f"Polynomial degree {degree}",
            degree + 1,
            float(np.mean((predictions - selected_mean_rates) ** 2)),
            "No",
            "No",
        )
    )

gaussian_parameters = fit_gaussian(orientation_degrees, selected_mean_rates)
gaussian_mse = float(
    np.mean(
        (
            gaussian_tuning(orientation_degrees, *gaussian_parameters)
            - selected_mean_rates
        )
        ** 2
    )
)
model_summary.append(("Gaussian", 4, gaussian_mse, "Yes", "No"))

von_mises_parameters = fit_von_mises(
    orientation_degrees, selected_mean_rates
)
von_mises_mse = float(
    np.mean(
        (
            von_mises_tuning(
                np.deg2rad(orientation_degrees), *von_mises_parameters
            )
            - selected_mean_rates
        )
        ** 2
    )
)
model_summary.append(("von Mises", 4, von_mises_mse, "Yes", "Yes"))

print("\nTask 6: Comparing the models")
print(f"6.1 Comparing different models for V1 neuron {selected_v1_neuron}")
print("Full-data model summary (MSE on the eight orientation means):")
print(
    f"{'Model':<25} {'Parameters':>10} {'MSE':>12} "
    f"{'Biological':>13} {'Circular':>10}"
)
for name, parameter_count, mse, interpretable, circular in model_summary:
    print(
        f"{name:<25} {parameter_count:>10} {mse:>12.6f} "
        f"{interpretable:>13} {circular:>10}"
    )
print("MSE units: (spikes/second)^2")

# The assignment asks for the selected neuron plus two noisier neurons.
# Compare linear, cubic, and Gaussian models using leave-one-orientation-out
# cross-validation, which evaluates prediction at an unseen orientation.
comparison_models = (
    "Polynomial degree 1",
    "Polynomial degree 3",
    "Gaussian",
)
training_mse_by_neuron = {}
cross_validation_mse = {}
for neuron_index in comparison_neuron_indices:
    rates = mean_rates[neuron_index]
    training_mse_by_neuron[neuron_index] = {}
    cross_validation_mse[neuron_index] = {}
    for model_name in comparison_models:
        full_fit_predictions = predict_model(
            model_name, orientation_degrees, rates, orientation_degrees
        )
        training_mse_by_neuron[neuron_index][model_name] = float(
            np.mean((full_fit_predictions - rates) ** 2)
        )
        squared_errors = []
        for held_out_index in range(orientation_degrees.size):
            training_mask = np.ones(orientation_degrees.size, dtype=bool)
            training_mask[held_out_index] = False
            prediction = predict_model(
                model_name,
                orientation_degrees[training_mask],
                rates[training_mask],
                orientation_degrees[~training_mask],
            )
            squared_errors.append(
                float((prediction[0] - rates[held_out_index]) ** 2)
            )
        cross_validation_mse[neuron_index][model_name] = float(
            np.mean(squared_errors)
        )

print("\n6.2 Comparing neurons with different noise levels")
print(
    f"Selected neuron: V1 neuron {selected_v1_neuron}; comparing it with "
    "the two neurons with the highest within-orientation variability."
)
print(
    "Fitting linear, third-order polynomial, and Gaussian regression models "
    "to all three neurons."
)
print("\nTraining MSE on the eight means for the three comparison models:")
print(
    f"{'Neuron':>8} {'Linear':>13} {'Cubic':>13} {'Gaussian':>13}"
)
for neuron_index in comparison_neuron_indices:
    errors = training_mse_by_neuron[neuron_index]
    print(
        f"{neuron_index + 1:>8} "
        f"{errors['Polynomial degree 1']:>13.3f} "
        f"{errors['Polynomial degree 3']:>13.3f} "
        f"{errors['Gaussian']:>13.3f}"
    )

print("\nNoise levels and leave-one-orientation-out test MSE:")
print("Noise is pooled within-orientation variance across trial firing rates.")
print(
    f"{'Neuron':>8} {'Noise SD (spikes/s)':>21} "
    f"{'Linear MSE':>13} {'Cubic MSE':>13} {'Gaussian MSE':>15}"
)
for neuron_index in comparison_neuron_indices:
    neuron_number = neuron_index + 1
    noise_sd = float(np.sqrt(within_orientation_variances[neuron_index]))
    errors = cross_validation_mse[neuron_index]
    print(
        f"{neuron_number:>8} {noise_sd:>21.3f} "
        f"{errors['Polynomial degree 1']:>13.3f} "
        f"{errors['Polynomial degree 3']:>13.3f} "
        f"{errors['Gaussian']:>15.3f}"
    )

print("\nTask 6 questions")
best_training_models_by_neuron = {
    index: min(
        training_mse_by_neuron[index],
        key=training_mse_by_neuron[index].get,
    )
    for index in comparison_neuron_indices
}
print(
    "1. The training-MSE and leave-one-orientation-out tables compare linear, "
    "cubic, and Gaussian models on the selected neuron and the two noisier "
    "neurons. The lowest in-sample MSE is from "
    + ", ".join(
        f"{best_training_models_by_neuron[index]} for neuron {index + 1}"
        for index in comparison_neuron_indices
    )
    + "; held-out performance is listed separately."
)
print(
    "2. The noisier neurons (31 and 101; within-orientation SDs "
    f"{np.sqrt(within_orientation_variances[noisier_neuron_indices[0]]):.2f} "
    "and "
    f"{np.sqrt(within_orientation_variances[noisier_neuron_indices[1]]):.2f} "
    "spikes/s) have more variable trial responses than neuron 98 "
    f"({np.sqrt(within_orientation_variances[selected_v1_index]):.2f} spikes/s). "
    "Higher noise makes the mean responses less certain, but it does not "
    "necessarily produce a larger held-out MSE: response shape also matters. "
    "The best held-out model is cubic for neuron 98 and Gaussian for neurons "
    "31 and 101."
)
best_in_sample_model = min(model_summary, key=lambda row: row[2])[0]
print(
    f"3. No. The lowest training MSE is from {best_in_sample_model}, but "
    "a highly flexible model can fit the observed means and still predict "
    "held-out orientations poorly. Leave-one-orientation-out MSE is the "
    "more relevant comparison for predicting a new orientation."
)
pooled_cv_mse = {
    model_name: float(
        np.mean(
            [
                cross_validation_mse[index][model_name]
                for index in comparison_neuron_indices
            ]
        )
    )
    for model_name in comparison_models
}
recommended_model = min(pooled_cv_mse, key=pooled_cv_mse.get)
print(
    f"4. For selected neuron {selected_v1_neuron}, "
    f"{min(cross_validation_mse[selected_v1_index], key=cross_validation_mse[selected_v1_index].get)} "
    "has the lowest leave-one-orientation-out MSE among the three models "
    f"({min(cross_validation_mse[selected_v1_index].values()):.3f} "
    "(spikes/second)^2). Across all three neurons, "
    f"{recommended_model} has the lowest mean held-out MSE "
    f"({pooled_cv_mse[recommended_model]:.3f} (spikes/second)^2). Prefer a "
    "simpler model when its test error is similar, especially for noisier "
    "neurons, because added complexity can fit noise. The conclusion is "
    "limited to these neurons and sampled orientations."
)

# Plot full-data model fits and held-out-orientation error together.
plot_angles = np.linspace(orientation_degrees.min(), orientation_degrees.max(), 500)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for ax, neuron_index in zip(axes, comparison_neuron_indices):
    rates = mean_rates[neuron_index]
    ax.scatter(
        orientation_degrees,
        rates,
        color="black",
        zorder=5,
        label="Experimental means",
    )
    for model_name in comparison_models:
        predictions = predict_model(
            model_name, orientation_degrees, rates, plot_angles
        )
        ax.plot(
            plot_angles,
            predictions,
            label=(
                f"{model_name} "
                f"(LOO MSE={cross_validation_mse[neuron_index][model_name]:.2f})"
            ),
        )
    ax.set_title(
        f"V1 neuron {neuron_index + 1}\n"
        f"within-orientation SD="
        f"{np.sqrt(within_orientation_variances[neuron_index]):.2f} spikes/s"
    )
    ax.set_xlabel("Stimulus orientation (degrees)")
    ax.set_xticks(orientation_degrees)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Mean firing rate (spikes/second)")
axes[-1].legend(fontsize=8)
fig.suptitle("Model fits for selected and noisier neurons")
fig.tight_layout()

fit_figure_path = project_dir / "task5and6_model_fits_by_noise.png"
fig.savefig(fit_figure_path, dpi=150)
plt.close(fig)
print(f"\nSaved model-comparison figure: {fit_figure_path}")

fig, ax = plt.subplots(figsize=(9, 5.5))
positions = np.arange(len(comparison_neuron_indices))
bar_width = 0.24
bar_containers = []
for model_offset, model_name in enumerate(comparison_models):
    values = [
        cross_validation_mse[index][model_name]
        for index in comparison_neuron_indices
    ]
    bar_containers.append(
        ax.bar(
            positions + (model_offset - 1) * bar_width,
            values,
            bar_width,
            label=model_name,
        )
    )
ax.set_xticks(
    positions,
    [f"V1 neuron {index + 1}" for index in comparison_neuron_indices],
)
ax.set_ylabel("Leave-one-orientation-out MSE ((spikes/second)^2, log scale)")
ax.set_yscale("log")
ax.set_title("Prediction error at held-out stimulus orientations")
ax.grid(axis="y", alpha=0.3)
ax.legend(handles=bar_containers, labels=["Linear", "Cubic", "Gaussian"])
fig.tight_layout()

cv_figure_path = project_dir / "task5and6_generalization_mse.png"
fig.savefig(cv_figure_path, dpi=150)
plt.close(fig)
print(f"Saved generalization figure: {cv_figure_path}")