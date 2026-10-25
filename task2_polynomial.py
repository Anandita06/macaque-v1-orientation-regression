from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial


project_dir = Path(__file__).parent
recording_path = (
    project_dir
    / "dataset"
    / "dataset"
    / "dataset1"
    / "106r001p26_spikes.mat"
)

selected_v1_neuron = 98
selected_v1_index = selected_v1_neuron - 1
stimulus_duration_seconds = 1.28

# MATLAB stores spikes as trials x milliseconds x neurons.
with h5py.File(recording_path, "r") as recording:
    v1_spikes = recording[recording["spikes"][0, 0]][()].transpose(2, 1, 0)
    stimulus_codes = recording["stim"][()].reshape(-1).astype(int)

# Stimulus IDs and their angles for session 106r001p26.
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

# Convert each trial's spike total to a firing rate, then average that neuron's
# trial rates at each orientation to get the eight experimental points.
trial_spike_counts = v1_spikes[selected_v1_index].sum(axis=0)
trial_firing_rates = trial_spike_counts / stimulus_duration_seconds
stimulus_codes_present = np.unique(stimulus_codes)
orientation_degrees = np.array(
    [degrees_by_code[code] for code in stimulus_codes_present],
    dtype=float,
)
mean_responses = np.array(
    [
        trial_firing_rates[stimulus_codes == code].mean()
        for code in stimulus_codes_present
    ]
)
orientation_order = np.argsort(orientation_degrees)
orientation_degrees = orientation_degrees[orientation_order]
mean_responses = mean_responses[orientation_order]

# Fit all four polynomial degrees listed in the assignment. Polynomial.fit
# scales the input domain internally to improve numerical conditioning.
polynomial_degrees = (1, 3, 4, 7)
fitted_polynomials = {
    degree: Polynomial.fit(orientation_degrees, mean_responses, degree)
    for degree in polynomial_degrees
}
mse_by_degree = {
    degree: float(
        np.mean(
            (fitted_polynomials[degree](orientation_degrees) - mean_responses) ** 2
        )
    )
    for degree in polynomial_degrees
}

print(f"Task 2: polynomial regression for V1 neuron {selected_v1_neuron}")
print("Experimental mean responses (spikes per second):")
for angle, response in zip(orientation_degrees, mean_responses):
    print(f"  {angle:g} degrees: {response:.3f}")
print("Training MSE on the eight orientation means:")
for degree in polynomial_degrees:
    print(f"  Degree {degree}: {mse_by_degree[degree]:.6f}")
print(
    "The assignment lists four polynomial models; all four are included "
    "in the plot."
)

plot_angles = np.linspace(orientation_degrees.min(), orientation_degrees.max(), 500)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    orientation_degrees,
    mean_responses,
    color="black",
    zorder=5,
    label="Experimental means",
)
for degree in polynomial_degrees:
    ax.plot(
        plot_angles,
        fitted_polynomials[degree](plot_angles),
        label=f"Degree {degree} (MSE={mse_by_degree[degree]:.3f})",
    )

ax.set_title(f"Polynomial fits for V1 neuron {selected_v1_neuron}")
ax.set_xlabel("Stimulus orientation (degrees)")
ax.set_ylabel("Mean firing rate (spikes/second)")
ax.set_xticks(orientation_degrees)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

figure_path = project_dir / "task2_polynomial_fits_neuron98.png"
fig.savefig(figure_path, dpi=150)
plt.close(fig)
print(f"Saved polynomial-fit figure: {figure_path}")

best_training_degree = min(mse_by_degree, key=mse_by_degree.get)
print("\nTask 2 questions")
print(
    f"1. Degree {best_training_degree} gives the lowest training MSE "
    f"({mse_by_degree[best_training_degree]:.6f}) on these eight means. "
    "This is the best in-sample fit, not necessarily the best predictor "
    "for new data."
)
print(
    "2. As degree increases from 1 to 3 to 4, the curve becomes more flexible "
    "and the training MSE decreases. Degree 7 interpolates all eight means, "
    "giving approximately zero training MSE; the added flexibility can also "
    "produce unnecessary bends between measured orientations."
)
print(
    "3. The degree controls the highest power of orientation included, and "
    "therefore the number of coefficients and the curve's flexibility. "
    "A degree-d polynomial has d+1 coefficients and can have at most d turning points."
)
print(
    "4. Higher degrees can represent more complex shapes and reduce "
    "underfitting. Their drawbacks include fitting noise, oscillating between "
    "data points, unstable extrapolation, and poor handling of the circular "
    "orientation boundary."
)
print(
    "5. The polynomial coefficients do not have direct biological meanings. "
    "They jointly describe the curve and depend on the angle units and "
    "polynomial basis, unlike parameters such as preferred orientation or "
    "tuning width in a biologically motivated model."
)