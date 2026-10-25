from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.discrete.discrete_model import NegativeBinomial


project_dir = Path(__file__).parent
recording_path = (
    project_dir
    / "dataset"
    / "dataset"
    / "dataset1"
    / "106r001p26_spikes.mat"
)

# MATLAB stores this dataset as trials x milliseconds x neurons.
with h5py.File(recording_path, "r") as recording:
    v1_spikes = recording[recording["spikes"][0, 0]][()].transpose(2, 1, 0)
    stimulus_codes = recording["stim"][()].reshape(-1).astype(int)

# Unnumbered Task 1: count spikes during the 1280 ms stimulus period.
v1_trial_counts = v1_spikes.sum(axis=1)
recorded_v1_count = v1_spikes.shape[0]
stimulus_codes_present, trials_at_each_code = np.unique(
    stimulus_codes, return_counts=True
)

# Convert stimulus IDs to degrees according to sessions_description.pdf.
degrees_for_code = {
    1: 0,
    2: 22.5,
    3: 45,
    4: 67.5,
    5: 90,
    6: 112.5,
    7: 135,
    8: 157.5,
}
degrees_at_each_code = np.array(
    [degrees_for_code[code] for code in stimulus_codes_present]
)
orientation_order = np.argsort(degrees_at_each_code)
orientations_degrees = degrees_at_each_code[orientation_order]

# Unnumbered Task 1: mean response for every V1 neuron and stimulus orientation.
mean_v1_response = np.stack(
    [
        v1_trial_counts[:, stimulus_codes == code].mean(axis=1)
        for code in stimulus_codes_present
    ],
    axis=1,
)

print("Task 1 data preparation")
print(
    f"Per-trial spike counts: {v1_trial_counts.shape} "
    "(V1 neurons x trials; summed across 1280 ms)"
)
print(
    f"Mean responses: {mean_v1_response.shape} "
    "(V1 neurons x stimulus orientations)"
)

# Question 1.
print(f"\nQuestion 1: {recorded_v1_count} V1 neurons were recorded.")

# Question 2.
print("Question 2: trials presented per orientation:")
for code, degrees, trial_count in zip(
    stimulus_codes_present, degrees_at_each_code, trials_at_each_code
):
    print(f"  {degrees:g} degrees: {trial_count} trials")

# Question 3.
print(
    "Question 3: stimulus orientations were "
    + ", ".join(f"{degrees:g} degrees" for degrees in orientations_degrees)
    + "."
)

# Question 4: show example tuning curves before choosing a neuron.
example_v1_indices = [0, 1, 2, 3, 4, 5]
figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
for v1_index, axis in zip(example_v1_indices, axes.flat):
    axis.plot(
        orientations_degrees,
        mean_v1_response[v1_index, orientation_order],
        marker="o",
    )
    axis.set_title(f"V1 neuron {v1_index + 1}")
    axis.set_xlabel("Stimulus orientation (degrees)")
    axis.set_ylabel("Mean spikes per trial")
    axis.set_xticks(orientations_degrees)
    axis.grid(True, alpha=0.3)

figure.suptitle("Mean orientation tuning curves for example V1 neurons")
figure.tight_layout()
examples_path = project_dir / "task1_v1_tuning_curves.png"
figure.savefig(examples_path, dpi=150)
plt.close(figure)
print(f"\nQuestion 4: example tuning curves saved to {examples_path}")

# Question 5: select using stratified five-fold cross-validation with an NB2
# count model. In each fold, select on training trials and rank on held-out
# trials, with an intercept-only model as the no-orientation-effect baseline.
fold_count = 5
fold_rng = np.random.default_rng(20261026)
fold_for_trial = np.empty(stimulus_codes.shape, dtype=int)
for code in stimulus_codes_present:
    code_trial_indices = np.flatnonzero(stimulus_codes == code)
    fold_rng.shuffle(code_trial_indices)
    fold_for_trial[code_trial_indices] = (
        np.arange(code_trial_indices.size) % fold_count
    )


def nb_orientation_likelihood_ratios(
    counts_for_neurons: np.ndarray, trial_codes: np.ndarray
) -> np.ndarray:
    """Return NB2 likelihood-ratio scores for adding categorical orientation."""
    orientation_design = np.column_stack(
        [
            np.ones(trial_codes.size),
            *[
                (trial_codes == code).astype(float)
                for code in stimulus_codes_present[1:]
            ],
        ]
    )
    baseline_design = np.ones((trial_codes.size, 1))
    likelihood_ratio_scores = np.empty(counts_for_neurons.shape[0])

    for v1_index, trial_counts in enumerate(counts_for_neurons):
        orientation_fit = NegativeBinomial(
            trial_counts, orientation_design, loglike_method="nb2"
        ).fit(disp=False, maxiter=80)
        baseline_fit = NegativeBinomial(
            trial_counts, baseline_design, loglike_method="nb2"
        ).fit(disp=False, maxiter=80)
        if not orientation_fit.mle_retvals.get("converged", False):
            raise RuntimeError(
                f"NB orientation model did not converge for V1 neuron {v1_index + 1}."
            )
        if not baseline_fit.mle_retvals.get("converged", False):
            raise RuntimeError(
                f"NB baseline model did not converge for V1 neuron {v1_index + 1}."
            )
        likelihood_ratio_scores[v1_index] = max(
            0.0, 2 * (orientation_fit.llf - baseline_fit.llf)
        )

    return likelihood_ratio_scores


fold_selected_indices = []
fold_selected_heldout_ranks = []
fold_heldout_scores = []
fold_heldout_rankings = []
for fold_number in range(fold_count):
    test_trials = fold_for_trial == fold_number
    train_trials = ~test_trials

    train_scores = nb_orientation_likelihood_ratios(
        v1_trial_counts[:, train_trials], stimulus_codes[train_trials]
    )
    test_scores = nb_orientation_likelihood_ratios(
        v1_trial_counts[:, test_trials], stimulus_codes[test_trials]
    )

    selected_index = int(np.argmax(train_scores))
    heldout_ranking = np.argsort(test_scores)[::-1]
    selected_heldout_rank = (
        int(np.flatnonzero(heldout_ranking == selected_index)[0]) + 1
    )
    fold_selected_indices.append(selected_index)
    fold_selected_heldout_ranks.append(selected_heldout_rank)
    fold_heldout_scores.append(test_scores)
    fold_heldout_rankings.append(heldout_ranking)

selection_counts = np.bincount(fold_selected_indices, minlength=recorded_v1_count)
selected_v1_index = int(np.argmax(selection_counts))
selected_v1_number = selected_v1_index + 1
selected_fold_ranks = [
    int(np.flatnonzero(ranking == selected_v1_index)[0]) + 1
    for ranking in fold_heldout_rankings
]
selected_fold_scores = [
    float(scores[selected_v1_index]) for scores in fold_heldout_scores
]

print(f"\nQuestion 5: selected V1 neuron {selected_v1_number}.")
print("NB2 cross-validation results (orientation model vs intercept-only baseline):")
for fold_number, (fold_index, heldout_rank) in enumerate(
    zip(fold_selected_indices, fold_selected_heldout_ranks), start=1
):
    print(
        f"  Fold {fold_number}: training selected neuron {fold_index + 1}; "
        f"that neuron's held-out NB score rank was "
        f"{heldout_rank}/{recorded_v1_count}."
    )
print(
    f"  Neuron {selected_v1_number} was selected in "
    f"{selection_counts[selected_v1_index]}/{fold_count} training folds."
)
print(
    f"  Its held-out NB score rank was "
    f"{min(selected_fold_ranks)}-{max(selected_fold_ranks)} "
    f"(median {np.median(selected_fold_ranks):.0f}); "
    f"held-out LR scores: {np.round(selected_fold_scores, 1).tolist()}."
)
print(
    "Reason for selection: it showed a clear orientation-dependent response "
    "pattern and was the most consistently selected by the count-based "
    "cross-validation comparison."
)

# Exploratory cross-check only: compare F-score and modulation index for the
# NB-selected neuron on one separate, stratified half-split. These metrics do
# not determine the selected neuron.
diagnostic_rng = np.random.default_rng(42)
diagnostic_train_mask = np.zeros(stimulus_codes.shape, dtype=bool)
for code in stimulus_codes_present:
    code_trial_indices = np.flatnonzero(stimulus_codes == code)
    diagnostic_rng.shuffle(code_trial_indices)
    diagnostic_train_mask[
        code_trial_indices[: len(code_trial_indices) // 2]
    ] = True
diagnostic_test_mask = ~diagnostic_train_mask


def f_score_and_modulation_index(
    neuron_trial_counts: np.ndarray, trial_codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate orientation F-scores and max-min modulation indices."""
    orientation_means = np.stack(
        [
            neuron_trial_counts[:, trial_codes == code].mean(axis=1)
            for code in stimulus_codes_present
        ],
        axis=1,
    )
    orientation_sample_counts = np.array(
        [(trial_codes == code).sum() for code in stimulus_codes_present]
    )
    overall_means = neuron_trial_counts.mean(axis=1)
    between_ss = (
        orientation_sample_counts
        * (orientation_means - overall_means[:, np.newaxis]) ** 2
    ).sum(axis=1)
    within_ss = np.zeros(neuron_trial_counts.shape[0])
    for orientation_index, code in enumerate(stimulus_codes_present):
        group_counts = neuron_trial_counts[:, trial_codes == code]
        within_ss += (
            group_counts - orientation_means[:, orientation_index, np.newaxis]
        ).__pow__(2).sum(axis=1)

    f_scores = (between_ss / (len(stimulus_codes_present) - 1)) / (
        within_ss / (neuron_trial_counts.shape[1] - len(stimulus_codes_present))
    )
    maximum = orientation_means.max(axis=1)
    minimum = orientation_means.min(axis=1)
    modulation_indices = np.divide(
        maximum - minimum,
        maximum + minimum,
        out=np.zeros_like(maximum),
        where=(maximum + minimum) != 0,
    )
    return f_scores, modulation_indices


diagnostic_train_f, diagnostic_train_mi = f_score_and_modulation_index(
    v1_trial_counts[:, diagnostic_train_mask],
    stimulus_codes[diagnostic_train_mask],
)
diagnostic_test_f, diagnostic_test_mi = f_score_and_modulation_index(
    v1_trial_counts[:, diagnostic_test_mask],
    stimulus_codes[diagnostic_test_mask],
)


def descending_rank(scores: np.ndarray, index: int) -> int:
    return int(np.sum(scores > scores[index]) + 1)


print(
    "Exploratory held-out F-score / modulation-index check "
    "(one stratified half-split; not used for selection):"
)
print(
    f"  F-score: train={diagnostic_train_f[selected_v1_index]:.2f} "
    f"(rank {descending_rank(diagnostic_train_f, selected_v1_index)}/"
    f"{recorded_v1_count}); held-out={diagnostic_test_f[selected_v1_index]:.2f} "
    f"(rank {descending_rank(diagnostic_test_f, selected_v1_index)}/"
    f"{recorded_v1_count})."
)
print(
    f"  Modulation index (max-min)/(max+min): "
    f"train={diagnostic_train_mi[selected_v1_index]:.4f} "
    f"(rank {descending_rank(diagnostic_train_mi, selected_v1_index)}/"
    f"{recorded_v1_count}); held-out={diagnostic_test_mi[selected_v1_index]:.4f} "
    f"(rank {descending_rank(diagnostic_test_mi, selected_v1_index)}/"
    f"{recorded_v1_count})."
)

# Plot the selected neuron's full-session tuning curve after the examples
# and selection procedure, with trial-to-trial SD shown for context.
selected_trial_means = np.array(
    [
        v1_trial_counts[selected_v1_index, stimulus_codes == code].mean()
        for code in stimulus_codes_present
    ]
)
selected_trial_stds = np.array(
    [
        v1_trial_counts[selected_v1_index, stimulus_codes == code].std(ddof=1)
        for code in stimulus_codes_present
    ]
)
figure, axis = plt.subplots(figsize=(8, 5))
axis.errorbar(
    orientations_degrees,
    selected_trial_means[orientation_order],
    yerr=selected_trial_stds[orientation_order],
    marker="o",
    capsize=4,
)
axis.set_title(f"Selected V1 neuron {selected_v1_number}: mean +/- SD")
axis.set_xlabel("Stimulus orientation (degrees)")
axis.set_ylabel("Spikes per trial")
axis.set_xticks(orientations_degrees)
axis.grid(True, alpha=0.3)
figure.tight_layout()
selected_figure_path = (
    project_dir / f"task1_selected_v1_neuron{selected_v1_number}.png"
)
figure.savefig(selected_figure_path, dpi=150)
plt.close(figure)
print(f"Selected-neuron figure saved to {selected_figure_path}")
