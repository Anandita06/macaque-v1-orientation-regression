# Regression Analysis of Macaque V1 Orientation Responses

This project analyzes electrophysiological spike recordings from macaque
primary visual cortex (V1) during presentation of drifting gratings at eight
orientations. It uses regression models to characterize how neuronal firing
rates vary with stimulus orientation and to compare model fit and
generalization. The main analyses focus on neuron 98, with additional neurons
included for comparing responses at different noise levels.

## Analysis scripts

| Script | Analysis |
| --- | --- |
| `task1_tuning_curve.py` | Trial responses, orientation tuning, and selection of neuron 98 |
| `task2_polynomial.py` | Linear and polynomial regression |
| `task3_nonlinear_reg.py` | Gaussian tuning fit |
| `task4_prob_w_orientation.py` | Circular orientation and von Mises fit |
| `task5and6_model_eval_and_generalization.py` | Model comparison and generalization across neurons with different noise levels |

The scripts save figures in the project directory. The tracked figures provide
the supporting plots for the analyses.

## Figures

- [Task 1: tuning curves across V1 neurons](task1_v1_tuning_curves.png)
- [Task 1: selected neuron 98](task1_selected_v1_neuron98.png)
- [Task 2: polynomial fits](task2_polynomial_fits_neuron98.png)
- [Task 3: Gaussian tuning fit](task3_gaussian_fit_neuron98.png)
- [Task 4: von Mises fit](task4_von_mises_neuron98.png)
- [Tasks 5–6: fits across noise levels](task5and6_model_fits_by_noise.png)
- [Tasks 5–6: held-out orientation errors](task5and6_generalization_mse.png)

## Setup and run

Install Python, then from this directory run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\task1_tuning_curve.py
```

Run the other task scripts the same way, substituting their filenames in the
last command. Calling the environment's Python directly avoids requiring shell
activation. The dataset must be downloaded separately and placed at the path
shown below before running the scripts.

## Dataset

The raw recordings are not included in this repository. Download the dataset
from [Dryad](https://doi.org/10.5061/dryad.h9w0vt4s0) and place
`106r001p26_spikes.mat` at:

```text
dataset/dataset/dataset1/106r001p26_spikes.mat
```

The dataset is described in the documentation below. Course assignment PDFs
and raw data files are intentionally excluded from the public repository.

## Dataset documentation

### Monkey V1/V2 neuronal population response to drifting gratings with varying orientation and contrast

[https://doi.org/10.5061/dryad.h9w0vt4s0](https://doi.org/10.5061/dryad.h9w0vt4s0)

These folders contain 1) electrophysiology recordings from Monkey V1 and V2 in response to drift gratings with different orientations and contrast levels 2) parameters of statistical models fit to the recordings. For details of recordings and statistical models, please check the paper [https://doi.org/10.1016/j.isci.2024.110512](https://doi.org/10.1016/j.isci.2024.110512 "Persistent link using digital object identifier")[. ](https://www.biorxiv.org/content/10.1101/2023.12.19.572428v1.)(Xia et al 2024)

#### Description of the data and file structure

###### dataset/dataset1:&#x20;

* 105l001p16_spikes.mat
* 106r001p26_spikes.mat
* 106r002p70_spikes.mat
* 107l002p67_spikes.mat
* 107l003p143_spikes.mat

This folder contains 5 .mat files. Each contains a  session of simultaneously recorded neuronal population responses from V1 (recorded using Utah array) and V2 (recorded using tetrodes) to oriented grating with 8 orientations. This dataset was originally published at [https://crcns.org/data-sets/vc/v1v2-1/about_v1v2-1](https://crcns.org/data-sets/vc/v1v2-1/about_v1v2-1), here we present a mildly preprocessed version which can easily be loaded by MATLAB or using python code from [https://github.com/tinaxia2016/NeuronalVariabilityStatsModels/blob/main/stats_model_fit_dataset1.ipynb](https://github.com/tinaxia2016/NeuronalVariabilityStatsModels/blob/main/stats_model_fit_dataset1.ipynb). 

In each matlab file, there are two variables:

* spikes: a cell contains two arrays
  * The first element (#n1 neurons x 1280 ms x 3200 trials) contain number of spikes of each V1 neuron at each trial
  * The second element (#n2 neurons x 1280 ms x 3200 trials) contain number of spikes of each V2 neuron at each trial
* stim: an (3200 trials x 1) array
  * an array indicates the orientation of drifting grating stimulus on each trial; in total there are 400 trials per orientation ordered randomly

###### dataset/dataset2:

* V1_V2_contrast_150r002p017_combine_2_trials.pkl

This folder contain 1 .pkl file. It contains a session of simultaneously recorded spike data from V1 and V2 (recorded by neuropixel probes) in response to drifting gratings with 8 orientations and 3 contrast levels. This dataset can be loaded using code from [https://github.com/tinaxia2016/NeuronalVariabilityStatsModels/blob/main/stats_model_fit_dataset2.ipynb](https://github.com/tinaxia2016/NeuronalVariabilityStatsModels/blob/main/stats_model_fit_dataset2.ipynb). 

Here is an example of how to load the .pkl file in python:

```
import pickle
with open('filename.pkl', 'rb') as f:
    data  = pickle.load(f)
```

###### stats\_model\_fitting\_res/stats\_model\_res:&#x20;

* dataset0_multidim_5_fold_cv_OTI_035.pkl
* dataset1_multidim_5_fold_cv_OTI_035.pkl
* dataset2_multidim_5_fold_cv_OTI_035.pkl
* dataset3_multidim_5_fold_cv_OTI_035.pkl
* dataset4_multidim_5_fold_cv_OTI_035.pkl
* dataset2/
  * 150_pytorch_neuropixel_all_contrasts_5_fold_cv_exclude_OTI_less_035_multidim_w_aff_con.pkl

There are 5 .pkl files containing parameters of fitted statistical models fit to each of the 5 sessions from dataset 1 and 1 folder with 1 .pkl file for the 1 session from dataset 2. To access the object stored in the pickle files, please refer to code in [https://github.com/tinaxia2016/NeuronalVariabilityStatsModels](https://github.com/tinaxia2016/NeuronalVariabilityStatsModels)

###### joint\_stats\_model\_fitting\_res:&#x20;

* dataset0_joint_stats_models_5_fold_cv.pkl
* dataset1_joint_stats_models_5_fold_cv.pkl
* dataset2_joint_stats_models_5_fold_cv.pkl
* dataset3_joint_stats_models_5_fold_cv.pkl
* dataset4_joint_stats_models_5_fold_cv.pkl

5 .pkl files contain parameters of fitted joint statistical models to each of the 5 sessions from dataset 1. To access the object stored in the pickle files, please refer to code in [https://github.com/tinaxia2016/NeuronalVariabilityStatsModels](https://github.com/tinaxia2016/NeuronalVariabilityStatsModels)

#### Code/Software

You may use code from [https://github.com/tinaxia2016/NeuronalVariabilityStatsModels](https://github.com/tinaxia2016/NeuronalVariabilityStatsModels) to analyze and visualize the dataset.
