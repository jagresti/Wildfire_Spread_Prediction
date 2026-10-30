# Does Cleaning Implausible Inputs Improve Next-Day Wildfire Spread Prediction?

A controlled comparison of two preprocessing pipelines for pixel-level next-day wildfire spread prediction, using a 1x1 CNN baseline, a multilayer CNN, and U-Net models on the Next Day Wildfire Spread dataset. 

This repository extends a UC Berkeley MIDS team project. The original team built the models and an initial preprocessing pipeline; this follow-up replaces that preprocessing with physically based cleaning of implausible values, fixes several evaluation issues, and compares both pipelines under identical training conditions. 

## Key Findings
- Cleaning did not measurably change performance. F1 and IoU were essentially unchanged across all five models. PR-AUC was slightly higher with the cleaned data for most models (about +0.01 to +0.02), but within the run-to-run variation observed with a single training seed (about ±0.02).
- The raw data contained many physically impossible values (ex: wind directions of -505,870°, minimum temperatures of -444 K, negative wind speed and humidity). Left in place, they inflated the wind-direction scaling factor from about 70° to 3,435°, compressing real wind directions to nearly zero. The models appear robust to these errors, likely because they affect under 0.5% of pixels and the strongest predictor (the previous day's fire mask) needed no cleaning.
- Evaluation choices mattered more than preprocessing. With fire pixels making up about 1.4% of valid pixels, PR-AUC is more informative than ROC-AUC. Under PR-AUC, the dropout U-Net outperforms the model originally selected as final by ROC-AUC.
- Fixed a bug in the original baseline, which applied a sigmoid both in the model and in the loss, preventing it from learning effectively.
   
  ## Repository Structure
├── Preprocessing_Jordan.ipynb               # Loading, EDA, revised cleaning, saving arrays
├── Models_TeamPreprocessing.ipynb           # Modeling + evaluation on the original team preprocessing
├── Models_JordanPreprocessing.ipynb         # Identical notebook, run on the revised preprocessing
├── Original Project/
│   ├── WildfirePrediction_Aria_preprocessing.ipynb    # Original team preprocessing
│   └── WildfirePrediction_FinalModel_SUBMITTED.ipynb  # Original team modeling notebook, as submitted
└── runs/
    ├── team_seed1234/      # Training histories and results.csv (team preprocessing)
    └── jordan_seed1234/    # Training histories and results.csv (revised preprocessing)

The two Models_* notebooks are identical except for their input data path (DATA_PATH) and output folder (RUN_DIR). 

## Data
The dataset is not included in this repository because of its size. It is the Next Day Wildfire Spread dataset (Hout et al., 2022):
- Paper: Next Day Wildfire Spread: A Machine Learning Data Set to Predict Wildfire Spreading from Remote-Sensing Data
- Download: https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread/data

It contains 18, 545 samples of 64x64 km regions at 1km resolution, with 12 input features (weather, drought, vegetation, terrain, population, previous-day fire mask) and a next-day fire mask as the target. The dataset comes pre-split into training (14,979), validation (1,877), and test (1,689) sets. 

## How to Reproduce
1. Download the dataset's .tfrecord files into a local folder.
2. Preprocessing: In Preprocessing_Jordan.ipynb, set data_path to that folder and SAVE_PATH to where the cleaned arrays should go, then run all cells.
3. Original preprocessing: Run Original Project/WildfirePrediction_Aria_preprocessing.ipynb to produce the team's arrays.
4. Modeling: In each Models_* notebook, set DATA_PATH to the matching array folder and RUN_DIR to an output folder, then run all cells top to bottom.
Requirements: Python 3.12, TensorFlow, NumPy, pandas, scikit-learn, matplotlib, seaborn. The U-Net models train slowly on CPU; a GPU is strongly recommended.

## Methods Summary

Revised Preprocessing:
- Defined physically plausible bounds for every input feature, based on physical limits and inspection of training data.
- Treated out-of-bounds values and the dataset's shared "no data" pattern as missing, and imputed them with training-set medians computed after invalid values were removed. Values just past a bound (rounding noise) were set to the bound.
- Kept physically real values such as negative NDVI, and applied a log transform to precipitation and population.
- Fit all fill values and scaling statistics on the training set only.

Evaluation (applied to both pipelines):
- PR-AUC, ROC-AUC, and precision, recall, F1, and IoU at each model's best F1 threshold chosen on the validation set.
- Training/validation loss curves and confusion matrices for every model.
- Test set evaluated once, for the final model only, using the validation threshold.
- Unlabeled pixels (-1) excluded from training loss and evaluation.

## Limitations
- All results come from a single training seed per model. Differences smaller than about 0.02 cannot be distinguished from run-to-run noise.
- In the original modeling notebook, the plain U-Net (learning rate 1e-3) and the learning-rate experiment (3e-4) use different early-stopping patience (5 vs 7), so comparisons between those two learning rates change two settings at once.
- Samples in the dataset can overlap in space and time, so training, validation, and test samples may not be fully independent.

## Credits
Original team project: Aria Koul, Aida Mohasesi, Jordan Agresti, Molly Jenkins; UC Berkeley MIDS, DATASCI 207. Revised preprocessing, evaluation changes, and controlled comparison: Jordan Agresti. 

Dataset: F. Huot, R. L. Hu, N. Goyal, T. Sankar, M. Ihme, Y.-F. Chen, "Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data," IEEE Transactions on Geoscience and Remote Sensing, 2022.
