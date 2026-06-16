# Cognitive Assessment from Self-Figure Drawings

A PyTorch research pipeline for analyzing self-figure drawings and investigating whether visual information in the drawings is associated with cognitive assessment scores.

The project was developed using a non-public dataset collected by a research team from participants in Israel, India, and Thailand. At the time of the documented experiments, 2,075 drawings had been collected and approximately 1,600 were used for model development.

The main task presented in this repository is formulated as **regression**: given a self-figure drawing, the model predicts a continuous cognitive assessment score. The repository also contains support for related experimental formulations, including three-class cognitive-status classification.

> **Research disclaimer:** This repository contains exploratory research code. It is not a medical diagnostic system, has not been clinically validated, and must not be used to make medical or clinical decisions.

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Dataset](#dataset)
- [Main Challenges](#main-challenges)
- [Problem Formulation](#problem-formulation)
- [Project Pipeline](#project-pipeline)
- [Image Preprocessing](#image-preprocessing)
- [Transfer Learning with TU-Berlin](#transfer-learning-with-tu-berlin)
- [Model Architectures](#model-architectures)
- [Training and Evaluation](#training-and-evaluation)
- [Experimental Results](#experimental-results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Running the Code](#running-the-code)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Data Privacy](#data-privacy)
- [Acknowledgments](#acknowledgments)

---

## Research Motivation

Cognitive decline can affect memory, attention, orientation, executive functioning, visuospatial ability, language, and other aspects of daily functioning.

Clinical cognitive assessment is usually based on questionnaires, interviews, and additional medical examinations. These procedures may require trained professionals and can be affected by language, cultural background, time constraints, and communication barriers.

Self-figure drawings have previously been studied in several psychological and neurological contexts. This project explores whether deep-learning models can learn visual patterns from self-figure drawings that are associated with cognitive assessment scores.

The goal is not to replace established clinical assessment. Instead, the project investigates whether drawings may provide an additional computational signal for future research.

---

## Dataset

The research dataset consists of scanned self-figure drawings collected from participants in:

- Israel
- India
- Thailand

Each participant was asked to draw themself and completed a shortened version of the Montreal Cognitive Assessment (MoCA).

The shortened assessment produced a raw score in the range of 0–15. The score was doubled to obtain a 0–30 scale, and an education-based adjustment was applied where appropriate, except when the maximum score had already been reached.

At the time of the documented experiments:

- **2,075 drawings** had been collected.
- Approximately **1,600 drawings** were used in the experiments.
- The broader data-collection goal was approximately 3,000 drawings.

The dataset is not included in this repository because it contains non-public participant data.

---

## Main Challenges

The collected dataset presents several challenges.

### Limited dataset size

Approximately 1,600 samples are relatively few for training modern deep neural networks, especially large CNN architectures.

### Scan artifacts

The drawings were scanned using different tools and applications. The scans may contain:

- non-uniform or shaded backgrounds;
- colored stains and paper texture;
- borders and printed instructions;
- scanner-application logos;
- parts of tables or surrounding surfaces;
- multiple objects in the same image;
- inconsistent image sizes and orientations.

### Drawing-style variation

Participants differ in artistic ability, drawing habits, age, gender, culture, and geographic background. These factors may introduce variation that is unrelated to cognitive functioning.

### Weak visual-label correspondence

The relationship between a drawing and its associated cognitive score is difficult to interpret visually. In some cases, experts could not reliably estimate the participant's score from the drawing alone.

### Domain mismatch

Common pretrained computer-vision models are trained on natural RGB photographs, while the target data consists primarily of grayscale scanned sketches.

---

## Problem Formulation

The main experiments treat cognitive-score prediction as a supervised regression task.

For an input drawing $x$, the network predicts one continuous score:

$$
\hat{y} = f_{\theta}(x)
$$

where:

- $x$ is the preprocessed self-figure drawing;
- $f_{\theta}$ is the neural network;
- $\hat{y}$ is the predicted cognitive score;
- $y$ is the participant's ground-truth score.

The primary regression objective is the L1 loss:

$$
L_1 = \frac{1}{N}\sum_{i=1}^{N}\left|\hat{y}_i-y_i\right|
$$

This loss corresponds to the mean absolute prediction error measured in score points and is less sensitive to large individual errors than squared-error loss.

Two regression target settings were explored:

1. **Original score range:** prediction on the 0–30 cognitive-score scale.
2. **Adjusted lower range:** scores in the interval 0–12 were transformed into the interval 12–18 using:

```text
adjusted_score = original_score / 2 + 12
```

The adjustment was explored because very low scores represented only a small portion of the dataset.

The data loader also contains support for alternative target formulations, including three cognitive-status classes:

- Healthy Control (HC)
- Mild Cognitive Impairment (MCI)
- Alzheimer's Dementia (AD)

The principal results described below focus on continuous score regression.

---

## Project Pipeline

The project implements the following experimental pipeline:

1. Collect scanned self-figure drawings and matching cognitive scores.
2. Crop irrelevant regions and resize the images.
3. Reduce scan artifacts using classical and deep-learning preprocessing methods.
4. Create deterministic training, validation, and test splits.
5. Optionally pretrain CNN models on the TU-Berlin sketch dataset.
6. Replace the classification head with a regression output layer.
7. Fine-tune the model on the self-figure drawing dataset.
8. Evaluate models using held-out validation and test data.
9. Save checkpoints, losses, accuracy values, and plots for experiment comparison.

Most experiment settings are supplied through YAML configuration files, allowing model, optimizer, loss, augmentation, paths, and transfer-learning settings to be changed without editing the training loop.

---

## Image Preprocessing

### Cropping and resizing

Images were first cropped to remove irrelevant regions such as scanner logos, surrounding surfaces, and unused page areas. They were then resized to a common resolution of **512 × 512** pixels.

### Classical thresholding

Two classical background-removal approaches were examined:

- **Otsu thresholding**, which selects a global threshold by minimizing within-class pixel variance;
- **Adaptive thresholding**, which calculates local thresholds for different image regions.

Thresholding removed part of the background noise but could leave gray artifacts or remove weak drawing details.

### Dichotomous Image Segmentation

Two pretrained models for high-accuracy foreground segmentation were tested:

- **IS-Net**
- **BiRefNet**

IS-Net did not consistently preserve or isolate the drawing in every sample. BiRefNet was generally more robust, but its direct output sometimes increased the contrast too strongly.

### Selected preprocessing output

The chosen representation was a **blended image combining the cropped original image with the BiRefNet output**.

This compromise reduced background artifacts while retaining more of the original line intensity and drawing appearance.

The preprocessing implementation itself is not fully included in the current public repository. The training code expects access to the resulting preprocessed images.

---

## Transfer Learning with TU-Berlin

Because the target dataset is small, the project investigates transfer learning from the TU-Berlin sketch dataset.

TU-Berlin contains:

- 20,000 rasterized sketches;
- 250 object categories;
- 80 drawings per category.

Although TU-Berlin contains object sketches rather than self-figure drawings, it is visually closer to the target domain than natural-image datasets.

### TU-Berlin split

The 20,000 images were partitioned as follows:

- **4,000 images (20%)** were reserved as a stratified test set;
- the remaining **16,000 images** were divided using stratified five-fold cross-validation;
- in each fold:
  - 12,800 images were used for training;
  - 3,200 images were used for validation.

### Pretraining strategies

The following initialization strategies were compared:

- random initialization;
- initialization with ImageNet weights;
- limited-layer fine-tuning;
- full training on TU-Berlin before transfer to the cognitive-score task.

The first convolutional layer was adapted for one-channel images, and the final layer was replaced with a 250-class output layer during TU-Berlin pretraining.

### TU-Berlin augmentation

The TU-Berlin training pipeline includes augmentations intended to increase variation in drawing appearance:

- elastic transformations;
- Gaussian blur;
- random erasing;
- horizontal and vertical flips;
- perspective transformations;
- affine rotation, translation, and scaling.

These augmentations are applied only during training.

---

## Model Architectures

The repository supports the following convolutional architectures:

- ResNet-50
- ResNet-101
- ConvNeXt Small
- ConvNeXt Base

For TU-Berlin classification:

- the first convolutional layer is adapted for grayscale input;
- the final layer outputs 250 class scores.

For cognitive-score regression:

- the final classification layer is replaced by a single-output regression layer.

The implementation supports both ImageNet initialization and loading a checkpoint obtained from TU-Berlin pretraining.

---

## Training and Evaluation

### TU-Berlin training

The reported TU-Berlin experiments used:

- 50 epochs;
- Adam for ResNet models;
- AdamW for ConvNeXt models;
- target learning rate of `1e-4`;
- weight decay of `1e-4`;
- 10 warm-up epochs;
- `ReduceLROnPlateau` after warm-up;
- cross-entropy or contrastive-center loss.

Contrastive-center loss was evaluated because sketches may exhibit:

- high intra-class variation: drawings from the same category may look different;
- inter-class similarity: drawings from different categories may look similar.

### Cognitive-score training

The cognitive-score pipeline supports:

- L1 or L2 regression loss;
- Adam or AdamW;
- optional ImageNet initialization;
- optional TU-Berlin checkpoint initialization;
- warm-up scheduling;
- `ReduceLROnPlateau`;
- configurable image size and batch size;
- optional affine, brightness, gamma, and horizontal-flip augmentation;
- checkpoint saving based on validation performance.

### Evaluation metrics

For TU-Berlin classification, models are compared using:

- training accuracy;
- validation accuracy;
- test accuracy;
- mean and standard deviation across five folds.

For cognitive-score regression, models are compared using:

- training L1 loss;
- validation L1 loss;
- test L1 loss;
- mean and standard deviation across repeated splits or runs.

An L1 result of `2.8`, for example, represents an average absolute error of approximately 2.8 score points.

---

## Experimental Results

## TU-Berlin classification

### Cross-entropy loss

| Model | ImageNet init. | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| ConvNeXt Small | Yes | 81.54 ± 0.67% | 81.49 ± 0.50% |
| ConvNeXt Small | No | 65.20 ± 0.72% | 64.20 ± 0.28% |
| ConvNeXt Base | Yes | **82.30 ± 0.45%** | 81.71 ± 0.26% |
| ConvNeXt Base | No | 65.09 ± 1.25% | 64.22 ± 0.83% |
| ResNet-50 | Yes | 77.11 ± 0.01% | 76.96 ± 0.61% |
| ResNet-50 | No | 68.81 ± 0.94% | 68.62 ± 0.69% |
| ResNet-101 | Yes | 79.21 ± 0.52% | 79.40 ± 0.64% |
| ResNet-101 | No | 70.16 ± 1.69% | 69.67 ± 1.65% |

### Contrastive-center loss

| Model | ImageNet init. | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| ConvNeXt Small | Yes | 81.71 ± 1.00% | **81.97 ± 0.81%** |
| ConvNeXt Small | No | 65.13 ± 1.37% | 64.68 ± 0.53% |
| ConvNeXt Base | Yes | 81.71 ± 0.50% | 81.86 ± 0.34% |
| ConvNeXt Base | No | 65.32 ± 0.32% | 64.76 ± 0.80% |
| ResNet-50 | Yes | 76.93 ± 0.77% | 77.03 ± 0.20% |
| ResNet-50 | No | 71.65 ± 1.60% | 71.88 ± 1.18% |
| ResNet-101 | Yes | 78.98 ± 1.52% | 79.12 ± 1.07% |
| ResNet-101 | No | 71.30 ± 0.89% | 71.70 ± 1.41% |

### TU-Berlin observations

- ImageNet initialization consistently improved validation and test performance.
- ConvNeXt models generally outperformed the tested ResNet variants.
- Contrastive-center loss did not produce a consistent improvement over cross-entropy.
- Training accuracy was substantially higher than validation and test accuracy, indicating overfitting.

---

## Cognitive-score regression

ConvNeXt Base was selected for the documented cognitive-score experiments because of its strong TU-Berlin performance.

| TU-Berlin source | ImageNet initialization | Augmentation | Train L1 | Validation L1 | Test L1 |
|---|---|---:|---:|---:|---:|
| None | No | No | 2.83 ± 0.04 | 2.89 ± 0.08 | **2.88 ± 0.09** |
| None | No | Yes | 3.04 ± 0.01 | 3.03 ± 0.14 | 3.05 ± 0.08 |
| None | Yes | No | 3.01 ± 0.06 | 2.99 ± 0.11 | 3.00 ± 0.08 |
| None | Yes | Yes | 3.03 ± 0.02 | 3.01 ± 0.13 | 3.03 ± 0.10 |
| TU-Berlin, ImageNet initialized | N/A | No | 1.53 ± 0.44 | 2.75 ± 0.15 | 2.87 ± 0.07 |
| TU-Berlin, ImageNet initialized | N/A | Yes | 2.49 ± 0.26 | 2.84 ± 0.17 | **2.82 ± 0.02** |
| TU-Berlin, random initialization | N/A | No | 0.66 ± 0.05 | 2.75 ± 0.11 | 2.90 ± 0.16 |
| TU-Berlin, random initialization | N/A | Yes | 2.79 ± 0.06 | 2.81 ± 0.11 | 2.89 ± 0.12 |

`TU-Berlin, ImageNet initialized` means that the network started from ImageNet weights during TU-Berlin pretraining and was then transferred to the cognitive-score task.

### Regression observations

- The reported test error was generally between approximately **2.82 and 3.05 score points**.
- TU-Berlin pretraining strongly reduced training loss in some settings.
- The much larger gap between training and validation loss in those settings indicates overfitting.
- TU-Berlin transfer learning did not provide a clear, consistent improvement on the held-out test set.
- The best reported test result was obtained with TU-Berlin pretraining initialized from ImageNet weights and augmentation, but the improvement over the non-transfer baseline was small.
- The experiments suggest that dataset quality, sample size, and weak drawing-score correspondence may be more limiting than model capacity alone.

---

## Key Findings

1. **Domain-specific pretraining was not automatically sufficient.**  
   TU-Berlin is closer to the target domain than natural photographs, but pretraining on object sketches did not clearly improve cognitive-score generalization.

2. **ImageNet initialization remained useful.**  
   ImageNet-pretrained models achieved substantially better TU-Berlin classification accuracy than randomly initialized models.

3. **ConvNeXt performed strongly on sketch classification.**  
   ConvNeXt Small and ConvNeXt Base outperformed the tested ResNet models on TU-Berlin.

4. **Overfitting was a central challenge.**  
   Large train-validation gaps appeared in both TU-Berlin and cognitive-score experiments.

5. **Preprocessing required a trade-off.**  
   Strong foreground segmentation removed background artifacts but could alter the original stroke appearance. Blending the original crop with the BiRefNet output preserved more visual information.

6. **The target signal is difficult.**  
   Even human experts found it difficult to infer the questionnaire score from drawings alone, which limits the expected predictability of the task.

---

## Repository Structure

```text
cognitive-decline-research/
├── models/                         # Model definitions and wrappers
├── model_scores/                   # Saved experiment summaries
├── train_configs/                  # YAML files for training experiments
├── test_configs/                   # YAML files for evaluation experiments
├── utils/                          # Model, loss, plotting, and LR utilities
├── alz_regression_train.py         # Cognitive-score training entry point
├── alz_regression_test.py          # Cognitive-score evaluation entry point
├── alz_sketch_data_loader.py       # Self-figure drawing dataset loader
├── tu_berlin_train.py              # TU-Berlin cross-validation training
├── tu_berlin_test.py               # TU-Berlin test evaluation
├── tu_berlin_dataloader.py         # TU-Berlin loader and augmentations
├── plot_results_summary.py         # Experiment-summary plots
├── sbatch_script_alzheimer.sh      # Example Slurm training script
├── sbatch_script_pretraining.sh    # Example Slurm pretraining script
└── alz_proj.yml                    # Conda environment definition
```

---

## Environment Setup

The included Conda environment uses Python 3.10, PyTorch 1.13.1, torchvision 0.14.1, and CUDA 11.7.

Clone the repository:

```bash
git clone https://github.com/LIOR-YAACOV/cognitive-decline-research.git
cd cognitive-decline-research
```

Create the environment:

```bash
conda env create -f alz_proj.yml
conda activate alz_proj
```

The environment includes the main dependencies used by the project:

- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- scikit-image
- matplotlib
- PyYAML
- tqdm

A CUDA-capable GPU is recommended for training the larger models.

---

## Data Preparation

## Self-figure drawing data

The participant dataset is private and is therefore not distributed with this repository.

The current data loader expects:

1. preprocessed images stored locally;
2. text files containing labels and image filenames;
3. paths to those list files supplied through the YAML configuration.

A list entry is parsed as a whitespace-separated line containing a target value and an image filename. Some experiment files may contain an additional field.

Example:

```text
24.0 participant_001.png
```

Before running the public code on a different dataset, update the image-root path in `alz_sketch_data_loader.py` or refactor it into the experiment configuration.

## TU-Berlin data

The TU-Berlin loader expects rasterized PNG images organized by class:

```text
tu_berlin/
├── airplane/
│   ├── image_001.png
│   └── ...
├── alarm_clock/
│   ├── image_001.png
│   └── ...
└── ...
```

The current loader asserts that each category contains exactly 80 PNG images.

The dataset is not bundled with this repository and must be obtained separately in accordance with its original license and terms.

---

## Running the Code

The commands below show the entry points used by the repository. The supplied configuration files contain local paths from the original research environment and will need to be updated before execution.

### Train on the cognitive-score dataset

```bash
python alz_regression_train.py \
  --config_file train_configs/<training-config>.yml
```

The configuration controls, among other values:

- training and validation list paths;
- model architecture;
- number of outputs;
- optimizer;
- learning rate;
- weight decay;
- loss function;
- warm-up settings;
- augmentation parameters;
- checkpoint path;
- optional TU-Berlin checkpoint.

### Evaluate a cognitive-score model

```bash
python alz_regression_test.py \
  --config_file test_configs/<test-config>.yml
```

The evaluation code loads the test-list files and checkpoints defined in the selected YAML file.

### Train on TU-Berlin

```bash
python tu_berlin_train.py \
  --config_file train_configs/<tu-berlin-config>.yml
```

The training script performs stratified cross-validation and supports cross-entropy and contrastive-center loss.

### Evaluate TU-Berlin checkpoints

```bash
python tu_berlin_test.py \
  --config_file test_configs/<tu-berlin-test-config>.yml
```

### Slurm execution

Example Slurm scripts are included:

```bash
sbatch sbatch_script_pretraining.sh
sbatch sbatch_script_alzheimer.sh
```

The scripts may contain cluster-specific paths and resource settings and should be reviewed before use.

---

## Limitations

This project has several important limitations:

- The participant dataset is relatively small.
- The score distribution is imbalanced, especially in the lowest score range.
- Drawings were collected across different countries and scanning environments.
- Scan artifacts may be correlated with collection site or participant groups.
- Drawing style can be influenced by culture, education, age, motor ability, and artistic experience.
- The shortened and transformed assessment score is not identical to a standard full clinical MoCA evaluation.
- The project does not demonstrate clinical validity.
- No external clinical cohort was used for independent validation.
- The direct relationship between drawing appearance and cognitive score is uncertain.
- Transfer learning from object sketches may not match the semantic features needed for self-figure analysis.
- Strong preprocessing can unintentionally remove potentially relevant drawing characteristics.
- Some file paths and experiment configurations are specific to the original research environment.
- Reproducibility is limited because the private dataset and preprocessing code are not fully public.

The reported results should therefore be interpreted as exploratory machine-learning findings rather than evidence of diagnostic performance.

---

## Future Work

Potential directions include:

- collecting additional participant data;
- evaluating on an independent external cohort;
- improving participant-level split control and demographic balancing;
- analyzing performance separately by country and collection site;
- adding stronger baselines such as mean-score prediction and classical image features;
- reporting additional regression metrics such as RMSE, $R^2$, and correlation;
- testing confidence intervals and statistical significance across splits;
- investigating smaller architectures and stronger regularization;
- evaluating self-supervised learning on unlabeled sketches;
- pretraining on the Sketchy or ImageNet-Sketch datasets;
- comparing sketch-domain pretraining with direct ImageNet fine-tuning;
- testing preprocessing methods that preserve faint strokes more reliably;
- studying diffusion-based restoration carefully to avoid hallucinating or changing clinically relevant structure;
- adding explainability methods to inspect which image regions influence predictions;
- moving dataset roots and all experiment paths into YAML configuration;
- adding fixed random seeds and experiment tracking;
- adding automated tests and reproducible example data;
- packaging preprocessing, training, and evaluation into a clearer end-to-end pipeline.

---

## Data Privacy

The self-figure drawings and participant assessment data are not included in this repository.

Do not upload participant drawings, identifiers, questionnaire answers, or derived files that could expose private research data. Any future public example should use synthetic or explicitly authorized material.

---

## Acknowledgments

The participant drawings and cognitive-assessment data were collected in collaboration with a research team from the University of Haifa.

This repository contains the deep-learning experimentation pipeline used to study preprocessing, model selection, transfer learning, sketch classification, and cognitive-score regression.

---

## Citation and Use

No formal software release or citation format is currently provided.

When using ideas, datasets, or external model implementations referenced by this repository, consult and cite the corresponding original papers and licenses.

---

## License

No license file is currently included in this repository. Until a license is added, the code should not be assumed to be available for unrestricted reuse.
