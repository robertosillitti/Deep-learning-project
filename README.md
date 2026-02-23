# Multitask facial analysis: gender/ethinicty classification and age regression using neural networks

In this project me and @AsiaGrillo implemented a multitask classification model for predicting both age and gender, and a regression model specifically designed to estimate age, using a modified version of ResNet18.

## Dataset description

The dataset (sourced from kaggle: https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv) contains 23,705 rows of pixel values corresponding to 48×48 grayscale images. We removed duplicate photos to prevent data leakage during the train/validation/test split, and then analyzed the distributions of the target variables: gender, ethnicity (grouped into five categories), and age (ranging from 1 to 116 years).

## Goal of the analysis

The goal of our project is to predict a person’s gender, ethnicity, and exact age from facial images. To achieve this, we designed a modified version of ResNet-18 tailored to the characteristics of our dataset. We then evaluated the model using quantitative metrics and visualizations, and complemented the results with additional analyses, such as clustering and Grad-CAM, to uncover patterns and better understand the model’s errors.

## Methodology and results

First, we implemented a multitask model to jointly predict gender and ethnicity. For age prediction, we built a regression model that reuses the classification backbone, leveraging the facial features already learned during multitask training.

### Data preprocessing

- We removed duplicates using hashing and cosine similarity;
- To address the strong class imbalance in ethnicity, we applied **data augmentation**, more aggressive for minority classes;
- We also implemented a **weighted random sampler** that changes how batches are drawn.

### Model architecture and training

- We designed a modified version of **ResNet18** adapted for 48x48 greyscale images, o simultaneously classify ethnicity and gender;
- We then trained the model with appropriate **regularization techniques** (early stopping, dropout, weight decay, label smoothing) and **optimization strategies** (warmup, cosine annealing);
- Optimization is performed using **AdamW optimizer** and we used **Kaiming (He) weights initialization**.
- For the regression task we adopted a two-stage pipeline:
  - A **coarse classifier** to assign each face to one of six equally populated age bins (reaching about 58% accuracy). The model reuses the convolutional backbone of the multitask model trained earlier. The model benefits from the pretrained multitask network, which has already learned rich representations of faces. Only the final classification head is new and trainable. So, we trained only final layers since the pretrained multitask backbone already learned
rich features (faces, structure, and so on) and then we fine tuned the entire network: after the classifier head learns to use the backbone features, we unfreeze the entire model to extract features specifically to predict age.
  - A **dedicated regressor** refined the prediction within the selected bin. Six independent regressors are trained, one for each coarse age bin. Each regressor adds a small fully connected head on top of the shared backbone (each regressor is trained only on images from its own bin).
Only the regression heads are updated. Coarse classification learns global age features; bin-specific regressors refine age estimation locally. This strategy proved particularly effective for dealing with the highly skewed age distribution of the dataset.

### Model evaluation and results

- We evaluated performance using confusion matrices and other relevant metrics (weighted F1, precision, recall, balanced accuracy and NIR, ROC curve for gender) for the classification task;
- We tested the model on both dataset images and external images for both tasks;
- To better understand the model’s decision process (in the classification task), we implemented **Grad-CAM** and performed **cluster analysis** on misclassifications;
- The multitask classifier achieved solid accuracy on both gender and ethnicity, performing reliably even in the presence of several dataset limitations such as low-resolution grayscale format 48×48 pixels, poor contrast, and substantial class imbalance;
- The final regression model achieved a global MAE of roughly 3–4 years, which is competitive considering the quality of the images and the limited sample diversity.

In conclusion, this analysis achieved reasonably good results given the limitations of the dataset. However, using higher-quality images (RGB, better resolution, less noise) and a more balanced distribution of samples across ethnicity and age would likely lead to significantly better performance.

## Repository structure

In this repository you will find:

- A Jupyter notebook with all the code needed to preprocess the data, define, train, and evaluate the models, and generate plots. It also includes comments and interpretations of the results.

- A PDF presentation summarizing the project, with key visualizations, plots, and a detailed discussion of the main findings.

- A README file providing an overview of the project structure, goals, and methodology.
