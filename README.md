# Multitask facial analysis: gender/ethinicty classification and age regression using neural networks

In this project me and @AsiaGrillo implemented a multitask classification model for predicting both age and gender, and a regression model specifically designed to estimate age, using a modified version of ResNet18.

## Dataset description

The dataset (sourced from kaggle: https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv) contains 23,705 rows of pixel values corresponding to 48×48 grayscale images. We removed duplicate photos to prevent data leakage during the train/validation/test split, and then analyzed the distributions of the target variables: gender, ethnicity (grouped into five categories), and age (ranging from 1 to 116 years).

## Goal of the analysis

The goal of our project is to predict a person’s gender, ethnicity, and exact age from facial images. To achieve this, we designed a modified version of ResNet-18 tailored to the characteristics of our dataset. We then evaluated the model using quantitative metrics and visualizations, and complemented the results with additional analyses, such as clustering and Grad-CAM, to uncover patterns and better understand the model’s errors.

## Methodology and results

First, we implemented a multitask model to jointly predict gender and ethnicity. To address the strong class imbalance in ethnicity, we applied **data augmentation**, more aggressive for minority classes, and used a **weighted random sampler**. We then trained the model with appropriate **regularization techniques** (early stopping, dropout, weight decay, label smoothing) and **optimization strategies** (warmup, cosine annealing).

We evaluated performance using confusion matrices and other relevant metrics, plotted the ROC curve for gender, and tested the model on both dataset images and external images. To better understand the model’s decision process, we implemented **Grad-CAM** and performed **cluster analysis** on misclassifications.

For age prediction, we built a regression model that reuses the classification backbone, leveraging the facial features already learned during multitask training.
We adopted a two-stage pipeline: first, a **coarse classifier** assigned each face to one of six equally populated age bins (reaching about 58% accuracy), and then a **dedicated regressor** refined the prediction within the selected bin. This strategy proved particularly effective for dealing with the highly skewed age distribution of the dataset. 

The final regression model achieved a global MAE of roughly 3–4 years, which is competitive considering the quality of the images and the limited sample diversity.
Also the multitask classifier achieved solid accuracy on both gender and ethnicity, performing reliably even in the presence of several dataset limitations such as low-resolution grayscale format 48×48 pixels, poor contrast, and substantial class imbalance.    

In conclusion, this analysis achieved reasonably good results given the limitations of the dataset. However, using higher-quality images (RGB, better resolution, less noise) and a more balanced distribution of samples across ethnicity and age would likely lead to significantly better performance.

## Repository structure

In this repository you will find:

- A Jupyter notebook with all the code needed to preprocess the data, define, train, and evaluate the models, and generate plots. It also includes comments and interpretations of the results.

- A PDF presentation summarizing the project, with key visualizations, plots, and a detailed discussion of the main findings.

- A README file providing an overview of the project structure, goals, and methodology.
