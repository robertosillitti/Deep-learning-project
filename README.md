# Multitask facial analysis: gender/ethinicty classification and age regression using neural networks

In this project me and @AsiaGrillo implemented a multitask classification model for predicting both age and gender, and a regression model specifically designed to estimate age, using a modified version of ResNet18.

## Dataset description

The dataset (sourced from kaggle: https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv) contains 23,705 rows of pixel values corresponding to 48×48 grayscale images. We removed duplicate photos to prevent data leakage during the train/validation/test split, and then analyzed the distributions of the target variables: gender, ethnicity (grouped into five categories), and age (ranging from 1 to 116 years).

## Goal of the analysis

The goal of our project is to predict a person’s gender, ethnicity, and exact age from facial images. To achieve this, we designed a modified version of ResNet-18 tailored to the characteristics of our dataset. We then evaluated the model using quantitative metrics and visualizations, and complemented the results with additional analyses, such as clustering and Grad-CAM, to uncover patterns and better understand the model’s errors.

## Methodology and results

First, we implemented a multitask model to predict gender and ethinicity at the same time. To solve the strong unbalnce of the ethnicity class we made some data augmentation (stronger for minority classes) and we used a weighted random sampler. Then we trained the model using proper regularization techniques. We evluated the results by computing confusion metrices and othere useful metrics, ROC curve for gender, and we tested the model on both dataset images and esternal images. To better understand how the model makes prediction we implemented a Grad-CUM and we also performed cluster analysis on the errors. The regression model uses the backbone of the classification model since has already learned some important feature of the faces. 
For age estimation, we adopted a two-stage pipeline: first, a coarse classifier assigned each face to one of six equally populated age bins (reaching about 58% accuracy), and then a dedicated regressor refined the prediction within the selected bin. This strategy proved particularly effective for dealing with the highly skewed age distribution of the dataset, where some age groups contain thousands of samples and others only a few dozen. The final regression model achieved a global MAE of roughly 3–4 years, which is competitive considering the quality of the images and the limited sample diversity.
Also the multitask classifier achieved solid accuracy on both gender and ethnicity, performing reliably even in the presence of several dataset limitations such as low-resolution grayscale format 48×48 pixels, poor contrast, and substantial class imbalance.    

In conclusion, this analysis achieved reasonably good results given the limitations of the dataset. However, using higher-quality images (RGB, better resolution, less noise) and a more balanced distribution of samples across ethnicity and age would likely lead to significantly better performance.
