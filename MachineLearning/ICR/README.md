# ICR-Identifying Age Related Conditions

Identifying Age-Related Conditions

## Goal of the Competition

The goal of this competition is to predict if a person has any of three medical conditions. You are being asked to predict if the person has one or more of any of the three medical conditions (Class 1), or none of the three medical conditions (Class 0). You will create a model trained on measurements of health characteristics.

To determine if someone has these medical conditions requires a long and intrusive process to collect information from patients. With predictive models, we can shorten this process and keep patient details private by collecting key characteristics relative to the conditions, then encoding these characteristics.

Your work will help researchers discover the relationship between measurements of certain characteristics and potential patient conditions.

## Context

They say age is just a number but a whole host of health issues come with aging. From heart disease and dementia to hearing loss and arthritis, aging is a risk factor for numerous diseases and complications. The growing field of bioinformatics includes research into interventions that can help slow and reverse biological aging and prevent major age-related ailments. Data science could have a role to play in developing new methods to solve problems with diverse data, even if the number of samples is small.

Currently, models like XGBoost and random forest are used to predict medical conditions yet the models' performance is not good enough. Dealing with critical problems where lives are on the line, models need to make correct predictions reliably and consistently between different cases.

Founded in 2015, competition host InVitro Cell Research, LLC (ICR) is a privately funded company focused on `regenerative and preventive personalized medicine`. Their offices and labs in the greater New York City area offer state-of-the-art research space. InVitro Cell Research's Scientists are what set them apart, helping guide and defining their mission of researching how to repair aging people fast.

In this competition, you’ll work with measurements of health characteristic data to solve critical problems in bioinformatics. Based on minimal training, you’ll create a model to predict if a person has any of three medical conditions, with an aim to improve on existing methods.

You could help advance the growing field of bioinformatics and explore new methods to solve complex problems with diverse data.

---
## Evaluation

Submissions are evaluated using a `balanced logarithmic loss`. The overall effect is such that each class is roughly equally important for the final score.

Each observation is either of class 0 or of class 1. For each observation, you must submit a probability for each class. The formula is then:

$\text{Log Loss} = \frac{−\frac{w_0}{N0} \sum_{i=0}^ {N_0}y_{0i}log(p_{0i}) - \frac{w_1}{N_1}\sum_{i=1}^ {N_1}y_{1i}log(p_{1i})}{w_0 + w_1}$

where $N_c$ is the number of observations of class c, log is the natural logarithm, $y_{ci}$ is 1 if observation i belongs to class c and 0 otherwise, $p_{ci}$ is the predicted probability that observation i belongs to class c. The weights $w_c$ are equal to the inverse prevalence of that class. 

The submitted probabilities for a given row are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, each predicted probability p is replaced with $max(min(p,1−10−15),10−15)$.


## Dataset Description

The competition data comprises over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to `predict whether a subject has or has not been diagnosed with one of these conditions` -- a `binary classification problem`.

## Files and Field Descriptions

###  train.csv - The training set.

- Id Unique identifier for each observation.
    
- AB-GL `Fifty-six anonymized health characteristics`. All are 
numeric except for EJ, which is categorical.
    
- Class A binary target: 1 indicates the subject has been 
diagnosed with one of the three conditions, 0 indicates they have not.

### test.csv - The test set. Your goal is to predict the probability that a subject in this set belongs to each of the two classes.

### greeks.csv - Supplemental metadata, only available for the training set.

-  `Alpha` Identifies the type of age-related condition, if present.
    - A No age-related condition. Corresponds to class 0.
    - B, D, G The three age-related conditions. Correspond to class 1.
- `Beta`, `Gamma`, `Delta` Three experimental characteristics.
- `Epsilon` The date the data for this subject was collected. Note that all of the data in the test set was collected after the training set was collected.

### sample_submission.csv - A sample submission file in the correct format. See the Evaluation page for more details.