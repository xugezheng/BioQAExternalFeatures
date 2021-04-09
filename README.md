# BioQA_externalFeatures

This repository is the official implementation of paper "External Features Enriched Model for Biomedical Question Answering". 

The code is based on original [BERT Repository released by Google](https://github.com/google-research/bert)

## Preparations

### Data and Model Params

[Trained Models's parameters](https://drive.google.com/drive/folders/1mQ68-CIsz3izoj_yuzVE86o8URN2o4SD?usp=sharing)
 and [processed data (added POS and NER labels)](https://drive.google.com/drive/folders/1rFeVTIjSiTXV_M4_4iGhvQXqbYtt3nTn?usp=sharing) can be directly downloaded from Google Drive.

For the trained models:

* We firstly fine-tuned [BERT (BERT-Base, Multilingual Cased)](https://github.com/google-research/bert) under our enriched external-feature framework on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/), 
of which the param could be found under `model/squad`

* Next, we further fine-tuned the model respectively on Biomedical Training Sets (6b, 7b and 8b), and the well trained models could be 
downloaded from `model/6b`, `model/7b`, `model/8b`.

For the data utilized in our experiments:

* The data mainly come from `SQuAD` and `BioASQ Challenge`;

* You can get the feature-enriched training data under two ways

   * Directly download the processed data from our [google drive link](https://drive.google.com/drive/folders/1rFeVTIjSiTXV_M4_4iGhvQXqbYtt3nTn?usp=sharing)
   
   * Use the provided scripts to process the NER, POS and BioNER feature on your own side.  

### Requirements

### Model Training

### Evaluation 

## Results


