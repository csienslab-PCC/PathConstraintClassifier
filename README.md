# Fuzbolic Project: Path Constraint Classifier 
Author: wombatwen (r05922009@ntu.edu.tw)
Email: csienslab.pcc@gmail.com


## Introduction

This repository provides the implementation of Path Constraint Classifier (PCC), which is a novel symbolic execution module to support solver selection functionality.
Given a set of path constraint, PCC will predict a fatest solver according the characteristics of the input path constraint.

## Usage & Simple Example

Please take a look of the tutorial scripts in the ```./tutorial``` directory.
It will tell you how to perform PCC to:
- Extract path constraint features
- Evaluate solver performance
- Generate machine learning data for training / testing
- Train and test the performance of a model

To reproduce the experiment results in the thesis, please take a look at the script in ```script/ExpUtils```.

## Architecture

Main Components:
- Feature Extractor (```feature.FeatureExtractor```)
- Constraint Feature Cache (```feature.FeatureCacheManager```)
- Machine Learning Module (Implement in each model in ```./model```. E.g. ```DNN.py```, ```RandomForest.py```)
- Solver Selection Interface/Module (```model.ModelManager.ModelManager```)
- Classifier: Not explicitly implemented, but one can use ```model.test()``` to perform similar operation. E.g. ```DNNModel.test()```

Others:
- FeatureProcessor (```feature.FeatureProcessor```): Normalize the numeric value of features to avoid training failure.
- SolverEvaluator (```evalueate.SolverEvaluator```)
- AnswerEvaluator (```evaluate.AnswerEvaluator```)
- PredictionEvaluator (```evaluate.PredicitonEvaluator```)
- DataPartitioner (```partition.DataPartitioner``` and ```partition.DataGrouper```)
- DataCombiner (```combine.DataCombiner```)

## Implementation Details

To be continue.

## Experiments

To be continue.

## Futrue Work
- Data and Solvers
    - Collect more symbolic data (more binaries)
    - Import more SMT solvers
- Machine Learning
    - Feature engineering / minimization (Dominating features)
    - Model design and training
        a. More DNN Layer?
        b. Better loss function?
- Integrate to an exisiting symbolic execution engine
    - Angr or KLEE?
- Solving Procedure Design
    - Backup Solver: More backup solver to use?
- Logic Selection
- Other Optimization
    
