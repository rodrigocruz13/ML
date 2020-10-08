# Titanic

![Titanic](https://i.imgur.com/3CXi7mb.png)

## ML Starter Repo

### Motivation

Machine Learning has matured in the industry and like other areas it greatly benefits from having standardized procedures and best practices whenever possible. On the other hand, most of the available educational material on ML is about the theory but rarely are these best practices taught

## Goal

In this project we will focus on one such best practice: building a suitable production-ready project structure for Machine Learning applications.

## Values

- Reproducibility: you should be able to reproduce any past experiment.
- Production Ready: the models trained by your code should be able to easily be put into a production environment.
- Visibility: you should be able to easily inspect the results, metrics, and parameters used for each experiment.
- Generality: the code base should serve as a template for future machine learning projects
- Speed: on your next project the template should drastically cut your time to production.

## Objectives

Train a model for the Titanic Dataset. The model will not be your focus but rather an excuse to create the project structure.

Your training code should be able to take command line arguments so it's easily usable from bash. Important parameters are:

- data_path: Input data should not be a constant since the repo should be general.
- debug: (optional) whether you are in debug mode.=- model_type: (optional) you can support changing the model used for training.

On each run / experiment your code should do the following tasks:

- Serialize/store the exact input parameters used for the experiment.
- Serialize/store the resulting metrics from experiment.
- Serialize/store the trained model plus the exact preprocessing procedure such that inference can be made without the original codebase. [pickle, model.save]

Train -> save model to storage -> load model in sever

Your code should serialize/store the exact code used in the experiment.

At the end of the project create a separate repo with the same code and remove any project specific parts, add comments of where the next user should probably insert important code. Create a README telling users how to easily use the template.

## Bonus

Try to have a way to visualize the parameters and metrics given by various experiments so you can compare them.
Try to separate code that sets up the experiment (which should be generic) from the code that does the preprocessing procedure and model definition (which is project specific).
Add nice features to the template such as data splitting, automatic exploration of the data, hyper parameter tuning, etc.

## Tips

Check out tools / services like ML Flow and Weights and Biases.
Scikit Learn’s custom transformers are a great way to perform complex preprocessing.
Structure idea
/src # your actual code
   ….
/results  #<- gitignore
   experiment1/ # serialized stuff goes here
       …
   experiment2/
