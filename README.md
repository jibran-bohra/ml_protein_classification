# ML Researcher Hands-on Challenge

## Overview

This repository provides an overview of the code development process, emphasizing crucial functions and scripts. It serves as a guide to the methodology adopted, facilitating an understanding of the implemented features and their outcomes.

## Getting Started

The code development process is based on an Object-Oriented approach, initiated with an `iPython` notebook. The resulting script, named `main.py`, preprocesses an arbitrary dataset for utilization in a machine learning model. Key functions in `main.py` include:

- **`generate_features()`**: Reads structural data (coordinates, occupancy, and $b$-factor) from `.pdb` files and stores the information in `all_data.json`.

- **`preprocess()`**: Gathers columns from `all_data.json` for preprocessing. It tokenizes protein sequence data, normalizes protein structure data (e.g., coordinates), normalizes numerical data (e.g., coordinate mean and standard deviations, occupancy, $b$-factor), and prepares class label data for one-hot encoding. Preprocessed data is saved in `structure.npy` and `preprocessed.json`.

## General Model and Results

A comprehensive model for classifying protein architectures is developed using protein sequence, protein structure, and various numerical data. The model architecture can be found in `all_data_classifier.py`. The model achieved a minimum of 90% accuracy on unseen samples.


