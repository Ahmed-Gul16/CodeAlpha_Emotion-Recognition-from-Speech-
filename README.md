# CodeAlpha_Emotion-Recognition-from-Speech-
This repository contains a deep learning project developed during my ML Internship, focused on identifying human emotions from speech audio using advanced signal processing and neural networks.
The project builds a complete end-to-end pipeline that transforms raw audio waveforms into statistical frequency features to classify distinct emotional states.

# Key Features
Dataset: Evaluated on the structured RAVDESS (Audio-Speech-Actors) dataset, encompassing 8 distinct emotional categories.

# Audio Processing: 
Utilizes librosa to extract 40 Mel-Frequency Cepstral Coefficients (MFCCs), capturing how human ears perceive changes in vocal pitch, energy, and tone pitch over time.

# Architecture:
Implements a Multi-Layer Perceptron (Dense Neural Network) in PyTorch using object-oriented design patterns, featuring dropout regularization to prevent overfitting.

# Custom Inference: 
Includes an inference script to test the trained model on custom .wav voice recordings with an automated probability breakdown.

# Performance Baseline
The baseline Multi-Layer Perceptron model achieves ~69.10% validation accuracy across 8 emotion classes within 50 epochs, establishing a strong foundation for future convolutional (1D/2D CNN) or recurrent (LSTM) architecture upgrades.

# Tech Stack
Language: Python

Deep Learning Framework: PyTorch

Audio & Signal Processing: Librosa, Soundfile

Data Science Tools: NumPy, Scikit-Learn
