# Movie recommendation system from scratch

This repository contains the code and documentation for a movie recommendation system. The system uses the MovieLens 25M dataset to provide movie recommendations based on user ratings.


## Dataset
MovieLens 25M dataset from the [GroupLens website](https://grouplens.org/datasets/movielens/25m/).


## Description

This project implements a collaborative filtering recommendation system using matrix factorization with latent vectors. The process includes:
1.  **Data Loading and Preprocessing**: Loading the 25 million ratings and creating efficient data structures for training.
2.  **Exploratory Data Analysis (EDA)**: Visualizing the rating and degree distributions to understand the dataset's characteristics.
3.  **Model Training**: Implementing and training a matrix factorization model using Alternating Least Squares (ALS) to learn user/item biases and latent vectors.
4.  **Embedding Visualization**: Using PCA to reduce the dimensionality of the learned item vectors to 2D for visualization.
5.  **Prediction**: Creating a function to provide real-time movie recommendations for a new user based on a single rated movie.
