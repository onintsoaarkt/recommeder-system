# functions.py

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.decomposition import PCA
import os

def load_and_preprocess_data(ratings_file="ratings.csv"):
    """Loads ratings data and creates necessary mapping and data structures."""
    print("Loading and preprocessing data...")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"{ratings_file} not found. Please download the MovieLens 25M dataset and place it in the root directory.")

    with open(ratings_file, encoding='utf-8') as r:
        ratings_reader = csv.reader(r)
        next(ratings_reader)  # Skip header
        ratings_data = list(ratings_reader)

    map_user_to_index = {}
    map_index_to_user = []
    map_movie_to_index = {}
    map_index_to_movie = []
    data_by_user_index = []
    data_by_movie_index = []

    user_index = 0
    movie_index = 0

    for i in ratings_data:
        us_index = int(i[0])
        mov_index = int(i[1])
        rat_index = float(i[2])

        if us_index not in map_user_to_index:
            map_user_to_index[us_index] = user_index
            map_index_to_user.append(us_index)
            data_by_user_index.append([])
            user_index += 1

        if mov_index not in map_movie_to_index:
            map_movie_to_index[mov_index] = movie_index
            map_index_to_movie.append(mov_index)
            data_by_movie_index.append([])
            movie_index += 1

        up_user_index = map_user_to_index[us_index]
        up_movie_index = map_movie_to_index[mov_index]

        data_by_user_index[up_user_index].append((up_movie_index, rat_index))
        data_by_movie_index[up_movie_index].append((up_user_index, rat_index))

    print(f"Data loaded. Found {len(map_user_to_index)} users and {len(map_movie_to_index)} movies.")
    
    return (ratings_data, map_user_to_index, map_index_to_user, map_movie_to_index, 
            map_index_to_movie, data_by_user_index, data_by_movie_index)


def create_eda_plots(data_by_user_index, data_by_movie_index, ratings_data, output_dir="images"):
    """Generates and saves EDA plots for degree and rating distributions."""
    print("Performing exploratory data analysis...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Degree distribution
    degree_user = [len(i) for i in data_by_user_index]
    degree_item = [len(j) for j in data_by_movie_index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(degree_user, bins=100, log=True, label='User', alpha=0.7)
    ax.hist(degree_item, bins=100, log=True, label='Item', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Degree (Number of Ratings)')
    ax.set_ylabel('Frequency')
    ax.set_title('Degree Distribution')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'degree_distribution.pdf'), format='pdf')
    plt.close()
    print(f"Saved degree distribution plot to '{output_dir}/'.")

    # Rating distribution
    all_ratings = [float(i[2]) for i in ratings_data]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(all_ratings, bins=np.arange(0.25, 5.75, 0.5), alpha=0.7, rwidth=0.8)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count (in millions)')
    ax.set_title('Ratings Distribution')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1_000_000:.1f}M'))
    plt.savefig(os.path.join(output_dir, 'rating_distribution.pdf'), format='pdf')
    plt.close()
    print(f"Saved rating distribution plot to '{output_dir}/'.")


def split_into_train_and_test(data, split_value=0.1):
    """Splits a list of lists into training and testing sets."""
    print(f"Splitting data with test ratio {split_value}...")
    data_train = [[] for _ in range(len(data))]
    data_test = [[] for _ in range(len(data))]
    for i in range(len(data)):
        for j in data[i]:
            if np.random.rand() < split_value:
                data_test[i].append(j)
            else:
                data_train[i].append(j)
    return data_train, data_test


def compute_loss_and_rmse_with_embeddings(data_by_usr, data_by_itm, K=10, epochs=10, lambd=0.01, gamma=0.0001, tau=0.9):
    """Trains a matrix factorization model and returns the learned parameters."""
    print(f"Training Matrix Factorization model with K={K}, epochs={epochs}...")
    user_biases = np.zeros(len(data_by_usr))
    item_biases = np.zeros(len(data_by_itm))
    user_vector = np.random.normal(0, 1/np.sqrt(K), (len(data_by_usr), K))
    item_vector = np.random.normal(0, 1/np.sqrt(K), (len(data_by_itm), K))
    
    for epoch in range(epochs):
        # Update user biases and vectors
        for m, user_ratings in enumerate(data_by_usr):
            if not user_ratings: continue
            
            predictions = [(user_vector[m] @ item_vector[n]) + item_biases[n] for n, r in user_ratings]
            errors = [r - p for (n, r), p in zip(user_ratings, predictions)]
            bias_sum = np.sum(errors)
            user_biases[m] = bias_sum / (len(user_ratings) + gamma)
            
            sum1 = np.sum([np.outer(item_vector[n], item_vector[n]) for n, r in user_ratings], axis=0)
            sum2 = np.sum([item_vector[n] * (r - user_biases[m] - item_biases[n]) for n, r in user_ratings], axis=0)
            user_vector[m] = np.linalg.solve(lambd * sum1 + tau * np.identity(K), lambd * sum2)

        # Update item biases and vectors
        for n, item_ratings in enumerate(data_by_itm):
            if not item_ratings: continue

            predictions = [(user_vector[m] @ item_vector[n]) + user_biases[m] for m, r in item_ratings]
            errors = [r - p for (m, r), p in zip(item_ratings, predictions)]
            bias_sum = np.sum(errors)
            item_biases[n] = bias_sum / (len(item_ratings) + gamma)

            sum1 = np.sum([np.outer(user_vector[m], user_vector[m]) for m, r in item_ratings], axis=0)
            sum2 = np.sum([user_vector[m] * (r - user_biases[m] - item_biases[n]) for m, r in item_ratings], axis=0)
            item_vector[n] = np.linalg.solve(lambd * sum1 + tau * np.identity(K), lambd * sum2)

        # Compute and print loss and RMSE for the epoch
        rmse_one, count = 0, 0
        for m, user_ratings in enumerate(data_by_usr):
            for n, r in user_ratings:
                error = r - (user_vector[m] @ item_vector[n] + user_biases[m] + item_biases[n])
                rmse_one += error**2
                count += 1
        
        rmse = np.sqrt(rmse_one / count) if count > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}: rmse = {rmse:.4f}")
        
    return user_biases, item_biases, user_vector, item_vector


def visualize_embeddings(item_vec, map_movie_to_index, output_dir="images"):
    """Reduces item vector dimensionality with PCA and plots the results."""
    print("Visualizing 2D embeddings...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pca = PCA(n_components=2)
    item_vectors_2d = pca.fit_transform(item_vec)

    movie_labels = {
        1: "Toy Story", 6377: "Finding Nemo", 2011: "Back to the Future I",
        3949: "Requiem for a Dream", 858: "The Godfather", 1035: "The Sound of Music",
        6016: "City of God", 899: "Singin' in the Rain", 527: "Schindler's list", 364: "The Lion King",
    }

    plt.figure(figsize=(10, 8))
    plt.scatter(item_vectors_2d[:, 0], item_vectors_2d[:, 1], alpha=0.5, s=10, color='#1f77b4')

    for movie_id, label in movie_labels.items():
        if movie_id in map_movie_to_index:
            idx = map_movie_to_index[movie_id]
            if idx < len(item_vectors_2d):
                x, y = item_vectors_2d[idx]
                plt.text(x + 0.02, y + 0.02, label, fontsize=9)

    plt.title(r"2D Embeddings of Item Trait Vectors $v_n$")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '2d_embedding.pdf'), format='pdf')
    plt.close()
    print(f"Saved 2D embedding plot to '{output_dir}/'.")


def list_possible_movies(movies_df, title_part: str):
    """Returns a dataframe of movies matching a title part."""
    return movies_df[movies_df['title'].str.contains(title_part, case=False)]


def predict_similar_movies(movie_id, rating, num_pred, movies_df, map_movie_to_index, map_index_to_movie, item_vec, item_bi, K=10, lambd=0.01, tau=0.9):
    """Predicts similar movies for a new user given a single movie rating."""
    if movie_id not in map_movie_to_index:
        return "Movie not found in the dataset."

    movie_idx = map_movie_to_index[movie_id]
    
    # Train a new user vector for a few epochs on the single given rating
    new_user_vector = np.random.normal(0, 1/np.sqrt(K), (K,))
    for _ in range(5):
        sum1 = np.outer(item_vec[movie_idx], item_vec[movie_idx])
        sum2 = item_vec[movie_idx] * (rating - item_bi[movie_idx])
        A = lambd * sum1 + tau * np.identity(K)
        b = lambd * sum2
        new_user_vector = np.linalg.solve(A, b)

    # Score all movies for this new user
    scores = item_vec @ new_user_vector + item_bi
    
    # Get top N recommendations, excluding the movie they already rated
    top_indices = np.argsort(scores)[::-1]
    
    recommended_movie_ids = []
    for i in top_indices:
        if i != movie_idx:
            recommended_movie_ids.append(map_index_to_movie[i])
        if len(recommended_movie_ids) == num_pred:
            break
            
    return movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
