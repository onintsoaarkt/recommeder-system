# main.py

import pandas as pd
import functions as fn
import os

def main():
    # --- Configuration ---
    RATINGS_FILE = "ratings.csv"
    MOVIES_FILE = "movies.csv"
    OUTPUT_DIR = "images"

    # --- 1. Data Loading and Preprocessing ---
    try:
        (ratings_data, map_user_to_index, _, map_movie_to_index, 
         map_index_to_movie, data_by_user_index, data_by_movie_index) = fn.load_and_preprocess_data(RATINGS_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    # --- 2. Exploratory Data Analysis ---
    fn.create_eda_plots(data_by_user_index, data_by_movie_index, ratings_data, OUTPUT_DIR)

    # --- 3. Split Data into Training and Test Sets ---
    data_by_user_train, _ = fn.split_into_train_and_test(data_by_user_index)
    data_by_item_train, _ = fn.split_into_train_and_test(data_by_movie_index)

    # --- 4. Model Training ---
    user_bi, item_bi, user_vec, item_vec = fn.compute_loss_and_rmse_with_embeddings(
        data_by_user_train, 
        data_by_item_train, 
        K=10, 
        epochs=10
    )

    # --- 5. 2D Embedding Visualization ---
    fn.visualize_embeddings(item_vec, map_movie_to_index, OUTPUT_DIR)

    # --- 6. Predictions ---
    print("\n" + "="*25)
    print("  Recommendation Examples  ")
    print("="*25 + "\n")

    try:
        movies_df = pd.read_csv(MOVIES_FILE)
    except FileNotFoundError:
        print(f"{MOVIES_FILE} not found. Cannot run prediction examples.")
        return

    # Example 1: Lord of the Rings
    print(">>> Query: 'Lord of the Rings'")
    print(fn.list_possible_movies(movies_df, 'Lord of the Rings'))
    print("\nRecommendations for a user who rated 'The Fellowship of the Ring' (ID 4993) a 5/5:")
    recommendations_lotr = fn.predict_similar_movies(
        movie_id=4993, rating=5, num_pred=10, movies_df=movies_df, 
        map_movie_to_index=map_movie_to_index, map_index_to_movie=map_index_to_movie,
        item_vec=item_vec, item_bi=item_bi
    )
    print(recommendations_lotr)
    # print("\n" + "-"*50 + "\n")

    # Example 2: Toy Story
    print(">>> Query: 'Toy Story'")
    print(fn.list_possible_movies(movies_df, 'Toy Story'))
    print("\nRecommendations for a user who rated 'Toy Story' (ID 1) a 5/5:")
    recommendations_ts = fn.predict_similar_movies(
        movie_id=1, rating=5, num_pred=10, movies_df=movies_df, 
        map_movie_to_index=map_movie_to_index, map_index_to_movie=map_index_to_movie,
        item_vec=item_vec, item_bi=item_bi
    )
    print(recommendations_ts)
    # print("\n" + "-"*50 + "\n")

if __name__ == '__main__':
    main()
