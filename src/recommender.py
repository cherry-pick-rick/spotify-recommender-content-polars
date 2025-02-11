import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from src.data import load_spotify_data

class SpotifyRecommender:
    def __init__(self):
        # Read and prepare data once
        self.df = load_spotify_data()
        self.features = ["danceability", "energy", "loudness", "speechiness",
                        "acousticness", "instrumentalness", "liveness", "valence"]
        
        # Normalize features once
        scaler = MinMaxScaler()
        self.features_normalized = scaler.fit_transform(
            self.df.select(self.features).to_numpy()
        )
        
        # Precompute similarity matrix once
        self.similarity_matrix = cosine_similarity(self.features_normalized)

    def get_recommendations(self, track_name: str, n_recommendations: int = 5) -> pl.DataFrame:
        track_idx = self.df.with_row_count().filter(
            pl.col("track_name") == track_name
        ).select("row_nr").item()
        
        similarities = self.similarity_matrix[track_idx]
        similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        
        return self.df.select([
            "track_name",
            "artists",
            "track_genre",
            "popularity"
        ]).filter(pl.int_range(0, len(self.df)).is_in(similar_indices))

    def get_batch_recommendations(self, track_names: list[str], n_recommendations: int = 5) -> pl.DataFrame:
        return pl.DataFrame({
            'track_name': track_names
        }).select([
            pl.col('track_name'),
            pl.col('track_name').map_batches(
                lambda x: [self.get_recommendations(name, n_recommendations) 
                          for name in x],
                return_dtype=pl.List(pl.DataFrame)
            ).alias('recommendations')
        ])

# Example usage
if __name__ == "__main__":
    recommender = SpotifyRecommender()
    
    # Get batch recommendations
    sample_tracks = recommender.df.select("track_name").sample(n=3).to_series().to_list()
    recommendations = recommender.get_batch_recommendations(sample_tracks)
    print(recommendations)