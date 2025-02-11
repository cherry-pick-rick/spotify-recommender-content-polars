import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from data import load_spotify_data

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
        ).astype(np.float32)  # Convert to float32 for memory efficiency
        
        # Remove the precomputation of similarity matrix
        # self.similarity_matrix = cosine_similarity(self.features_normalized)  # Remove this line

    def get_recommendations(self, track_name: str, n_recommendations: int = 5) -> pl.DataFrame:
        track_idx = self.df.with_row_count().filter(
            pl.col("track_name") == track_name
        ).select("row_nr").item()
        
        # Get the features for the target track
        track_features = self.features_normalized[track_idx].reshape(1, -1)
        
        # Calculate similarities only for this track
        similarities = cosine_similarity(track_features, self.features_normalized)[0]
        similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        
        return self.df.select([
            "track_name",
            "artists",
            "track_genre",
            "popularity"
        ]).filter(pl.int_range(0, len(self.df)).is_in(similar_indices))

    def get_batch_recommendations(self, track_names: list[str], n_recommendations: int = 5) -> pl.DataFrame:
        # Get recommendations one by one and convert to list of records
        all_recommendations = []
        for track in track_names:
            recs = self.get_recommendations(track, n_recommendations).to_dict(as_series=False)
            all_recommendations.append({
                'input_track': track,
                'recommended_tracks': recs['track_name'],
                'artists': recs['artists'],
                'genres': recs['track_genre'],
                'popularity': recs['popularity']
            })
        
        return pl.DataFrame(all_recommendations)

# Example usage
if __name__ == "__main__":
    recommender = SpotifyRecommender()
    
    # Get recommendations for specific tracks
    sample_tracks = recommender.df.select("track_name").sample(n=3).to_series().to_list()
    recommendations = recommender.get_batch_recommendations(sample_tracks)
    
    # Print in a more readable format
    for track in sample_tracks:
        print(f"\nRecommendations for '{track}':")
        track_recs = recommendations.filter(pl.col("input_track") == track)
        print(track_recs)