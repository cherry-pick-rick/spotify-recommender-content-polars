import pytest
from src.data import load_spotify_data

def test_load_spotify_data():
    df = load_spotify_data()
    required_columns = [
        "track_name", "artists", "track_genre", "popularity",
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence"
    ]
    assert all(col in df.columns for col in required_columns)
    assert len(df) > 0