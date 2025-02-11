import pytest
import polars as pl
from src.recommender import SpotifyRecommender

@pytest.fixture
def sample_data():
    return pl.DataFrame({
        "track_name": ["Song1", "Song2", "Song3"],
        "artists": ["Artist1", "Artist2", "Artist3"],
        "track_genre": ["Pop", "Rock", "Jazz"],
        "popularity": [80, 75, 70],
        "danceability": [0.8, 0.7, 0.6],
        "energy": [0.9, 0.8, 0.7],
        "loudness": [-5.0, -6.0, -7.0],
        "speechiness": [0.1, 0.2, 0.3],
        "acousticness": [0.2, 0.3, 0.4],
        "instrumentalness": [0.0, 0.1, 0.2],
        "liveness": [0.1, 0.2, 0.3],
        "valence": [0.8, 0.7, 0.6]
    })

@pytest.fixture
def recommender(monkeypatch, sample_data):
    monkeypatch.setattr(pl, "read_csv", lambda _: sample_data)
    return SpotifyRecommender()

def test_get_recommendations(recommender):
    recommendations = recommender.get_recommendations("Song1", n_recommendations=2)
    assert len(recommendations) == 2
    assert "track_name" in recommendations.columns
    assert "Song1" not in recommendations["track_name"]

def test_get_batch_recommendations(recommender):
    track_names = ["Song1", "Song2"]
    recommendations = recommender.get_batch_recommendations(track_names, n_recommendations=2)
    assert len(recommendations) == 2
    assert "recommendations" in recommendations.columns
    assert all(len(rec) == 2 for rec in recommendations["recommendations"])

def test_invalid_track_name(recommender):
    with pytest.raises(Exception):
        recommender.get_recommendations("NonexistentSong")