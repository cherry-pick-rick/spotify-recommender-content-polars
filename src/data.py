import polars as pl

def load_spotify_data() -> pl.DataFrame:
   return pl.read_csv('hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv')