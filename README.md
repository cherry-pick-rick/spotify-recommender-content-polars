# Spotify Content Based Recommender

Music recommendation system using Polars, Scikit-learn, and Numpy. Uses track data from Hugging Face datasets to provide personalized music recommendations based on audio features.

## Setup with uv

```bash
# Install uv
pip install uv

# Create virtual environment
uv venv
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt

# Install development dependencies
uv pip install -r requirements-dev.txt
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_data.py

# Run tests with print statements
pytest -s tests/
```

## Usage

```python
from src.recommender import SpotifyRecommender

recommender = SpotifyRecommender()
sample_tracks = recommender.df.select("track_name").sample(n=3).to_series().to_list()
recommendations = recommender.get_batch_recommendations(sample_tracks)
print(recommendations)
```

## Dependencies

- polars
- scikit-learn
- numpy
- pytest
- pytest-mock

## Project Structure

```
spotify-recommender/
├── src/
│   ├── __init__.py
│   ├── recommender.py   # Main recommendation logic
│   └── data.py         # Data loading from Hugging Face
├── tests/
│   ├── __init__.py
│   ├── test_recommender.py
│   └── test_data.py
├── README.md
├── requirements.txt
```

## License

MIT License