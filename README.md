# Steam Game Recommendation System

A comprehensive recommendation system for Steam games built using Python and Streamlit, based on data from the [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) Kaggle dataset.

## Project Overview

This system provides game recommendations using multiple approaches:

1. **Collaborative Filtering**: Recommends games based on user behavior patterns and similarities between users' preferences
2. **Content-Based Filtering**: Recommends games similar to a selected game based on game features and characteristics
3. **Hybrid Recommender**: Combines collaborative and content-based approaches for better recommendations
4. **Popularity-Based**: Recommends top-rated and widely reviewed games

## Features

- **Interactive Web Interface**: Built with Streamlit for an easy-to-use experience
- **Multiple Recommendation Algorithms**: Collaborative filtering, content-based, hybrid, and popularity-based
- **Game Explorer**: Search and browse games by title, popularity, release date, or price range
- **Visualization Tools**: Visual representations of recommendations and dataset distributions
- **Model Evaluation**: Compare performance of different recommendation algorithms

## Project Structure

```
steam-game-recommender/
├── app.py                  # Main Streamlit application
├── utils/
│   └── data_loader.py      # Data loading and preprocessing
├── models/
│   ├── recommendation_models.py  # Implementation of recommendation algorithms
│   └── model_evaluation.py      # Evaluation of recommendation algorithms
├── datasets/               # Dataset files
│   ├── games.csv           # Game information
│   ├── users.csv           # User information
│   ├── recommendations.csv  # User game interactions and recommendations
│   └── games_metadata.json  # Additional game metadata
└── README.md               # Project documentation
```

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Place the dataset files in the `datasets/` directory (games.csv, users.csv, recommendations.csv, games_metadata.json)

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will be accessible in your web browser. The first run may take some time as the system needs to train the recommendation models.

## Recommendation Algorithms

### Collaborative Filtering

Uses K-Nearest Neighbors to find similar games based on user interactions. The system analyzes patterns in user interactions to recommend games that users with similar preferences have enjoyed.

### Content-Based Filtering

Recommends games based on game features such as tags, platforms, price category, and other attributes. This approach finds games that are similar to a selected game based on their characteristics.

### Hybrid Approach

Combines collaborative and content-based approaches by weighting the recommendations from both methods to provide more robust recommendations that leverage the strengths of both approaches.

### Popularity-Based

Simple recommendation approach that suggests the most popular games based on user reviews and positive rating ratios.

## Model Evaluation

The system includes a comprehensive evaluation module that allows comparing different recommendation approaches using:

- Hit Rate: How many recommended games match a user's positively rated games
- Top-N Analysis: How performance varies with different numbers of recommendations
- User-specific evaluation: Analyzing recommendation quality for individual users

## Dataset

The dataset includes:

- Game information (title, rating, price, platforms, etc.)
- User information
- User-game interactions (recommendations, hours played)
- Additional game metadata

## Future Improvements

- Implement matrix factorization algorithms (SVD, ALS)
- Add personalized recommendations based on user profiles
- Improve model training time with efficient algorithms
- Add more visualization options
- Implement real-time recommendation updates

## License

This project is for educational purposes. Dataset provided by [Anton Kozyriev on Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam).