import os
import pickle

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_loader import DataLoader


class RecommendationModels:
    def __init__(self, data_loader=None):
        if data_loader is None:
            self.data_loader = DataLoader()
            self.data = self.data_loader.load_all_data()
        else:
            self.data_loader = data_loader
            self.data = data_loader.load_all_data()

        self.user_game_matrix = None
        self.hours_matrix = None
        self.game_similarity_matrix = None
        self.popular_games = None
        self.knn_model = None
        self.kmeans_model = None
        self.nb_model = None
        self.rf_model = None

    def create_matrices(self):
        self.user_game_matrix, self.hours_matrix = self.prepare_user_game_matrix()
        return self.user_game_matrix, self.hours_matrix

    def prepare_user_game_matrix(self):
        if self.user_game_matrix is None or self.hours_matrix is None:
            self.user_game_matrix, self.hours_matrix = self.data_loader.prepare_user_game_matrix(max_users=5000, max_games=3000)

        return self.user_game_matrix, self.hours_matrix

    def compute_game_similarity(self):
        if self.user_game_matrix is None:
            self.create_matrices()

        binary_matrix = self.hours_matrix.astype(bool).astype(int)
        game_similarity = cosine_similarity(binary_matrix.T)
        self.game_similarity_matrix = pd.DataFrame(game_similarity, index=binary_matrix.columns, columns=binary_matrix.columns)

        return self.game_similarity_matrix

    def build_knn_model(self, n_neighbors=10):
        if self.hours_matrix is None:
            self.create_matrices()

        sparse_matrix = csr_matrix(self.hours_matrix.values)
        self.knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn_model.fit(sparse_matrix.T)

        return self.knn_model

    def build_kmeans_model(self, n_clusters=10):
        if self.hours_matrix is None:
            self.create_matrices()

        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(self.hours_matrix.values)

        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans_model.fit(scaled_matrix)

        return self.kmeans_model

    def build_naive_bayes_model(self):
        if self.hours_matrix is None:
            self.create_matrices()

        X = self.hours_matrix.values

        user_ids = self.hours_matrix.index.tolist()
        user_game_stats = self.data["recommendations"][self.data["recommendations"]["user_id"].isin(user_ids)].groupby("user_id").agg({"hours": "sum", "app_id": "count", "is_recommended": "mean"}).reset_index()

        user_game_stats.columns = ["user_id", "total_hours", "games_owned", "positive_ratio"]

        user_game_stats["user_category"] = "Medium"
        user_game_stats.loc[(user_game_stats["total_hours"] < user_game_stats["total_hours"].quantile(0.33)) & (user_game_stats["games_owned"] < user_game_stats["games_owned"].quantile(0.33)), "user_category"] = "Casual"
        user_game_stats.loc[(user_game_stats["total_hours"] > user_game_stats["total_hours"].quantile(0.66)) & (user_game_stats["games_owned"] > user_game_stats["games_owned"].quantile(0.66)), "user_category"] = "Hardcore"
        user_game_stats.loc[(user_game_stats["total_hours"] < user_game_stats["total_hours"].quantile(0.33)) & (user_game_stats["games_owned"] > user_game_stats["games_owned"].quantile(0.66)), "user_category"] = "Collector"
        user_game_stats.loc[(user_game_stats["total_hours"] > user_game_stats["total_hours"].quantile(0.66)) & (user_game_stats["games_owned"] < user_game_stats["games_owned"].quantile(0.33)), "user_category"] = "Selective"
        user_categories = pd.DataFrame(index=self.hours_matrix.index)
        user_categories = user_categories.reset_index().merge(user_game_stats[["user_id", "user_category"]], left_on="user_id", right_on="user_id", how="left").set_index("user_id")["user_category"].fillna("Medium").values

        y = user_categories

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.nb_model = MultinomialNB()
        self.nb_model.fit(X_train, y_train)

        return self.nb_model

    def build_random_forest_model(self):
        if self.hours_matrix is None:
            self.create_matrices()

        X = self.hours_matrix.values

        user_ids = self.hours_matrix.index.tolist()
        user_game_stats = self.data["recommendations"][self.data["recommendations"]["user_id"].isin(user_ids)].groupby("user_id").agg({"hours": "sum", "app_id": "count", "is_recommended": "mean"}).reset_index()

        user_game_stats.columns = ["user_id", "total_hours", "games_owned", "positive_ratio"]

        user_game_stats["user_category"] = "Medium"

        user_game_stats.loc[(user_game_stats["total_hours"] < user_game_stats["total_hours"].quantile(0.33)) & (user_game_stats["games_owned"] < user_game_stats["games_owned"].quantile(0.33)), "user_category"] = "Casual"
        user_game_stats.loc[(user_game_stats["total_hours"] > user_game_stats["total_hours"].quantile(0.66)) & (user_game_stats["games_owned"] > user_game_stats["games_owned"].quantile(0.66)), "user_category"] = "Hardcore"
        user_game_stats.loc[(user_game_stats["total_hours"] < user_game_stats["total_hours"].quantile(0.33)) & (user_game_stats["games_owned"] > user_game_stats["games_owned"].quantile(0.66)), "user_category"] = "Collector"
        user_game_stats.loc[(user_game_stats["total_hours"] > user_game_stats["total_hours"].quantile(0.66)) & (user_game_stats["games_owned"] < user_game_stats["games_owned"].quantile(0.33)), "user_category"] = "Selective"
        user_categories = pd.DataFrame(index=self.hours_matrix.index)
        user_categories = user_categories.reset_index().merge(user_game_stats[["user_id", "user_category"]], left_on="user_id", right_on="user_id", how="left").set_index("user_id")["user_category"].fillna("Medium").values

        y = user_categories

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.rf_model = RandomForestClassifier(random_state=42)
        self.rf_model.fit(X_train, y_train)

        return self.rf_model

    def content_based_recommendations(self, game_id, top_n=10):
        game_features = self.data_loader.get_game_features()

        if "tags" in game_features.columns and isinstance(game_features["tags"].iloc[0], list):
            all_tags = set()
            for tags in game_features["tags"]:
                if isinstance(tags, list):
                    all_tags.update(tags)

            for tag in all_tags:
                game_features[f"tag_{tag}"] = game_features["tags"].apply(lambda x: 1 if tag in x else 0 if isinstance(x, list) else 0)

        numeric_features = game_features.select_dtypes(include=["int64", "float64", "bool"]).drop(["app_id", "price_original", "price_final", "discount"], axis=1, errors="ignore")
        similarities = cosine_similarity(numeric_features, numeric_features)
        sim_df = pd.DataFrame(similarities, index=game_features["app_id"], columns=game_features["app_id"])
        similar_games = sim_df[game_id].sort_values(ascending=False)[1 : top_n + 1]
        recommended_games = game_features[game_features["app_id"].isin(similar_games.index)]
        
        return recommended_games[["app_id", "title", "rating", "positive_ratio", "user_reviews", "price_final"]]

    def save_models(self, base_path="models"):
        os.makedirs(base_path, exist_ok=True)

        if self.user_game_matrix is None:
            self.create_matrices()

        if self.game_similarity_matrix is None:
            self.compute_game_similarity()

        model_files = {
            "user_game_matrix.pkl": self.user_game_matrix,
            "hours_matrix.pkl": self.hours_matrix,
            "game_similarity_matrix.pkl": self.game_similarity_matrix,
            "knn_model.pkl": self.knn_model if self.knn_model is not None else self.build_knn_model(),
            "kmeans_model.pkl": self.kmeans_model if self.kmeans_model is not None else self.build_kmeans_model(),
            "nb_model.pkl": self.nb_model if self.nb_model is not None else self.build_naive_bayes_model(),
            "rf_model.pkl": self.rf_model if self.rf_model is not None else self.build_random_forest_model(),
        }

        for filename, model in model_files.items():
            with open(os.path.join(base_path, filename), "wb") as f:
                pickle.dump(model, f)

    def load_models(self, base_path="models"):
        try:
            model_files = {
                "user_game_matrix.pkl": "user_game_matrix",
                "hours_matrix.pkl": "hours_matrix",
                "game_similarity_matrix.pkl": "game_similarity_matrix",
                "knn_model.pkl": "knn_model",
                "kmeans_model.pkl": "kmeans_model",
                "nb_model.pkl": "nb_model",
                "rf_model.pkl": "rf_model",
            }

            for filename, attr_name in model_files.items():
                file_path = os.path.join(base_path, filename)
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        setattr(self, attr_name, pickle.load(f))

            return True
        except FileNotFoundError:
            print("Models not found. Please train the models first.")
            return False
