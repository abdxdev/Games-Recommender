import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
import os
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

    def compute_popular_games(self, min_reviews=100):
        games_df = self.data["games"].copy()

        filtered_games = games_df[games_df["user_reviews"] >= min_reviews]

        self.popular_games = filtered_games.sort_values(by=["positive_ratio", "user_reviews"], ascending=False).reset_index(drop=True)

        return self.popular_games

    def build_knn_model(self, n_neighbors=10):
        if self.hours_matrix is None:
            self.create_matrices()

        sparse_matrix = csr_matrix(self.hours_matrix.values)

        self.knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn_model.fit(sparse_matrix.T)

        return self.knn_model

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

    def collaborative_filtering_recommendations(self, game_id, top_n=10):
        if self.knn_model is None:
            self.build_knn_model()

        if game_id not in self.hours_matrix.columns:
            return pd.DataFrame()

        game_idx = self.hours_matrix.columns.get_loc(game_id)

        distances, indices = self.knn_model.kneighbors(self.hours_matrix.values.T[game_idx].reshape(1, -1), n_neighbors=top_n + 1)

        similar_game_indices = indices.flatten()[1:]
        similar_game_ids = [self.hours_matrix.columns[idx] for idx in similar_game_indices]

        similarity_scores = 1 - distances.flatten()[1:]

        recommendations = pd.DataFrame({"app_id": similar_game_ids, "similarity_score": similarity_scores})

        recommended_games = pd.merge(
            recommendations,
            self.data["games"][
                [
                    "app_id",
                    "title",
                    "rating",
                    "positive_ratio",
                    "user_reviews",
                    "price_final",
                ]
            ],
            on="app_id",
        ).sort_values("similarity_score", ascending=False)

        return recommended_games

    def hybrid_recommendations(self, game_id, top_n=10, cf_weight=0.7, cb_weight=0.3):
        cf_recs = self.collaborative_filtering_recommendations(game_id, top_n=top_n * 2)
        cb_recs = self.content_based_recommendations(game_id, top_n=top_n * 2)

        if cf_recs.empty or cb_recs.empty:
            return cf_recs if not cf_recs.empty else cb_recs

        cf_recs["cf_score"] = cf_recs["similarity_score"] / cf_recs["similarity_score"].max()

        merged_recs = pd.merge(cf_recs[["app_id", "cf_score"]], cb_recs[["app_id"]], on="app_id", how="outer").fillna(0)

        cb_scores = {}
        for app_id in merged_recs["app_id"]:
            if app_id in cb_recs["app_id"].values:
                cb_scores[app_id] = 1.0 - cb_recs[cb_recs["app_id"] == app_id].index.item() / len(cb_recs)
            else:
                cb_scores[app_id] = 0

        merged_recs["cb_score"] = merged_recs["app_id"].map(cb_scores)

        merged_recs["hybrid_score"] = (cf_weight * merged_recs["cf_score"]) + (cb_weight * merged_recs["cb_score"])

        top_recs = merged_recs.sort_values("hybrid_score", ascending=False).head(top_n)

        recommended_games = pd.merge(
            top_recs[["app_id", "hybrid_score"]],
            self.data["games"][
                [
                    "app_id",
                    "title",
                    "rating",
                    "positive_ratio",
                    "user_reviews",
                    "price_final",
                ]
            ],
            on="app_id",
        ).sort_values("hybrid_score", ascending=False)

        return recommended_games

    def save_models(self, base_path="models"):
        os.makedirs(base_path, exist_ok=True)

        if self.user_game_matrix is None:
            self.create_matrices()

        if self.game_similarity_matrix is None:
            self.compute_game_similarity()

        if self.popular_games is None:
            self.compute_popular_games()

        if self.knn_model is None:
            self.build_knn_model()

        with open(os.path.join(base_path, "user_game_matrix.pkl"), "wb") as f:
            pickle.dump(self.user_game_matrix, f)

        with open(os.path.join(base_path, "hours_matrix.pkl"), "wb") as f:
            pickle.dump(self.hours_matrix, f)

        with open(os.path.join(base_path, "game_similarity_matrix.pkl"), "wb") as f:
            pickle.dump(self.game_similarity_matrix, f)

        with open(os.path.join(base_path, "popular_games.pkl"), "wb") as f:
            pickle.dump(self.popular_games, f)

        with open(os.path.join(base_path, "knn_model.pkl"), "wb") as f:
            pickle.dump(self.knn_model, f)

    def load_models(self, base_path="models"):
        try:
            with open(os.path.join(base_path, "user_game_matrix.pkl"), "rb") as f:
                self.user_game_matrix = pickle.load(f)

            with open(os.path.join(base_path, "hours_matrix.pkl"), "rb") as f:
                self.hours_matrix = pickle.load(f)

            with open(os.path.join(base_path, "game_similarity_matrix.pkl"), "rb") as f:
                self.game_similarity_matrix = pickle.load(f)

            with open(os.path.join(base_path, "popular_games.pkl"), "rb") as f:
                self.popular_games = pickle.load(f)

            with open(os.path.join(base_path, "knn_model.pkl"), "rb") as f:
                self.knn_model = pickle.load(f)

            return True
        except FileNotFoundError:
            print("Models not found. Please train the models first.")
            return False

    def get_game_title_by_id(self, game_id):
        game = self.data["games"][self.data["games"]["app_id"] == game_id]
        if not game.empty:
            return game["title"].iloc[0]
        return None


if __name__ == "__main__":
    models = RecommendationModels()
    models.create_matrices()
    models.compute_game_similarity()
    models.build_knn_model()

    test_game_id = models.data["games"]["app_id"].iloc[0]
    print(f"Getting recommendations for game: {models.get_game_title_by_id(test_game_id)}")

    print("\nCollaborative Filtering Recommendations:")
    cf_recs = models.collaborative_filtering_recommendations(test_game_id)
    print(cf_recs.head())

    print("\nContent-Based Recommendations:")
    cb_recs = models.content_based_recommendations(test_game_id)
    print(cb_recs.head())

    print("\nHybrid Recommendations:")
    hybrid_recs = models.hybrid_recommendations(test_game_id)
    print(hybrid_recs.head())
