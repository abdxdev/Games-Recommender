import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from models.recommendation_models import RecommendationModels


class ModelEvaluation:
    def __init__(self, recommendation_models=None):
        if recommendation_models is None:
            self.rec_models = RecommendationModels()
        else:
            self.rec_models = recommendation_models

        if not hasattr(self.rec_models, "data") or self.rec_models.data is None:
            self.rec_models.data = self.rec_models.data_loader.load_all_data()

        if self.rec_models.user_game_matrix is None or self.rec_models.hours_matrix is None:
            self.rec_models.create_matrices()

        if self.rec_models.knn_model is None:
            self.rec_models.build_knn_model()

        if self.rec_models.kmeans_model is None:
            self.rec_models.build_kmeans_model()

        if self.rec_models.nb_model is None:
            self.rec_models.build_naive_bayes_model()

        if self.rec_models.rf_model is None:
            self.rec_models.build_random_forest_model()

        self.evaluation_results = {}

    def evaluate_classification_models(self):
        if self.rec_models.hours_matrix is None:
            self.rec_models.create_matrices()

        X = self.rec_models.hours_matrix.values

        user_ids = self.rec_models.hours_matrix.index.tolist()

        user_game_stats = self.rec_models.data["recommendations"][self.rec_models.data["recommendations"]["user_id"].isin(user_ids)].groupby("user_id").agg({"hours": "sum", "app_id": "count", "is_recommended": "mean"}).reset_index()

        user_game_stats.columns = ["user_id", "total_hours", "games_owned", "positive_ratio"]

        user_game_stats["user_category"] = "Medium"

        user_game_stats.loc[(user_game_stats["total_hours"] < user_game_stats["total_hours"].quantile(0.33)) & (user_game_stats["games_owned"] < user_game_stats["games_owned"].quantile(0.33)), "user_category"] = "Casual"
        user_game_stats.loc[(user_game_stats["total_hours"] > user_game_stats["total_hours"].quantile(0.66)) & (user_game_stats["games_owned"] > user_game_stats["games_owned"].quantile(0.66)), "user_category"] = "Hardcore"
        user_game_stats.loc[(user_game_stats["total_hours"] < user_game_stats["total_hours"].quantile(0.33)) & (user_game_stats["games_owned"] > user_game_stats["games_owned"].quantile(0.66)), "user_category"] = "Collector"
        user_game_stats.loc[(user_game_stats["total_hours"] > user_game_stats["total_hours"].quantile(0.66)) & (user_game_stats["games_owned"] < user_game_stats["games_owned"].quantile(0.33)), "user_category"] = "Selective"

        user_categories = pd.DataFrame(index=self.rec_models.hours_matrix.index)
        user_categories = user_categories.reset_index().merge(user_game_stats[["user_id", "user_category"]], left_on="user_id", right_on="user_id", how="left").set_index("user_id")["user_category"].fillna("Medium")

        y = user_categories.values

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        if self.rec_models.nb_model is None:
            self.rec_models.build_naive_bayes_model()

        y_pred_nb = self.rec_models.nb_model.predict(X_test)

        if hasattr(y_pred_nb[0], "dtype") and np.issubdtype(y_pred_nb[0].dtype, np.str_):
            y_pred_nb = label_encoder.transform(y_pred_nb)

        nb_results = {
            "accuracy": accuracy_score(y_test, y_pred_nb),
            "precision": precision_score(y_test, y_pred_nb, average="weighted"),
            "recall": recall_score(y_test, y_pred_nb, average="weighted"),
            "f1": f1_score(y_test, y_pred_nb, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, y_pred_nb),
        }

        if self.rec_models.rf_model is None:
            self.rec_models.build_random_forest_model()

        y_pred_rf = self.rec_models.rf_model.predict(X_test)

        if not isinstance(y_pred_rf[0], (np.integer, int)):
            y_pred_rf = label_encoder.transform(y_pred_rf)

        rf_results = {
            "accuracy": accuracy_score(y_test, y_pred_rf),
            "precision": precision_score(y_test, y_pred_rf, average="weighted"),
            "recall": recall_score(y_test, y_pred_rf, average="weighted"),
            "f1": f1_score(y_test, y_pred_rf, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, y_pred_rf),
        }

        from sklearn.neighbors import KNeighborsClassifier

        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(X_train, y_train)
        y_pred_knn = knn_classifier.predict(X_test)

        knn_results = {
            "accuracy": accuracy_score(y_test, y_pred_knn),
            "precision": precision_score(y_test, y_pred_knn, average="weighted"),
            "recall": recall_score(y_test, y_pred_knn, average="weighted"),
            "f1": f1_score(y_test, y_pred_knn, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, y_pred_knn),
        }

        self.evaluation_results["classification"] = {"nb": nb_results, "rf": rf_results, "knn": knn_results, "label_mapping": dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}

        return self.evaluation_results["classification"]

    def evaluate_clustering(self, n_clusters=5):
        if self.rec_models.hours_matrix is None:
            self.rec_models.create_matrices()

        X = self.rec_models.hours_matrix.values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.rec_models.kmeans_model is None or self.rec_models.kmeans_model.n_clusters != n_clusters:
            self.rec_models.build_kmeans_model(n_clusters=n_clusters)

        cluster_labels = self.rec_models.kmeans_model.predict(X_scaled)

        silhouette = -1
        try:
            silhouette = silhouette_score(X_scaled, cluster_labels)
        except Exception as e:
            print(f"Could not compute silhouette score: {str(e)}")

        self.evaluation_results["clustering"] = {
            "kmeans": {
                "inertia": self.rec_models.kmeans_model.inertia_,
                "silhouette": silhouette,
                "n_clusters": n_clusters,
                "cluster_centers": self.rec_models.kmeans_model.cluster_centers_,
                "cluster_distribution": np.bincount(cluster_labels).tolist(),
            }
        }

        return self.evaluation_results["clustering"]

    def compare_algorithms(self):
        if "classification" not in self.evaluation_results:
            self.evaluate_classification_models()

        metrics = ["accuracy", "precision", "recall", "f1"]
        algorithms = ["knn", "nb", "rf"]
        comparison = pd.DataFrame(index=metrics, columns=algorithms)

        for algo in algorithms:
            for metric in metrics:
                comparison.loc[metric, algo] = self.evaluation_results["classification"][algo][metric]

        self.evaluation_results["comparison"] = comparison.to_dict()

        return comparison

    def visualize_clusters(self, save_path="models", return_data=False):
        if "clustering" not in self.evaluation_results:
            self.evaluate_clustering()

        X = self.rec_models.hours_matrix.values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        cluster_labels = self.rec_models.kmeans_model.predict(X_scaled)

        if return_data:
            return {
                "pca_components": X_pca,
                "cluster_labels": cluster_labels,
                "cluster_centers": pca.transform(self.rec_models.kmeans_model.cluster_centers_),
                "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
                "cluster_distribution": np.bincount(cluster_labels).tolist(),
            }

        return {"pca_explained_variance": pca.explained_variance_ratio_.tolist(), "cluster_distribution": np.bincount(cluster_labels).tolist()}

    def explain_algorithm_choice(self):
        explanation = """
        # Algorithm Selection and Comparison
        
        ## Why Random Forest?
        
        Random Forest was selected as our fourth algorithm for several reasons:
        
        1. **Robustness to Overfitting**: Unlike single decision trees, Random Forest reduces overfitting by averaging multiple decision trees trained on different subsets of the data.
        2. **Feature Importance**: It provides insights into which features are most important for classification, helping us understand user behavior patterns.
        3. **Handles Mixed Data Types**: Our dataset contains various metrics about user behavior, and Random Forest naturally handles mixed data types and scales.
        4. **Non-linearity**: User behaviors are often non-linear, and Random Forest captures these complex relationships better than linear models.
        5. **Accuracy**: Random Forest typically achieves high accuracy in classification tasks compared to simpler models.
        
        ## Algorithm Comparison
        
        Based on our evaluation metrics (F1 Score, Precision, Recall, and Accuracy):
        
        - **KNN**: Simple but effective for neighbor-based classification, useful as a baseline but sensitive to the choice of K.
        - **Naive Bayes**: Fast and efficient with lower memory requirements, but makes strong independence assumptions that may not hold for user behavior.
        - **K-Means**: Provides unsupervised clustering to identify natural groupings in the data, complementing our supervised methods.
        - **Random Forest**: Typically achieves the highest accuracy and F1 score among the tested algorithms, making it particularly valuable for predicting user preferences.
        
        ## Future Improvements
        
        To further improve results, we could:
        
        1. Implement hyperparameter tuning for each algorithm
        2. Feature engineering to extract more meaningful patterns from data
        3. Create ensemble methods combining multiple algorithms
        4. Use dimensionality reduction to improve model performance
        5. Collect more diverse training data to improve generalization
        """

        return explanation


if __name__ == "__main__":
    evaluator = ModelEvaluation()

    classification_results = evaluator.evaluate_classification_models()
    print("\nClassification Results:")
    for model, results in classification_results.items():
        if model != "label_mapping":
            print(f"\n{model.upper()} Metrics:")
            for metric, value in results.items():
                if metric != "confusion_matrix":
                    print(f"{metric}: {value:.4f}")

    clustering_results = evaluator.evaluate_clustering()
    print("\nClustering Results:")
    print(f"Inertia: {clustering_results['kmeans']['inertia']:.4f}")
    print(f"Silhouette Score: {clustering_results['kmeans']['silhouette']:.4f}")

    comparison = evaluator.compare_algorithms()
    print("\nAlgorithm Comparison:")
    print(comparison)

    evaluator.visualize_clusters()
    print("\nCluster visualizations saved to 'models' directory")
