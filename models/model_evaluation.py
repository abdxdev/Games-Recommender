import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from models.recommendation_models import RecommendationModels
from utils.data_loader import DataLoader
import os

class ModelEvaluation:
    def __init__(self, recommendation_models=None):
        """
        Initialize the model evaluation class
        
        Parameters:
        -----------
        recommendation_models : RecommendationModels
            Instance of RecommendationModels class
        """
        if recommendation_models is None:
            self.rec_models = RecommendationModels()
        else:
            self.rec_models = recommendation_models
            
        # Make sure data is loaded
        if not hasattr(self.rec_models, 'data') or self.rec_models.data is None:
            self.rec_models.data = self.rec_models.data_loader.load_all_data()
            
        # Create matrices if not already created
        if self.rec_models.user_game_matrix is None or self.rec_models.hours_matrix is None:
            self.rec_models.create_matrices()
            
    def evaluate_popular_recommendations(self, top_n=10, min_reviews=100):
        """
        Evaluate popularity-based recommendations
        
        Parameters:
        -----------
        top_n : int
            Number of top games to recommend
        min_reviews : int
            Minimum number of reviews to consider a game popular
        """
        # Compute popular games if not already computed
        if self.rec_models.popular_games is None:
            self.rec_models.compute_popular_games(min_reviews=min_reviews)
            
        # Get top N popular games
        top_games = self.rec_models.popular_games.head(top_n)
        
        # Calculate average metrics of top games
        avg_positive = top_games['positive_ratio'].mean()
        avg_reviews = top_games['user_reviews'].mean()
        
        # Return evaluation metrics
        return {
            'avg_positive_ratio': avg_positive,
            'avg_user_reviews': avg_reviews,
            'top_games': top_games[['app_id', 'title', 'positive_ratio', 'user_reviews']]
        }
    
    def evaluate_recommendations_for_user(self, user_id, method='hybrid', top_n=10):
        """
        Evaluate recommendations for a specific user
        
        Parameters:
        -----------
        user_id : int
            ID of the user
        method : str
            Recommendation method to evaluate ('collaborative', 'content', 'hybrid')
        top_n : int
            Number of recommendations to generate
        """
        # Get user's actual game interactions
        user_games = self.rec_models.data['recommendations'][
            self.rec_models.data['recommendations']['user_id'] == user_id
        ]
        
        if user_games.empty:
            return {'error': f'User {user_id} not found in the dataset'}
        
        # Get user's positively recommended games
        positive_games = user_games[user_games['is_recommended']]['app_id'].tolist()
        
        # Get a game the user has interacted with to use as a seed for recommendations
        seed_game = user_games['app_id'].iloc[0]
        
        # Generate recommendations based on the specified method
        if method == 'collaborative':
            recs = self.rec_models.collaborative_filtering_recommendations(seed_game, top_n=top_n)
        elif method == 'content':
            recs = self.rec_models.content_based_recommendations(seed_game, top_n=top_n)
        elif method == 'hybrid':
            recs = self.rec_models.hybrid_recommendations(seed_game, top_n=top_n)
        else:
            return {'error': f'Unknown recommendation method: {method}'}
        
        # Calculate hit rate (how many recommended games are in the user's positively rated games)
        recommended_games = recs['app_id'].tolist()
        hits = [game for game in recommended_games if game in positive_games]
        hit_rate = len(hits) / min(len(positive_games), top_n) if positive_games else 0
        
        return {
            'user_id': user_id,
            'method': method,
            'seed_game': {
                'app_id': seed_game,
                'title': self.rec_models.get_game_title_by_id(seed_game)
            },
            'hit_rate': hit_rate,
            'hits': hits,
            'recommendations': recs,
            'positive_games': positive_games
        }
    
    def evaluate_all_methods(self, user_sample=100, top_n=10):
        """
        Evaluate all recommendation methods on a sample of users
        
        Parameters:
        -----------
        user_sample : int
            Number of users to sample for evaluation
        top_n : int
            Number of recommendations to generate for each user
        """
        # Sample users
        all_users = self.rec_models.data['recommendations']['user_id'].unique()
        if len(all_users) > user_sample:
            np.random.seed(42)  # For reproducibility
            sampled_users = np.random.choice(all_users, user_sample, replace=False)
        else:
            sampled_users = all_users
        
        # Evaluate each method
        results = {
            'collaborative': [],
            'content': [],
            'hybrid': []
        }
        
        for user_id in sampled_users:
            for method in results.keys():
                try:
                    eval_result = self.evaluate_recommendations_for_user(user_id, method=method, top_n=top_n)
                    if 'error' not in eval_result:
                        results[method].append(eval_result['hit_rate'])
                except Exception as e:
                    print(f"Error evaluating {method} for user {user_id}: {str(e)}")
        
        # Calculate average hit rates
        avg_results = {
            method: np.mean(hit_rates) if hit_rates else 0 
            for method, hit_rates in results.items()
        }
        
        return {
            'average_hit_rates': avg_results,
            'detailed_results': results
        }
    
    def plot_evaluation_results(self, eval_results, save_path=None):
        """
        Plot evaluation results for different recommendation methods
        
        Parameters:
        -----------
        eval_results : dict
            Results from evaluate_all_methods
        save_path : str
            Path to save the plot image (if None, plot is displayed)
        """
        # Extract average hit rates
        methods = list(eval_results['average_hit_rates'].keys())
        hit_rates = list(eval_results['average_hit_rates'].values())
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=methods, y=hit_rates)
        plt.title('Average Hit Rate by Recommendation Method', fontsize=16)
        plt.xlabel('Method', fontsize=14)
        plt.ylabel('Average Hit Rate', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, rate in enumerate(hit_rates):
            plt.text(i, rate + 0.01, f'{rate:.3f}', ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def evaluate_top_n_coverage(self, methods=['collaborative', 'content', 'hybrid'], top_n_values=[5, 10, 20]):
        """
        Evaluate how top-N recommendation performance varies with different N values
        
        Parameters:
        -----------
        methods : list
            List of recommendation methods to evaluate
        top_n_values : list
            List of top-N values to test
        """
        # Sample users
        all_users = self.rec_models.data['recommendations']['user_id'].unique()
        user_sample = min(50, len(all_users))  # Limit to 50 users for efficiency
        np.random.seed(42)  # For reproducibility
        sampled_users = np.random.choice(all_users, user_sample, replace=False)
        
        # Evaluate each method with different top-N values
        results = {}
        for method in methods:
            method_results = []
            for n in top_n_values:
                hit_rates = []
                for user_id in sampled_users:
                    try:
                        eval_result = self.evaluate_recommendations_for_user(user_id, method=method, top_n=n)
                        if 'error' not in eval_result:
                            hit_rates.append(eval_result['hit_rate'])
                    except Exception as e:
                        print(f"Error evaluating {method} with top-{n} for user {user_id}: {str(e)}")
                
                avg_hit_rate = np.mean(hit_rates) if hit_rates else 0
                method_results.append(avg_hit_rate)
            
            results[method] = method_results
        
        return {
            'top_n_values': top_n_values,
            'results': results
        }
    
    def plot_top_n_results(self, top_n_results, save_path=None):
        """
        Plot top-N evaluation results
        
        Parameters:
        -----------
        top_n_results : dict
            Results from evaluate_top_n_coverage
        save_path : str
            Path to save the plot image (if None, plot is displayed)
        """
        plt.figure(figsize=(12, 8))
        
        top_n_values = top_n_results['top_n_values']
        for method, hit_rates in top_n_results['results'].items():
            plt.plot(top_n_values, hit_rates, marker='o', linewidth=2, label=method)
        
        plt.title('Average Hit Rate vs. Number of Recommendations (Top-N)', fontsize=16)
        plt.xlabel('Top-N', fontsize=14)
        plt.ylabel('Average Hit Rate', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluation()
    
    # Evaluate popular recommendations
    popular_eval = evaluator.evaluate_popular_recommendations()
    print("Popular Recommendations Evaluation:")
    print(f"Average Positive Ratio: {popular_eval['avg_positive_ratio']:.2f}")
    print(f"Average User Reviews: {popular_eval['avg_user_reviews']:.2f}")
    
    # Evaluate for a sample user
    user_sample = evaluator.rec_models.data['recommendations']['user_id'].unique()[0]
    user_eval = evaluator.evaluate_recommendations_for_user(user_sample, method='hybrid')
    print(f"\nUser {user_sample} Evaluation (Hybrid Method):")
    print(f"Hit Rate: {user_eval['hit_rate']:.2f}")
    
    # Evaluate all methods
    all_methods_eval = evaluator.evaluate_all_methods(user_sample=20)
    print("\nAll Methods Evaluation:")
    for method, hit_rate in all_methods_eval['average_hit_rates'].items():
        print(f"{method.capitalize()}: {hit_rate:.3f}")