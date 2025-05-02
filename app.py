import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import numpy as np
from utils.data_loader import DataLoader
from models.recommendation_models import RecommendationModels
from models.model_evaluation import ModelEvaluation

# Set page configuration
st.set_page_config(
    page_title="Steam Game Recommendation System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5e35b1;
        margin-bottom: 1rem;
    }
    .stat-container {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        padding: 1rem;
        border-radius: 5px;
        background-color: #e8f0fe;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1e88e5;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading to improve performance
@st.cache_data(ttl=3600)
def load_data():
    """
    Load and cache the dataset
    """
    # Use memory-efficient loading with a reasonable sample size for recommendations
    loader = DataLoader(sample_size=10000, memory_efficient=True)
    with st.spinner("Loading dataset... This might take a moment."):
        data = loader.load_all_data()
    return data, loader

@st.cache_resource(hash_funcs={DataLoader: lambda _: None})
def load_recommendation_models(_data_loader):
    """
    Load and cache recommendation models
    
    Parameters:
    -----------
    _data_loader : DataLoader
        Instance of DataLoader class (with underscore to prevent hashing)
    """
    models = RecommendationModels(data_loader=_data_loader)
    
    # Try to load pre-trained models first
    if not models.load_models():
        with st.spinner("Training recommendation models... This might take a moment."):
            # Prepare data for collaborative filtering
            models.create_matrices()
            
            # Build similarity matrices
            models.compute_game_similarity()
            
            # Build KNN model
            models.build_knn_model()
            
            # Compute popular games
            models.compute_popular_games()
            
            # Save models for future use
            models.save_models()
    
    return models

def run_app():
    """
    Main function to run the Streamlit app
    """
    st.markdown('<h1 class="main-header">Steam Game Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://cdn.cloudflare.steamstatic.com/store/home/store_home_share.jpg", width=300)
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Select a Page", 
        ["Home", "Game Explorer", "Recommendation Engine", "Model Evaluation"]
    )
    
    # Load data and models
    data, loader = load_data()
    recommendation_models = load_recommendation_models(loader)
    
    # Display dataset overview on Home page
    if page == "Home":
        st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="stat-container">', unsafe_allow_html=True)
            st.metric("Total Games", f"{len(data['games']):,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="stat-container">', unsafe_allow_html=True)
            st.metric("Total Users", f"{len(data['users']):,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="stat-container">', unsafe_allow_html=True)
            st.metric("Total Recommendations", f"{len(data['recommendations']):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Dataset samples
        st.markdown('<h3>Game Sample</h3>', unsafe_allow_html=True)
        st.dataframe(data['games'].head(10), hide_index=True)
        
        st.markdown('<h3>User Recommendations Sample</h3>', unsafe_allow_html=True)
        st.dataframe(data['recommendations'].head(10), hide_index=True)
        
        # Distribution plots
        st.markdown('<h2 class="sub-header">Data Distributions</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['games']['positive_ratio'].dropna(), bins=20, kde=True, ax=ax)
            ax.set_title('Distribution of Positive Ratio', fontsize=14)
            ax.set_xlabel('Positive Ratio (%)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['games']['price_final'].clip(upper=60), bins=20, kde=True, ax=ax)
            ax.set_title('Distribution of Game Prices', fontsize=14)
            ax.set_xlabel('Price ($)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            st.pyplot(fig)
            
        # Platform statistics
        st.markdown('<h3>Platform Support</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            win_percent = data['games']['win'].mean() * 100
            st.metric("Windows Support", f"{win_percent:.1f}%")
            
        with col2:
            mac_percent = data['games']['mac'].mean() * 100
            st.metric("Mac Support", f"{mac_percent:.1f}%")
            
        with col3:
            linux_percent = data['games']['linux'].mean() * 100
            st.metric("Linux Support", f"{linux_percent:.1f}%")
        
    # Game Explorer page
    elif page == "Game Explorer":
        st.markdown('<h2 class="sub-header">Game Explorer</h2>', unsafe_allow_html=True)
        
        # Search options
        search_option = st.radio(
            "Search by", 
            ["Game Title", "Popular Games", "Recent Releases", "Price Range"]
        )
        
        if search_option == "Game Title":
            # Search by title
            search_query = st.text_input("Enter game title to search")
            
            if search_query:
                filtered_games = data['games'][
                    data['games']['title'].str.contains(search_query, case=False, na=False)
                ]
                
                if not filtered_games.empty:
                    st.dataframe(
                        filtered_games[['app_id', 'title', 'rating', 'positive_ratio', 'user_reviews', 'price_final']], 
                        hide_index=True
                    )
                    
                    # Select a game for more details
                    selected_game_id = st.selectbox(
                        "Select a game to see more details",
                        filtered_games['app_id'].tolist(),
                        format_func=lambda x: filtered_games[filtered_games['app_id'] == x]['title'].iloc[0]
                    )
                    
                    # Show game details
                    game_details = filtered_games[filtered_games['app_id'] == selected_game_id].iloc[0]
                    
                    st.markdown(f"<h3>{game_details['title']}</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
                        st.metric("Rating", game_details['rating'])
                        st.metric("Positive Ratio", f"{game_details['positive_ratio']}%")
                        st.metric("User Reviews", f"{game_details['user_reviews']:,}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
                        st.metric("Price", f"${game_details['price_final']:.2f}")
                        
                        platforms = []
                        if game_details['win']:
                            platforms.append("Windows")
                        if game_details['mac']:
                            platforms.append("Mac")
                        if game_details['linux']:
                            platforms.append("Linux")
                            
                        st.metric("Platforms", ", ".join(platforms))
                        st.metric("Steam Deck Compatible", "Yes" if game_details['steam_deck'] else "No")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    # Show similar games
                    st.markdown("<h4>Similar Games</h4>", unsafe_allow_html=True)
                    similar_games = recommendation_models.content_based_recommendations(selected_game_id, top_n=5)
                    st.dataframe(similar_games[['title', 'rating', 'positive_ratio', 'price_final']], hide_index=True)
                    
                else:
                    st.warning(f"No games found matching '{search_query}'")
                    
        elif search_option == "Popular Games":
            # Show popular games
            if recommendation_models.popular_games is None:
                recommendation_models.compute_popular_games()
                
            min_reviews = st.slider("Minimum Reviews", 10, 1000, 100, step=10)
            top_n = st.slider("Number of Games", 5, 50, 20, step=5)
            
            popular_games = recommendation_models.compute_popular_games(min_reviews=min_reviews)
            st.dataframe(
                popular_games.head(top_n)[['app_id', 'title', 'rating', 'positive_ratio', 'user_reviews', 'price_final']], 
                hide_index=True
            )
            
        elif search_option == "Recent Releases":
            # Show recent releases
            years = sorted(data['games']['date_release'].dt.year.dropna().unique(), reverse=True)
            selected_year = st.selectbox("Select Year", years)
            
            recent_games = data['games'][
                data['games']['date_release'].dt.year == selected_year
            ].sort_values('date_release', ascending=False)
            
            st.dataframe(
                recent_games[['app_id', 'title', 'date_release', 'rating', 'positive_ratio', 'price_final']], 
                hide_index=True
            )
            
        elif search_option == "Price Range":
            # Filter by price range
            min_price, max_price = st.slider(
                "Price Range ($)", 
                0.0, 100.0, (0.0, 50.0), 
                step=5.0
            )
            
            # Filter games within the selected price range
            price_filtered_games = data['games'][
                (data['games']['price_final'] >= min_price) & 
                (data['games']['price_final'] <= max_price)
            ].sort_values('positive_ratio', ascending=False)
            
            st.dataframe(
                price_filtered_games[['app_id', 'title', 'rating', 'positive_ratio', 'user_reviews', 'price_final']], 
                hide_index=True
            )
            
    # Recommendation Engine page
    elif page == "Recommendation Engine":
        st.markdown('<h2 class="sub-header">Game Recommendation Engine</h2>', unsafe_allow_html=True)
        
        # Recommendation method selection
        method = st.radio(
            "Select Recommendation Method",
            ["Collaborative Filtering", "Content-Based", "Hybrid", "Popular Games"]
        )
        
        if method != "Popular Games":
            # Search for a seed game
            search_query = st.text_input("Search for a game to get recommendations")
            
            if search_query:
                # Find matching games
                matched_games = data['games'][
                    data['games']['title'].str.contains(search_query, case=False, na=False)
                ]
                
                if not matched_games.empty:
                    # Select a game
                    selected_game_id = st.selectbox(
                        "Select a game",
                        matched_games['app_id'].tolist(),
                        format_func=lambda x: matched_games[matched_games['app_id'] == x]['title'].iloc[0]
                    )
                    
                    # Number of recommendations
                    top_n = st.slider("Number of Recommendations", 5, 20, 10, step=5)
                    
                    # Get recommendations based on the selected method
                    if st.button("Get Recommendations"):
                        with st.spinner("Generating recommendations..."):
                            if method == "Collaborative Filtering":
                                recommendations = recommendation_models.collaborative_filtering_recommendations(
                                    selected_game_id, top_n=top_n
                                )
                                score_col = 'similarity_score'
                                
                            elif method == "Content-Based":
                                recommendations = recommendation_models.content_based_recommendations(
                                    selected_game_id, top_n=top_n
                                )
                                score_col = None
                                
                            else:  # Hybrid
                                recommendations = recommendation_models.hybrid_recommendations(
                                    selected_game_id, top_n=top_n
                                )
                                score_col = 'hybrid_score'
                        
                        # Display selected game details
                        selected_game = matched_games[matched_games['app_id'] == selected_game_id].iloc[0]
                        st.markdown(f"<h3>Selected Game: {selected_game['title']}</h3>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rating", selected_game['rating'])
                            st.metric("Positive Ratio", f"{selected_game['positive_ratio']}%")
                        with col2:
                            st.metric("User Reviews", f"{selected_game['user_reviews']:,}")
                            st.metric("Price", f"${selected_game['price_final']:.2f}")
                        
                        # Display recommendations
                        st.markdown("<h3>Recommended Games</h3>", unsafe_allow_html=True)
                        
                        # Create columns to display based on method
                        display_cols = ['title', 'rating', 'positive_ratio', 'user_reviews', 'price_final']
                        if score_col:
                            recommendations[score_col] = recommendations[score_col].round(3)
                            display_cols = [score_col] + display_cols
                            
                        st.dataframe(recommendations[display_cols], hide_index=True)
                        
                        # Visualize recommendations
                        if len(recommendations) >= 5:
                            st.markdown("<h3>Recommendation Visualization</h3>", unsafe_allow_html=True)
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Create visualization based on rating and reviews
                            scatter = ax.scatter(
                                recommendations['positive_ratio'],
                                recommendations['user_reviews'].clip(upper=10000),
                                s=recommendations['price_final'] * 20 + 50,
                                alpha=0.7,
                                c=range(len(recommendations)),
                                cmap='viridis'
                            )
                            
                            # Add labels
                            for i, row in recommendations.iterrows():
                                ax.annotate(
                                    row['title'][:20] + ('...' if len(row['title']) > 20 else ''),
                                    (row['positive_ratio'], min(row['user_reviews'], 10000)),
                                    fontsize=9,
                                    ha='center',
                                    va='center',
                                    xytext=(0, 10),
                                    textcoords='offset points'
                                )
                            
                            ax.set_xlabel('Positive Ratio (%)', fontsize=12)
                            ax.set_ylabel('User Reviews (capped at 10k)', fontsize=12)
                            ax.set_title('Recommended Games by Rating and Popularity', fontsize=14)
                            ax.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                            
                else:
                    st.warning(f"No games found matching '{search_query}'")
        else:
            # Popular games recommendations
            min_reviews = st.slider("Minimum Reviews", 100, 5000, 1000, step=100)
            top_n = st.slider("Number of Games", 5, 50, 10, step=5)
            
            if st.button("Show Popular Games"):
                with st.spinner("Finding popular games..."):
                    popular_games = recommendation_models.compute_popular_games(min_reviews=min_reviews)
                    top_popular = popular_games.head(top_n)[['app_id', 'title', 'rating', 'positive_ratio', 'user_reviews', 'price_final']]
                
                st.markdown("<h3>Most Popular Games</h3>", unsafe_allow_html=True)
                st.dataframe(top_popular, hide_index=True)
                
                # Visualize popular games
                st.markdown("<h3>Popular Games Visualization</h3>", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(
                    top_popular['title'].str[:20] + '...',
                    top_popular['positive_ratio'],
                    color=plt.cm.viridis(np.linspace(0, 1, len(top_popular)))
                )
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 1,
                        f'{height:.0f}%',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )
                
                ax.set_xlabel('Game', fontsize=12)
                ax.set_ylabel('Positive Ratio (%)', fontsize=12)
                ax.set_title('Top Popular Games by Positive Ratio', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
    
    # Model Evaluation page
    elif page == "Model Evaluation":
        st.markdown('<h2 class="sub-header">Model Evaluation</h2>', unsafe_allow_html=True)
        
        # Initialize model evaluator
        evaluator = ModelEvaluation(recommendation_models=recommendation_models)
        
        # Select evaluation method
        eval_method = st.radio(
            "Select Evaluation Method",
            ["Compare All Methods", "Evaluate for Specific User", "Top-N Analysis"]
        )
        
        if eval_method == "Compare All Methods":
            # Evaluate all recommendation methods
            user_sample = st.slider("Number of Users to Sample", 5, 100, 20, step=5)
            top_n = st.slider("Number of Recommendations (Top-N)", 5, 20, 10, step=5)
            
            if st.button("Run Evaluation"):
                with st.spinner("Evaluating all methods... This may take a while."):
                    eval_results = evaluator.evaluate_all_methods(user_sample=user_sample, top_n=top_n)
                
                st.markdown("<h3>Average Hit Rates</h3>", unsafe_allow_html=True)
                
                # Display results in a table
                results_df = pd.DataFrame({
                    'Method': list(eval_results['average_hit_rates'].keys()),
                    'Average Hit Rate': list(eval_results['average_hit_rates'].values())
                })
                
                st.dataframe(results_df, hide_index=True)
                
                # Plot results
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(
                    results_df['Method'],
                    results_df['Average Hit Rate'],
                    color=plt.cm.viridis(np.linspace(0, 1, len(results_df)))
                )
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.01,
                        f'{height:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=12
                    )
                
                ax.set_xlabel('Method', fontsize=14)
                ax.set_ylabel('Average Hit Rate', fontsize=14)
                ax.set_title(f'Average Hit Rate by Method (Users: {user_sample}, Top-{top_n})', fontsize=16)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
                
        elif eval_method == "Evaluate for Specific User":
            # Evaluate recommendations for a specific user
            user_id = st.number_input("Enter User ID", min_value=1, step=1)
            
            if st.button("Evaluate for User"):
                if user_id > 0:
                    with st.spinner("Evaluating recommendations for user..."):
                        # Check if user exists
                        user_exists = user_id in recommendation_models.data['recommendations']['user_id'].values
                        
                        if user_exists:
                            # Evaluate all methods for user
                            cf_eval = evaluator.evaluate_recommendations_for_user(user_id, method='collaborative')
                            cb_eval = evaluator.evaluate_recommendations_for_user(user_id, method='content')
                            hybrid_eval = evaluator.evaluate_recommendations_for_user(user_id, method='hybrid')
                            
                            # Display user's game interactions
                            user_games = recommendation_models.data['recommendations'][
                                recommendation_models.data['recommendations']['user_id'] == user_id
                            ]
                            
                            st.markdown(f"<h3>User {user_id}'s Game Interactions</h3>", unsafe_allow_html=True)
                            st.dataframe(
                                user_games[['app_id', 'is_recommended', 'hours']].merge(
                                    recommendation_models.data['games'][['app_id', 'title']],
                                    on='app_id'
                                )[['title', 'is_recommended', 'hours']],
                                hide_index=True
                            )
                            
                            # Display evaluation results
                            st.markdown("<h3>Recommendation Performance</h3>", unsafe_allow_html=True)
                            
                            results_df = pd.DataFrame({
                                'Method': ['Collaborative Filtering', 'Content-Based', 'Hybrid'],
                                'Hit Rate': [cf_eval['hit_rate'], cb_eval['hit_rate'], hybrid_eval['hit_rate']]
                            })
                            
                            st.dataframe(results_df, hide_index=True)
                            
                            # Plot results
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(
                                results_df['Method'],
                                results_df['Hit Rate'],
                                color=plt.cm.viridis(np.linspace(0, 1, len(results_df)))
                            )
                            
                            # Add values on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width()/2.,
                                    height + 0.01,
                                    f'{height:.3f}',
                                    ha='center',
                                    va='bottom',
                                    fontsize=12
                                )
                            
                            ax.set_xlabel('Method', fontsize=14)
                            ax.set_ylabel('Hit Rate', fontsize=14)
                            ax.set_title(f'Hit Rate by Method for User {user_id}', fontsize=16)
                            ax.grid(axis='y', linestyle='--', alpha=0.3)
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Display recommended games for best method
                            best_method = results_df.loc[results_df['Hit Rate'].idxmax(), 'Method']
                            best_eval = None
                            
                            if best_method == 'Collaborative Filtering':
                                best_eval = cf_eval
                            elif best_method == 'Content-Based':
                                best_eval = cb_eval
                            else:
                                best_eval = hybrid_eval
                                
                            st.markdown(f"<h3>Recommendations from Best Method ({best_method})</h3>", unsafe_allow_html=True)
                            st.dataframe(best_eval['recommendations'][['title', 'rating', 'positive_ratio', 'price_final']], hide_index=True)
                            
                        else:
                            st.error(f"User ID {user_id} not found in the dataset.")
                
        elif eval_method == "Top-N Analysis":
            # Analyze how performance varies with different top-N values
            if st.button("Run Top-N Analysis"):
                with st.spinner("Analyzing performance across different Top-N values... This may take a while."):
                    top_n_results = evaluator.evaluate_top_n_coverage(top_n_values=[5, 10, 15, 20, 25])
                
                st.markdown("<h3>Performance vs. Number of Recommendations</h3>", unsafe_allow_html=True)
                
                # Create dataframe for results
                results = []
                for method, hit_rates in top_n_results['results'].items():
                    for i, n in enumerate(top_n_results['top_n_values']):
                        results.append({
                            'Method': method,
                            'Top-N': n,
                            'Hit Rate': hit_rates[i]
                        })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, hide_index=True)
                
                # Plot results
                fig, ax = plt.subplots(figsize=(12, 8))
                
                for method in results_df['Method'].unique():
                    method_data = results_df[results_df['Method'] == method]
                    ax.plot(
                        method_data['Top-N'],
                        method_data['Hit Rate'],
                        marker='o',
                        linewidth=2,
                        label=method
                    )
                
                ax.set_xlabel('Number of Recommendations (Top-N)', fontsize=14)
                ax.set_ylabel('Average Hit Rate', fontsize=14)
                ax.set_title('Hit Rate vs. Number of Recommendations', fontsize=16)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(fontsize=12)
                plt.tight_layout()
                
                st.pyplot(fig)
    
    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>Steam Game Recommendation System | Powered by Python, Pandas, Scikit-learn, and Streamlit</p>
            <p>Data source: <a href="https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam" target="_blank">Game Recommendations on Steam (Kaggle)</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    run_app()