import streamlit as st
import pandas as pd
from utils.data_loader import DataLoader
from models.recommendation_models import RecommendationModels
from models.model_evaluation import ModelEvaluation

st.set_page_config(page_title="Algorithm Comparison for Classification and Clustering", layout="wide")


@st.cache_data(ttl=3600)
def load_data():
    loader = DataLoader(sample_size=10000, memory_efficient=True)
    with st.spinner("Loading dataset... This might take a moment."):
        data = loader.load_all_data()
    return data, loader


@st.cache_resource(hash_funcs={DataLoader: lambda _: None})
def load_models(_data_loader):
    models = RecommendationModels(data_loader=_data_loader)

    if not models.load_models():
        with st.spinner("Training models... This might take a moment."):
            models.create_matrices()
            models.build_knn_model()
            models.build_kmeans_model()
            models.build_naive_bayes_model()
            models.build_random_forest_model()
            models.save_models()

    return models


def run_app():
    st.markdown(
        """
        <style>
        .steam-banner {
            position: relative;
            width: 100%;
            margin-bottom: 20px;
        }
        .steam-banner img {
            width: 100%;
            height: 500px;
            object-fit: cover;
        }
        .steam-banner::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(to bottom, rgba(0,0,0,0) 0%, rgba(0,0,0,0.7) 100%);
        }
        </style>
        <div class="steam-banner">
            <img src="https://cdn.cloudflare.steamstatic.com/store/home/store_home_share.jpg" alt="Steam Banner">
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.title("Algorithm Comparison for Classification and Clustering")

    tab1, tab2, tab3 = st.tabs(["Recommendation Engine", "Model Evaluation", "Algorithm Comparison"])

    data, loader = load_data()
    models = load_models(loader)
    
    with tab1:
        st.header("Recommendation Engine")
        st.write("This section demonstrates model-based recommendations.")

        search_query = st.text_input("Search for a game to get recommendations")

        if search_query:
            matched_games = data["games"][data["games"]["title"].str.contains(search_query, case=False, na=False)]

            if not matched_games.empty:
                selected_game_id = st.selectbox("Select a game", matched_games["app_id"].tolist(), 
                                            format_func=lambda x: matched_games[matched_games["app_id"] == x]["title"].iloc[0])

                top_n = st.slider("Number of Recommendations", 5, 20, 10, step=5)

                if st.button("Get Recommendations"):
                    with st.spinner("Generating recommendations..."):
                        recommendations = models.content_based_recommendations(selected_game_id, top_n=top_n)

                    selected_game = matched_games[matched_games["app_id"] == selected_game_id].iloc[0]
                    st.subheader(f"Selected Game: {selected_game['title']}")

                    cols = st.columns(4)
                    metrics = [
                        ("Rating", selected_game["rating"]),
                        ("Positive Ratio", f"{selected_game['positive_ratio']}%"),
                        ("User Reviews", f"{selected_game['user_reviews']:,}"),
                        ("Price", f"${selected_game['price_final']:.2f}"),
                    ]
                    for i, (label, value) in enumerate(metrics):
                        with cols[i]:
                            st.metric(label, value)

                    st.subheader("Recommended Games")
                    display_cols = ["title", "rating", "positive_ratio", "user_reviews", "price_final"]
                    st.dataframe(recommendations[display_cols], hide_index=True)
            else:
                st.warning(f"No games found matching '{search_query}'")

    with tab2:
        st.header("Model Evaluation")
        evaluator = ModelEvaluation(recommendation_models=models)
        
        st.subheader("Model Performance Metrics")
        
        if st.button("Evaluate Classification Models"):
            with st.spinner("Evaluating models... This might take a while."):
                classification_results = evaluator.evaluate_classification_models()
                
                results_df = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
                for model, metrics in classification_results.items():
                    if model != 'label_mapping':
                        results_df[model.upper()] = [
                            metrics['accuracy'], 
                            metrics['precision'],
                            metrics['recall'],
                            metrics['f1']
                        ]
                
                st.dataframe(results_df.round(3), use_container_width=True)
                
                st.subheader("Performance Metrics Comparison")
                
                chart_data = results_df.transpose()
                st.bar_chart(chart_data, use_container_width=True)
                
                st.subheader("Best Performing Algorithm")
                best_metrics = pd.DataFrame({
                    'Metric': results_df.index,
                    'Best Algorithm': [results_df.columns[results_df.loc[metric].argmax()] for metric in results_df.index],
                    'Value': [results_df.loc[metric].max() for metric in results_df.index]
                })
                
                for i, row in best_metrics.iterrows():
                    st.metric(
                        label=f"Best for {row['Metric']}", 
                        value=row['Best Algorithm'], 
                        delta=f"{row['Value']:.3f}"
                    )

    with tab3:
        st.header("Algorithm Comparison")
        evaluator = ModelEvaluation(recommendation_models=models)
        
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Classification Comparison")
            
            if st.button("Compare Classification Algorithms"):
                with st.spinner("Comparing classification algorithms..."):
                    comparison = evaluator.compare_algorithms()
                    
                    st.dataframe(comparison)
                    st.bar_chart(comparison)
        
        with cols[1]:
            st.subheader("Clustering Evaluation")
            
            n_clusters = st.slider("Number of Clusters", 3, 10, 5, step=1)
            
            if st.button("Evaluate Clustering"):
                with st.spinner("Evaluating KMeans clustering..."):
                    models.build_kmeans_model(n_clusters=n_clusters)
                    clustering_results = evaluator.evaluate_clustering(n_clusters=n_clusters)
                    
                    kmeans_results = clustering_results['kmeans']
                    
                    st.metric("Inertia (Lower is better)", f"{kmeans_results['inertia']:.2f}")
                    st.metric("Silhouette Score (-1 to 1, higher is better)", f"{kmeans_results['silhouette']:.3f}")
                    
                    cluster_data = evaluator.visualize_clusters(return_data=True)
                    
                    scatter_data = pd.DataFrame({
                        'PCA Component 1': cluster_data['pca_components'][:, 0],
                        'PCA Component 2': cluster_data['pca_components'][:, 1],
                        'Cluster': cluster_data['cluster_labels']
                    })
                    
                    st.scatter_chart(
                        scatter_data,
                        x='PCA Component 1',
                        y='PCA Component 2',
                        color='Cluster'
                    )
        
        st.subheader("Algorithm Selection Reasoning")
        
        if st.button("Show Algorithm Comparison Analysis"):
            explanation = evaluator.explain_algorithm_choice()
            st.markdown(explanation)
            
            st.subheader("Strategies to Improve Results")
            
            improvement_strategies = """
            ## Strategies for Improving Model Performance
            
            ### 1. Feature Engineering
            - Create composite features that better capture user preferences
            - Apply dimensionality reduction to focus on most important features
            - Add temporal features to capture gaming trends over time
            
            ### 2. Hyperparameter Tuning
            - Use grid search or randomized search for optimal parameters
            - Optimize K for KNN classifier
            - Tune number of clusters for K-Means
            - Adjust alpha for Naive Bayes
            - Optimize tree depth and number of estimators for Random Forest
            
            ### 3. Data Enhancement
            - Balance class distributions for better classification
            - Filter outliers that may skew model learning
            - Include additional data sources for richer feature set
            
            ### 4. Ensemble Methods
            - Combine multiple models through voting or stacking
            - Create a meta-classifier using predictions from base models
            """
            
            st.markdown(improvement_strategies)


if __name__ == "__main__":
    run_app()
