import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from models.recommendation_models import RecommendationModels
from models.model_evaluation import ModelEvaluation

st.set_page_config(page_title="Steam Game Recommendation System", page_icon="🎮", layout="wide")


@st.cache_data(ttl=3600)
def load_data():
    loader = DataLoader(sample_size=10000, memory_efficient=True)
    with st.spinner("Loading dataset... This might take a moment."):
        data = loader.load_all_data()
    return data, loader


@st.cache_resource(hash_funcs={DataLoader: lambda _: None})
def load_recommendation_models(_data_loader):
    models = RecommendationModels(data_loader=_data_loader)

    if not models.load_models():
        with st.spinner("Training recommendation models... This might take a moment."):
            models.create_matrices()
            models.compute_game_similarity()
            models.build_knn_model()
            models.compute_popular_games()
            models.save_models()

    return models


def run_app():
    st.title("Steam Game Recommendation System")

    st.sidebar.image("https://cdn.cloudflare.steamstatic.com/store/home/store_home_share.jpg", width=300)
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Select a Page", ["Home", "Game Explorer", "Recommendation Engine", "Model Evaluation"])

    data, loader = load_data()
    recommendation_models = load_recommendation_models(loader)

    if page == "Home":
        st.header("Dataset Overview")

        cols = st.columns(3)
        metrics = [("Total Games", f"{len(data['games']):,}"), ("Total Users", f"{len(data['users']):,}"), ("Total Recommendations", f"{len(data['recommendations']):,}")]
        for i, (label, value) in enumerate(metrics):
            with cols[i]:
                st.metric(label, value)

        st.subheader("Game Sample")
        st.dataframe(data["games"].head(10), hide_index=True)

        st.subheader("User Recommendations Sample")
        st.dataframe(data["recommendations"].head(10), hide_index=True)

        st.header("Data Distributions")

        cols = st.columns(2)
        plots = [("positive_ratio", "Distribution of Positive Ratio", "Positive Ratio (%)"), ("price_final", "Distribution of Game Prices", "Price ($)")]

        for i, (field, title, xlabel) in enumerate(plots):
            with cols[i]:
                data_to_plot = data["games"][field].clip(upper=60) if field == "price_final" else data["games"][field].dropna()

                st.subheader(title)

                hist, bin_edges = np.histogram(data_to_plot, bins=20)

                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                chart_data = pd.DataFrame({"Bin": bin_centers, "Count": hist})

                st.bar_chart(chart_data.set_index("Bin"))

        st.subheader("Platform Support")

        platform_data = {"Platform": ["Windows", "Mac", "Linux"], "Support (%)": [data["games"]["win"].mean() * 100, data["games"]["mac"].mean() * 100, data["games"]["linux"].mean() * 100]}
        platform_df = pd.DataFrame(platform_data)
        st.bar_chart(platform_df, x="Platform", y="Support (%)")

        cols = st.columns(3)
        platforms = [("Windows Support", data["games"]["win"].mean() * 100), ("Mac Support", data["games"]["mac"].mean() * 100), ("Linux Support", data["games"]["linux"].mean() * 100)]
        for i, (label, value) in enumerate(platforms):
            with cols[i]:
                st.metric(label, f"{value:.1f}%")

    elif page == "Game Explorer":
        st.header("Game Explorer")

        search_option = st.radio("Search by", ["Game Title", "Popular Games", "Recent Releases", "Price Range"])

        if search_option == "Game Title":
            search_query = st.text_input("Enter game title to search")

            if search_query:
                filtered_games = data["games"][data["games"]["title"].str.contains(search_query, case=False, na=False)]

                if not filtered_games.empty:
                    st.dataframe(filtered_games[["app_id", "title", "rating", "positive_ratio", "user_reviews", "price_final"]], hide_index=True)

                    selected_game_id = st.selectbox(
                        "Select a game to see more details",
                        filtered_games["app_id"].tolist(),
                        format_func=lambda x: filtered_games[filtered_games["app_id"] == x]["title"].iloc[0],
                    )

                    game_details = filtered_games[filtered_games["app_id"] == selected_game_id].iloc[0]
                    st.subheader(game_details["title"])

                    cols = st.columns(2)
                    metrics = [
                        [
                            ("Rating", game_details["rating"]),
                            ("Positive Ratio", f"{game_details['positive_ratio']}%"),
                            ("User Reviews", f"{game_details['user_reviews']:,}"),
                        ],
                        [
                            ("Price", f"${game_details['price_final']:.2f}"),
                            ("Platforms", ", ".join([p for p, v in zip(["Windows", "Mac", "Linux"], [game_details["win"], game_details["mac"], game_details["linux"]]) if v])),
                            ("Steam Deck Compatible", "Yes" if game_details["steam_deck"] else "No"),
                        ],
                    ]

                    for i, col_metrics in enumerate(metrics):
                        with cols[i]:
                            for label, value in col_metrics:
                                st.metric(label, value)

                    st.subheader("Similar Games")
                    similar_games = recommendation_models.content_based_recommendations(selected_game_id, top_n=5)
                    st.dataframe(similar_games[["title", "rating", "positive_ratio", "price_final"]], hide_index=True)
                else:
                    st.warning(f"No games found matching '{search_query}'")

        elif search_option == "Popular Games":
            if recommendation_models.popular_games is None:
                recommendation_models.compute_popular_games()

            min_reviews = st.slider("Minimum Reviews", 10, 1000, 100, step=10)
            top_n = st.slider("Number of Games", 5, 50, 20, step=5)

            popular_games = recommendation_models.compute_popular_games(min_reviews=min_reviews)
            st.dataframe(popular_games.head(top_n)[["app_id", "title", "rating", "positive_ratio", "user_reviews", "price_final"]], hide_index=True)

        elif search_option == "Recent Releases":
            years = sorted(data["games"]["date_release"].dt.year.dropna().unique(), reverse=True)
            selected_year = st.selectbox("Select Year", years)

            recent_games = data["games"][data["games"]["date_release"].dt.year == selected_year].sort_values("date_release", ascending=False)

            st.dataframe(recent_games[["app_id", "title", "date_release", "rating", "positive_ratio", "price_final"]], hide_index=True)

        elif search_option == "Price Range":
            min_price, max_price = st.slider("Price Range ($)", 0.0, 100.0, (0.0, 50.0), step=5.0)

            price_filtered_games = data["games"][(data["games"]["price_final"] >= min_price) & (data["games"]["price_final"] <= max_price)].sort_values("positive_ratio", ascending=False)

            st.dataframe(price_filtered_games[["app_id", "title", "rating", "positive_ratio", "user_reviews", "price_final"]], hide_index=True)

    elif page == "Recommendation Engine":
        st.header("Game Recommendation Engine")

        method = st.radio("Select Recommendation Method", ["Collaborative Filtering", "Content-Based", "Hybrid", "Popular Games"])

        if method != "Popular Games":
            search_query = st.text_input("Search for a game to get recommendations")

            if search_query:
                matched_games = data["games"][data["games"]["title"].str.contains(search_query, case=False, na=False)]

                if not matched_games.empty:
                    selected_game_id = st.selectbox("Select a game", matched_games["app_id"].tolist(), format_func=lambda x: matched_games[matched_games["app_id"] == x]["title"].iloc[0])

                    top_n = st.slider("Number of Recommendations", 5, 20, 10, step=5)

                    if st.button("Get Recommendations"):
                        with st.spinner("Generating recommendations..."):
                            method_map = {
                                "Collaborative Filtering": ("collaborative_filtering_recommendations", "similarity_score"),
                                "Content-Based": ("content_based_recommendations", None),
                                "Hybrid": ("hybrid_recommendations", "hybrid_score"),
                            }

                            method_func, score_col = method_map[method]
                            recommendations = getattr(recommendation_models, method_func)(selected_game_id, top_n=top_n)

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
                        if score_col:
                            recommendations[score_col] = recommendations[score_col].round(3)
                            display_cols = [score_col] + display_cols

                        st.dataframe(recommendations[display_cols], hide_index=True)
                else:
                    st.warning(f"No games found matching '{search_query}'")
        else:
            min_reviews, top_n = st.slider("Minimum Reviews", 100, 5000, 1000, step=100), st.slider("Number of Games", 5, 50, 10, step=5)

            if st.button("Show Popular Games"):
                with st.spinner("Finding popular games..."):
                    popular_games = recommendation_models.compute_popular_games(min_reviews=min_reviews)
                    st.dataframe(popular_games.head(top_n)[["title", "rating", "positive_ratio", "user_reviews", "price_final"]], hide_index=True)

    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        evaluator = ModelEvaluation(recommendation_models=recommendation_models)

        eval_method = st.radio("Select Evaluation Method", ["Compare All Methods", "Evaluate for Specific User", "Top-N Analysis"])

        if eval_method == "Compare All Methods":
            user_sample = st.slider("Number of Users to Sample", 5, 100, 20, step=5)
            top_n = st.slider("Number of Recommendations (Top-N)", 5, 20, 10, step=5)

            if st.button("Run Evaluation"):
                with st.spinner("Evaluating all methods... This may take a while."):
                    eval_results = evaluator.evaluate_all_methods(user_sample=user_sample, top_n=top_n)

                results_df = pd.DataFrame({"Method": list(eval_results["average_hit_rates"].keys()), "Average Hit Rate": list(eval_results["average_hit_rates"].values())})
                st.dataframe(results_df, hide_index=True)

                st.subheader("Method Comparison")
                st.bar_chart(results_df, x="Method", y="Average Hit Rate")

        elif eval_method == "Evaluate for Specific User":
            user_id = st.number_input("Enter User ID", min_value=1, step=1)

            if st.button("Evaluate for User") and user_id > 0:
                with st.spinner("Evaluating recommendations for user..."):
                    if user_id in recommendation_models.data["recommendations"]["user_id"].values:
                        methods = {"collaborative": "Collaborative Filtering", "content": "Content-Based", "hybrid": "Hybrid"}
                        evals = {m: evaluator.evaluate_recommendations_for_user(user_id, method=m) for m in methods.keys()}

                        user_games = recommendation_models.data["recommendations"][recommendation_models.data["recommendations"]["user_id"] == user_id]

                        st.subheader(f"User {user_id}'s Game Interactions")
                        st.dataframe(user_games[["app_id", "is_recommended", "hours"]].merge(recommendation_models.data["games"][["app_id", "title"]], on="app_id")[["title", "is_recommended", "hours"]], hide_index=True)

                        st.subheader("Recommendation Performance")
                        results_df = pd.DataFrame({"Method": list(methods.values()), "Hit Rate": [evals[m]["hit_rate"] for m in methods.keys()]})
                        st.dataframe(results_df, hide_index=True)

                        best_method_key = max(methods.keys(), key=lambda m: evals[m]["hit_rate"])
                        st.subheader(f"Recommendations from Best Method ({methods[best_method_key]})")
                        st.dataframe(evals[best_method_key]["recommendations"][["title", "rating", "positive_ratio", "price_final"]], hide_index=True)
                    else:
                        st.error(f"User ID {user_id} not found in the dataset.")

        elif eval_method == "Top-N Analysis":
            if st.button("Run Top-N Analysis"):
                with st.spinner("Analyzing performance across different Top-N values... This may take a while."):
                    top_n_results = evaluator.evaluate_top_n_coverage(top_n_values=[5, 10, 15, 20, 25])

                st.subheader("Performance vs. Number of Recommendations")
                results = []
                for method, hit_rates in top_n_results["results"].items():
                    for i, n in enumerate(top_n_results["top_n_values"]):
                        results.append({"Method": method, "Top-N": n, "Hit Rate": hit_rates[i]})

                results_df = pd.DataFrame(results)
                st.dataframe(results_df, hide_index=True)

                st.subheader("Hit Rate Trends by Top-N Value")
                pivot_df = results_df.pivot(index="Top-N", columns="Method", values="Hit Rate")
                st.line_chart(pivot_df)

    st.caption("Steam Game Recommendation System | Data: Game Recommendations on Steam (Kaggle)")


if __name__ == "__main__":
    run_app()
