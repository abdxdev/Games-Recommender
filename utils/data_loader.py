import os
import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, data_path="dataset", sample_size=None, memory_efficient=True):
        self.data_path = data_path
        self.sample_size = sample_size
        self.memory_efficient = memory_efficient
        self.games_df = None
        self.users_df = None
        self.recommendations_df = None
        self.games_metadata = None
        self.train_data = None
        self.test_data = None
        self.user_game_matrix = None
        self.hours_matrix = None

    def _optimize_dtypes(self, df):
        if df is None or not self.memory_efficient:
            return df

        int_cols = df.select_dtypes(include=["int64"]).columns
        for col in int_cols:
            max_val = df[col].max()
            min_val = df[col].min()

            if min_val >= 0:
                if max_val < 255:
                    df[col] = df[col].astype(np.uint8)
                elif max_val < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif max_val < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if min_val > -128 and max_val < 127:
                    df[col] = df[col].astype(np.int8)
                elif min_val > -32768 and max_val < 32767:
                    df[col] = df[col].astype(np.int16)
                elif min_val > -2147483648 and max_val < 2147483647:
                    df[col] = df[col].astype(np.int32)

        float_cols = df.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            df[col] = df[col].astype(np.float32)

        return df

    def load_games(self):
        file_path = os.path.join(self.data_path, "games.csv")

        if self.sample_size and self.memory_efficient:
            self.games_df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and np.random.random() > self.sample_size / 100)
        else:
            self.games_df = pd.read_csv(file_path)

        self.games_df = self._optimize_dtypes(self.games_df)
        self.games_df["date_release"] = pd.to_datetime(self.games_df["date_release"], errors="coerce")

        bool_cols = ["win", "mac", "linux", "steam_deck"]
        for col in bool_cols:
            self.games_df[col] = self.games_df[col].astype(bool)

        return self.games_df

    def load_users(self):
        file_path = os.path.join(self.data_path, "users.csv")

        if self.sample_size and self.memory_efficient:
            self.users_df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and np.random.random() > self.sample_size / 100)
        else:
            self.users_df = pd.read_csv(file_path)

        self.users_df = self._optimize_dtypes(self.users_df)

        return self.users_df

    def load_recommendations(self):
        file_path = os.path.join(self.data_path, "recommendations.csv")

        try:
            dtypes = {"app_id": "int32", "helpful": "int16", "funny": "int16", "review_id": "int32", "user_id": "int32"}

            if self.memory_efficient:
                if self.sample_size:
                    with open(file_path, "r") as f:
                        total_rows = sum(1 for _ in f) - 1

                    skip_rate = max(1, int(total_rows / self.sample_size))

                    chunks = []
                    for chunk in pd.read_csv(file_path, dtype=dtypes, chunksize=50000, skiprows=lambda i: i > 0 and i % skip_rate != 0):
                        chunks.append(chunk)

                    if chunks:
                        self.recommendations_df = pd.concat(chunks, ignore_index=True)
                    else:
                        self.recommendations_df = pd.DataFrame()
                else:
                    chunks = []
                    for chunk in pd.read_csv(file_path, dtype=dtypes, chunksize=100000):
                        chunks.append(chunk)
                    self.recommendations_df = pd.concat(chunks, ignore_index=True)
            else:
                self.recommendations_df = pd.read_csv(file_path)

            self.recommendations_df["date"] = pd.to_datetime(self.recommendations_df["date"], errors="coerce")
            self.recommendations_df["is_recommended"] = self.recommendations_df["is_recommended"].astype(bool)
            self.recommendations_df = self._optimize_dtypes(self.recommendations_df)

            print(f"Loaded recommendations data with shape: {self.recommendations_df.shape}")

            return self.recommendations_df

        except Exception as e:
            print(f"Error loading recommendations file: {str(e)}")
            self.recommendations_df = pd.DataFrame(
                columns=[
                    "app_id",
                    "helpful",
                    "funny",
                    "date",
                    "is_recommended",
                    "hours",
                    "user_id",
                    "review_id",
                ]
            )
            return self.recommendations_df

    def load_games_metadata(self):
        file_path = os.path.join(self.data_path, "games_metadata.json")
        metadata_list = []

        try:
            if self.sample_size and self.memory_efficient:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    total_lines = sum(1 for _ in f)
                sample_rate = max(1, int(total_lines / self.sample_size))
            else:
                sample_rate = 1

            line_count = 0
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_count += 1
                    if self.sample_size and line_count % sample_rate != 0:
                        continue

                    try:
                        metadata_list.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON line: {line[:50]}...")
                        continue
        except UnicodeDecodeError:
            line_count = 0
            with open(file_path, "r", encoding="latin-1") as f:
                for line in f:
                    line_count += 1
                    if self.sample_size and line_count % sample_rate != 0:
                        continue

                    try:
                        metadata_list.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON line: {line[:50]}...")
                        continue

        self.games_metadata = pd.DataFrame(metadata_list)
        self.games_metadata = self._optimize_dtypes(self.games_metadata)

        return self.games_metadata

    def merge_game_data(self):
        if self.games_df is None:
            self.load_games()

        if self.games_metadata is None:
            self.load_games_metadata()

        merged_games = pd.merge(self.games_df, self.games_metadata, on="app_id", how="left")

        return merged_games

    def prepare_user_game_matrix(self, max_users=10000, max_games=5000):
        if self.recommendations_df is None:
            self.load_recommendations()

        if self.users_df is None:
            self.load_users()

        if self.recommendations_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.memory_efficient:
            top_users = self.recommendations_df["user_id"].value_counts().head(max_users).index
            top_games = self.recommendations_df["app_id"].value_counts().head(max_games).index

            filtered_recs = self.recommendations_df[self.recommendations_df["user_id"].isin(top_users) & self.recommendations_df["app_id"].isin(top_games)]
        else:
            filtered_recs = self.recommendations_df

        user_game_matrix = filtered_recs.pivot_table(index="user_id", columns="app_id", values="is_recommended", aggfunc="first").fillna(False)

        hours_matrix = filtered_recs.pivot_table(index="user_id", columns="app_id", values="hours", aggfunc="sum").fillna(0)

        print(f"Created user-game matrix with shape: {user_game_matrix.shape}")
        print(f"Created hours matrix with shape: {hours_matrix.shape}")

        self.user_game_matrix = user_game_matrix
        self.hours_matrix = hours_matrix

        return user_game_matrix, hours_matrix

    def create_train_test_split(self, test_size=0.2, random_state=42):
        if self.recommendations_df is None:
            self.load_recommendations()

        if self.recommendations_df.empty:
            self.train_data = pd.DataFrame(columns=self.recommendations_df.columns)
            self.test_data = pd.DataFrame(columns=self.recommendations_df.columns)
            return self.train_data, self.test_data

        user_ids = self.recommendations_df["user_id"].unique()
        train_users, test_users = train_test_split(user_ids, test_size=test_size, random_state=random_state)

        self.train_data = self.recommendations_df[self.recommendations_df["user_id"].isin(train_users)]
        self.test_data = self.recommendations_df[self.recommendations_df["user_id"].isin(test_users)]

        return self.train_data, self.test_data

    def load_all_data(self):
        self.load_games()
        self.load_users()
        self.load_recommendations()
        self.load_games_metadata()
        self.create_train_test_split()

        return {"games": self.games_df, "users": self.users_df, "recommendations": self.recommendations_df, "games_metadata": self.games_metadata, "train_data": self.train_data, "test_data": self.test_data}

    def get_game_features(self):
        if self.games_df is None or self.games_metadata is None:
            merged_games = self.merge_game_data()
        else:
            merged_games = self.merge_game_data()

        features = merged_games.copy()
        features["description"] = features.get("description", pd.Series("")).fillna("")

        features["platforms"] = features.apply(lambda x: sum([x["win"], x["mac"], x["linux"]]), axis=1)

        current_date = datetime.datetime.now()
        features["age_days"] = (current_date - features["date_release"]).dt.days

        features["has_discount"] = features["discount"] > 0
        features["price_category"] = pd.cut(features["price_final"], bins=[0, 5, 10, 20, 50, 100, float("inf")], labels=["Free/Very Low", "Low", "Medium", "High", "Very High", "Premium"])

        if "tags" in features.columns:
            features["tag_count"] = features["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)

        return features


def load_data(sample_size=None, memory_efficient=True):
    loader = DataLoader(sample_size=sample_size, memory_efficient=memory_efficient)
    return loader.load_all_data()
