import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import os
import datetime

class DataLoader:
    def __init__(self, data_path="datasets", sample_size=None, memory_efficient=True):
        """
        Initialize the DataLoader with the path to the dataset files
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing the dataset files
        sample_size : int, optional
            Number of rows to sample from large files (None=all data)
        memory_efficient : bool
            Whether to use memory-efficient loading techniques
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.memory_efficient = memory_efficient
        self.games_df = None
        self.users_df = None
        self.recommendations_df = None
        self.games_metadata = None
        self.train_data = None
        self.test_data = None
        
    def _optimize_dtypes(self, df):
        """
        Optimize data types to reduce memory usage
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to optimize
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with optimized data types
        """
        if df is None or not self.memory_efficient:
            return df
            
        # Optimize integer columns
        int_cols = df.select_dtypes(include=['int64']).columns
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
        
        # Optimize float columns
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype(np.float32)
            
        return df
    
    def load_games(self):
        """
        Load the games.csv file into a pandas DataFrame
        """
        file_path = os.path.join(self.data_path, "games.csv")
        
        # Read the CSV file (with sampling if specified)
        if self.sample_size and self.memory_efficient:
            # Randomly sample rows
            self.games_df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and np.random.random() > self.sample_size/100)
        else:
            self.games_df = pd.read_csv(file_path)
        
        # Optimize data types
        self.games_df = self._optimize_dtypes(self.games_df)
        
        # Convert date_release to datetime
        self.games_df['date_release'] = pd.to_datetime(self.games_df['date_release'], errors='coerce')
        
        # Convert boolean columns to proper booleans
        bool_cols = ['win', 'mac', 'linux', 'steam_deck']
        for col in bool_cols:
            self.games_df[col] = self.games_df[col].astype(bool)
            
        return self.games_df
    
    def load_users(self):
        """
        Load the users.csv file into a pandas DataFrame
        """
        file_path = os.path.join(self.data_path, "users.csv")
        
        # Read the CSV file (with sampling if specified)
        if self.sample_size and self.memory_efficient:
            # Randomly sample rows
            self.users_df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and np.random.random() > self.sample_size/100)
        else:
            self.users_df = pd.read_csv(file_path)
            
        # Optimize data types
        self.users_df = self._optimize_dtypes(self.users_df)
        
        return self.users_df
    
    def load_recommendations(self):
        """
        Load the recommendations.csv file into a pandas DataFrame
        """
        file_path = os.path.join(self.data_path, "recommendations.csv")
        
        try:
            # Define optimal dtypes to reduce memory usage
            dtypes = {
                'app_id': 'int32',
                'helpful': 'int16',
                'funny': 'int16',
                'review_id': 'int32',
                'user_id': 'int32'
            }
            
            if self.memory_efficient:
                if self.sample_size:
                    # For very large files, sample a percentage of the rows
                    # First count total rows without loading the data
                    with open(file_path, 'r') as f:
                        total_rows = sum(1 for _ in f) - 1  # Subtract header row
                    
                    # Calculate row skip rate based on sample size
                    skip_rate = max(1, int(total_rows / self.sample_size))
                    
                    # Use chunking to read the file in parts
                    chunks = []
                    for chunk in pd.read_csv(file_path, dtype=dtypes, chunksize=50000, skiprows=lambda i: i > 0 and i % skip_rate != 0):
                        chunks.append(chunk)
                    
                    if chunks:
                        self.recommendations_df = pd.concat(chunks, ignore_index=True)
                    else:
                        self.recommendations_df = pd.DataFrame()
                else:
                    # Read the file in chunks to save memory
                    chunks = []
                    for chunk in pd.read_csv(file_path, dtype=dtypes, chunksize=100000):
                        chunks.append(chunk)
                    self.recommendations_df = pd.concat(chunks, ignore_index=True)
            else:
                # Regular loading (may cause memory issues with large files)
                self.recommendations_df = pd.read_csv(file_path)
        
            # Convert date to datetime
            self.recommendations_df['date'] = pd.to_datetime(self.recommendations_df['date'], errors='coerce')
            
            # Convert is_recommended to boolean
            self.recommendations_df['is_recommended'] = self.recommendations_df['is_recommended'].astype(bool)
            
            # Optimize other data types
            self.recommendations_df = self._optimize_dtypes(self.recommendations_df)
            
            print(f"Loaded recommendations data with shape: {self.recommendations_df.shape}")
            
            return self.recommendations_df
        
        except Exception as e:
            print(f"Error loading recommendations file: {str(e)}")
            # Create a minimal DataFrame to allow the app to function
            self.recommendations_df = pd.DataFrame(columns=['app_id', 'helpful', 'funny', 'date', 'is_recommended', 'hours', 'user_id', 'review_id'])
            return self.recommendations_df
    
    def load_games_metadata(self):
        """
        Load the games_metadata.json file line by line as it's a JSON lines file
        """
        file_path = os.path.join(self.data_path, "games_metadata.json")
        metadata_list = []
        
        try:
            # Count total lines to estimate sampling
            if self.sample_size and self.memory_efficient:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines = sum(1 for _ in f)
                sample_rate = max(1, int(total_lines / self.sample_size))
            else:
                sample_rate = 1
            
            # Read the file with utf-8 encoding and handle lines individually
            line_count = 0
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_count += 1
                    # Sample data if needed
                    if self.sample_size and line_count % sample_rate != 0:
                        continue
                        
                    try:
                        metadata_list.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON line: {line[:50]}...")
                        continue
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding if utf-8 fails
            line_count = 0
            with open(file_path, 'r', encoding='latin-1') as f:
                for line in f:
                    line_count += 1
                    # Sample data if needed
                    if self.sample_size and line_count % sample_rate != 0:
                        continue
                        
                    try:
                        metadata_list.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON line: {line[:50]}...")
                        continue
                        
        self.games_metadata = pd.DataFrame(metadata_list)
        
        # Optimize data types
        self.games_metadata = self._optimize_dtypes(self.games_metadata)
        
        return self.games_metadata
    
    def merge_game_data(self):
        """
        Merge game data with metadata to create a comprehensive dataset
        """
        if self.games_df is None:
            self.load_games()
            
        if self.games_metadata is None:
            self.load_games_metadata()
            
        # Merge games data with metadata on app_id
        merged_games = pd.merge(
            self.games_df, 
            self.games_metadata, 
            on='app_id', 
            how='left'
        )
        
        return merged_games
    
    def prepare_user_game_matrix(self, max_users=10000, max_games=5000):
        """
        Create a user-game interaction matrix for collaborative filtering
        based on user recommendations
        
        Parameters:
        -----------
        max_users : int
            Maximum number of users to include in the matrix
        max_games : int
            Maximum number of games to include in the matrix
        """
        if self.recommendations_df is None:
            self.load_recommendations()
            
        if self.users_df is None:
            self.load_users()
        
        if self.recommendations_df.empty:
            # Return empty matrices if no recommendation data
            return pd.DataFrame(), pd.DataFrame()
            
        # When using memory efficient mode, limit the size of the matrices
        if self.memory_efficient:
            # Get the most active users and most popular games to reduce matrix size
            top_users = self.recommendations_df['user_id'].value_counts().head(max_users).index
            top_games = self.recommendations_df['app_id'].value_counts().head(max_games).index
            
            filtered_recs = self.recommendations_df[
                self.recommendations_df['user_id'].isin(top_users) & 
                self.recommendations_df['app_id'].isin(top_games)
            ]
        else:
            filtered_recs = self.recommendations_df
        
        # Create a user-item matrix where each row is a user and each column is a game
        # Value can be either the hours played or a binary indicator of whether the game is recommended
        
        # For simplicity, we'll use binary indicators first
        user_game_matrix = filtered_recs.pivot_table(
            index='user_id',
            columns='app_id',
            values='is_recommended',
            aggfunc='first'  # Use first occurrence if there are duplicates
        ).fillna(False)
        
        # We can also create a matrix with hours played
        hours_matrix = filtered_recs.pivot_table(
            index='user_id',
            columns='app_id',
            values='hours',
            aggfunc='sum'  # Sum up hours if there are multiple entries
        ).fillna(0)
        
        print(f"Created user-game matrix with shape: {user_game_matrix.shape}")
        print(f"Created hours matrix with shape: {hours_matrix.shape}")
        
        return user_game_matrix, hours_matrix
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split the recommendation data into training and testing sets
        
        Parameters:
        -----------
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random seed for reproducibility
        """
        if self.recommendations_df is None:
            self.load_recommendations()
        
        if self.recommendations_df.empty:
            # Create empty DataFrames if no recommendation data
            self.train_data = pd.DataFrame(columns=self.recommendations_df.columns)
            self.test_data = pd.DataFrame(columns=self.recommendations_df.columns)
            return self.train_data, self.test_data
            
        # Split the data by user
        user_ids = self.recommendations_df['user_id'].unique()
        train_users, test_users = train_test_split(
            user_ids, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Create train and test datasets
        self.train_data = self.recommendations_df[self.recommendations_df['user_id'].isin(train_users)]
        self.test_data = self.recommendations_df[self.recommendations_df['user_id'].isin(test_users)]
        
        return self.train_data, self.test_data
    
    def load_all_data(self):
        """
        Load all data files and return them
        """
        self.load_games()
        self.load_users()
        self.load_recommendations()
        self.load_games_metadata()
        self.create_train_test_split()
        
        return {
            'games': self.games_df,
            'users': self.users_df,
            'recommendations': self.recommendations_df,
            'games_metadata': self.games_metadata,
            'train_data': self.train_data,
            'test_data': self.test_data
        }
    
    def get_game_features(self):
        """
        Extract and prepare features from game data for model training
        """
        if self.games_df is None or self.games_metadata is None:
            merged_games = self.merge_game_data()
        else:
            merged_games = self.merge_game_data()
            
        # Extract features that might be useful for recommendations
        features = merged_games.copy()
        
        # Handle missing values
        features['description'] = features.get('description', pd.Series('')).fillna('')
        
        # Create platform feature
        features['platforms'] = features.apply(
            lambda x: sum([x['win'], x['mac'], x['linux']]), 
            axis=1
        )
        
        # Compute age of the game in days
        current_date = datetime.datetime.now()
        features['age_days'] = (current_date - features['date_release']).dt.days
        
        # Create price features
        features['has_discount'] = features['discount'] > 0
        features['price_category'] = pd.cut(
            features['price_final'],
            bins=[0, 5, 10, 20, 50, 100, float('inf')],
            labels=['Free/Very Low', 'Low', 'Medium', 'High', 'Very High', 'Premium']
        )
        
        # Extract tag counts if tags column exists
        if 'tags' in features.columns:
            features['tag_count'] = features['tags'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        return features

# Helper function for easy data access
def load_data(sample_size=None, memory_efficient=True):
    """
    Utility function to load all data at once
    
    Parameters:
    -----------
    sample_size : int, optional
        Number of rows to sample from large files (None=all data)
    memory_efficient : bool
        Whether to use memory-efficient loading techniques
    """
    loader = DataLoader(sample_size=sample_size, memory_efficient=memory_efficient)
    return loader.load_all_data()

# Example usage
if __name__ == "__main__":
    # Use memory-efficient loading with 10% sampling for testing
    loader = DataLoader(sample_size=10, memory_efficient=True)
    data = loader.load_all_data()
    print(f"Loaded {len(data['games'])} games")
    print(f"Loaded {len(data['users'])} users")
    print(f"Loaded {len(data['recommendations'])} recommendations")
    print(f"Loaded {len(data['games_metadata'])} game metadata entries")
    print(f"Train set size: {len(data['train_data'])}")
    print(f"Test set size: {len(data['test_data'])}")