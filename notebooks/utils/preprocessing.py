import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

class AmazonDataPreprocessor:
    """
    Class for preprocessing Amazon UK data.
    Contains all the necessary functions to clean and prepare the data.
    """
    def __init__(self):
        self.mms = MinMaxScaler()
        
    def clean_dataset(self, df):
        """
        Cleans the dataset by removing invalid and extreme values
        
        Parameters:
            df (pd.DataFrame): Raw DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Filtering valid data
        mask = (
            (df_clean['reviews'] > 0) & 
            (df_clean['price'] > 0) & 
            (df_clean['stars'] > 0) &
            (df_clean['stars'] <= 5) &
            (df_clean['price'] <= df_clean['price'].quantile(0.99))
        )
        df_clean = df_clean[mask]
        
        # Log transformation of prices
        df_clean['price_log'] = np.log1p(df_clean['price'])
        
        return df_clean
    
    def create_features(self, df):
        """
        Creates features for analysis and modeling
        
        Parameters:
            df (pd.DataFrame): Cleaned DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with new features
        """
        df_featured = df.copy()
        
        # Log transformation of reviews
        df_featured['reviews_log'] = np.log1p(df_featured['reviews'])
        
        # Popularity score (enhanced version)
        reviews_norm = self.mms.fit_transform(df_featured['reviews_log'].values.reshape(-1, 1))
        stars_norm = df_featured['stars'] / 5
        df_featured['popularity_score'] = 0.7 * reviews_norm.ravel() + 0.3 * stars_norm
        
        # Price categories
        df_featured['price_category'] = pd.qcut(
            df_featured['price_log'],
            q=5,
            labels=['very_cheap', 'cheap', 'medium', 'expensive', 'very_expensive']
        )
        
        # Price features by category
        df_featured['price_cat_mean'] = df_featured.groupby('categoryName')['price'].transform('mean')
        df_featured['price_ratio_to_category'] = df_featured['price'] / df_featured['price_cat_mean']
        
        # Enhanced value for money
        max_reviews = df_featured['reviews'].max()
        df_featured['value_for_money'] = (
            df_featured['stars'] / (df_featured['price_log'] + 1) * 
            (1 + np.log1p(df_featured['reviews']) / max_reviews)
        )
        df_featured['value_for_money'] = self.mms.fit_transform(
            df_featured['value_for_money'].values.reshape(-1, 1)
        )
        
        # Additional features
        df_featured['price_segment'] = pd.qcut(df_featured['price_log'], q=10, labels=False)
        df_featured['is_high_rated'] = (df_featured['stars'] >= 4).astype(int)
        df_featured['review_segment'] = pd.qcut(
            df_featured['reviews_log'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        return df_featured
    
    def prepare_data(self, df_raw, verbose=True):
        """
        Complete data preparation pipeline
        
        Parameters:
            df_raw (pd.DataFrame): Raw DataFrame
            verbose (bool): Display statistics
            
        Returns:
            pd.DataFrame: Prepared DataFrame for modeling
        """
        if verbose:
            print("Starting processing...")
            print(f"Initial number of entries: {len(df_raw)}")
        
        # Cleaning
        df_cleaned = self.clean_dataset(df_raw)
        if verbose:
            print(f"After cleaning: {len(df_cleaned)} entries")
        
        # Feature creation
        df_final = self.create_features(df_cleaned)
        if verbose:
            print("Features created")
        
        return df_final
    
    def get_feature_stats(self, df):
        """
        Returns statistics of the main features
        """
        features = ['popularity_score', 'value_for_money', 'stars', 'price_log']
        return {feature: df[feature].describe() for feature in features}
    


def plot_final_distributions(df):
    """
    Visualization of final distributions

    Parameters:
        df (pd.DataFrame): Processed DataFrame to visualize
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution of popularity score
    sns.histplot(data=df, x='popularity_score', bins=50, ax=axes[0,0])
    axes[0,0].set_title('Distribution of Popularity Score (normalized)')
    
    # Distribution of value for money ratio
    sns.histplot(data=df, x='value_for_money', bins=50, ax=axes[0,1])
    axes[0,1].set_title('Distribution of Value for Money Ratio (normalized)')
    
    # Log-transformed prices
    sns.histplot(data=df, x='price_log', bins=50, ax=axes[1,0])
    axes[1,0].set_title('Distribution of Prices (log)')
    
    # Correlation
    features = ['price_log', 'stars', 'reviews_log', 'popularity_score', 'value_for_money']
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
    axes[1,1].set_title('Correlations')
    
    plt.tight_layout()
    plt.show()
