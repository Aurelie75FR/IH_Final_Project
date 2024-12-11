import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler

class AmazonDataPreprocessor:
    """
    Classe pour le prétraitement des données Amazon UK.
    Contient toutes les fonctions nécessaires pour nettoyer et préparer les données.
    """
    def __init__(self):
        self.mms = MinMaxScaler()
        
    def clean_dataset(self, df):
        """
        Nettoie le dataset en supprimant les valeurs invalides et extrêmes
        
        Parameters:
            df (pd.DataFrame): DataFrame brut
            
        Returns:
            pd.DataFrame: DataFrame nettoyé
        """
        df_clean = df.copy()
        
        # Filtrage des données valides
        mask = (
            (df_clean['reviews'] > 0) & 
            (df_clean['price'] > 0) & 
            (df_clean['stars'] > 0) &
            (df_clean['stars'] <= 5) &
            (df_clean['price'] <= df_clean['price'].quantile(0.99))
        )
        df_clean = df_clean[mask]
        
        # Transformation logarithmique des prix
        df_clean['price_log'] = np.log1p(df_clean['price'])
        
        return df_clean
    
    def create_features(self, df):
        """
        Crée les features pour l'analyse et la modélisation
        
        Parameters:
            df (pd.DataFrame): DataFrame nettoyé
            
        Returns:
            pd.DataFrame: DataFrame avec nouvelles features
        """
        df_featured = df.copy()
        
        # Transformation logarithmique des reviews
        df_featured['reviews_log'] = np.log1p(df_featured['reviews'])
        
        # Score de popularité (version améliorée)
        reviews_norm = self.mms.fit_transform(df_featured['reviews_log'].values.reshape(-1, 1))
        stars_norm = df_featured['stars'] / 5
        df_featured['popularity_score'] = 0.7 * reviews_norm.ravel() + 0.3 * stars_norm
        
        # Catégories de prix
        df_featured['price_category'] = pd.qcut(
            df_featured['price_log'],
            q=5,
            labels=['very_cheap', 'cheap', 'medium', 'expensive', 'very_expensive']
        )
        
        # Features de prix par catégorie
        df_featured['price_cat_mean'] = df_featured.groupby('categoryName')['price'].transform('mean')
        df_featured['price_ratio_to_category'] = df_featured['price'] / df_featured['price_cat_mean']
        
        # Value for money amélioré
        max_reviews = df_featured['reviews'].max()
        df_featured['value_for_money'] = (
            df_featured['stars'] / (df_featured['price_log'] + 1) * 
            (1 + np.log1p(df_featured['reviews']) / max_reviews)
        )
        df_featured['value_for_money'] = self.mms.fit_transform(
            df_featured['value_for_money'].values.reshape(-1, 1)
        )
        
        # Features supplémentaires
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
        Pipeline complet de préparation des données
        
        Parameters:
            df_raw (pd.DataFrame): DataFrame brut
            verbose (bool): Afficher les statistiques
            
        Returns:
            pd.DataFrame: DataFrame préparé pour la modélisation
        """
        if verbose:
            print("Début du traitement...")
            print(f"Nombre initial d'entrées: {len(df_raw)}")
        
        # Nettoyage
        df_cleaned = self.clean_dataset(df_raw)
        if verbose:
            print(f"Après nettoyage: {len(df_cleaned)} entrées")
        
        # Création des features
        df_final = self.create_features(df_cleaned)
        if verbose:
            print("Features créées")
        
        return df_final
    
    def get_feature_stats(self, df):
        """
        Retourne les statistiques des features principales
        """
        features = ['popularity_score', 'value_for_money', 'stars', 'price_log']
        return {feature: df[feature].describe() for feature in features}
    


def plot_final_distributions(df):
    """
    Visualisation des distributions finales

    Parameters:
        df (pd.DataFrame): DataFrame traité à visualiser
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution du score de popularité
    sns.histplot(data=df, x='popularity_score', bins=50, ax=axes[0,0])
    axes[0,0].set_title('Distribution du score de popularité (normalisé)')
    
    # Distribution du ratio qualité/prix
    sns.histplot(data=df, x='value_for_money', bins=50, ax=axes[0,1])
    axes[0,1].set_title('Distribution du ratio qualité/prix (normalisé)')
    
    # Prix log-transformés
    sns.histplot(data=df, x='price_log', bins=50, ax=axes[1,0])
    axes[1,0].set_title('Distribution des prix (log)')
    
    # Corrélation
    features = ['price_log', 'stars', 'reviews_log', 'popularity_score', 'value_for_money']
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
    axes[1,1].set_title('Corrélations')
    
    plt.tight_layout()
    plt.show()