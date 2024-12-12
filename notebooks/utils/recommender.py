import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from warnings import filterwarnings
filterwarnings('ignore', category=UserWarning)

class AmazonRecommender:
    """
    Système de recommandation pour les produits Amazon
    """
    def __init__(self):
        self.knn_model = None
        self.scaler = MinMaxScaler()
        self.product_features = None
        self.product_data = None
    
    def _calculate_category_similarity(self, cat1, cat2):
        """
        Calcule la similarité entre deux catégories basée sur des règles métier
        """
        electronics_categories = {
            'Hi-Fi Speakers', 'PC & Video Games', 'PC Gaming Accessories',
            'Headphones', 'Home Audio & Theater', 'TV & Home Cinema',
            'Home Entertainment', 'Electrical', 'Consumer Electronics'
        }
        
        home_categories = {
            'Home & Kitchen', 'Furniture', 'Home Entertainment Furniture',
            'Home Improvement', 'Home Accessories', 'Home Storage'
        }
        
        if cat1 == cat2:
            return 1.0
        elif cat1 in electronics_categories and cat2 in electronics_categories:
            if 'Speakers' in cat1 and 'Speakers' in cat2:
                return 0.8
            return 0.6
        elif cat1 in home_categories and cat2 in home_categories:
            return 0.6
        else:
            return 0.3
    
    def create_product_features(self, df):
        """
        Crée les features avec pondérations adaptatives
        """
        feature_df = pd.DataFrame(index=df.index)
        
        # Features numériques normalisées
        feature_df['price_norm'] = np.log1p(df['price']) / np.log1p(df['price'].max())
        feature_df['stars_norm'] = df['stars'] / 5
        feature_df['reviews_norm'] = np.log1p(df['reviews']) / np.log1p(df['reviews'].max())
        
        # Segmentation des prix
        price_quantiles = pd.qcut(df['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        price_dummies = pd.get_dummies(price_quantiles, prefix='price')
        
        # Segmentation des notes
        rating_cats = pd.cut(df['stars'], 
                           bins=[0, 3.5, 4.0, 4.5, 5.0], 
                           labels=['low', 'medium', 'high', 'very_high'])
        rating_dummies = pd.get_dummies(rating_cats, prefix='rating')
        
        # Catégories principales
        top_categories = df['categoryName'].value_counts().nlargest(50).index
        df_filtered = df.copy()
        df_filtered.loc[~df_filtered['categoryName'].isin(top_categories), 'categoryName'] = 'Other'
        category_dummies = pd.get_dummies(df_filtered['categoryName'], prefix='category')
        
        self.product_features = pd.concat([
            feature_df,
            price_dummies,
            rating_dummies,
            category_dummies
        ], axis=1)
        
        self.product_features = pd.DataFrame(
            self.scaler.fit_transform(self.product_features),
            columns=self.product_features.columns,
            index=df.index
        )
        
        return self.product_features

    def get_similar_products(self, product_id, n=5):
        """
        Version raffinée avec recommandations plus pertinentes
        """
        original = self.product_data.loc[product_id]
        recommendations = []
        
        # Définir les limites de ratio de prix
        max_price_ratio = 3.0
        min_price_ratio = 0.2
        
        # 1. Produits similaires mais plus diversifiés
        similar_products = self.product_data[
            (self.product_data.index != product_id) &
            (~self.product_data['title'].str.lower().str.contains(
                original['title'].lower().split('|')[0].strip()
            )) &
            (self.product_data['price'] >= original['price'] * min_price_ratio) &
            (self.product_data['price'] <= original['price'] * max_price_ratio)
        ].copy()
        
        # Calcul des scores de similarité de catégorie
        similar_products['category_similarity'] = similar_products['categoryName'].apply(
            lambda x: self._calculate_category_similarity(x, original['categoryName'])
        )
        
        # Calcul des scores de prix
        similar_products['price_ratio'] = similar_products['price'] / original['price']
        similar_products['price_score'] = 1 - np.abs(np.log(similar_products['price_ratio']))
        similar_products['price_score'] = similar_products['price_score'].clip(0, 1)
        
        # Score initial
        similar_products['initial_score'] = (
            similar_products['category_similarity'] * 0.4 +
            similar_products['price_score'] * 0.3 +
            (similar_products['stars'] / 5) * 0.3
        )
        
        # Sélection des recommandations par segments
        segments = [
            ("Même catégorie, prix différent", 
             lambda x: (x['category_similarity'] > 0.9) & 
                      (0.5 < x['price_ratio']) & (x['price_ratio'] < 2.0), 1),
            
            ("Catégorie similaire", 
             lambda x: (x['category_similarity'] > 0.6) & 
                      (0.3 < x['price_ratio']) & (x['price_ratio'] < 2.5) &
                      (x['stars'] >= 4.2), 2),
            
            ("Différent mais pertinent", 
             lambda x: (x['category_similarity'] > 0.4) & 
                      (0.2 < x['price_ratio']) & (x['price_ratio'] < 3.0) &
                      (x['stars'] >= 4.2), 2)
        ]
        
        for desc, condition, count in segments:
            segment_products = similar_products[condition(similar_products)].copy()
            
            if not segment_products.empty:
                weights = (
                    segment_products['initial_score'] * 
                    np.log1p(segment_products['reviews'])
                )
                weights = weights / weights.sum()
                
                selected = segment_products.sample(
                    n=min(count, len(segment_products)), 
                    weights=weights,
                    replace=False
                )
                
                for _, product in selected.iterrows():
                    recommendations.append(product)
        
        if not recommendations:
            return pd.DataFrame()
        
        result = pd.DataFrame(recommendations)
        
        # Scores de diversité finaux
        result['price_div'] = np.clip(
            abs(result['price'] - original['price']) / original['price'],
            0, 1
        )
        result['rating_div'] = abs(result['stars'] - original['stars']) / 2
        result['cat_div'] = 1 - result['category_similarity']
        result['pop_div'] = np.log1p(result['reviews']) / np.log1p(result['reviews'].max())
        
        # Score final équilibré
        result['final_score'] = (
            0.35 * result['cat_div'] +
            0.25 * result['price_div'] +
            0.20 * result['rating_div'] +
            0.20 * result['pop_div']
        )
        
        return result.nlargest(min(n, len(result)), 'final_score')[
            ['title', 'categoryName', 'price', 'stars', 'reviews', 'final_score']
        ]
    
    def get_category_recommendations(self, category, n=5):
        """
        Recommande les meilleurs produits d'une catégorie
        """
        category_products = self.product_data[
            self.product_data['categoryName'] == category
        ].copy()
        
        if len(category_products) == 0:
            return pd.DataFrame()
        
        category_products['value_score'] = (
            category_products['stars'] / 5 * 0.4 +
            (1 - category_products['price'] / category_products['price'].max()) * 0.3 +
            (np.log1p(category_products['reviews']) / 
             np.log1p(category_products['reviews'].max())) * 0.3
        )
        
        return category_products.nlargest(n, 'value_score')[[
            'title', 'categoryName', 'price', 'stars', 'value_score'
        ]]
    
    def get_personalized_recommendations(self, user_prefs, n=5):
        """
        Recommandations personnalisées basées sur les préférences utilisateur
        """
        mask = (
            self.product_data['categoryName'].isin(user_prefs['categories']) &
            (self.product_data['price'] >= user_prefs['min_price']) &
            (self.product_data['price'] <= user_prefs['max_price']) &
            (self.product_data['stars'] >= user_prefs['min_rating'])
        )
        
        candidates = self.product_data[mask].copy()
        if len(candidates) == 0:
            return pd.DataFrame()
        
        candidates['pref_score'] = (
            candidates['stars'] / 5 * 0.4 +
            (1 - candidates['price'] / candidates['price'].max()) * 0.3 +
            (np.log1p(candidates['reviews']) / 
             np.log1p(candidates['reviews'].max())) * 0.3
        )
        
        return candidates.nlargest(n, 'pref_score')[[
            'title', 'categoryName', 'price', 'stars', 'pref_score'
        ]]
    
    def fit(self, df, verbose=True):
        """
        Entraîne le système de recommandation
        """
        if verbose:
            print("Creating features...")
            
        self.product_data = df
        self.create_product_features(df)
        
        if verbose:
            print("Training the KNN model...")
            
        self.knn_model = NearestNeighbors(
            n_neighbors=50,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
        self.knn_model.fit(self.product_features)
        
        if verbose:
            print("Training completed!")
            
        return self