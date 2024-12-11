import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import random

class AmazonRecommender:
    def __init__(self, price_variation=0.3, random_ratio=0.2):
        self.knn_model = None
        self.scaler = MinMaxScaler()
        self.product_features = None
        self.product_data = None
        self.price_variation = price_variation
        self.random_ratio = random_ratio  # Ratio of random recommendations
        
    def create_product_features(self, df):
        """
        Features with dynamic weighting
        """
        numeric_features = {
            'stars': 0.8,  # Reduced to allow more diversity
            'price_log': 0.7,
            'popularity_score': 0.6,
            'value_for_money': 0.9,
            'price_ratio_to_category': 0.5
        }
        
        # Dynamic normalization of features
        numeric_df = df[numeric_features.keys()].copy()
        for feature, weight in numeric_features.items():
            feature_std = numeric_df[feature].std()
            adjusted_weight = weight * (1 + feature_std)  # Adjusts the weight according to variance
            numeric_df[feature] = numeric_df[feature] * adjusted_weight
        
        # Categories with diversification
        top_categories = df['categoryName'].value_counts().nlargest(100).index  # Increased from 50 to 100
        df_filtered = df.copy()
        df_filtered.loc[~df_filtered['categoryName'].isin(top_categories), 'categoryName'] = 'Other'
        category_dummies = pd.get_dummies(df_filtered['categoryName'], prefix='category')
        
        # Dynamic adjustment of category weights
        category_weights = 1.0 + (df_filtered['categoryName'].value_counts(normalize=True) * 0.5)
        for cat in category_dummies.columns:
            cat_name = cat.split('_', 1)[1]
            if cat_name in category_weights:
                category_dummies[cat] = category_dummies[cat] * category_weights[cat_name]
        
        self.product_features = pd.concat([
            numeric_df,
            category_dummies,
            pd.get_dummies(df['price_category'], prefix='price_segment'),
            pd.get_dummies(df['review_segment'], prefix='review_segment')
        ], axis=1)
        
        self.product_features = pd.DataFrame(
            self.scaler.fit_transform(self.product_features),
            columns=self.product_features.columns,
            index=df.index
        )
        
        return self.product_features

    
    def get_category_recommendations(self, category, n=5, min_price=5.0, include_related=True):
        try:
            # Strict category selection
            categories = [category]
            if include_related:
                related_cats = self.find_similar_categories(category)[:2]
                # Additional relevance check
                related_cats = [cat for cat in related_cats 
                            if self.check_category_relevance(category, cat)]
                categories.extend(related_cats)
                    
            # Filtering with stricter price constraints
            mask = (
                (self.product_data['categoryName'].isin(categories)) & 
                (self.product_data['price'] >= min_price) &
                (self.product_data['price'] <= min_price * 20)  # Reasonable max price
            )
            category_products = self.product_data[mask].copy()
                
            # Finer price segmentation
            category_products['price_segment'] = pd.qcut(
                category_products['price'], 
                q=5, 
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
                
            # Enhanced value-for-money score
            price_mean = category_products['price'].mean()
            category_products['price_score'] = 1 / (1 + np.abs(np.log(category_products['price'] / price_mean)))
                
            # Bonus for popular but not too expensive products
            review_score = np.log1p(category_products['stars']) / np.log(6)  # Smoother rating normalization
            category_products['value_score'] = review_score * category_products['price_score']
                
            # Improved diversity
            segment_counts = category_products.groupby(['price_segment', 'categoryName']).size()
            category_products['diversity_bonus'] = category_products.apply(
                lambda x: 1 / np.log1p(segment_counts[x['price_segment']][x['categoryName']]),
                axis=1
            )
                
            # Final score with more components
            category_products['final_score'] = (
                0.35 * category_products['value_score'] +
                0.25 * category_products['stars'] / 5 +
                0.25 * category_products['price_score'] +
                0.15 * category_products['diversity_bonus']
            )
                
            # Improved stratified selection
            recommendations = []
            for segment in category_products['price_segment'].unique():
                segment_products = category_products[
                    (category_products['price_segment'] == segment) &
                    (category_products['categoryName'] == category)  # Priority to the main category
                ]
                if len(segment_products) > 0:
                    recommendations.append(segment_products.nlargest(1, 'final_score'))
                
            # Fill in if necessary
            if len(recommendations) < n:
                remaining = category_products[
                    ~category_products.index.isin(pd.concat(recommendations).index)
                ]
                recommendations.append(remaining.nlargest(n - len(recommendations), 'final_score'))
            
            final_recommendations = pd.concat(recommendations)
            return final_recommendations.nlargest(n, 'final_score')[
                ['title', 'categoryName', 'price', 'stars', 'price_segment', 'final_score']
            ]
        
        except Exception as e:
            print(f"Error in get_category_recommendations: {str(e)}")
            return None


    def check_category_relevance(self, category1, category2):
        """
        Checks the relevance between two categories
        """
        # Calculate price statistics
        cat1_stats = self.product_data[self.product_data['categoryName'] == category1]['price'].describe()
        cat2_stats = self.product_data[self.product_data['categoryName'] == category2]['price'].describe()
        
        # Check compatibility of price ranges
        price_overlap = (
            (cat1_stats['75%'] >= cat2_stats['25%']) and 
            (cat1_stats['25%'] <= cat2_stats['75%'])
        )
        
        # Check average ratings
        cat1_rating = self.product_data[self.product_data['categoryName'] == category1]['stars'].mean()
        cat2_rating = self.product_data[self.product_data['categoryName'] == category2]['stars'].mean()
        rating_similarity = abs(cat1_rating - cat2_rating) < 1.0
        
        return price_overlap and rating_similarity

    def get_similar_products(self, product_id, n=5):
        """
        Improved version with more diversity
        """
        if product_id not in self.product_data.index:
            raise ValueError("Product ID not found in dataset")
        
        # Increase the initial pool
        initial_n = min(n * 30, len(self.product_data)-1)
        product_features = self.product_features.loc[product_id].values.reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(
            product_features,
            n_neighbors=initial_n+1
        )
        
        recommendations = self.product_data.iloc[indices.flatten()[1:]].copy()
        recommendations['similarity_score'] = 1 - distances.flatten()[1:]
        
        # Price diversification
        original_price = self.product_data.loc[product_id, 'price']
        recommendations['price_ratio'] = recommendations['price'] / original_price
        recommendations['price_diversity'] = 1 - np.abs(np.log(recommendations['price_ratio']))
        
        # Category diversification
        original_category = self.product_data.loc[product_id, 'categoryName']
        recommendations['same_category'] = (recommendations['categoryName'] == original_category)
        category_counts = recommendations.groupby('categoryName')['title'].transform('count')
        recommendations['category_diversity'] = 1 / np.log1p(category_counts)
        
        # Bonus for different categories
        recommendations.loc[~recommendations['same_category'], 'category_diversity'] *= 1.5
        
        # Final score
        recommendations['final_score'] = (
            0.35 * recommendations['similarity_score'] +
            0.35 * recommendations['category_diversity'] +
            0.30 * recommendations['price_diversity']
        )
        
        # Selection with diversity control
        final_recs = []
        
        # At least one product from the same category
        same_cat = recommendations[recommendations['same_category']].nlargest(1, 'final_score')
        if not same_cat.empty:
            final_recs.append(same_cat)
        
        # The rest of the products with the best diversity
        diff_cat = recommendations[~recommendations['same_category']].nlargest(n-len(final_recs), 'final_score')
        if not diff_cat.empty:
            final_recs.append(diff_cat)
        
        final_recommendations = pd.concat(final_recs)
        return final_recommendations[
            ['title', 'categoryName', 'price', 'stars', 
            'similarity_score', 'category_diversity', 'price_diversity', 'final_score']
        ]

    def get_personalized_recommendations(self, user_preferences, n=5):
        """
        Personalized recommendations with exploration
        """
        # Broadening the criteria
        extended_categories = set(user_preferences['preferred_categories'])
        for category in user_preferences['preferred_categories']:
            similar_categories = self.find_similar_categories(category)
            extended_categories.update(similar_categories[:2])  # Adds 2 similar categories
        
        price_range = user_preferences['price_range']
        extended_price_range = (
            price_range[0] * 0.8,  # 20% lower
            price_range[1] * 1.2   # 20% higher
        )
        
        mask = (
            self.product_data['categoryName'].isin(extended_categories) &
            (self.product_data['price'] >= extended_price_range[0]) &
            (self.product_data['price'] <= extended_price_range[1]) &
            (self.product_data['stars'] >= user_preferences['min_rating'] * 0.9)  # 10% more lenient
        )
        
        filtered_products = self.product_data[mask].copy()
        
        if len(filtered_products) == 0:
            raise ValueError("No products found matching the specified criteria")
        
        # Enhanced personalized scores
        filtered_products['diversity_score'] = 1 / np.log1p(
            filtered_products.groupby(['categoryName', 'price_category'])['title'].transform('count')
        )
        
        filtered_products['exploration_bonus'] = np.random.uniform(0.8, 1.2, size=len(filtered_products))
        
        filtered_products['personalized_score'] = (
            0.25 * filtered_products['popularity_score'] +
            0.25 * filtered_products['value_for_money'] +
            0.20 * filtered_products['stars'] / 5 +
            0.15 * filtered_products['diversity_score'] +
            0.15 * filtered_products['exploration_bonus']
        )
        
        # Balanced selection by category
        recommendations = []
        remaining_n = n
        
        for category in user_preferences['preferred_categories']:
            category_products = filtered_products[
                filtered_products['categoryName'] == category
            ].nlargest(max(1, remaining_n // len(user_preferences['preferred_categories'])), 'personalized_score')
            
            recommendations.append(category_products)
            remaining_n -= len(category_products)
        
        # Add recommendations from extended categories if needed
        if remaining_n > 0:
            other_recommendations = filtered_products[
                ~filtered_products['categoryName'].isin(user_preferences['preferred_categories'])
            ].nlargest(remaining_n, 'personalized_score')
            recommendations.append(other_recommendations)
        
        final_recommendations = pd.concat(recommendations)
        return final_recommendations[
            ['title', 'categoryName', 'price', 'stars', 'value_for_money']
        ]

    
    def find_similar_categories(self, category):
        """
        Finds similar categories based solely on price and ratings
        
        Parameters:
            category (str): The reference category
            
        Returns:
            list: List of similar categories sorted by similarity
        """
        # Calculate statistics by category
        category_stats = self.product_data.groupby('categoryName').agg({
            'price': ['mean', 'std'],
            'stars': ['mean', 'std']
        })
        
        # Flatten the multi-level index
        category_stats.columns = ['price_mean', 'price_std', 'stars_mean', 'stars_std']
        
        # Normalize the statistics
        stats_scaled = pd.DataFrame()
        for col in category_stats.columns:
            stats_scaled[col] = (category_stats[col] - category_stats[col].mean()) / category_stats[col].std()
        
        # Calculate distances with the reference category
        target_stats = stats_scaled.loc[category]
        distances = pd.Series(index=stats_scaled.index)
        
        for cat in stats_scaled.index:
            if cat != category:
                # Weighted Euclidean distance
                price_distance = np.sqrt(
                    (stats_scaled.loc[cat, 'price_mean'] - target_stats['price_mean'])**2 +
                    (stats_scaled.loc[cat, 'price_std'] - target_stats['price_std'])**2
                )
                stars_distance = np.sqrt(
                    (stats_scaled.loc[cat, 'stars_mean'] - target_stats['stars_mean'])**2 +
                    (stats_scaled.loc[cat, 'stars_std'] - target_stats['stars_std'])**2
                )
                
                distances[cat] = 0.6 * price_distance + 0.4 * stars_distance
        
        # Sort categories by similarity
        similar_categories = distances.sort_values().index.tolist()
        
        return similar_categories

    def fit(self, df, verbose=True):
        """
        Training with verification
        """
        if verbose:
            print("Creating features...")
        self.product_data = df
        self.create_product_features(df)
        
        if verbose:
            print("Training the KNN model...")
        self.knn_model = NearestNeighbors(
            n_neighbors=min(50, len(df)),  # Increased for more diversity
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
        self.knn_model.fit(self.product_features)
        
        if verbose:
            print("Training completed!")
        return self
