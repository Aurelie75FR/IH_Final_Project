import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class RecommenderEvaluator:
    def __init__(self, recommender, data):
        """
        Initialise l'évaluateur
        
        Parameters:
            recommender: Instance de AmazonRecommender
            data: DataFrame contenant les données complètes
        """
        self.recommender = recommender
        self.data = data
        
    def calculate_diversity_metrics(self, recommendations):
        """
        Calcule les métriques de diversité
        """
        # Diversité des catégories
        category_diversity = len(recommendations['categoryName'].unique()) / len(recommendations)
        
        # Diversité des prix
        price_range = recommendations['price'].max() - recommendations['price'].min()
        price_diversity = price_range / recommendations['price'].mean()
        
        # Diversité des notes
        rating_diversity = recommendations['stars'].std()
        
        return {
            'category_diversity': category_diversity,
            'price_diversity': price_diversity,
            'rating_diversity': rating_diversity
        }
    
    def calculate_coverage(self, n_recommendations=5, n_samples=100):
        """
        Calcule la couverture du catalogue
        """
        # Échantillonnage aléatoire de produits
        sampled_products = np.random.choice(self.data.index, size=n_samples, replace=False)
        recommended_products = set()
        
        for product_id in sampled_products:
            try:
                recs = self.recommender.get_similar_products(product_id, n=n_recommendations)
                recommended_products.update(recs.index)
            except Exception:
                continue
        
        coverage = len(recommended_products) / len(self.data)
        return coverage
    
    def calculate_novelty(self, recommendations, popularity_threshold=0.8):
        """
        Calcule la nouveauté des recommandations
        """
        # Calcul de la popularité relative
        popularity = self.data['reviews'].rank(pct=True)
        
        # Identification des produits non populaires
        novel_recommendations = recommendations[
            recommendations.index.map(lambda x: popularity[x] < popularity_threshold)
        ]
        
        novelty_score = len(novel_recommendations) / len(recommendations)
        return novelty_score
    
    def calculate_relevance_metrics(self, recommendations, product_id):
        """
        Calcule les métriques de pertinence
        """
        original_category = self.data.loc[product_id, 'categoryName']
        original_price = self.data.loc[product_id, 'price']
        
        # Pertinence de la catégorie
        category_relevance = (recommendations['categoryName'] == original_category).mean()
        
        # Pertinence du prix (±30%)
        price_range = (0.7 * original_price, 1.3 * original_price)
        price_relevance = (
            (recommendations['price'] >= price_range[0]) & 
            (recommendations['price'] <= price_range[1])
        ).mean()
        
        # Pertinence des notes
        rating_relevance = (recommendations['stars'] >= self.data.loc[product_id, 'stars']).mean()
        
        return {
            'category_relevance': category_relevance,
            'price_relevance': price_relevance,
            'rating_relevance': rating_relevance
        }
    
    def evaluate_similar_products(self, n_samples=100, n_recommendations=5):
        """
        Évalue les recommandations de produits similaires
        """
        sampled_products = np.random.choice(self.data.index, size=n_samples, replace=False)
        metrics = {
            'diversity': {'category': [], 'price': [], 'rating': []},
            'relevance': {'category': [], 'price': [], 'rating': []},
            'novelty': []
        }
        
        for product_id in sampled_products:
            try:
                recommendations = self.recommender.get_similar_products(product_id, n=n_recommendations)
                
                # Calcul des métriques de diversité
                diversity = self.calculate_diversity_metrics(recommendations)
                metrics['diversity']['category'].append(diversity['category_diversity'])
                metrics['diversity']['price'].append(diversity['price_diversity'])
                metrics['diversity']['rating'].append(diversity['rating_diversity'])
                
                # Calcul des métriques de pertinence
                relevance = self.calculate_relevance_metrics(recommendations, product_id)
                metrics['relevance']['category'].append(relevance['category_relevance'])
                metrics['relevance']['price'].append(relevance['price_relevance'])
                metrics['relevance']['rating'].append(relevance['rating_relevance'])
                
                # Calcul de la nouveauté
                metrics['novelty'].append(self.calculate_novelty(recommendations))
                
            except Exception:
                continue
        
        return {
            'diversity': {k: np.mean(v) for k, v in metrics['diversity'].items()},
            'relevance': {k: np.mean(v) for k, v in metrics['relevance'].items()},
            'novelty': np.mean(metrics['novelty']),
            'coverage': self.calculate_coverage()
        }
    
    def plot_evaluation_results(self, results):
        """
        Visualise les résultats de l'évaluation
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Métriques de diversité
        diversity_data = pd.Series(results['diversity'])
        sns.barplot(x=diversity_data.index, y=diversity_data.values, ax=axes[0,0])
        axes[0,0].set_title('Métriques de Diversité')
        axes[0,0].set_ylim(0, 1)
        
        # Métriques de pertinence
        relevance_data = pd.Series(results['relevance'])
        sns.barplot(x=relevance_data.index, y=relevance_data.values, ax=axes[0,1])
        axes[0,1].set_title('Métriques de Pertinence')
        axes[0,1].set_ylim(0, 1)
        
        # Nouveauté et Couverture
        other_metrics = pd.Series({
            'Novelty': results['novelty'],
            'Coverage': results['coverage']
        })
        sns.barplot(x=other_metrics.index, y=other_metrics.values, ax=axes[1,0])
        axes[1,0].set_title('Nouveauté et Couverture')
        axes[1,0].set_ylim(0, 1)
        
        # Score global
        global_score = np.mean([
            np.mean(list(results['diversity'].values())),
            np.mean(list(results['relevance'].values())),
            results['novelty'],
            results['coverage']
        ])
        axes[1,1].text(0.5, 0.5, f'Score Global:\n{global_score:.3f}', 
                      ha='center', va='center', fontsize=20)
        axes[1,1].set_title('Score Global')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return global_score