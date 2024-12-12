import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class RecommenderEvaluator:
    """
    Classe d'évaluation pour le système de recommandation Amazon
    """
    def __init__(self, recommender, data):
        self.recommender = recommender
        self.data = data
        self.all_recommendations = set()
    
    def evaluate_diversity(self, original_product, recommendations):
        """
        Calcule les métriques de diversité pour un ensemble de recommandations
        """
        try:
            metrics = {
                'category_diversity': len(recommendations['categoryName'].unique()) / len(recommendations),
                'price_range_ratio': (recommendations['price'].max() - recommendations['price'].min()) / original_product['price'],
                'price_distance': abs(recommendations['price'] - original_product['price']).mean() / original_product['price'],
                'rating_spread': recommendations['stars'].max() - recommendations['stars'].min()
            }
            return metrics
        except Exception as e:
            print(f"Erreur dans evaluate_diversity: {str(e)}")
            return None
    
    def evaluate_relevance(self, recommendations):
        """
        Calcule les métriques de pertinence pour un ensemble de recommandations
        """
        try:
            metrics = {
                'avg_rating': recommendations['stars'].mean(),
                'avg_reviews': recommendations['reviews'].mean(),
                'min_rating': recommendations['stars'].min(),
                'weighted_rating': (recommendations['stars'] * np.log1p(recommendations['reviews'])).mean() / 
                                 np.log1p(recommendations['reviews']).mean()
            }
            return metrics
        except Exception as e:
            print(f"Erreur dans evaluate_relevance: {str(e)}")
            return None
    
    def evaluate_coverage(self, n_samples=50):
        """
        Calcule les métriques de couverture pour le système
        """
        try:
            total_categories = len(self.data['categoryName'].unique())
            total_price_range = self.data['price'].max() - self.data['price'].min()
            
            # Échantillonnage stratifié
            price_bins = pd.qcut(self.data['price'], q=5)
            sample_products = []
            
            for bin_label in price_bins.unique():
                bin_products = self.data[price_bins == bin_label].sample(
                    n=min(n_samples // 5, len(self.data[price_bins == bin_label]))
                ).index
                sample_products.extend(bin_products)
            
            covered_categories = set()
            price_ranges = []
            successful_recs = 0
            
            for product_id in sample_products:
                try:
                    recs = self.recommender.get_similar_products(product_id)
                    if not recs.empty:
                        self.all_recommendations.update(recs.index)
                        covered_categories.update(recs['categoryName'].unique())
                        price_ranges.extend([recs['price'].min(), recs['price'].max()])
                        successful_recs += 1
                except Exception as e:
                    continue
            
            metrics = {
                'category_coverage': len(covered_categories) / total_categories,
                'price_range_coverage': (max(price_ranges) - min(price_ranges)) / total_price_range if price_ranges else 0,
                'unique_items_ratio': len(set(self.all_recommendations)) / len(self.all_recommendations) if self.all_recommendations else 0,
                'success_rate': successful_recs / len(sample_products)
            }
            return metrics
        except Exception as e:
            print(f"Erreur dans evaluate_coverage: {str(e)}")
            return None
    
    def evaluate_system(self, n_samples=50):
        """
        Évaluation complète du système
        """
        try:
            # Échantillonnage stratifié par gamme de prix
            price_bins = pd.qcut(self.data['price'], q=5)
            sample_products = []
            
            for bin_label in price_bins.unique():
                bin_products = self.data[price_bins == bin_label].sample(
                    n=min(n_samples // 5, len(self.data[price_bins == bin_label]))
                ).index
                sample_products.extend(bin_products)
            
            metrics = {
                'diversity': [],
                'relevance': [],
            }
            
            for product_id in sample_products:
                try:
                    original = self.data.loc[product_id]
                    recs = self.recommender.get_similar_products(product_id)
                    
                    if not recs.empty:
                        diversity_metrics = self.evaluate_diversity(original, recs)
                        relevance_metrics = self.evaluate_relevance(recs)
                        
                        if diversity_metrics and relevance_metrics:
                            metrics['diversity'].append(diversity_metrics)
                            metrics['relevance'].append(relevance_metrics)
                except Exception as e:
                    continue
            
            coverage_metrics = self.evaluate_coverage(n_samples)
            
            # Calcul des moyennes
            avg_metrics = {
                'diversity': {k: np.mean([d[k] for d in metrics['diversity']]) 
                             for k in metrics['diversity'][0].keys()},
                'relevance': {k: np.mean([r[k] for r in metrics['relevance']]) 
                             for k in metrics['relevance'][0].keys()},
                'coverage': coverage_metrics
            }
            
            return avg_metrics
        
        except Exception as e:
            print(f"Erreur dans evaluate_system: {str(e)}")
            return None

    def plot_evaluation_results(self, results):
        """
        Visualisation des résultats d'évaluation
        """
        if not results:
            print("Pas de résultats à afficher")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Diversité
        diversity_data = pd.Series(results['diversity'])
        sns.barplot(x=diversity_data.index, y=diversity_data.values, ax=axes[0,0])
        axes[0,0].set_title('Diversity Metrics')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Pertinence
        relevance_data = pd.Series(results['relevance'])
        sns.barplot(x=relevance_data.index, y=relevance_data.values, ax=axes[0,1])
        axes[0,1].set_title('Pertinance Metrics')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Couverture
        coverage_data = pd.Series(results['coverage'])
        sns.barplot(x=coverage_data.index, y=coverage_data.values, ax=axes[1,0])
        axes[1,0].set_title('Coverage Metrics')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Score global
        global_scores = {
            'Diversity': np.mean(list(results['diversity'].values())),
            'Relevance': np.mean(list(results['relevance'].values())),
            'Coverage': np.mean(list(results['coverage'].values()))
        }
        global_score = np.mean(list(global_scores.values()))
        
        axes[1,1].text(0.5, 0.5, f'Score Global:\n{global_score:.3f}', 
                      ha='center', va='center', fontsize=20)
        axes[1,1].set_title('Score Global')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return global_score