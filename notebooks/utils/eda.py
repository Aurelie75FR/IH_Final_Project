import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AmazonEDA:
    """
    Classe pour l'analyse exploratoire des données Amazon UK
    """
    def __init__(self):
        plt.style.use('seaborn')
    
    def analyze_categories(self, df):
        """
        Analyse la distribution et les caractéristiques des catégories
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Distribution des produits par catégorie
        category_counts = df['categoryName'].value_counts()
        sns.barplot(x=category_counts.values[:10], y=category_counts.index[:10], ax=axes[0,0])
        axes[0,0].set_title('Top 10 des catégories par nombre de produits')
        
        # Prix moyen par catégorie
        cat_prices = df.groupby('categoryName')['price'].mean().sort_values(ascending=False)
        sns.barplot(x=cat_prices.values[:10], y=cat_prices.index[:10], ax=axes[0,1])
        axes[0,1].set_title('Top 10 des catégories par prix moyen')
        
        # Notes moyennes par catégorie
        cat_ratings = df.groupby('categoryName')['stars'].mean().sort_values(ascending=False)
        sns.barplot(x=cat_ratings.values[:10], y=cat_ratings.index[:10], ax=axes[1,0])
        axes[1,0].set_title('Top 10 des catégories par note moyenne')
        
        # Nombre de best-sellers par catégorie
        bestsellers = df[df['isBestSeller']]['categoryName'].value_counts()
        sns.barplot(x=bestsellers.values[:10], y=bestsellers.index[:10], ax=axes[1,1])
        axes[1,1].set_title('Top 10 des catégories par nombre de best-sellers')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'total_categories': len(category_counts),
            'top_categories': category_counts.head(10).to_dict(),
            'avg_products_per_category': len(df) / len(category_counts)
        }
    
    def analyze_price_distribution(self, df):
        """
        Analyse détaillée de la distribution des prix
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution générale des prix
        sns.histplot(data=df, x='price_log', bins=50, ax=axes[0,0])
        axes[0,0].set_title('Distribution des prix (log)')
        
        # Prix par segment de popularité
        sns.boxplot(data=df, x='review_segment', y='price', ax=axes[0,1])
        axes[0,1].set_title('Distribution des prix par segment de reviews')
        
        # Prix des best-sellers vs autres
        sns.boxplot(data=df, x='isBestSeller', y='price', ax=axes[1,0])
        axes[1,0].set_title('Prix: Best-sellers vs Autres')
        
        # Corrélation prix/notes par catégorie
        top_cats = df['categoryName'].value_counts().head(10).index
        corrs = []
        for cat in top_cats:
            corr = df[df['categoryName'] == cat]['price'].corr(df[df['categoryName'] == cat]['stars'])
            corrs.append(corr)
        
        sns.barplot(x=corrs, y=top_cats, ax=axes[1,1])
        axes[1,1].set_title('Corrélation prix/notes par catégorie')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_ratings_and_reviews(self, df):
        """
        Analyse des notes et des reviews
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution des notes
        sns.histplot(data=df, x='stars', bins=10, ax=axes[0,0])
        axes[0,0].set_title('Distribution des notes')
        
        # Distribution des reviews (log)
        sns.histplot(data=df, x='reviews_log', bins=50, ax=axes[0,1])
        axes[0,1].set_title('Distribution du nombre de reviews (log)')
        
        # Notes moyennes par segment de prix
        sns.boxplot(data=df, x='price_category', y='stars', ax=axes[1,0])
        axes[1,0].set_title('Notes par catégorie de prix')
        
        # Evolution des notes selon le nombre de reviews
        review_bins = pd.qcut(df['reviews'], q=10, labels=['0-10%', '10-20%', '20-30%', '30-40%', 
                                                          '40-50%', '50-60%', '60-70%', '70-80%', 
                                                          '80-90%', '90-100%'])
        df['review_bin'] = review_bins
        sns.boxplot(data=df, x='review_bin', y='stars', ax=axes[1,1])
        axes[1,1].set_title('Notes selon le nombre de reviews')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_bestsellers(self, df):
        """
        Analyse spécifique des best-sellers
        """
        bestsellers = df[df['isBestSeller']]
        others = df[~df['isBestSeller']]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Comparaison des notes
        sns.histplot(data=bestsellers, x='stars', bins=10, color='green', 
                    alpha=0.5, label='Best-sellers', ax=axes[0,0])
        sns.histplot(data=others, x='stars', bins=10, color='blue', 
                    alpha=0.5, label='Autres', ax=axes[0,0])
        axes[0,0].set_title('Distribution des notes: Best-sellers vs Autres')
        axes[0,0].legend()
        
        # Comparaison des prix
        sns.histplot(data=bestsellers, x='price_log', bins=50, color='green', 
                    alpha=0.5, label='Best-sellers', ax=axes[0,1])
        sns.histplot(data=others, x='price_log', bins=50, color='blue', 
                    alpha=0.5, label='Autres', ax=axes[0,1])
        axes[0,1].set_title('Distribution des prix: Best-sellers vs Autres')
        axes[0,1].legend()
        
        # Ratio de best-sellers par catégorie de prix
        bestseller_ratio = df.groupby('price_category')['isBestSeller'].mean()
        sns.barplot(x=bestseller_ratio.index, y=bestseller_ratio.values, ax=axes[1,0])
        axes[1,0].set_title('Ratio de best-sellers par catégorie de prix')
        
        # Comparaison des reviews
        sns.histplot(data=bestsellers, x='reviews_log', bins=50, color='green', 
                    alpha=0.5, label='Best-sellers', ax=axes[1,1])
        sns.histplot(data=others, x='reviews_log', bins=50, color='blue', 
                    alpha=0.5, label='Autres', ax=axes[1,1])
        axes[1,1].set_title('Distribution des reviews: Best-sellers vs Autres')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()