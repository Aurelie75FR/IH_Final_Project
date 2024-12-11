import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    has_seaborn = True
except ImportError:
    has_seaborn = False

class AmazonEDA:
    """
    Class for exploratory data analysis of Amazon UK data
    """
    def __init__(self):
        # Use a standard matplotlib style if seaborn is not available
        if has_seaborn:
            sns.set_theme()  # Using sns.set_theme() instead of plt.style.use()
        else:
            plt.style.use('default')
    
    def analyze_categories(self, df):
        """
        Analyzes the distribution and characteristics of categories
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Distribution of products by category
        category_counts = df['categoryName'].value_counts()
        if has_seaborn:
            sns.barplot(x=category_counts.values[:10], y=category_counts.index[:10], ax=axes[0,0])
        else:
            axes[0,0].bar(range(10), category_counts.values[:10])
            axes[0,0].set_yticks(range(10))
            axes[0,0].set_yticklabels(category_counts.index[:10])
        axes[0,0].set_title('Top 10 Categories by Number of Products')
        
        # Average price by category
        cat_prices = df.groupby('categoryName')['price'].mean().sort_values(ascending=False)
        if has_seaborn:
            sns.barplot(x=cat_prices.values[:10], y=cat_prices.index[:10], ax=axes[0,1])
        else:
            axes[0,1].bar(range(10), cat_prices.values[:10])
            axes[0,1].set_yticks(range(10))
            axes[0,1].set_yticklabels(cat_prices.index[:10])
        axes[0,1].set_title('Top 10 Categories by Average Price')
        
        # Average ratings by category
        cat_ratings = df.groupby('categoryName')['stars'].mean().sort_values(ascending=False)
        if has_seaborn:
            sns.barplot(x=cat_ratings.values[:10], y=cat_ratings.index[:10], ax=axes[1,0])
        else:
            axes[1,0].bar(range(10), cat_ratings.values[:10])
            axes[1,0].set_yticks(range(10))
            axes[1,0].set_yticklabels(cat_ratings.index[:10])
        axes[1,0].set_title('Top 10 Categories by Average Rating')
        
        # Number of bestsellers by category
        bestsellers = df[df['isBestSeller']]['categoryName'].value_counts()
        if has_seaborn:
            sns.barplot(x=bestsellers.values[:10], y=bestsellers.index[:10], ax=axes[1,1])
        else:
            axes[1,1].bar(range(10), bestsellers.values[:10])
            axes[1,1].set_yticks(range(10))
            axes[1,1].set_yticklabels(bestsellers.index[:10])
        axes[1,1].set_title('Top 10 Categories by Number of Bestsellers')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'total_categories': len(category_counts),
            'top_categories': category_counts.head(10).to_dict(),
            'avg_products_per_category': len(df) / len(category_counts)
        }
