import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class AmazonExploration:
    def __init__(self, df):
        self.df = df

    def dataset_summary(self):
        """Prints dataset information and checks for null, NaN, and duplicate values."""
        print("Dataset Information:")
        self.df.info()
        print("\nNull Values:")
        print(self.df.isnull().sum())
        print("\nNaN Values:")
        print(self.df.isna().sum())
        print("\nDuplicate Entries:")
        print(self.df.duplicated().sum())

    def descriptive_analysis(self):
        """Performs descriptive statistics and visualizations on the dataset."""
        print("\nDescriptive Statistics:")
        print(self.df.describe(include='all'))

    def visualize_stars_distribution(self):
        """Visualizes the distribution of stars."""
        self.df['stars'].hist(bins=50)
        plt.title('Distribution of Scores')
        plt.xlabel('Stars')
        plt.ylabel('Frequency')
        plt.show()

    def visualize_top_categories(self):
        """Visualizes the top 20 categories."""
        self.df['categoryName'].value_counts().head(20).plot(kind='bar')
        plt.title('Top 20 Categories')
        plt.xlabel('Category')
        plt.ylabel('Number of Products')
        plt.show()

    def plot_correlation_matrix(self):
        """Plots a heatmap of the correlation matrix."""
        corr = self.df[['stars', 'reviews', 'price', 'boughtInLastMonth']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def analyze_textual_variables(self):
        """Analyzes textual variables such as title lengths and generates a word cloud."""
        self.df['title_length'] = self.df['title'].apply(len)
        self.df['title_length'].hist(bins=50)
        plt.title('Title Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.show()

        text = ' '.join(self.df['title'].dropna().tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def visualize_outliers(self):
        """Visualizes outliers in numerical variables using box plots."""
        sns.boxplot(x=self.df['price'])
        plt.title('Price Box Plot')
        plt.show()

    def analyze_relationships(self):
        """Analyzes relationships between variables such as price and stars, and reviews and stars."""
        sns.scatterplot(x='price', y='stars', data=self.df.sample(min(10000, len(self.df))))
        plt.title('Relation between Price and Stars')
        plt.xlabel('Price')
        plt.ylabel('Stars')
        plt.show()

        sns.scatterplot(x='reviews', y='stars', data=self.df.sample(min(10000, len(self.df))))
        plt.title('Relation between Reviews and Stars')
        plt.xlabel('Reviews')
        plt.ylabel('Stars')
        plt.show()

