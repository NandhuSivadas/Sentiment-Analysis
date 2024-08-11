import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load the data
df = pd.read_csv('reviews.csv')

# Perform sentiment analysis
def analyze_sentiment(review):
    analysis = TextBlob(review)
    # Classify sentiment as Positive, Negative, or Neutral based on polarity
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['review'].apply(analyze_sentiment)

# Count sentiment occurrences
sentiment_counts = df['sentiment'].value_counts()

# Plot sentiment distribution
plt.figure(figsize=(12, 8))

# Bar chart
plt.subplot(2, 2, 1)
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Pie chart
plt.subplot(2, 2, 2) 
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', n_colors=len(sentiment_counts)))
plt.title('Sentiment Proportion')

# Stacked Bar Chart
# Data preparation for stacked bar chart
sentiment_data = pd.DataFrame({'Sentiment': sentiment_counts.index, 'Count': sentiment_counts.values})
sentiment_data['Percentage'] = (sentiment_data['Count'] / sentiment_data['Count'].sum()) * 100

# Create a DataFrame for stacked bar chart
df_stacked = pd.DataFrame({
    'Sentiment': sentiment_data['Sentiment'],
    'Count': sentiment_data['Count'],
    'Percentage': sentiment_data['Percentage']
})

plt.subplot(2, 2, 3)
df_stacked.plot(kind='bar', x='Sentiment', stacked=True, color=sns.color_palette('viridis', n_colors=len(sentiment_counts)))
plt.title('Sentiment Stacked Bar Chart')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.legend(['Count'])

# Adjust layout
plt.tight_layout()
plt.show()
