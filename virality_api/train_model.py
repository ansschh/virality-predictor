import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import joblib
import shutil
import os


class ViralityAnalyzer:
    def __init__(self, interactions_df, articles_df):
        self.interactions_df = interactions_df
        self.articles_df = articles_df
        self.weights = {
            'VIEW': 1,
            'LIKE': 4,
            'COMMENT CREATED': 10,
            'FOLLOW': 25,
            'BOOKMARK': 100
        }
        
    def analyze_temporal_patterns(self):
        """Analyze posting times and engagement patterns"""
        self.articles_df['datetime'] = pd.to_datetime(self.articles_df['timestamp'], unit='s')
        self.interactions_df['datetime'] = pd.to_datetime(self.interactions_df['timestamp'], unit='s')
        
        hourly_posts = self.articles_df['datetime'].dt.hour.value_counts().sort_index()
        daily_posts = self.articles_df['datetime'].dt.dayofweek.value_counts().sort_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.barplot(x=hourly_posts.index, y=hourly_posts.values, ax=ax1)
        ax1.set_title('Distribution of Posts by Hour')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Posts')
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sns.barplot(x=daily_posts.index, y=daily_posts.values, ax=ax2)
        ax2.set_xticklabels(days, rotation=45)
        ax2.set_title('Distribution of Posts by Day')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Number of Posts')
        
        plt.tight_layout()
        return fig

def main():
    # Create output directory for plots
    if not os.path.exists('analysis_plots'):
        os.makedirs('analysis_plots')

    print("Loading data...")
    interactions_df = pd.read_csv('user_interactions.csv')
    articles_df = pd.read_csv('shared_articles.csv')
    
    print(f"Interactions shape: {interactions_df.shape}")
    print(f"Articles shape: {articles_df.shape}")
    
    # Initialize analyzer
    analyzer = ViralityAnalyzer(interactions_df, articles_df)
    
    # Generate and save temporal analysis
    temporal_fig = analyzer.analyze_temporal_patterns()
    temporal_fig.savefig('analysis_plots/temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close(temporal_fig)
    
    # Calculate virality scores
    print("\nCalculating virality scores...")
    weights = {
        'VIEW': 1,
        'LIKE': 4,
        'COMMENT CREATED': 10,
        'FOLLOW': 25,
        'BOOKMARK': 100
    }
    
    interaction_scores = interactions_df.groupby('contentId').apply(
        lambda x: pd.Series({
            'virality_score': np.log1p(sum(x['eventType'].map(weights))),
            'total_views': sum(x['eventType'] == 'VIEW'),
            'total_likes': sum(x['eventType'] == 'LIKE'),
            'total_comments': sum(x['eventType'] == 'COMMENT CREATED'),
            'total_follows': sum(x['eventType'] == 'FOLLOW'),
            'total_bookmarks': sum(x['eventType'] == 'BOOKMARK'),
            'unique_users': len(set(x['personId'])),
            'unique_countries': len(set(x['userCountry'])),
            'engagement_duration': x['timestamp'].max() - x['timestamp'].min()
        })
    ).reset_index()
    
    # Create content features
    print("\nCreating features...")
    content_features = pd.DataFrame({
        'contentId': articles_df['contentId'],
        'text_length': articles_df['text'].str.len(),
        'title_length': articles_df['title'].str.len(),
        'language': articles_df['lang'],
        'post_hour': pd.to_datetime(articles_df['timestamp'], unit='s').dt.hour,
        'post_day': pd.to_datetime(articles_df['timestamp'], unit='s').dt.dayofweek,
        'is_weekend': pd.to_datetime(articles_df['timestamp'], unit='s').dt.dayofweek.isin([5, 6]).astype(int)
    })
    
    # Visualize content length distributions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=content_features, x='text_length', bins=50)
    plt.title('Distribution of Text Length')
    plt.subplot(1, 2, 2)
    sns.histplot(data=content_features, x='title_length', bins=50)
    plt.title('Distribution of Title Length')
    plt.tight_layout()
    plt.savefig('analysis_plots/content_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Merge features with interaction scores
    final_data = content_features.merge(interaction_scores, on='contentId', how='inner')
    
    # Handle categorical variables
    for col in ['language']:
        if col in final_data.columns:
            le = LabelEncoder()
            final_data[col] = le.fit_transform(final_data[col].fillna('unknown'))
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = final_data.drop(['contentId'], axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('analysis_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prepare data for modeling
    y = final_data['virality_score']
    X = final_data.drop(['contentId', 'virality_score'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    print("\nBuilding and training model...")
    input_dim = X_train.shape[1]
    
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('analysis_plots/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model evaluation
    train_loss = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Sample predictions
    print("\nSample predictions vs actual (converted from log space):")
    sample_predictions = np.expm1(model.predict(X_test_scaled[:5]))
    actual_values = np.expm1(y_test[:5].values)
    
    for pred, actual in zip(sample_predictions, actual_values):
        print(f"Predicted: {pred[0]:.2f}, Actual: {actual:.2f}")
    
    # Feature importance analysis
    print("\nTop 10 most important features (by correlation with virality):")
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [abs(X[col].corr(y)) for col in X.columns]
    }).sort_values('correlation', ascending=False).head(10)
    print(correlations)
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=correlations, x='correlation', y='feature')
    plt.title('Top 10 Features by Correlation with Virality')
    plt.tight_layout()
    plt.savefig('analysis_plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the trained model and scaler
    model.save('trained_model.keras')
    joblib.dump(scaler, 'scaler.pkl')

    os.makedirs('../backend/trained_model', exist_ok=True)
    
    # Save model and scaler in the api directory
    model.save('../backend/trained_model.keras')
    joblib.dump(scaler, '../backend/scaler.pkl')
    
    # Copy the training plots to frontend/public/analysis
    if not os.path.exists('../frontend/public/analysis'):
        os.makedirs('../frontend/public/analysis')
    
    for plot in os.listdir('analysis_plots'):
        shutil.copy(
            f'analysis_plots/{plot}', 
            f'../frontend/public/analysis/{plot}'
        )
    
    return model, analyzer

if __name__ == "__main__":
    model, analyzer = main()