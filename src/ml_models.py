"""
Machine Learning Models for Disaster Risk Prediction
Aligned with Sendai Framework for Disaster Risk Reduction 2015-2030

This module provides:
- Event frequency forecasting
- Impact severity prediction
- Risk score prediction
- Seasonal pattern detection
- Model training, evaluation, and persistence
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_DIR


class DisasterFrequencyPredictor:
    """
    Predicts the frequency of disaster events based on historical patterns
    Uses time series and seasonal features
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        """
        Create features for frequency prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            Disaster events data with year, month, hazardtype
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (event count)
        """
        # Aggregate by year, month, hazard type
        freq_data = df.groupby(['year', 'month', 'hazardtype']).size().reset_index(name='event_count')
        
        # Create time-based features
        freq_data['year_normalized'] = (freq_data['year'] - freq_data['year'].min()) / (freq_data['year'].max() - freq_data['year'].min())
        freq_data['month_sin'] = np.sin(2 * np.pi * freq_data['month'] / 12)
        freq_data['month_cos'] = np.cos(2 * np.pi * freq_data['month'] / 12)
        
        # One-hot encode hazard types
        hazard_dummies = pd.get_dummies(freq_data['hazardtype'], prefix='hazard')
        
        # Historical features (lagged event counts)
        freq_data = freq_data.sort_values(['hazardtype', 'year', 'month'])
        freq_data['lag_1'] = freq_data.groupby('hazardtype')['event_count'].shift(1).fillna(0)
        freq_data['lag_3'] = freq_data.groupby('hazardtype')['event_count'].shift(3).fillna(0)
        freq_data['lag_12'] = freq_data.groupby('hazardtype')['event_count'].shift(12).fillna(0)
        
        # Rolling averages
        freq_data['rolling_mean_3'] = freq_data.groupby('hazardtype')['event_count'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        freq_data['rolling_mean_12'] = freq_data.groupby('hazardtype')['event_count'].transform(
            lambda x: x.rolling(window=12, min_periods=1).mean()
        )
        
        # Combine features
        X = pd.concat([
            freq_data[['year_normalized', 'month_sin', 'month_cos', 'lag_1', 'lag_3', 'lag_12', 
                       'rolling_mean_3', 'rolling_mean_12']],
            hazard_dummies
        ], axis=1)
        
        y = freq_data['event_count']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, df, test_size=0.2):
        """
        Train the frequency prediction model
        
        Parameters:
        -----------
        df : pd.DataFrame
            Disaster events data
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        dict : Training metrics
        """
        X, y = self.prepare_features(df)
        
        # Time series split (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        else:
            self.model = LinearRegression()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        # Train on full data
        self.model.fit(X_scaled, y)
        
        # Predictions
        y_pred = self.model.predict(X_scaled)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std()
        }
        
        return metrics
    
    def predict(self, year, month, hazard_type, historical_data=None):
        """
        Predict event frequency for a given period
        
        Parameters:
        -----------
        year : int
            Target year
        month : int
            Target month
        hazard_type : str
            Hazard type
        historical_data : dict
            Historical event counts for lag features
            
        Returns:
        --------
        float : Predicted event count
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create feature vector
        year_normalized = (year - 2010) / 15  # Assuming 2010-2025 range
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Default lag values if not provided
        if historical_data is None:
            historical_data = {'lag_1': 0, 'lag_3': 0, 'lag_12': 0, 'rolling_mean_3': 0, 'rolling_mean_12': 0}
        
        # Create feature dictionary
        features = {
            'year_normalized': year_normalized,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'lag_1': historical_data.get('lag_1', 0),
            'lag_3': historical_data.get('lag_3', 0),
            'lag_12': historical_data.get('lag_12', 0),
            'rolling_mean_3': historical_data.get('rolling_mean_3', 0),
            'rolling_mean_12': historical_data.get('rolling_mean_12', 0)
        }
        
        # Add hazard type one-hot encoding
        for fname in self.feature_names:
            if fname.startswith('hazard_'):
                features[fname] = 1 if fname == f'hazard_{hazard_type}' else 0
        
        # Create DataFrame with correct column order
        X_pred = pd.DataFrame([features])[self.feature_names]
        X_pred_scaled = self.scaler.transform(X_pred)
        
        prediction = self.model.predict(X_pred_scaled)[0]
        return max(0, prediction)  # Ensure non-negative
    
    def save(self, filename='frequency_predictor.pkl'):
        """Save trained model"""
        model_path = MODELS_DIR / filename
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, model_path)
        return model_path
    
    def load(self, filename='frequency_predictor.pkl'):
        """Load trained model"""
        model_path = MODELS_DIR / filename
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']


class ImpactSeverityPredictor:
    """
    Predicts the severity of disaster impacts (deaths, affected, economic loss)
    """
    
    def __init__(self, target='deaths', model_type='random_forest'):
        self.target = target
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        """
        Create features for impact prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            Disaster events data
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        """
        # Filter out events with missing target
        df_clean = df[df[self.target].notna()].copy()
        
        # Create features
        features = pd.DataFrame()
        
        # Temporal features
        features['year'] = df_clean['year']
        features['month'] = df_clean['month']
        features['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)
        
        # Duration (if available)
        if 'durationdays' in df_clean.columns:
            features['duration'] = df_clean['durationdays'].fillna(1)
        else:
            features['duration'] = 1
        
        # Hazard type encoding
        hazard_dummies = pd.get_dummies(df_clean['hazardtype'], prefix='hazard')
        
        # Location encoding (if available)
        if 'level2' in df_clean.columns:
            location_dummies = pd.get_dummies(df_clean['level2'], prefix='location')
            # Limit to top 20 locations to avoid too many features
            top_locations = location_dummies.sum().nlargest(20).index
            location_dummies = location_dummies[top_locations]
        else:
            location_dummies = pd.DataFrame()
        
        # Cross-impact features (if predicting one, use others as features)
        if self.target == 'deaths' and 'affected' in df_clean.columns:
            features['affected_log'] = np.log1p(df_clean['affected'].fillna(0))
        elif self.target == 'affected' and 'deaths' in df_clean.columns:
            features['deaths_log'] = np.log1p(df_clean['deaths'].fillna(0))
        
        if 'usdvalue' in df_clean.columns and self.target != 'usdvalue':
            features['economic_loss_log'] = np.log1p(df_clean['usdvalue'].fillna(0))
        
        # Combine all features
        X = pd.concat([features, hazard_dummies, location_dummies], axis=1)
        
        # Target variable (log transform for better distribution)
        y = np.log1p(df_clean[self.target])
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, df, test_size=0.2):
        """
        Train the impact prediction model
        
        Parameters:
        -----------
        df : pd.DataFrame
            Disaster events data
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        dict : Training metrics
        """
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=7, random_state=42)
        else:
            self.model = LinearRegression()
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Convert back from log scale for metrics
        y_train_actual = np.expm1(y_train)
        y_train_pred_actual = np.expm1(y_train_pred)
        y_test_actual = np.expm1(y_test)
        y_test_pred_actual = np.expm1(y_test_pred)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train_actual, y_train_pred_actual),
            'test_mae': mean_absolute_error(y_test_actual, y_test_pred_actual),
            'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def predict(self, event_features):
        """
        Predict impact severity for a new event
        
        Parameters:
        -----------
        event_features : dict
            Event characteristics (year, month, hazardtype, etc.)
            
        Returns:
        --------
        float : Predicted impact (in original scale)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create feature vector matching training features
        features = {}
        for fname in self.feature_names:
            if fname in event_features:
                features[fname] = event_features[fname]
            elif fname.startswith('hazard_'):
                hazard = fname.replace('hazard_', '')
                features[fname] = 1 if event_features.get('hazardtype') == hazard else 0
            elif fname.startswith('location_'):
                location = fname.replace('location_', '')
                features[fname] = 1 if event_features.get('level2') == location else 0
            else:
                features[fname] = 0
        
        X_pred = pd.DataFrame([features])[self.feature_names]
        X_pred_scaled = self.scaler.transform(X_pred)
        
        prediction_log = self.model.predict(X_pred_scaled)[0]
        prediction = np.expm1(prediction_log)
        
        return max(0, prediction)
    
    def save(self, filename=None):
        """Save trained model"""
        if filename is None:
            filename = f'{self.target}_predictor.pkl'
        model_path = MODELS_DIR / filename
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target': self.target,
            'model_type': self.model_type
        }, model_path)
        return model_path
    
    def load(self, filename=None):
        """Load trained model"""
        if filename is None:
            filename = f'{self.target}_predictor.pkl'
        model_path = MODELS_DIR / filename
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.target = data['target']
        self.model_type = data['model_type']


class RiskScorePredictor:
    """
    Predicts overall risk scores for locations based on historical patterns
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        """
        Create features for risk score prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            Disaster events data with location information
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (risk score)
        """
        # Aggregate by location
        location_features = df.groupby('level2').agg({
            'serial': 'count',  # Event frequency
            'deaths': 'sum',
            'affected': 'sum',
            'usdvalue': 'sum',
            'hazardtype': lambda x: x.nunique()  # Hazard diversity
        }).reset_index()
        
        location_features.columns = ['location', 'event_count', 'total_deaths', 
                                     'total_affected', 'total_loss', 'hazard_diversity']
        
        # Calculate risk score (composite of normalized metrics)
        location_features['deaths_norm'] = (location_features['total_deaths'] - location_features['total_deaths'].min()) / \
                                           (location_features['total_deaths'].max() - location_features['total_deaths'].min() + 1)
        location_features['affected_norm'] = (location_features['total_affected'] - location_features['total_affected'].min()) / \
                                             (location_features['total_affected'].max() - location_features['total_affected'].min() + 1)
        location_features['loss_norm'] = (location_features['total_loss'] - location_features['total_loss'].min()) / \
                                         (location_features['total_loss'].max() - location_features['total_loss'].min() + 1)
        
        # Risk score: weighted average
        location_features['risk_score'] = (
            0.4 * location_features['deaths_norm'] +
            0.3 * location_features['affected_norm'] +
            0.3 * location_features['loss_norm']
        )
        
        # Features
        X = location_features[['event_count', 'total_deaths', 'total_affected', 
                               'total_loss', 'hazard_diversity']]
        y = location_features['risk_score']
        
        self.feature_names = X.columns.tolist()
        
        return X, y, location_features['location']
    
    def train(self, df):
        """Train the risk score prediction model"""
        X, y, locations = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        else:
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        # Train
        self.model.fit(X_scaled, y)
        
        # Predictions
        y_pred = self.model.predict(X_scaled)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
        
        return metrics
    
    def predict(self, event_count, total_deaths, total_affected, total_loss, hazard_diversity):
        """Predict risk score for a location"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_pred = pd.DataFrame([[event_count, total_deaths, total_affected, total_loss, hazard_diversity]],
                              columns=self.feature_names)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        prediction = self.model.predict(X_pred_scaled)[0]
        return np.clip(prediction, 0, 1)  # Ensure 0-1 range
    
    def save(self, filename='risk_score_predictor.pkl'):
        """Save trained model"""
        model_path = MODELS_DIR / filename
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, model_path)
        return model_path
    
    def load(self, filename='risk_score_predictor.pkl'):
        """Load trained model"""
        model_path = MODELS_DIR / filename
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']


def train_all_models(df, save_models=True):
    """
    Train all ML models and return metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster events data
    save_models : bool
        Whether to save trained models
        
    Returns:
    --------
    dict : All models and their metrics
    """
    results = {}
    
    print("Training Frequency Predictor...")
    freq_model = DisasterFrequencyPredictor(model_type='random_forest')
    freq_metrics = freq_model.train(df)
    results['frequency'] = {'model': freq_model, 'metrics': freq_metrics}
    if save_models:
        freq_model.save()
    print(f"✓ Frequency Predictor - MAE: {freq_metrics['mae']:.2f}, R²: {freq_metrics['r2']:.3f}")
    
    print("\nTraining Impact Predictors...")
    for target in ['deaths', 'affected', 'usdvalue']:
        if target in df.columns and df[target].notna().sum() > 10:
            impact_model = ImpactSeverityPredictor(target=target, model_type='random_forest')
            impact_metrics = impact_model.train(df)
            results[target] = {'model': impact_model, 'metrics': impact_metrics}
            if save_models:
                impact_model.save()
            print(f"✓ {target.title()} Predictor - Test MAE: {impact_metrics['test_mae']:.2f}, R²: {impact_metrics['test_r2']:.3f}")
    
    print("\nTraining Risk Score Predictor...")
    if 'level2' in df.columns:
        risk_model = RiskScorePredictor(model_type='random_forest')
        risk_metrics = risk_model.train(df)
        results['risk_score'] = {'model': risk_model, 'metrics': risk_metrics}
        if save_models:
            risk_model.save()
        print(f"✓ Risk Score Predictor - MAE: {risk_metrics['mae']:.3f}, R²: {risk_metrics['r2']:.3f}")
    
    print("\n" + "="*80)
    print("✓ All models trained successfully!")
    if save_models:
        print(f"✓ Models saved to: {MODELS_DIR}")
    
    return results
