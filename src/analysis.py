"""
Statistical analysis functions for disaster data
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import poisson, nbinom, expon, gamma
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import logger


def calculate_descriptive_statistics(df: pd.DataFrame, 
                                     group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate descriptive statistics for disaster impacts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    group_by : str, optional
        Column to group by (e.g., 'hazardtype')
        
    Returns:
    --------
    pd.DataFrame
        Descriptive statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    impact_cols = [col for col in numeric_cols if col in [
        'deaths', 'injured', 'missing', 'affected', 'displaced',
        'housesdestroyed', 'housesdamaged', 'usdvalue', 'total_deaths',
        'total_affected'
    ]]
    
    if not impact_cols:
        logger.warning("No impact columns found for statistics")
        return pd.DataFrame()
    
    if group_by and group_by in df.columns:
        stats_df = df.groupby(group_by)[impact_cols].agg([
            'count', 'sum', 'mean', 'median', 'std', 'min', 'max'
        ])
    else:
        stats_df = df[impact_cols].agg([
            'count', 'sum', 'mean', 'median', 'std', 'min', 'max'
        ]).T
    
    return stats_df


def analyze_temporal_trends(df: pd.DataFrame, 
                           date_col: str = 'year',
                           value_col: str = 'serial') -> pd.DataFrame:
    """
    Analyze temporal trends in disaster events.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    date_col : str
        Date column name
    value_col : str
        Value column to analyze
        
    Returns:
    --------
    pd.DataFrame
        Temporal trends with statistics
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return pd.DataFrame()
    
    if value_col == 'serial':
        trends = df.groupby(date_col).size().reset_index(name='count')
    else:
        trends = df.groupby(date_col)[value_col].sum().reset_index()
    
    trends = trends.sort_values(date_col)
    
    if len(trends) > 1:
        value_name = 'count' if value_col == 'serial' else value_col
        
        x = np.arange(len(trends))
        y = trends[value_name].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        trends['trend_line'] = slope * x + intercept
        trends['trend_slope'] = slope
        trends['trend_r_squared'] = r_value ** 2
        trends['trend_p_value'] = p_value
    
    return trends


def analyze_seasonal_patterns(df: pd.DataFrame,
                              date_col: str = 'date',
                              value_col: str = 'serial',
                              freq: int = 12) -> Dict:
    """
    Analyze seasonal patterns in disaster events.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data with datetime column
    date_col : str
        Date column name
    value_col : str
        Value column to analyze
    freq : int
        Frequency for seasonal decomposition (12 for monthly)
        
    Returns:
    --------
    dict
        Seasonal decomposition results
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return {}
    
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    
    if value_col == 'serial':
        ts = df_temp.set_index(date_col).resample('M').size()
    else:
        ts = df_temp.set_index(date_col).resample('M')[value_col].sum()
    
    if len(ts) < 2 * freq:
        logger.warning(f"Insufficient data for seasonal decomposition (need at least {2*freq} periods)")
        return {'time_series': ts}
    
    try:
        decomposition = seasonal_decompose(ts, model='additive', period=freq)
        
        return {
            'time_series': ts,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
    except Exception as e:
        logger.warning(f"Seasonal decomposition failed: {str(e)}")
        return {'time_series': ts}


def fit_frequency_distribution(event_counts: pd.Series,
                               distributions: List[str] = ['poisson', 'nbinom']) -> Dict:
    """
    Fit probability distributions to event frequency data.
    
    Parameters:
    -----------
    event_counts : pd.Series
        Event counts per time period
    distributions : list
        List of distributions to fit
        
    Returns:
    --------
    dict
        Fitted distribution parameters and goodness-of-fit statistics
    """
    results = {}
    
    data = event_counts.values
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return results
    
    if 'poisson' in distributions:
        lambda_param = np.mean(data)
        
        # Use integer bins for count data
        min_val = int(np.floor(data.min()))
        max_val = int(np.ceil(data.max()))
        bins = np.arange(min_val, max_val + 2) - 0.5
        
        observed_freq, _ = np.histogram(data, bins=bins)
        
        # Calculate expected frequencies for each integer value
        k_values = np.arange(min_val, max_val + 1)
        expected_freq = poisson.pmf(k_values, lambda_param) * len(data)
        
        # Ensure sums match by normalizing expected frequencies
        expected_freq = expected_freq * (observed_freq.sum() / expected_freq.sum())
        
        # Combine bins with expected frequency < 5 for valid chi-square test
        mask = expected_freq >= 5
        if mask.sum() >= 2:  # Need at least 2 bins
            observed_combined = observed_freq[mask]
            expected_combined = expected_freq[mask]
            
            chi2_stat, p_value = stats.chisquare(observed_combined, expected_combined)
        else:
            # Not enough data for chi-square test
            chi2_stat, p_value = np.nan, np.nan
        
        results['poisson'] = {
            'lambda': lambda_param,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'aic': 2 * 1 - 2 * np.sum(poisson.logpmf(data, lambda_param))
        }
    
    if 'nbinom' in distributions:
        mean = np.mean(data)
        var = np.var(data)
        
        if var > mean:
            p = mean / var
            n = mean * p / (1 - p)
            
            results['nbinom'] = {
                'n': n,
                'p': p,
                'mean': mean,
                'variance': var
            }
    
    return results


def calculate_correlation_matrix(df: pd.DataFrame,
                                 columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix for impact variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    columns : list, optional
        Specific columns to include
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    if columns is None:
        columns = [col for col in df.columns if col in [
            'deaths', 'injured', 'missing', 'affected', 'displaced',
            'housesdestroyed', 'housesdamaged', 'usdvalue',
            'total_deaths', 'total_affected'
        ]]
    
    available_cols = [col for col in columns if col in df.columns]
    
    if not available_cols:
        logger.warning("No valid columns for correlation analysis")
        return pd.DataFrame()
    
    corr_matrix = df[available_cols].corr()
    
    return corr_matrix


def perform_clustering_analysis(df: pd.DataFrame,
                                features: List[str],
                                n_clusters: int = 5,
                                method: str = 'kmeans') -> pd.DataFrame:
    """
    Perform clustering analysis on disaster events.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    features : list
        Features to use for clustering
    n_clusters : int
        Number of clusters (for k-means)
    method : str
        Clustering method ('kmeans' or 'dbscan')
        
    Returns:
    --------
    pd.DataFrame
        Data with cluster assignments
    """
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        logger.warning("No valid features for clustering")
        return df
    
    df_cluster = df.copy()
    
    X = df_cluster[available_features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        df_cluster['cluster'] = clusterer.fit_predict(X_scaled)
        df_cluster['cluster_distance'] = clusterer.transform(X_scaled).min(axis=1)
    
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        df_cluster['cluster'] = clusterer.fit_predict(X_scaled)
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    logger.info(f"Clustering complete. Found {df_cluster['cluster'].nunique()} clusters")
    
    return df_cluster


def detect_anomalies(df: pd.DataFrame,
                    column: str,
                    method: str = 'iqr',
                    threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalous disaster events.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    column : str
        Column to analyze for anomalies
    method : str
        Detection method ('iqr', 'zscore', or 'isolation_forest')
    threshold : float
        Threshold for anomaly detection
        
    Returns:
    --------
    pd.DataFrame
        Data with anomaly flags
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found")
        return df
    
    df_anomaly = df.copy()
    
    if method == 'iqr':
        Q1 = df_anomaly[column].quantile(0.25)
        Q3 = df_anomaly[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_anomaly[f'{column}_anomaly'] = (
            (df_anomaly[column] < lower_bound) | 
            (df_anomaly[column] > upper_bound)
        )
    
    elif method == 'zscore':
        mean = df_anomaly[column].mean()
        std = df_anomaly[column].std()
        
        z_scores = np.abs((df_anomaly[column] - mean) / std)
        df_anomaly[f'{column}_anomaly'] = z_scores > threshold
    
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        
        X = df_anomaly[[column]].fillna(0)
        
        clf = IsolationForest(contamination=0.1, random_state=42)
        predictions = clf.fit_predict(X)
        
        df_anomaly[f'{column}_anomaly'] = predictions == -1
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    anomaly_count = df_anomaly[f'{column}_anomaly'].sum()
    logger.info(f"Detected {anomaly_count} anomalies in {column}")
    
    return df_anomaly


def analyze_hazard_interactions(df: pd.DataFrame,
                                location_col: str = 'level2',
                                time_window_days: int = 30) -> pd.DataFrame:
    """
    Analyze interactions between different hazard types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data with date column
    location_col : str
        Location column for spatial grouping
    time_window_days : int
        Time window for considering events as related
        
    Returns:
    --------
    pd.DataFrame
        Hazard interaction patterns
    """
    if 'date' not in df.columns or 'hazardtype' not in df.columns:
        logger.warning("Required columns not found for interaction analysis")
        return pd.DataFrame()
    
    df_sorted = df.sort_values(['date', location_col]).copy()
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])
    
    interactions = []
    
    for location in df_sorted[location_col].unique():
        location_data = df_sorted[df_sorted[location_col] == location]
        
        for i in range(len(location_data) - 1):
            event1 = location_data.iloc[i]
            event2 = location_data.iloc[i + 1]
            
            time_diff = (event2['date'] - event1['date']).days
            
            if 0 < time_diff <= time_window_days:
                interactions.append({
                    'location': location,
                    'hazard1': event1['hazardtype'],
                    'hazard2': event2['hazardtype'],
                    'time_diff_days': time_diff,
                    'date1': event1['date'],
                    'date2': event2['date']
                })
    
    if interactions:
        interaction_df = pd.DataFrame(interactions)
        
        interaction_summary = interaction_df.groupby(['hazard1', 'hazard2']).size().reset_index(name='count')
        interaction_summary = interaction_summary.sort_values('count', ascending=False)
        
        return interaction_summary
    
    return pd.DataFrame()


def calculate_event_frequency_by_period(df: pd.DataFrame,
                                       period: str = 'year',
                                       hazard_type: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate event frequency by time period.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    period : str
        Time period ('year', 'month', 'season')
    hazard_type : str, optional
        Specific hazard type to analyze
        
    Returns:
    --------
    pd.DataFrame
        Event frequency by period
    """
    df_freq = df.copy()
    
    if hazard_type:
        df_freq = df_freq[df_freq['hazardtype'] == hazard_type]
    
    if period == 'year':
        frequency = df_freq.groupby('year').size().reset_index(name='event_count')
    
    elif period == 'month':
        frequency = df_freq.groupby('month').size().reset_index(name='event_count')
    
    elif period == 'season':
        if 'season' in df_freq.columns:
            frequency = df_freq.groupby('season').size().reset_index(name='event_count')
        else:
            logger.warning("Season column not found")
            return pd.DataFrame()
    
    else:
        raise ValueError(f"Unknown period: {period}")
    
    return frequency


def perform_regression_analysis(df: pd.DataFrame,
                                dependent_var: str,
                                independent_vars: List[str]) -> Dict:
    """
    Perform regression analysis to identify impact factors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    dependent_var : str
        Dependent variable
    independent_vars : list
        Independent variables
        
    Returns:
    --------
    dict
        Regression results
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    
    available_vars = [var for var in independent_vars if var in df.columns]
    
    if dependent_var not in df.columns or not available_vars:
        logger.warning("Required variables not found for regression")
        return {}
    
    df_reg = df[[dependent_var] + available_vars].dropna()
    
    if len(df_reg) < 10:
        logger.warning("Insufficient data for regression analysis")
        return {}
    
    X = df_reg[available_vars]
    y = df_reg[dependent_var]
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    results = {
        'coefficients': dict(zip(available_vars, model.coef_)),
        'intercept': model.intercept_,
        'r_squared': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'n_observations': len(df_reg)
    }
    
    return results
