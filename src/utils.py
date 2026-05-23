"""
Utility functions for disaster risk analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import pickle
from typing import Union, List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_disaster_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load disaster data from various file formats.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the disaster data file
    **kwargs : dict
        Additional arguments to pass to pandas read functions
        
    Returns:
    --------
    pd.DataFrame
        Loaded disaster data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_extension = filepath.suffix.lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(filepath, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, **kwargs)
        elif file_extension == '.json':
            df = pd.read_json(filepath, **kwargs)
        elif file_extension == '.parquet':
            df = pd.read_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Successfully loaded {len(df)} records from {filepath}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise


def save_data(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """
    Save DataFrame to various file formats.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or Path
        Output file path
    **kwargs : dict
        Additional arguments to pass to pandas save functions
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    file_extension = filepath.suffix.lower()
    
    try:
        if file_extension == '.csv':
            df.to_csv(filepath, index=False, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            df.to_excel(filepath, index=False, **kwargs)
        elif file_extension == '.json':
            df.to_json(filepath, **kwargs)
        elif file_extension == '.parquet':
            df.to_parquet(filepath, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Successfully saved {len(df)} records to {filepath}")
    
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {str(e)}")
        raise


def create_date_column(df: pd.DataFrame, year_col: str = 'year', 
                       month_col: str = 'month', day_col: str = 'day') -> pd.Series:
    """
    Create a datetime column from separate year, month, day columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    year_col : str
        Name of year column
    month_col : str
        Name of month column
    day_col : str
        Name of day column
        
    Returns:
    --------
    pd.Series
        DateTime series
    """
    df_copy = df.copy()
    
    df_copy[month_col] = df_copy[month_col].fillna(1).astype(int)
    df_copy[day_col] = df_copy[day_col].fillna(1).astype(int)
    
    df_copy[month_col] = df_copy[month_col].clip(1, 12)
    df_copy[day_col] = df_copy[day_col].clip(1, 31)
    
    try:
        date_series = pd.to_datetime(
            df_copy[[year_col, month_col, day_col]].rename(
                columns={year_col: 'year', month_col: 'month', day_col: 'day'}
            ),
            errors='coerce'
        )
        return date_series
    except Exception as e:
        logger.warning(f"Error creating date column: {str(e)}")
        return pd.to_datetime(df_copy[year_col], format='%Y', errors='coerce')


def calculate_total_affected(df: pd.DataFrame) -> pd.Series:
    """
    Calculate total affected people from individual impact columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with impact columns
        
    Returns:
    --------
    pd.Series
        Total affected people
    """
    impact_cols = ['injured', 'affected', 'displaced']
    total = pd.Series(0, index=df.index)
    
    for col in impact_cols:
        if col in df.columns:
            total += df[col].fillna(0)
    
    return total


def calculate_total_deaths(df: pd.DataFrame) -> pd.Series:
    """
    Calculate total deaths including missing persons.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with death and missing columns
        
    Returns:
    --------
    pd.Series
        Total deaths
    """
    total = df['deaths'].fillna(0)
    if 'missing' in df.columns:
        total += df['missing'].fillna(0)
    
    return total


def get_season(month: int) -> str:
    """
    Get season from month number (Northern Hemisphere).
    
    Parameters:
    -----------
    month : int
        Month number (1-12)
        
    Returns:
    --------
    str
        Season name
    """
    if pd.isna(month):
        return 'Unknown'
    
    month = int(month)
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Unknown'


def classify_severity(value: float, thresholds: Dict[str, float]) -> str:
    """
    Classify severity based on thresholds.
    
    Parameters:
    -----------
    value : float
        Value to classify
    thresholds : dict
        Dictionary with severity thresholds
        
    Returns:
    --------
    str
        Severity classification
    """
    if pd.isna(value) or value < thresholds['minor']:
        return 'No Impact'
    elif value < thresholds['moderate']:
        return 'Minor'
    elif value < thresholds['severe']:
        return 'Moderate'
    elif value < thresholds['catastrophic']:
        return 'Severe'
    else:
        return 'Catastrophic'


def normalize_hazard_type(hazard: str) -> str:
    """
    Normalize hazard type names to standard categories.
    
    Parameters:
    -----------
    hazard : str
        Raw hazard type name
        
    Returns:
    --------
    str
        Normalized hazard type
    """
    if pd.isna(hazard):
        return 'UNKNOWN'
    
    hazard = str(hazard).upper().strip()
    
    flood_keywords = ['FLOOD', 'INUNDATION', 'OVERFLOW']
    storm_keywords = ['STORM', 'CYCLONE', 'HURRICANE', 'TYPHOON', 'TORNADO', 'WIND']
    drought_keywords = ['DROUGHT', 'DRY']
    fire_keywords = ['FIRE', 'WILDFIRE', 'FOREST FIRE']
    earthquake_keywords = ['EARTHQUAKE', 'SEISMIC', 'TREMOR']
    landslide_keywords = ['LANDSLIDE', 'MUDSLIDE', 'ROCKFALL', 'AVALANCHE']
    rain_keywords = ['RAIN', 'PRECIPITATION']
    
    for keyword in flood_keywords:
        if keyword in hazard:
            return 'FLOOD'
    for keyword in storm_keywords:
        if keyword in hazard:
            return 'STORM'
    for keyword in drought_keywords:
        if keyword in hazard:
            return 'DROUGHT'
    for keyword in fire_keywords:
        if keyword in hazard:
            return 'FIRE'
    for keyword in earthquake_keywords:
        if keyword in hazard:
            return 'EARTHQUAKE'
    for keyword in landslide_keywords:
        if keyword in hazard:
            return 'LANDSLIDE'
    for keyword in rain_keywords:
        if keyword in hazard:
            return 'HEAVY RAIN'
    
    return hazard


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : Any
        Trained model object
    filepath : str or Path
        Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: Union[str, Path]) -> Any:
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to saved model
        
    Returns:
    --------
    Any
        Loaded model object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    return model


def get_data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate data quality summary statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Data quality summary
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Null Percentage': (df.isnull().sum() / len(df) * 100).values,
        'Dtype': df.dtypes.values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
    return summary.sort_values('Null Percentage', ascending=False)


def filter_by_date_range(df: pd.DataFrame, date_col: str, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Filter DataFrame by date range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Name of date column
    start_date : str, optional
        Start date (YYYY-MM-DD format)
    end_date : str, optional
        End date (YYYY-MM-DD format)
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    df_filtered = df.copy()
    
    if start_date:
        df_filtered = df_filtered[df_filtered[date_col] >= pd.to_datetime(start_date)]
    
    if end_date:
        df_filtered = df_filtered[df_filtered[date_col] <= pd.to_datetime(end_date)]
    
    logger.info(f"Filtered from {len(df)} to {len(df_filtered)} records")
    return df_filtered


def aggregate_by_period(df: pd.DataFrame, date_col: str, 
                        value_cols: List[str], period: str = 'Y') -> pd.DataFrame:
    """
    Aggregate data by time period.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Name of date column
    value_cols : list
        List of columns to aggregate
    period : str
        Aggregation period ('Y', 'M', 'Q', 'W', 'D')
        
    Returns:
    --------
    pd.DataFrame
        Aggregated DataFrame
    """
    df_agg = df.copy()
    df_agg[date_col] = pd.to_datetime(df_agg[date_col])
    df_agg = df_agg.set_index(date_col)
    
    agg_dict = {col: 'sum' for col in value_cols if col in df_agg.columns}
    
    result = df_agg.resample(period).agg(agg_dict).reset_index()
    
    return result
