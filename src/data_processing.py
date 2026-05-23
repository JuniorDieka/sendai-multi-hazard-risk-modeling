"""
Data processing and cleaning functions for disaster data
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import HAZARD_TYPES, MIN_YEAR, MAX_YEAR
from src.utils import (create_date_column, normalize_hazard_type, 
                       get_data_quality_summary, logger)


def load_and_validate_disaster_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform initial validation of disaster data.
    
    Parameters:
    -----------
    filepath : str
        Path to disaster data file
        
    Returns:
    --------
    pd.DataFrame
        Validated disaster data
    """
    logger.info(f"Loading disaster data from {filepath}")
    
    df = pd.read_csv(filepath, low_memory=False)
    
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Columns: {list(df.columns)}")
    
    required_columns = ['year', 'hazardtype']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def clean_disaster_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize disaster data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw disaster data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned disaster data
    """
    logger.info("Starting data cleaning process")
    
    df_clean = df.copy()
    
    if 'hazardtype' in df_clean.columns:
        df_clean['hazardtype'] = df_clean['hazardtype'].apply(normalize_hazard_type)
    
    numeric_columns = [
        'deaths', 'injured', 'missing', 'affected', 'displaced',
        'housesdestroyed', 'housesdamaged', 'educationcenters', 
        'healthcenters', 'localvalue', 'usdvalue', 'durationdays',
        'latitude', 'longitude'
    ]
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].clip(lower=0)
    
    if 'year' in df_clean.columns:
        df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
        df_clean = df_clean[
            (df_clean['year'] >= MIN_YEAR) & 
            (df_clean['year'] <= MAX_YEAR)
        ]
    
    if 'month' in df_clean.columns:
        df_clean['month'] = pd.to_numeric(df_clean['month'], errors='coerce')
        df_clean['month'] = df_clean['month'].clip(1, 12)
    
    if 'day' in df_clean.columns:
        df_clean['day'] = pd.to_numeric(df_clean['day'], errors='coerce')
        df_clean['day'] = df_clean['day'].clip(1, 31)
    
    text_columns = ['eventname', 'location', 'level1', 'level2', 'level3', 'eventdescription']
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace(['nan', 'None', ''], np.nan)
    
    logger.info(f"Data cleaning complete. {len(df_clean)} records retained")
    
    return df_clean


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns for analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned disaster data
        
    Returns:
    --------
    pd.DataFrame
        Data with derived columns
    """
    logger.info("Adding derived columns")
    
    df_derived = df.copy()
    
    if all(col in df_derived.columns for col in ['year', 'month', 'day']):
        df_derived['date'] = create_date_column(df_derived)
    elif 'year' in df_derived.columns:
        df_derived['date'] = pd.to_datetime(df_derived['year'], format='%Y', errors='coerce')
    
    if 'month' in df_derived.columns:
        df_derived['season'] = df_derived['month'].apply(
            lambda x: get_season(x) if pd.notna(x) else 'Unknown'
        )
    
    impact_cols = ['injured', 'affected', 'displaced']
    if all(col in df_derived.columns for col in impact_cols):
        df_derived['total_affected'] = df_derived[impact_cols].fillna(0).sum(axis=1)
    
    if 'deaths' in df_derived.columns and 'missing' in df_derived.columns:
        df_derived['total_deaths'] = df_derived['deaths'].fillna(0) + df_derived['missing'].fillna(0)
    elif 'deaths' in df_derived.columns:
        df_derived['total_deaths'] = df_derived['deaths'].fillna(0)
    
    housing_cols = ['housesdestroyed', 'housesdamaged']
    if all(col in df_derived.columns for col in housing_cols):
        df_derived['total_houses_affected'] = df_derived[housing_cols].fillna(0).sum(axis=1)
    
    if 'usdvalue' in df_derived.columns:
        df_derived['has_economic_data'] = df_derived['usdvalue'].notna() & (df_derived['usdvalue'] > 0)
    
    if 'latitude' in df_derived.columns and 'longitude' in df_derived.columns:
        df_derived['has_coordinates'] = (
            df_derived['latitude'].notna() & 
            df_derived['longitude'].notna() &
            (df_derived['latitude'].abs() <= 90) &
            (df_derived['longitude'].abs() <= 180)
        )
    
    logger.info(f"Added derived columns: {[col for col in df_derived.columns if col not in df.columns]}")
    
    return df_derived


def get_season(month: int) -> str:
    """Get season from month (Northern Hemisphere)."""
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
    return 'Unknown'


def merge_with_population_data(disaster_df: pd.DataFrame, 
                                population_df: pd.DataFrame,
                                country_col: str = 'level1',
                                year_col: str = 'year') -> pd.DataFrame:
    """
    Merge disaster data with population data.
    
    Parameters:
    -----------
    disaster_df : pd.DataFrame
        Disaster data
    population_df : pd.DataFrame
        Population data with columns: country, year, population
    country_col : str
        Name of country column in disaster data
    year_col : str
        Name of year column
        
    Returns:
    --------
    pd.DataFrame
        Merged data
    """
    logger.info("Merging disaster data with population data")
    
    if country_col not in disaster_df.columns:
        logger.warning(f"Country column '{country_col}' not found in disaster data")
        return disaster_df
    
    if not all(col in population_df.columns for col in ['country', 'year', 'population']):
        logger.warning("Population data missing required columns")
        return disaster_df
    
    merged = disaster_df.merge(
        population_df,
        left_on=[country_col, year_col],
        right_on=['country', 'year'],
        how='left',
        suffixes=('', '_pop')
    )
    
    if 'country' in merged.columns and 'country' != country_col:
        merged = merged.drop(columns=['country'])
    
    logger.info(f"Merged data: {merged['population'].notna().sum()} records have population data")
    
    return merged


def merge_with_gdp_data(disaster_df: pd.DataFrame, 
                        gdp_df: pd.DataFrame,
                        country_col: str = 'level1',
                        year_col: str = 'year') -> pd.DataFrame:
    """
    Merge disaster data with GDP data.
    
    Parameters:
    -----------
    disaster_df : pd.DataFrame
        Disaster data
    gdp_df : pd.DataFrame
        GDP data with columns: country, year, gdp
    country_col : str
        Name of country column in disaster data
    year_col : str
        Name of year column
        
    Returns:
    --------
    pd.DataFrame
        Merged data
    """
    logger.info("Merging disaster data with GDP data")
    
    if country_col not in disaster_df.columns:
        logger.warning(f"Country column '{country_col}' not found in disaster data")
        return disaster_df
    
    if not all(col in gdp_df.columns for col in ['country', 'year', 'gdp']):
        logger.warning("GDP data missing required columns")
        return disaster_df
    
    merged = disaster_df.merge(
        gdp_df,
        left_on=[country_col, year_col],
        right_on=['country', 'year'],
        how='left',
        suffixes=('', '_gdp')
    )
    
    if 'country' in merged.columns and 'country' != country_col:
        merged = merged.drop(columns=['country'])
    
    logger.info(f"Merged data: {merged['gdp'].notna().sum()} records have GDP data")
    
    return merged


def remove_duplicates(df: pd.DataFrame, 
                      subset: Optional[List[str]] = None,
                      keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate records.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    subset : list, optional
        Columns to consider for identifying duplicates
    keep : str
        Which duplicates to keep ('first', 'last', False)
        
    Returns:
    --------
    pd.DataFrame
        Data with duplicates removed
    """
    initial_count = len(df)
    
    df_dedup = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = initial_count - len(df_dedup)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate records")
    
    return df_dedup


def handle_outliers(df: pd.DataFrame, 
                    columns: List[str],
                    method: str = 'iqr',
                    threshold: float = 3.0) -> pd.DataFrame:
    """
    Handle outliers in specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    columns : list
        Columns to check for outliers
    method : str
        Method to use ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    pd.DataFrame
        Data with outlier flags
    """
    df_out = df.copy()
    
    for col in columns:
        if col not in df_out.columns:
            continue
        
        if method == 'iqr':
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
        
        elif method == 'zscore':
            mean = df_out[col].mean()
            std = df_out[col].std()
            z_scores = np.abs((df_out[col] - mean) / std)
            outliers = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        df_out[f'{col}_outlier'] = outliers
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            logger.info(f"Identified {outlier_count} outliers in {col}")
    
    return df_out


def create_processed_dataset(raw_data_path: str, 
                            output_path: str,
                            population_data: Optional[pd.DataFrame] = None,
                            gdp_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Complete data processing pipeline.
    
    Parameters:
    -----------
    raw_data_path : str
        Path to raw disaster data
    output_path : str
        Path to save processed data
    population_data : pd.DataFrame, optional
        Population data
    gdp_data : pd.DataFrame, optional
        GDP data
        
    Returns:
    --------
    pd.DataFrame
        Processed disaster data
    """
    logger.info("Starting complete data processing pipeline")
    
    df = load_and_validate_disaster_data(raw_data_path)
    
    df = clean_disaster_data(df)
    
    df = add_derived_columns(df)
    
    df = remove_duplicates(df)
    
    if population_data is not None:
        df = merge_with_population_data(df, population_data)
    
    if gdp_data is not None:
        df = merge_with_gdp_data(df, gdp_data)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Processed data saved to {output_path}")
    logger.info(f"Final dataset: {len(df)} records, {len(df.columns)} columns")
    
    return df
