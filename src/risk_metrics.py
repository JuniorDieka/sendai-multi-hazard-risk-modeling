"""
Risk metrics and Sendai Framework indicator calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import SENDAI_INDICATORS, SEVERITY_THRESHOLDS
from src.utils import logger, classify_severity


def calculate_sendai_a1(deaths: float, missing: float, population: float) -> float:
    """
    Calculate Sendai Indicator A-1: Mortality rate per 100,000 population.
    
    Parameters:
    -----------
    deaths : float
        Number of deaths
    missing : float
        Number of missing persons
    population : float
        Total population
        
    Returns:
    --------
    float
        Mortality rate per 100,000
    """
    if pd.isna(population) or population == 0:
        return np.nan
    
    total_mortality = (deaths or 0) + (missing or 0)
    return (total_mortality / population) * 100000


def calculate_sendai_b1(injured: float, affected: float, displaced: float, 
                        population: float) -> float:
    """
    Calculate Sendai Indicator B-1: Affected rate per 100,000 population.
    
    Parameters:
    -----------
    injured : float
        Number of injured people
    affected : float
        Number of affected people
    displaced : float
        Number of displaced people
    population : float
        Total population
        
    Returns:
    --------
    float
        Affected rate per 100,000
    """
    if pd.isna(population) or population == 0:
        return np.nan
    
    total_affected = (injured or 0) + (affected or 0) + (displaced or 0)
    return (total_affected / population) * 100000


def calculate_sendai_c1(economic_loss: float, gdp: float) -> float:
    """
    Calculate Sendai Indicator C-1: Economic loss as % of GDP.
    
    Parameters:
    -----------
    economic_loss : float
        Direct economic loss in USD
    gdp : float
        GDP in USD
        
    Returns:
    --------
    float
        Economic loss as percentage of GDP
    """
    if pd.isna(gdp) or gdp == 0:
        return np.nan
    
    if pd.isna(economic_loss):
        return np.nan
    
    return (economic_loss / gdp) * 100


def calculate_sendai_d1(health_facilities: float, education_facilities: float,
                        population: float) -> float:
    """
    Calculate Sendai Indicator D-1: Infrastructure damage per 100,000 population.
    
    Parameters:
    -----------
    health_facilities : float
        Number of damaged/destroyed health facilities
    education_facilities : float
        Number of damaged/destroyed education facilities
    population : float
        Total population
        
    Returns:
    --------
    float
        Infrastructure damage per 100,000
    """
    if pd.isna(population) or population == 0:
        return np.nan
    
    total_infrastructure = (health_facilities or 0) + (education_facilities or 0)
    return (total_infrastructure / population) * 100000


def add_sendai_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all Sendai Framework indicators to disaster dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data with population and GDP
        
    Returns:
    --------
    pd.DataFrame
        Data with Sendai indicators
    """
    logger.info("Calculating Sendai Framework indicators")
    
    df_sendai = df.copy()
    
    if all(col in df_sendai.columns for col in ['deaths', 'missing', 'population']):
        df_sendai['sendai_a1_mortality_rate'] = df_sendai.apply(
            lambda row: calculate_sendai_a1(
                row.get('deaths', 0), 
                row.get('missing', 0), 
                row.get('population')
            ), axis=1
        )
    
    df_sendai['sendai_a2_deaths'] = df_sendai.get('deaths', 0)
    df_sendai['sendai_a3_missing'] = df_sendai.get('missing', 0)
    
    if all(col in df_sendai.columns for col in ['injured', 'affected', 'displaced', 'population']):
        df_sendai['sendai_b1_affected_rate'] = df_sendai.apply(
            lambda row: calculate_sendai_b1(
                row.get('injured', 0),
                row.get('affected', 0),
                row.get('displaced', 0),
                row.get('population')
            ), axis=1
        )
    
    df_sendai['sendai_b2_injured'] = df_sendai.get('injured', 0)
    df_sendai['sendai_b3_damaged_dwellings'] = df_sendai.get('housesdamaged', 0)
    df_sendai['sendai_b4_destroyed_dwellings'] = df_sendai.get('housesdestroyed', 0)
    df_sendai['sendai_b5_livelihoods_disrupted'] = df_sendai.get('affected', 0)
    
    if all(col in df_sendai.columns for col in ['usdvalue', 'gdp']):
        df_sendai['sendai_c1_economic_loss_pct_gdp'] = df_sendai.apply(
            lambda row: calculate_sendai_c1(
                row.get('usdvalue'),
                row.get('gdp')
            ), axis=1
        )
    
    if all(col in df_sendai.columns for col in ['healthcenters', 'educationcenters', 'population']):
        df_sendai['sendai_d1_infrastructure_damage_rate'] = df_sendai.apply(
            lambda row: calculate_sendai_d1(
                row.get('healthcenters', 0),
                row.get('educationcenters', 0),
                row.get('population')
            ), axis=1
        )
    
    df_sendai['sendai_d2_health_facilities'] = df_sendai.get('healthcenters', 0)
    df_sendai['sendai_d3_education_facilities'] = df_sendai.get('educationcenters', 0)
    
    sendai_cols = [col for col in df_sendai.columns if col.startswith('sendai_')]
    logger.info(f"Added {len(sendai_cols)} Sendai indicators: {sendai_cols}")
    
    return df_sendai


def calculate_hazard_exposure_index(df: pd.DataFrame, 
                                    location_col: str = 'level2') -> pd.DataFrame:
    """
    Calculate hazard exposure index by location.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    location_col : str
        Column name for location grouping
        
    Returns:
    --------
    pd.DataFrame
        Exposure index by location
    """
    if location_col not in df.columns:
        logger.warning(f"Location column '{location_col}' not found")
        return pd.DataFrame()
    
    exposure = df.groupby(location_col).agg({
        'serial': 'count',
        'hazardtype': lambda x: x.nunique(),
        'deaths': 'sum',
        'total_affected': 'sum',
        'usdvalue': 'sum'
    }).rename(columns={
        'serial': 'event_count',
        'hazardtype': 'hazard_diversity',
        'deaths': 'total_deaths',
        'total_affected': 'total_affected',
        'usdvalue': 'total_economic_loss'
    })
    
    for col in ['event_count', 'hazard_diversity', 'total_deaths', 
                'total_affected', 'total_economic_loss']:
        if col in exposure.columns:
            exposure[f'{col}_normalized'] = (
                (exposure[col] - exposure[col].min()) / 
                (exposure[col].max() - exposure[col].min())
            )
    
    normalized_cols = [col for col in exposure.columns if col.endswith('_normalized')]
    if normalized_cols:
        exposure['exposure_index'] = exposure[normalized_cols].mean(axis=1)
    
    return exposure.sort_values('exposure_index', ascending=False)


def calculate_vulnerability_index(df: pd.DataFrame,
                                  location_col: str = 'level2') -> pd.DataFrame:
    """
    Calculate vulnerability index by location.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data with Sendai indicators
    location_col : str
        Column name for location grouping
        
    Returns:
    --------
    pd.DataFrame
        Vulnerability index by location
    """
    if location_col not in df.columns:
        logger.warning(f"Location column '{location_col}' not found")
        return pd.DataFrame()
    
    vulnerability_metrics = []
    
    if 'sendai_a1_mortality_rate' in df.columns:
        vulnerability_metrics.append('sendai_a1_mortality_rate')
    if 'sendai_b1_affected_rate' in df.columns:
        vulnerability_metrics.append('sendai_b1_affected_rate')
    if 'sendai_c1_economic_loss_pct_gdp' in df.columns:
        vulnerability_metrics.append('sendai_c1_economic_loss_pct_gdp')
    
    if not vulnerability_metrics:
        logger.warning("No vulnerability metrics available")
        return pd.DataFrame()
    
    vulnerability = df.groupby(location_col)[vulnerability_metrics].mean()
    
    for col in vulnerability_metrics:
        vulnerability[f'{col}_normalized'] = (
            (vulnerability[col] - vulnerability[col].min()) / 
            (vulnerability[col].max() - vulnerability[col].min())
        )
    
    normalized_cols = [col for col in vulnerability.columns if col.endswith('_normalized')]
    if normalized_cols:
        vulnerability['vulnerability_index'] = vulnerability[normalized_cols].mean(axis=1)
    
    return vulnerability.sort_values('vulnerability_index', ascending=False)


def calculate_risk_score(exposure_index: float, vulnerability_index: float,
                        weight_exposure: float = 0.5,
                        weight_vulnerability: float = 0.5) -> float:
    """
    Calculate composite risk score.
    
    Parameters:
    -----------
    exposure_index : float
        Hazard exposure index (0-1)
    vulnerability_index : float
        Vulnerability index (0-1)
    weight_exposure : float
        Weight for exposure component
    weight_vulnerability : float
        Weight for vulnerability component
        
    Returns:
    --------
    float
        Risk score (0-1)
    """
    if pd.isna(exposure_index) or pd.isna(vulnerability_index):
        return np.nan
    
    risk = (weight_exposure * exposure_index + 
            weight_vulnerability * vulnerability_index)
    
    return risk


def calculate_multi_hazard_risk(df: pd.DataFrame,
                                location_col: str = 'level2') -> pd.DataFrame:
    """
    Calculate multi-hazard risk assessment by location.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data with Sendai indicators
    location_col : str
        Column name for location grouping
        
    Returns:
    --------
    pd.DataFrame
        Multi-hazard risk assessment
    """
    logger.info("Calculating multi-hazard risk assessment")
    
    exposure = calculate_hazard_exposure_index(df, location_col)
    vulnerability = calculate_vulnerability_index(df, location_col)
    
    risk = exposure[['exposure_index']].join(
        vulnerability[['vulnerability_index']], 
        how='outer'
    )
    
    risk['risk_score'] = risk.apply(
        lambda row: calculate_risk_score(
            row['exposure_index'], 
            row['vulnerability_index']
        ), axis=1
    )
    
    risk['risk_category'] = pd.cut(
        risk['risk_score'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    
    risk = risk.join(exposure[['event_count', 'hazard_diversity']])
    
    return risk.sort_values('risk_score', ascending=False)


def calculate_severity_index(deaths: float = 0, affected: float = 0, 
                            damage: float = 0) -> float:
    """
    Calculate composite severity index for a disaster event.
    
    Parameters:
    -----------
    deaths : float
        Number of deaths
    affected : float
        Number of people affected
    damage : float
        Economic damage in USD
        
    Returns:
    --------
    float
        Severity index (0-100)
    """
    # Normalize each component (log scale for large numbers)
    death_score = np.log10(deaths + 1) * 10 if deaths > 0 else 0
    affected_score = np.log10(affected + 1) * 5 if affected > 0 else 0
    damage_score = np.log10(damage + 1) * 3 if damage > 0 else 0
    
    # Weighted combination
    severity = (death_score * 0.5 + affected_score * 0.3 + damage_score * 0.2)
    
    # Normalize to 0-100 scale
    return min(severity, 100)


def classify_event_severity(df: pd.DataFrame, severity_col: str = 'severity_index') -> pd.DataFrame:
    """
    Classify event severity based on severity index.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing severity index
    severity_col : str
        Name of the severity index column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'overall_severity' column
    """
    df_result = df.copy()
    
    def _classify_single(severity_index: float) -> str:
        """Helper function to classify a single severity value."""
        if pd.isna(severity_index):
            return 'No Impact'
        elif severity_index < 10:
            return 'Minor'
        elif severity_index < 30:
            return 'Moderate'
        elif severity_index < 60:
            return 'Severe'
        else:
            return 'Catastrophic'
    
    # Apply classification to each row
    if severity_col in df_result.columns:
        df_result['overall_severity'] = df_result[severity_col].apply(_classify_single)
    else:
        logger.warning(f"Column '{severity_col}' not found in DataFrame")
        df_result['overall_severity'] = 'No Impact'
    
    return df_result


def classify_risk_level(risk_score: float) -> str:
    """
    Classify risk level based on risk score.
    
    Parameters:
    -----------
    risk_score : float
        Risk score value (normalized 0-100)
        
    Returns:
    --------
    str
        Risk level classification
    """
    if risk_score < 20:
        return 'Low'
    elif risk_score < 40:
        return 'Medium'
    elif risk_score < 60:
        return 'High'
    elif risk_score < 80:
        return 'Very High'
    else:
        return 'Critical'


def classify_event_severity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify event severity based on multiple impact dimensions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
        
    Returns:
    --------
    pd.DataFrame
        Data with severity classifications
    """
    logger.info("Classifying event severity")
    
    df_severity = df.copy()
    
    if 'total_deaths' in df_severity.columns:
        df_severity['severity_deaths'] = df_severity['total_deaths'].apply(
            lambda x: classify_severity(x, SEVERITY_THRESHOLDS['deaths'])
        )
    
    if 'total_affected' in df_severity.columns:
        df_severity['severity_affected'] = df_severity['total_affected'].apply(
            lambda x: classify_severity(x, SEVERITY_THRESHOLDS['affected'])
        )
    
    if 'usdvalue' in df_severity.columns:
        df_severity['severity_economic'] = df_severity['usdvalue'].apply(
            lambda x: classify_severity(x, SEVERITY_THRESHOLDS['economic_loss_usd'])
        )
    
    severity_cols = [col for col in df_severity.columns if col.startswith('severity_')]
    
    if severity_cols:
        severity_mapping = {
            'No Impact': 0,
            'Minor': 1,
            'Moderate': 2,
            'Severe': 3,
            'Catastrophic': 4
        }
        
        severity_numeric = df_severity[severity_cols].applymap(
            lambda x: severity_mapping.get(x, 0)
        )
        
        df_severity['overall_severity_score'] = severity_numeric.max(axis=1)
        
        reverse_mapping = {v: k for k, v in severity_mapping.items()}
        df_severity['overall_severity'] = df_severity['overall_severity_score'].map(reverse_mapping)
    
    return df_severity


def calculate_return_period(df: pd.DataFrame, 
                           hazard_type: str,
                           threshold_col: str,
                           threshold_value: float) -> float:
    """
    Calculate return period for events exceeding a threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    hazard_type : str
        Hazard type to analyze
    threshold_col : str
        Column to apply threshold to
    threshold_value : float
        Threshold value
        
    Returns:
    --------
    float
        Return period in years
    """
    hazard_data = df[df['hazardtype'] == hazard_type].copy()
    
    if len(hazard_data) == 0:
        return np.nan
    
    if threshold_col not in hazard_data.columns:
        return np.nan
    
    exceeding_events = hazard_data[hazard_data[threshold_col] >= threshold_value]
    
    if len(exceeding_events) == 0:
        return np.inf
    
    years_covered = hazard_data['year'].max() - hazard_data['year'].min() + 1
    
    return years_covered / len(exceeding_events)


def calculate_risk_trends(df: pd.DataFrame, 
                         window_years: int = 5) -> pd.DataFrame:
    """
    Calculate risk trends over time using rolling windows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data with date column
    window_years : int
        Size of rolling window in years
        
    Returns:
    --------
    pd.DataFrame
        Risk trends by year
    """
    if 'year' not in df.columns:
        logger.warning("Year column not found")
        return pd.DataFrame()
    
    trends = df.groupby('year').agg({
        'serial': 'count',
        'deaths': 'sum',
        'total_affected': 'sum',
        'usdvalue': 'sum'
    }).rename(columns={
        'serial': 'event_count',
        'deaths': 'total_deaths',
        'total_affected': 'total_affected',
        'usdvalue': 'total_economic_loss'
    })
    
    for col in ['event_count', 'total_deaths', 'total_affected', 'total_economic_loss']:
        if col in trends.columns:
            trends[f'{col}_rolling_avg'] = trends[col].rolling(
                window=window_years, 
                min_periods=1
            ).mean()
    
    return trends


def prioritize_locations_for_intervention(risk_df: pd.DataFrame,
                                         top_n: int = 10) -> pd.DataFrame:
    """
    Prioritize locations for disaster risk reduction interventions.
    
    Parameters:
    -----------
    risk_df : pd.DataFrame
        Multi-hazard risk assessment data
    top_n : int
        Number of top priority locations
        
    Returns:
    --------
    pd.DataFrame
        Prioritized locations
    """
    if 'risk_score' not in risk_df.columns:
        logger.warning("Risk score not found in data")
        return pd.DataFrame()
    
    priority = risk_df.copy()
    
    priority['priority_rank'] = priority['risk_score'].rank(ascending=False)
    
    priority = priority.sort_values('priority_rank').head(top_n)
    
    return priority
