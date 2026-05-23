"""
External data fetching module for population and GDP data
Integrates with World Bank API and other data sources
"""

import pandas as pd
import numpy as np
import requests
import wbgapi as wb
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (WORLD_BANK_INDICATORS, COUNTRY_ISO_CODES, 
                   MIN_YEAR, MAX_YEAR, API_CACHE_DAYS, EXTERNAL_DATA_DIR)
from src.utils import logger


class ExternalDataFetcher:
    """
    Fetches and caches external data from World Bank and other sources.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the data fetcher.
        
        Parameters:
        -----------
        cache_dir : Path, optional
            Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or EXTERNAL_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"External data fetcher initialized. Cache dir: {self.cache_dir}")
    
    def _get_cache_path(self, data_type: str, country: str = 'all') -> Path:
        """Get cache file path for a specific data type."""
        return self.cache_dir / f"{data_type}_{country}.csv"
    
    def _is_cache_valid(self, cache_path: Path, max_age_days: int = API_CACHE_DAYS) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age.days < max_age_days
    
    def fetch_population_data(self, countries: Optional[List[str]] = None,
                             start_year: int = MIN_YEAR,
                             end_year: int = MAX_YEAR,
                             use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch population data from World Bank API.
        
        Parameters:
        -----------
        countries : list, optional
            List of country ISO codes. If None, fetches all countries.
        start_year : int
            Start year for data
        end_year : int
            End year for data
        use_cache : bool
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            Population data with columns: country, country_code, year, population
        """
        cache_key = 'all' if countries is None else '_'.join(sorted(countries))
        cache_path = self._get_cache_path('population', cache_key)
        
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"Loading population data from cache: {cache_path}")
            return pd.read_csv(cache_path)
        
        logger.info("Fetching population data from World Bank API...")
        
        try:
            if countries is None:
                countries_to_fetch = 'all'
            else:
                countries_to_fetch = countries
            
            data_records = []
            
            data = wb.data.DataFrame(
                WORLD_BANK_INDICATORS['population'],
                countries_to_fetch,
                time=range(start_year, end_year + 1),
                labels=True
            )
            
            data = data.reset_index()
            
            if 'Country' in data.columns and 'time' in data.columns:
                data = data.rename(columns={
                    'Country': 'country',
                    'time': 'year',
                    WORLD_BANK_INDICATORS['population']: 'population'
                })
                
                if 'economy' in data.columns:
                    data = data.rename(columns={'economy': 'country_code'})
                
                data = data[['country', 'year', 'population']]
                data = data.dropna(subset=['population'])
                
                data.to_csv(cache_path, index=False)
                logger.info(f"Population data cached to {cache_path}")
                
                return data
            
        except Exception as e:
            logger.error(f"Error fetching population data from World Bank: {str(e)}")
            logger.info("Attempting alternative method...")
        
        try:
            import wbdata
            
            data_records = []
            
            countries_list = countries if countries else list(COUNTRY_ISO_CODES.values())
            
            for country_code in countries_list:
                try:
                    country_data = wbdata.get_dataframe(
                        {WORLD_BANK_INDICATORS['population']: 'population'},
                        country=country_code
                    )
                    
                    country_data = country_data.reset_index()
                    country_data['country_code'] = country_code
                    
                    country_name = [k for k, v in COUNTRY_ISO_CODES.items() if v == country_code]
                    country_data['country'] = country_name[0] if country_name else country_code
                    
                    if 'date' in country_data.columns:
                        country_data['year'] = pd.to_datetime(country_data['date']).dt.year
                    
                    data_records.append(country_data)
                    
                except Exception as country_error:
                    logger.warning(f"Could not fetch data for {country_code}: {str(country_error)}")
                    continue
            
            if data_records:
                df = pd.concat(data_records, ignore_index=True)
                df = df[['country', 'country_code', 'year', 'population']]
                df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
                df = df.dropna(subset=['population'])
                
                df.to_csv(cache_path, index=False)
                logger.info(f"Population data cached to {cache_path}")
                
                return df
        
        except Exception as e:
            logger.error(f"Alternative method also failed: {str(e)}")
        
        logger.warning("Creating sample population data for demonstration...")
        return self._create_sample_population_data(countries, start_year, end_year)
    
    def fetch_gdp_data(self, countries: Optional[List[str]] = None,
                       start_year: int = 1960,
                       end_year: int = MAX_YEAR,
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch GDP data from World Bank API.
        
        Parameters:
        -----------
        countries : list, optional
            List of country ISO codes
        start_year : int
            Start year for data (World Bank GDP data typically starts from 1960)
        end_year : int
            End year for data
        use_cache : bool
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            GDP data with columns: country, country_code, year, gdp
        """
        cache_key = 'all' if countries is None else '_'.join(sorted(countries))
        cache_path = self._get_cache_path('gdp', cache_key)
        
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"Loading GDP data from cache: {cache_path}")
            return pd.read_csv(cache_path)
        
        logger.info("Fetching GDP data from World Bank API...")
        
        try:
            if countries is None:
                countries_to_fetch = 'all'
            else:
                countries_to_fetch = countries
            
            data = wb.data.DataFrame(
                WORLD_BANK_INDICATORS['gdp'],
                countries_to_fetch,
                time=range(start_year, end_year + 1),
                labels=True
            )
            
            data = data.reset_index()
            
            if 'Country' in data.columns and 'time' in data.columns:
                data = data.rename(columns={
                    'Country': 'country',
                    'time': 'year',
                    WORLD_BANK_INDICATORS['gdp']: 'gdp'
                })
                
                if 'economy' in data.columns:
                    data = data.rename(columns={'economy': 'country_code'})
                
                data = data[['country', 'year', 'gdp']]
                data = data.dropna(subset=['gdp'])
                
                data.to_csv(cache_path, index=False)
                logger.info(f"GDP data cached to {cache_path}")
                
                return data
            
        except Exception as e:
            logger.error(f"Error fetching GDP data from World Bank: {str(e)}")
            logger.info("Attempting alternative method...")
        
        try:
            import wbdata
            
            data_records = []
            
            countries_list = countries if countries else list(COUNTRY_ISO_CODES.values())
            
            for country_code in countries_list:
                try:
                    country_data = wbdata.get_dataframe(
                        {WORLD_BANK_INDICATORS['gdp']: 'gdp'},
                        country=country_code
                    )
                    
                    country_data = country_data.reset_index()
                    country_data['country_code'] = country_code
                    
                    country_name = [k for k, v in COUNTRY_ISO_CODES.items() if v == country_code]
                    country_data['country'] = country_name[0] if country_name else country_code
                    
                    if 'date' in country_data.columns:
                        country_data['year'] = pd.to_datetime(country_data['date']).dt.year
                    
                    data_records.append(country_data)
                    
                except Exception as country_error:
                    logger.warning(f"Could not fetch GDP for {country_code}: {str(country_error)}")
                    continue
            
            if data_records:
                df = pd.concat(data_records, ignore_index=True)
                df = df[['country', 'country_code', 'year', 'gdp']]
                df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
                df = df.dropna(subset=['gdp'])
                
                df.to_csv(cache_path, index=False)
                logger.info(f"GDP data cached to {cache_path}")
                
                return df
        
        except Exception as e:
            logger.error(f"Alternative method also failed: {str(e)}")
        
        logger.warning("Creating sample GDP data for demonstration...")
        return self._create_sample_gdp_data(countries, start_year, end_year)
    
    def _create_sample_population_data(self, countries: Optional[List[str]],
                                      start_year: int, end_year: int) -> pd.DataFrame:
        """Create sample population data for demonstration purposes."""
        logger.info("Creating sample population data...")
        
        sample_countries = {
            'Bangladesh': {'code': 'BGD', 'base_pop': 150000000, 'growth': 0.012},
            'India': {'code': 'IND', 'base_pop': 1200000000, 'growth': 0.011},
            'Indonesia': {'code': 'IDN', 'base_pop': 250000000, 'growth': 0.010},
            'Nepal': {'code': 'NPL', 'base_pop': 28000000, 'growth': 0.013},
            'Pakistan': {'code': 'PAK', 'base_pop': 200000000, 'growth': 0.020},
            'Philippines': {'code': 'PHL', 'base_pop': 100000000, 'growth': 0.015},
        }
        
        data_records = []
        base_year = 2020
        
        for country, info in sample_countries.items():
            if countries and info['code'] not in countries:
                continue
            
            for year in range(start_year, end_year + 1):
                years_diff = year - base_year
                population = info['base_pop'] * ((1 + info['growth']) ** years_diff)
                
                data_records.append({
                    'country': country,
                    'country_code': info['code'],
                    'year': year,
                    'population': population
                })
        
        df = pd.DataFrame(data_records)
        cache_path = self._get_cache_path('population', 'sample')
        df.to_csv(cache_path, index=False)
        
        return df
    
    def _create_sample_gdp_data(self, countries: Optional[List[str]],
                               start_year: int, end_year: int) -> pd.DataFrame:
        """Create sample GDP data for demonstration purposes."""
        logger.info("Creating sample GDP data...")
        
        sample_countries = {
            'Bangladesh': {'code': 'BGD', 'base_gdp': 300e9, 'growth': 0.065},
            'India': {'code': 'IND', 'base_gdp': 2700e9, 'growth': 0.070},
            'Indonesia': {'code': 'IDN', 'base_gdp': 1000e9, 'growth': 0.055},
            'Nepal': {'code': 'NPL', 'base_gdp': 30e9, 'growth': 0.045},
            'Pakistan': {'code': 'PAK', 'base_gdp': 280e9, 'growth': 0.040},
            'Philippines': {'code': 'PHL', 'base_gdp': 350e9, 'growth': 0.060},
        }
        
        data_records = []
        base_year = 2020
        
        for country, info in sample_countries.items():
            if countries and info['code'] not in countries:
                continue
            
            for year in range(max(start_year, 1960), end_year + 1):
                years_diff = year - base_year
                gdp = info['base_gdp'] * ((1 + info['growth']) ** years_diff)
                
                data_records.append({
                    'country': country,
                    'country_code': info['code'],
                    'year': year,
                    'gdp': gdp
                })
        
        df = pd.DataFrame(data_records)
        cache_path = self._get_cache_path('gdp', 'sample')
        df.to_csv(cache_path, index=False)
        
        return df
    
    def interpolate_missing_years(self, df: pd.DataFrame,
                                 value_col: str,
                                 year_col: str = 'year',
                                 group_col: str = 'country') -> pd.DataFrame:
        """
        Interpolate missing years in time series data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with potential gaps
        value_col : str
            Column to interpolate
        year_col : str
            Year column name
        group_col : str
            Grouping column (e.g., country)
            
        Returns:
        --------
        pd.DataFrame
            Data with interpolated values
        """
        logger.info(f"Interpolating missing values in {value_col}...")
        
        interpolated_dfs = []
        
        for group_name, group_data in df.groupby(group_col):
            group_data = group_data.sort_values(year_col)
            
            year_range = range(group_data[year_col].min(), group_data[year_col].max() + 1)
            complete_years = pd.DataFrame({year_col: list(year_range)})
            
            merged = complete_years.merge(group_data, on=year_col, how='left')
            merged[group_col] = group_name
            
            merged[value_col] = merged[value_col].interpolate(method='linear')
            
            interpolated_dfs.append(merged)
        
        result = pd.concat(interpolated_dfs, ignore_index=True)
        
        return result
    
    def fetch_all_external_data(self, countries: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all external data sources.
        
        Parameters:
        -----------
        countries : list, optional
            List of country ISO codes
            
        Returns:
        --------
        dict
            Dictionary with 'population' and 'gdp' DataFrames
        """
        logger.info("Fetching all external data sources...")
        
        population_data = self.fetch_population_data(countries)
        gdp_data = self.fetch_gdp_data(countries)
        
        return {
            'population': population_data,
            'gdp': gdp_data
        }


def prepare_external_data_for_merge(population_df: pd.DataFrame,
                                   gdp_df: pd.DataFrame,
                                   country_mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare external data for merging with disaster data.
    
    Parameters:
    -----------
    population_df : pd.DataFrame
        Population data
    gdp_df : pd.DataFrame
        GDP data
    country_mapping : dict, optional
        Mapping from disaster data country names to ISO codes
        
    Returns:
    --------
    tuple
        Prepared (population_df, gdp_df)
    """
    if country_mapping is None:
        country_mapping = COUNTRY_ISO_CODES
    
    reverse_mapping = {v: k for k, v in country_mapping.items()}
    
    pop_prepared = population_df.copy()
    if 'country_code' in pop_prepared.columns:
        pop_prepared['country'] = pop_prepared['country_code'].map(reverse_mapping).fillna(pop_prepared['country'])
    
    gdp_prepared = gdp_df.copy()
    if 'country_code' in gdp_prepared.columns:
        gdp_prepared['country'] = gdp_prepared['country_code'].map(reverse_mapping).fillna(gdp_prepared['country'])
    
    return pop_prepared, gdp_prepared
