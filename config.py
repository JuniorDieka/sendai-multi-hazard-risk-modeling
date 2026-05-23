"""
Configuration parameters for Multi-Hazard Risk Modeling Project
Aligned with Sendai Framework for Disaster Risk Reduction 2015-2030
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

OUTPUTS_DIR = BASE_DIR / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
REPORTS_DIR = OUTPUTS_DIR / 'reports'
MODELS_DIR = OUTPUTS_DIR / 'models'

NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
SRC_DIR = BASE_DIR / 'src'

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

HAZARD_TYPES = [
    'FLOOD', 'DROUGHT', 'FIRE', 'STORM', 'EARTHQUAKE', 
    'LANDSLIDE', 'HEAVY RAIN', 'TSUNAMI', 'VOLCANIC ACTIVITY',
    'EXTREME TEMPERATURE', 'EPIDEMIC', 'INSECT INFESTATION'
]

SEVERITY_THRESHOLDS = {
    'deaths': {'minor': 0, 'moderate': 10, 'severe': 100, 'catastrophic': 1000},
    'affected': {'minor': 0, 'moderate': 1000, 'severe': 10000, 'catastrophic': 100000},
    'economic_loss_usd': {'minor': 0, 'moderate': 1e6, 'severe': 1e7, 'catastrophic': 1e8}
}

SENDAI_INDICATORS = {
    'A-1': 'Mortality rate per 100,000 population',
    'A-2': 'Number of deaths',
    'A-3': 'Number of missing persons',
    'B-1': 'Affected rate per 100,000 population',
    'B-2': 'Number of injured/sick people',
    'B-3': 'Number of people whose damaged dwellings attributed to disasters',
    'B-4': 'Number of people whose destroyed dwellings attributed to disasters',
    'B-5': 'Number of people whose livelihoods disrupted',
    'C-1': 'Economic loss as % of GDP',
    'C-2': 'Direct agricultural loss',
    'C-3': 'Direct loss to productive assets',
    'C-4': 'Direct loss in critical infrastructure',
    'C-5': 'Direct loss to cultural heritage',
    'D-1': 'Infrastructure damage per 100,000 population',
    'D-2': 'Number of destroyed/damaged health facilities',
    'D-3': 'Number of destroyed/damaged educational facilities'
}

WORLD_BANK_INDICATORS = {
    'population': 'SP.POP.TOTL',
    'gdp': 'NY.GDP.MKTP.CD',
    'gdp_per_capita': 'NY.GDP.PCAP.CD',
    'urban_population': 'SP.URB.TOTL.IN.ZS'
}

COUNTRY_ISO_CODES = {
    'Kenya': 'KEN',
    'Ethiopia': 'ETH',
    'Somalia': 'SOM',
    'Tanzania': 'TZA',
    'Uganda': 'UGA',
    'Mozambique': 'MOZ',
    'Madagascar': 'MDG',
    'South Africa': 'ZAF',
    'Nigeria': 'NGA',
    'Ghana': 'GHA'
}

PLOT_STYLE = {
    'figure_size': (12, 8),
    'dpi': 300,
    'font_size': 12,
    'title_size': 16,
    'color_palette': 'Set2'
}

RANDOM_SEED = 42

API_CACHE_DAYS = 30

MIN_YEAR = 1917
MAX_YEAR = 2025
