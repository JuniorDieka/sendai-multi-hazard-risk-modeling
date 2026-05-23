"""
Visualization functions for disaster risk analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PLOT_STYLE, HAZARD_TYPES
from src.utils import logger

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = PLOT_STYLE['figure_size']
plt.rcParams['figure.dpi'] = PLOT_STYLE['dpi']
plt.rcParams['font.size'] = PLOT_STYLE['font_size']


def plot_temporal_trends(df: pd.DataFrame, 
                        date_col: str = 'year',
                        value_col: str = 'serial',
                        title: str = 'Disaster Events Over Time',
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot temporal trends in disaster events.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    date_col : str
        Date column name
    value_col : str
        Value column to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if value_col == 'serial':
        trend_data = df.groupby(date_col).size().reset_index(name='count')
        y_col = 'count'
        y_label = 'Number of Events'
    else:
        trend_data = df.groupby(date_col)[value_col].sum().reset_index()
        y_col = value_col
        y_label = value_col.replace('_', ' ').title()
    
    ax.plot(trend_data[date_col], trend_data[y_col], marker='o', linewidth=2, markersize=4)
    
    if len(trend_data) > 1:
        z = np.polyfit(range(len(trend_data)), trend_data[y_col], 1)
        p = np.poly1d(z)
        ax.plot(trend_data[date_col], p(range(len(trend_data))), 
                "r--", alpha=0.7, linewidth=2, label='Trend Line')
        ax.legend()
    
    ax.set_xlabel(date_col.title(), fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_hazard_frequency(df: pd.DataFrame,
                         hazard_col: str = 'hazardtype',
                         title: str = 'Disaster Frequency by Hazard Type',
                         top_n: Optional[int] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot frequency of different hazard types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    hazard_col : str
        Hazard type column name
    title : str
        Plot title
    top_n : int, optional
        Show only top N hazards
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    hazard_counts = df[hazard_col].value_counts()
    
    if top_n:
        hazard_counts = hazard_counts.head(top_n)
    
    colors = sns.color_palette(PLOT_STYLE['color_palette'], len(hazard_counts))
    hazard_counts.plot(kind='bar', ax=ax, color=colors)
    
    ax.set_xlabel('Hazard Type', fontsize=14)
    ax.set_ylabel('Number of Events', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(hazard_counts.values):
        ax.text(i, v + max(hazard_counts.values) * 0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_impact_distribution(df: pd.DataFrame,
                            impact_col: str = 'total_deaths',
                            hazard_col: str = 'hazardtype',
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of impacts by hazard type using box plots.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    impact_col : str
        Impact column to plot
    hazard_col : str
        Hazard type column
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df_plot = df[df[impact_col] > 0].copy()
    
    if len(df_plot) == 0:
        logger.warning(f"No data available for {impact_col}")
        return fig
    
    hazard_order = df_plot.groupby(hazard_col)[impact_col].median().sort_values(ascending=False).index
    
    sns.boxplot(data=df_plot, x=hazard_col, y=impact_col, 
                order=hazard_order, palette=PLOT_STYLE['color_palette'], ax=ax)
    
    ax.set_yscale('log')
    ax.set_xlabel('Hazard Type', fontsize=14)
    ax.set_ylabel(impact_col.replace('_', ' ').title() + ' (log scale)', fontsize=14)
    
    if title is None:
        title = f'{impact_col.replace("_", " ").title()} Distribution by Hazard Type'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_seasonal_heatmap(df: pd.DataFrame,
                         month_col: str = 'month',
                         hazard_col: str = 'hazardtype',
                         title: str = 'Seasonal Pattern of Disasters',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot heatmap of disaster frequency by month and hazard type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    month_col : str
        Month column name
    hazard_col : str
        Hazard type column
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_clean = df[[month_col, hazard_col]].dropna()
    
    pivot_table = pd.crosstab(df_clean[hazard_col], df_clean[month_col])
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_table.columns = [month_names[int(m)-1] if m <= 12 else str(m) 
                          for m in pivot_table.columns]
    
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Events'}, ax=ax)
    
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Hazard Type', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_geographic_distribution(df: pd.DataFrame,
                                 location_col: str = 'level2',
                                 value_col: str = 'serial',
                                 title: str = 'Geographic Distribution of Disasters',
                                 top_n: int = 20,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot geographic distribution of disasters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    location_col : str
        Location column name
    value_col : str
        Value column to aggregate
    title : str
        Plot title
    top_n : int
        Number of top locations to show
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if value_col == 'serial':
        geo_data = df.groupby(location_col).size().reset_index(name='count')
        y_col = 'count'
        y_label = 'Number of Events'
    else:
        geo_data = df.groupby(location_col)[value_col].sum().reset_index()
        y_col = value_col
        y_label = value_col.replace('_', ' ').title()
    
    geo_data = geo_data.sort_values(y_col, ascending=True).tail(top_n)
    
    colors = sns.color_palette(PLOT_STYLE['color_palette'], len(geo_data))
    ax.barh(geo_data[location_col], geo_data[y_col], color=colors)
    
    ax.set_xlabel(y_label, fontsize=14)
    ax.set_ylabel('Location', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    for i, v in enumerate(geo_data[y_col].values):
        ax.text(v + max(geo_data[y_col].values) * 0.01, i, str(int(v)), 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def create_interactive_time_series(df: pd.DataFrame,
                                   date_col: str = 'year',
                                   hazard_col: str = 'hazardtype',
                                   title: str = 'Interactive Disaster Timeline') -> go.Figure:
    """
    Create interactive time series plot with Plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    date_col : str
        Date column name
    hazard_col : str
        Hazard type column
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure
    """
    trend_data = df.groupby([date_col, hazard_col]).size().reset_index(name='count')
    
    fig = px.line(trend_data, x=date_col, y='count', color=hazard_col,
                  title=title, labels={'count': 'Number of Events', date_col: date_col.title()})
    
    fig.update_layout(
        hovermode='x unified',
        legend_title_text='Hazard Type',
        font=dict(size=12),
        title_font_size=16
    )
    
    return fig


def create_interactive_map(df: pd.DataFrame,
                          lat_col: str = 'latitude',
                          lon_col: str = 'longitude',
                          popup_cols: Optional[List[str]] = None,
                          cluster: bool = True) -> folium.Map:
    """
    Create interactive map of disaster locations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data with coordinates
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    popup_cols : list, optional
        Columns to include in popup
    cluster : bool
        Whether to cluster markers
        
    Returns:
    --------
    folium.Map
        Folium map object
    """
    df_map = df[[lat_col, lon_col]].dropna()
    
    if len(df_map) == 0:
        logger.warning("No coordinate data available for mapping")
        center_lat, center_lon = 0, 0
    else:
        center_lat = df_map[lat_col].mean()
        center_lon = df_map[lon_col].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    
    if cluster:
        marker_cluster = plugins.MarkerCluster()
    
    for idx, row in df.iterrows():
        if pd.notna(row.get(lat_col)) and pd.notna(row.get(lon_col)):
            popup_text = f"Event: {row.get('eventname', 'Unknown')}<br>"
            popup_text += f"Hazard: {row.get('hazardtype', 'Unknown')}<br>"
            popup_text += f"Date: {row.get('year', 'Unknown')}<br>"
            
            if popup_cols:
                for col in popup_cols:
                    if col in row and pd.notna(row[col]):
                        popup_text += f"{col}: {row[col]}<br>"
            
            marker = folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign')
            )
            
            if cluster:
                marker.add_to(marker_cluster)
            else:
                marker.add_to(m)
    
    if cluster:
        marker_cluster.add_to(m)
    
    return m


def plot_sendai_indicators_comparison(df: pd.DataFrame,
                                     location_col: str = 'level1',
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of raw counts vs normalized Sendai indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with Sendai indicators
    location_col : str
        Location column for grouping
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if 'sendai_a1_mortality_rate' in df.columns:
        data_a = df.groupby(location_col).agg({
            'deaths': 'sum',
            'sendai_a1_mortality_rate': 'mean'
        }).sort_values('sendai_a1_mortality_rate', ascending=False).head(10)
        
        ax1 = axes[0, 0]
        x = range(len(data_a))
        width = 0.35
        
        ax1_twin = ax1.twinx()
        ax1.bar([i - width/2 for i in x], data_a['deaths'], width, 
                label='Raw Deaths', color='steelblue', alpha=0.7)
        ax1_twin.bar([i + width/2 for i in x], data_a['sendai_a1_mortality_rate'], width,
                     label='Mortality Rate per 100k', color='coral', alpha=0.7)
        
        ax1.set_xlabel('Location', fontsize=12)
        ax1.set_ylabel('Total Deaths', fontsize=12, color='steelblue')
        ax1_twin.set_ylabel('Mortality Rate per 100k', fontsize=12, color='coral')
        ax1.set_title('Target A: Mortality - Raw vs Normalized', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(data_a.index, rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1_twin.tick_params(axis='y', labelcolor='coral')
    
    if 'sendai_b1_affected_rate' in df.columns:
        data_b = df.groupby(location_col).agg({
            'total_affected': 'sum',
            'sendai_b1_affected_rate': 'mean'
        }).sort_values('sendai_b1_affected_rate', ascending=False).head(10)
        
        ax2 = axes[0, 1]
        x = range(len(data_b))
        
        ax2_twin = ax2.twinx()
        ax2.bar([i - width/2 for i in x], data_b['total_affected'], width,
                label='Raw Affected', color='steelblue', alpha=0.7)
        ax2_twin.bar([i + width/2 for i in x], data_b['sendai_b1_affected_rate'], width,
                     label='Affected Rate per 100k', color='coral', alpha=0.7)
        
        ax2.set_xlabel('Location', fontsize=12)
        ax2.set_ylabel('Total Affected', fontsize=12, color='steelblue')
        ax2_twin.set_ylabel('Affected Rate per 100k', fontsize=12, color='coral')
        ax2.set_title('Target B: Affected People - Raw vs Normalized', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(data_b.index, rotation=45, ha='right')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        ax2_twin.tick_params(axis='y', labelcolor='coral')
    
    if 'sendai_c1_economic_loss_pct_gdp' in df.columns:
        data_c = df.groupby(location_col).agg({
            'usdvalue': 'sum',
            'sendai_c1_economic_loss_pct_gdp': 'mean'
        }).sort_values('sendai_c1_economic_loss_pct_gdp', ascending=False).head(10)
        
        ax3 = axes[1, 0]
        x = range(len(data_c))
        
        ax3_twin = ax3.twinx()
        ax3.bar([i - width/2 for i in x], data_c['usdvalue']/1e6, width,
                label='Raw Loss (Million USD)', color='steelblue', alpha=0.7)
        ax3_twin.bar([i + width/2 for i in x], data_c['sendai_c1_economic_loss_pct_gdp'], width,
                     label='Loss as % of GDP', color='coral', alpha=0.7)
        
        ax3.set_xlabel('Location', fontsize=12)
        ax3.set_ylabel('Economic Loss (Million USD)', fontsize=12, color='steelblue')
        ax3_twin.set_ylabel('Loss as % of GDP', fontsize=12, color='coral')
        ax3.set_title('Target C: Economic Loss - Raw vs Normalized', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(data_c.index, rotation=45, ha='right')
        ax3.tick_params(axis='y', labelcolor='steelblue')
        ax3_twin.tick_params(axis='y', labelcolor='coral')
    
    if 'sendai_d1_infrastructure_damage_rate' in df.columns:
        data_d = df.groupby(location_col).agg({
            'healthcenters': 'sum',
            'educationcenters': 'sum',
            'sendai_d1_infrastructure_damage_rate': 'mean'
        }).sort_values('sendai_d1_infrastructure_damage_rate', ascending=False).head(10)
        
        data_d['total_infrastructure'] = data_d['healthcenters'] + data_d['educationcenters']
        
        ax4 = axes[1, 1]
        x = range(len(data_d))
        
        ax4_twin = ax4.twinx()
        ax4.bar([i - width/2 for i in x], data_d['total_infrastructure'], width,
                label='Raw Infrastructure', color='steelblue', alpha=0.7)
        ax4_twin.bar([i + width/2 for i in x], data_d['sendai_d1_infrastructure_damage_rate'], width,
                     label='Damage Rate per 100k', color='coral', alpha=0.7)
        
        ax4.set_xlabel('Location', fontsize=12)
        ax4.set_ylabel('Total Facilities Damaged', fontsize=12, color='steelblue')
        ax4_twin.set_ylabel('Damage Rate per 100k', fontsize=12, color='coral')
        ax4.set_title('Target D: Infrastructure - Raw vs Normalized', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(data_d.index, rotation=45, ha='right')
        ax4.tick_params(axis='y', labelcolor='steelblue')
        ax4_twin.tick_params(axis='y', labelcolor='coral')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_severity_distribution(df: pd.DataFrame,
                               severity_col: str = 'severity_class',
                               title: str = 'Distribution of Event Severity',
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of event severity classifications.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with severity classifications
    severity_col : str
        Severity classification column
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    severity_counts = df[severity_col].value_counts()
    severity_order = ['Minor', 'Moderate', 'Severe', 'Catastrophic']
    severity_counts = severity_counts.reindex([s for s in severity_order if s in severity_counts.index])
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    severity_counts.plot(kind='bar', ax=ax, color=colors[:len(severity_counts)])
    
    ax.set_xlabel('Severity Class', fontsize=14)
    ax.set_ylabel('Number of Events', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(severity_counts.values):
        ax.text(i, v + max(severity_counts.values) * 0.01, str(v),
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_impact_comparison(df: pd.DataFrame,
                          hazard_col: str = 'hazardtype',
                          impact_cols: Optional[List[str]] = None,
                          title: str = 'Impact Comparison by Hazard Type',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple impact metrics by hazard type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Disaster data
    hazard_col : str
        Hazard type column
    impact_cols : list, optional
        Impact columns to compare
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    if impact_cols is None:
        impact_cols = ['total_deaths', 'total_affected', 'total_damage']
    
    available_cols = [col for col in impact_cols if col in df.columns]
    
    if not available_cols:
        logger.warning("No impact columns available for comparison")
        return plt.figure()
    
    fig, axes = plt.subplots(1, len(available_cols), figsize=(6*len(available_cols), 6))
    
    if len(available_cols) == 1:
        axes = [axes]
    
    for idx, col in enumerate(available_cols):
        ax = axes[idx]
        
        impact_by_hazard = df.groupby(hazard_col)[col].sum().sort_values(ascending=False).head(10)
        
        colors = sns.color_palette(PLOT_STYLE['color_palette'], len(impact_by_hazard))
        impact_by_hazard.plot(kind='barh', ax=ax, color=colors)
        
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Hazard Type', fontsize=12)
        ax.set_title(f'{col.replace("_", " ").title()} by Hazard', fontsize=14, fontweight='bold')
        
        for i, v in enumerate(impact_by_hazard.values):
            ax.text(v + max(impact_by_hazard.values) * 0.01, i, f'{v:,.0f}',
                    va='center', fontsize=10)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_risk_matrix(df: pd.DataFrame,
                    frequency_col: str = 'frequency',
                    severity_col: str = 'severity_index',
                    label_col: Optional[str] = None,
                    title: str = 'Risk Matrix: Frequency vs Severity',
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot risk matrix showing frequency vs severity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Risk data
    frequency_col : str
        Frequency column
    severity_col : str
        Severity column
    label_col : str, optional
        Column for point labels
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_plot = df[[frequency_col, severity_col]].dropna()
    
    if len(df_plot) == 0:
        logger.warning("No data available for risk matrix")
        return fig
    
    if 'risk_score' in df.columns:
        scatter = ax.scatter(df_plot[frequency_col], df_plot[severity_col],
                           c=df.loc[df_plot.index, 'risk_score'],
                           s=100, alpha=0.6, cmap='YlOrRd', edgecolors='black')
        plt.colorbar(scatter, ax=ax, label='Risk Score')
    else:
        ax.scatter(df_plot[frequency_col], df_plot[severity_col],
                  s=100, alpha=0.6, color='coral', edgecolors='black')
    
    ax.axhline(y=df_plot[severity_col].median(), color='gray', linestyle='--', alpha=0.5, label='Median Severity')
    ax.axvline(x=df_plot[frequency_col].median(), color='gray', linestyle='--', alpha=0.5, label='Median Frequency')
    
    if label_col and label_col in df.columns:
        for idx in df_plot.index[:20]:
            ax.annotate(df.loc[idx, label_col],
                       (df.loc[idx, frequency_col], df.loc[idx, severity_col]),
                       fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel(frequency_col.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel(severity_col.replace('_', ' ').title(), fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_risk_heatmap(df: pd.DataFrame,
                     location_col: str = 'level2',
                     hazard_col: str = 'hazardtype',
                     risk_col: str = 'risk_score',
                     title: str = 'Risk Heatmap by Location and Hazard',
                     top_n: int = 15,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot heatmap of risk scores by location and hazard type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Risk data
    location_col : str
        Location column
    hazard_col : str
        Hazard type column
    risk_col : str
        Risk score column
    title : str
        Plot title
    top_n : int
        Number of top locations to show
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    pivot_table = df.pivot_table(
        values=risk_col,
        index=location_col,
        columns=hazard_col,
        aggfunc='mean',
        fill_value=0
    )
    
    top_locations = pivot_table.sum(axis=1).nlargest(top_n).index
    pivot_table = pivot_table.loc[top_locations]
    
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Risk Score'}, ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Hazard Type', fontsize=14)
    ax.set_ylabel('Location', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def create_risk_map(df: pd.DataFrame,
                   lat_col: str = 'latitude',
                   lon_col: str = 'longitude',
                   risk_col: str = 'risk_score',
                   location_col: str = 'level2',
                   title: str = 'Risk Map') -> folium.Map:
    """
    Create interactive risk map with color-coded markers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Risk data with coordinates
    lat_col : str
        Latitude column
    lon_col : str
        Longitude column
    risk_col : str
        Risk score column
    location_col : str
        Location name column
    title : str
        Map title
        
    Returns:
    --------
    folium.Map
        Folium map object
    """
    df_map = df[[lat_col, lon_col, risk_col]].dropna()
    
    if len(df_map) == 0:
        logger.warning("No coordinate data available for risk mapping")
        center_lat, center_lon = 0, 0
    else:
        center_lat = df_map[lat_col].mean()
        center_lon = df_map[lon_col].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
                   tiles='OpenStreetMap')
    
    if len(df_map) > 0:
        risk_min = df_map[risk_col].min()
        risk_max = df_map[risk_col].max()
        
        def get_color(risk_score):
            if risk_max > risk_min:
                normalized = (risk_score - risk_min) / (risk_max - risk_min)
            else:
                normalized = 0.5
            
            if normalized < 0.25:
                return 'green'
            elif normalized < 0.5:
                return 'orange'
            elif normalized < 0.75:
                return 'red'
            else:
                return 'darkred'
        
        for idx, row in df.iterrows():
            if pd.notna(row.get(lat_col)) and pd.notna(row.get(lon_col)) and pd.notna(row.get(risk_col)):
                popup_text = f"<b>{row.get(location_col, 'Unknown')}</b><br>"
                popup_text += f"Risk Score: {row[risk_col]:.2f}<br>"
                
                if 'hazardtype' in row:
                    popup_text += f"Hazard: {row['hazardtype']}<br>"
                if 'frequency' in row:
                    popup_text += f"Frequency: {row['frequency']:.2f}<br>"
                if 'severity_index' in row:
                    popup_text += f"Severity: {row['severity_index']:.2f}<br>"
                
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=8,
                    popup=folium.Popup(popup_text, max_width=300),
                    color=get_color(row[risk_col]),
                    fill=True,
                    fillColor=get_color(row[risk_col]),
                    fillOpacity=0.7
                ).add_to(m)
    
    return m


def create_risk_dashboard(risk_df: pd.DataFrame,
                         title: str = 'Multi-Hazard Risk Dashboard') -> go.Figure:
    """
    Create interactive risk dashboard with multiple visualizations.
    
    Parameters:
    -----------
    risk_df : pd.DataFrame
        Risk assessment data
    title : str
        Dashboard title
        
    Returns:
    --------
    go.Figure
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Score by Location', 'Exposure vs Vulnerability',
                       'Risk Categories', 'Hazard Diversity'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}],
               [{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    top_risks = risk_df.nlargest(15, 'risk_score')
    
    fig.add_trace(
        go.Bar(x=top_risks.index, y=top_risks['risk_score'],
               marker_color='crimson', name='Risk Score'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=risk_df['exposure_index'], y=risk_df['vulnerability_index'],
                   mode='markers', marker=dict(size=10, color=risk_df['risk_score'],
                   colorscale='Reds', showscale=True),
                   text=risk_df.index, name='Locations'),
        row=1, col=2
    )
    
    if 'risk_category' in risk_df.columns:
        category_counts = risk_df['risk_category'].value_counts()
        fig.add_trace(
            go.Pie(labels=category_counts.index, values=category_counts.values,
                   name='Risk Categories'),
            row=2, col=1
        )
    
    if 'hazard_diversity' in risk_df.columns:
        top_diversity = risk_df.nlargest(15, 'hazard_diversity')
        fig.add_trace(
            go.Bar(x=top_diversity.index, y=top_diversity['hazard_diversity'],
                   marker_color='teal', name='Hazard Types'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text=title, title_font_size=20)
    fig.update_xaxes(tickangle=45)
    
    return fig
