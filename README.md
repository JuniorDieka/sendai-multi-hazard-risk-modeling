# Multi-Hazard Risk Modeling for Evidence-Based Disaster Risk Reduction

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/JuniorDieka/sendai-multi-hazard-risk-modeling)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Public%20Good-green.svg)](LICENSE)

## Aligned with the Sendai Framework for Disaster Risk Reduction 2015-2030

---

## 🎯 Project Overview

### Purpose
This project demonstrates to **ministers** (Finance, Health, Economic Planning) and **decision-makers** how well-organized, country-maintained disaster data enables strategic decisions on:
- **Financing**: Budget allocation and climate finance access
- **Preparedness**: Resource pre-positioning and capacity planning
- **Early Warning**: Forecasting and alert systems
- **Planning**: Risk-informed development and land use

### Framework Compliance
✅ **Fully aligned** with international standards:
- **Sendai Framework for Disaster Risk Reduction 2015-2030** (all 7 targets, 4 priorities)
- **Global Disaster-Related Statistics Framework (G-DRSF)** (UN Statistical Commission)
- **Sustainable Development Goals** (SDG 1.5.1, 11.5.1, 13.1.1)
- **Climate Adaptation Agendas** (Global Goal on Adaptation)

📄 See `FRAMEWORK_ALIGNMENT.md` for detailed compliance documentation

### The Problem This Solves

**Current Gap:** Most disaster databases lack population and GDP metadata, making it impossible to calculate normalized Sendai Framework indicators.

**Consequence:** Countries can only report raw counts (e.g., "500 deaths") instead of normalized indicators (e.g., "5 deaths per 100,000 population"), which are required for:
- International climate finance applications
- Fair comparison between regions
- Evidence-based policy decisions
- Sendai Framework monitoring

**Solution:** This project automatically integrates World Bank/UN data to transform raw disaster counts into actionable decision-support metrics.

---

## 📊 Key Features

### 1. Automated External Data Integration
- **Population Data**: UN/World Bank API (annual population by country/region)
- **GDP Data**: World Bank API (annual GDP in current USD)
- **Smart Caching**: Downloaded data cached locally for 30 days
- **Interpolation**: Handles missing years automatically

### 2. Sendai Framework Indicator Calculations
Calculates ALL Sendai Framework indicators:

**Target A - Mortality:**
- A-1: Mortality rate per 100,000 population
- A-2: Number of deaths
- A-3: Number of missing persons

**Target B - Affected People:**
- B-1: Affected rate per 100,000 population
- B-2: Number of injured/sick people
- B-3: People with damaged dwellings
- B-4: People with destroyed dwellings
- B-5: People with disrupted livelihoods

**Target C - Economic Loss:**
- C-1: Economic loss as % of GDP
- C-2-C-5: Sectoral losses (where data available)

**Target D - Infrastructure:**
- D-1: Infrastructure damage per 100,000 population
- D-2: Damaged/destroyed health facilities
- D-3: Damaged/destroyed education facilities

### 3. Comprehensive Analysis
- **Frequency Analysis**: Event patterns, return periods, probability distributions
- **Severity Analysis**: Multi-dimensional impact classification
- **Spatial Analysis**: Geographic clustering and hotspot identification
- **Risk Modeling**: Exposure, vulnerability, and composite risk scores
- **Trend Analysis**: Temporal patterns and forecasting

### 4. Machine Learning Predictions
- **Event Frequency Forecasting**: Predict when disasters are likely to occur
- **Impact Severity Prediction**: Estimate deaths, affected people, and economic losses
- **Risk Score Prediction**: Identify high-risk locations using ML
- **Feature Importance Analysis**: Understand key risk factors
- **Model Persistence**: Save and load trained models for production use

### 5. Decision-Support Visualizations
- Time series plots with trend lines
- Interactive dashboards (Plotly)
- Geographic maps (Folium)
- Comparison charts (raw vs normalized indicators)
- Risk heatmaps and priority rankings

---

## 🗂️ Project Structure

```
sendai-multi-hazard-risk-modeling/
├── data/
│   ├── raw/                          # Original disaster data
│   │   └── disaster_events_sample.csv
│   ├── processed/                    # Cleaned and enriched data
│   └── external/                     # Population and GDP data (auto-fetched)
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and quality assessment
│   ├── 02_frequency_analysis.ipynb   # Event frequency and patterns
│   ├── 03_severity_analysis.ipynb    # Impact magnitude analysis
│   ├── 04_spatial_analysis.ipynb     # Geographic patterns
│   ├── 05_risk_modeling.ipynb        # Risk assessment and Sendai indicators
│   ├── 06_dashboard.ipynb            # Interactive dashboard
│   ├── 07_machine_learning.ipynb     # ML models for prediction
│   ├── USE_CASE_1_Financing.ipynb    # Budget allocation and climate finance
│   ├── USE_CASE_2_Preparedness.ipynb # Resource pre-positioning
│   ├── USE_CASE_3_EarlyWarning.ipynb # Forecasting and alerts
│   └── USE_CASE_4_Planning.ipynb     # Development planning
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Data cleaning and transformation
│   ├── external_data.py              # World Bank API integration
│   ├── analysis.py                   # Statistical analysis functions
│   ├── visualization.py              # Plotting functions
│   ├── risk_metrics.py               # Sendai indicators and risk scores
│   ├── ml_models.py                  # Machine learning prediction models
│   └── utils.py                      # Helper functions
├── outputs/
│   ├── figures/                      # Generated plots
│   ├── reports/                      # Analysis reports (CSV/Excel)
│   └── models/                       # Saved models
├── config.py                         # Configuration parameters
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/JuniorDieka/sendai-multi-hazard-risk-modeling.git
cd sendai-multi-hazard-risk-modeling
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Your Data

**Option A: Use Sample Data (African Context)**
- Sample data is provided in `data/raw/disaster_events_sample.csv`
- Features: Kenya, Ethiopia, Somalia, Tanzania, Uganda, Mozambique disasters (2010-2023)
- Rename to `disaster_events.csv` to use

**Option B: Use Your Own Data**
- Place your disaster data CSV in `data/raw/disaster_events.csv`
- **DesInventar users**: Export directly from DesInventar and use without modification!
- **Sendai Framework users**: System auto-maps to Sendai indicators
- Required columns: `year, hazardtype` (minimum)
- Recommended columns: `serial, year, month, day, eventname, level1, level2, level3, latitude, longitude, hazardtype, deaths, injured, missing, affected, displaced, housesdestroyed, housesdamaged, educationcenters, healthcenters, localvalue, usdvalue, durationdays, eventdescription`
- Optional columns are acceptable; the system adapts to available data

**Option C: Import from DesInventar**
```
DesInventar Export Format:
Event | DataCards | Deaths | Injured | Missing | Houses Destroyed | Houses Damaged | 
Directly affected | Relocated | Losses $USD | Education centers | Hospitals
```
→ System automatically maps to standard format (see `DATA_FORMAT_GUIDE.md`)

### Step 4: Run Notebooks
Execute notebooks in sequence:
```bash
jupyter notebook
```

1. Start with `01_data_exploration.ipynb`
2. Continue through `02-06` for analysis and visualization
3. Run `07_machine_learning.ipynb` to train predictive models
4. Explore use cases for decision-support demonstrations

---

## 📖 Usage Guide

### For Data Analysts

**Workflow:**
1. **Data Exploration** (Notebook 01): Assess data quality, coverage, and completeness
2. **Frequency Analysis** (Notebook 02): Understand event patterns and return periods
3. **Severity Analysis** (Notebook 03): Classify impacts and identify extreme events
4. **Spatial Analysis** (Notebook 04): Map hotspots and geographic patterns
5. **Risk Modeling** (Notebook 05): Calculate Sendai indicators and risk scores
6. **Machine Learning** (Notebook 07): Train predictive models for forecasting

**Key Functions:**
```python
from src.external_data import ExternalDataFetcher
from src.risk_metrics import add_sendai_indicators, calculate_multi_hazard_risk
from src.ml_models import DisasterFrequencyPredictor, ImpactSeverityPredictor

# Fetch external data
fetcher = ExternalDataFetcher()
external_data = fetcher.fetch_all_external_data()

# Calculate Sendai indicators
df_sendai = add_sendai_indicators(df)

# Assess multi-hazard risk
risk_assessment = calculate_multi_hazard_risk(df_sendai)

# Train ML models
freq_model = DisasterFrequencyPredictor()
freq_model.train(df)
freq_model.save()
```

### For Decision-Makers

**Use Case Demonstrations:**

1. **Financing (USE_CASE_1)**: 
   - Budget allocation priorities
   - Cost-benefit analysis of DRR investments
   - Climate finance application support
   - Economic loss trends

2. **Preparedness (USE_CASE_2)**:
   - High-frequency hazard identification
   - Resource pre-positioning optimization
   - Contingency planning priorities
   - Capacity gap analysis

3. **Early Warning (USE_CASE_3)**:
   - Historical patterns for forecasting
   - Seasonal risk windows
   - Impact thresholds for alerts
   - Vulnerable area mapping

4. **Planning (USE_CASE_4)**:
   - Land use zoning recommendations
   - Infrastructure investment priorities
   - Development policy risk-screening
   - Climate adaptation targeting

---

## 🎓 Understanding the Value of Complete Data

### Without Population/GDP Metadata

**What you can report:**
- "500 people died from floods in 2023"
- "Economic losses were $10 million"

**Limitations:**
- Cannot compare regions fairly (large vs small populations)
- Cannot access international climate finance
- Cannot calculate Sendai Framework indicators
- Cannot prioritize based on relative impact

### With Population/GDP Metadata

**What you can report:**
- "Mortality rate: 5 deaths per 100,000 population (Sendai A-1)"
- "Economic losses: 0.5% of GDP (Sendai C-1)"

**Advantages:**
- ✅ Fair comparison between regions
- ✅ Eligible for Green Climate Fund, Adaptation Fund
- ✅ Meets Sendai Framework reporting requirements
- ✅ Evidence-based prioritization
- ✅ International credibility

---

## 💡 Key Findings & Insights

### Demonstration Results (Using Sample Data)

**Economic Impact:**
- Total losses: $XXX million over XX years
- Average annual loss: $XX million (X.X% of GDP)
- ROI on DRR investment: 4:1 to 7:1

**Risk Priorities:**
- Top 20 high-risk locations identified
- Multi-hazard exposure mapped
- Seasonal patterns documented

**Data Quality:**
- XX% of events have population data
- XX% have economic loss data
- Demonstrates value of complete metadata

---

## 🌍 Benefits of Country-Owned Disaster Data Systems

### For Government

1. **Evidence-Based Policy**: Data-driven decisions on DRR investments
2. **Budget Justification**: Clear ROI for disaster risk reduction
3. **International Finance**: Access to climate adaptation funding
4. **Monitoring & Evaluation**: Track progress on Sendai Framework targets
5. **Institutional Memory**: Preserve knowledge across administrations

### For Citizens

1. **Better Preparedness**: Resources allocated to high-risk areas
2. **Improved Response**: Faster, more effective disaster response
3. **Reduced Losses**: Proactive risk reduction saves lives and livelihoods
4. **Transparency**: Public access to disaster risk information

### For International Partners

1. **Standardized Reporting**: Comparable data across countries
2. **Targeted Support**: Aid directed to highest-need areas
3. **Impact Measurement**: Track effectiveness of interventions
4. **Knowledge Sharing**: Best practices based on evidence

---

## 📚 Data Sources

### Disaster Data
- **Format**: **DesInventar-compatible CSV** (fully interoperable with existing systems)
- **Source**: National disaster databases, emergency management agencies, DesInventar exports
- **Coverage**: 1917-2025 (configurable)
- **Sample Data**: African countries (Kenya, Ethiopia, Somalia, Tanzania, Uganda, Mozambique, Madagascar)
- **See**: `DATA_FORMAT_GUIDE.md` for complete format specifications and DesInventar mapping

### Population Data
- **Source**: World Bank API (SP.POP.TOTL indicator)
- **Backup**: UN Population Division
- **Update Frequency**: Annual

### GDP Data
- **Source**: World Bank API (NY.GDP.MKTP.CD indicator)
- **Coverage**: 1960-present
- **Currency**: Current USD

### Geographic Boundaries
- **Source**: Natural Earth, country-specific GIS data
- **Use**: Spatial analysis and mapping

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Time period
MIN_YEAR = 1917
MAX_YEAR = 2025

# Severity thresholds
SEVERITY_THRESHOLDS = {
    'deaths': {'minor': 0, 'moderate': 10, 'severe': 100, 'catastrophic': 1000},
    'affected': {'minor': 0, 'moderate': 1000, 'severe': 10000, 'catastrophic': 100000},
    'economic_loss_usd': {'minor': 0, 'moderate': 1e6, 'severe': 1e7, 'catastrophic': 1e8}
}

# Country codes for external data
COUNTRY_ISO_CODES = {
    'Bangladesh': 'BGD',
    'India': 'IND',
    # Add your countries...
}
```

---

## 📊 Output Files

### Reports (CSV/Excel)
- `multi_hazard_risk_assessment.csv`: Risk scores by location
- `priority_locations.csv`: Top priority areas for intervention
- `UC1_financial_priorities.csv`: Budget allocation recommendations
- `climate_finance_data.csv`: Data package for funding applications

### Visualizations (PNG/HTML)
- Time series plots with trends
- Interactive dashboards (HTML)
- Geographic maps
- Comparison charts (raw vs normalized)
- Risk heatmaps

### Processed Data
- `disaster_events_processed.csv`: Cleaned data with derived columns
- `disaster_events_with_metadata.csv`: Merged with population/GDP
- `disaster_events_sendai.csv`: Complete with Sendai indicators

---

## 🎯 Ministerial Communication Guide

### Key Messages for Different Audiences

**Minister of Finance:**
- "Disasters cost us X% of GDP annually"
- "Every $1 in DRR saves $4-7 in losses"
- "This data qualifies us for $XX million in climate finance"

**Minister of Health:**
- "X deaths per 100,000 population (Sendai target)"
- "High-risk areas need health facility reinforcement"
- "Seasonal patterns inform medical preparedness"

**Minister of Planning:**
- "These areas should be excluded from development"
- "Infrastructure investments should prioritize high-risk zones"
- "Multi-hazard exposure mapped for land use planning"

**Cabinet/Prime Minister:**
- "Evidence-based DRR strategy will save lives and money"
- "We now meet international reporting standards"
- "Data-driven approach to disaster resilience"

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Real-time data integration via APIs
- [x] Machine learning for impact prediction ✅
- [ ] Mobile data collection app integration
- [ ] Automated report generation
- [ ] Multi-language support
- [ ] Integration with national early warning systems
- [ ] Climate change scenario modeling
- [ ] Social vulnerability indicators

### Advanced Analytics
- [x] Event frequency forecasting ✅
- [x] Impact severity prediction ✅
- [ ] Bayesian risk modeling
- [ ] Agent-based simulation
- [ ] Network analysis of cascading impacts
- [ ] Satellite imagery integration
- [ ] Social media sentiment analysis

---

## � System Interoperability

### Compatible with Existing Systems

This project is designed to work seamlessly with:

✅ **DesInventar** (90+ countries worldwide)
- Direct import of DesInventar CSV exports
- Automatic column mapping
- No data restructuring needed

✅ **Sendai Framework Monitoring**
- Auto-calculates all Sendai indicators
- UNDRR reporting format compatible
- Meets international standards

✅ **National Disaster Databases**
- Flexible column mapping
- Handles regional variations
- Supports custom fields

### Data Format Example (DesInventar Export)

```
Event          DataCards  Deaths  Injured  Houses Destroyed  Losses $USD
FLOOD          89         728     2070     7078              11280400000
DROUGHT        10         37      2483     30686             57600000000
EARTHQUAKE     2          0       8        656               0
FIRE           12         1025    3004     181               200000000000
```

**This system processes this format automatically!**

See `DATA_FORMAT_GUIDE.md` for complete specifications.

---

## �� References

### Sendai Framework
- [Sendai Framework for Disaster Risk Reduction 2015-2030](https://www.undrr.org/publication/sendai-framework-disaster-risk-reduction-2015-2030)
- [Sendai Framework Indicators](https://www.undrr.org/publication/technical-guidance-monitoring-and-reporting-progress-achieving-global-targets-sendai)

### Data Sources
- [World Bank Open Data](https://data.worldbank.org/)
- [UN Population Division](https://population.un.org/wpp/)
- [DesInventar Methodology](https://www.desinventar.net/)

### Framework Documents
- [Global Disaster-Related Statistics Framework (G-DRSF)](https://www.undrr.org/building-risk-knowledge/framework-disaster-statistics)
- [Technical Guidance on Sendai Framework Indicators](https://www.undrr.org/publication/technical-guidance-monitoring-and-reporting-progress-achieving-global-targets-sendai)
- [SDG Indicators Metadata](https://unstats.un.org/sdgs/metadata/)

### Research
- "The Economic Case for Disaster Risk Reduction" - UNDRR
- "Cost-Benefit Analysis of DRR Investments" - World Bank
- "Climate Finance for Adaptation" - Green Climate Fund
- "Measuring Disaster-Related Economic Losses" - IAEG-DRS Working Paper

---

## 🤝 Contributing

This project is designed for adaptation to country-specific contexts. To customize:

1. **Update country codes** in `config.py`
2. **Adjust severity thresholds** based on local context
3. **Add hazard types** specific to your region
4. **Customize visualizations** for your audience
5. **Translate** notebooks and outputs to local language

---

## 📄 License

This project is provided for disaster risk reduction purposes. Adapt and use freely for public good.

---

## 📞 Support

For technical assistance:
- Review notebook documentation
- Check `src/` module docstrings
- Examine sample data format
- Consult Sendai Framework technical guidance

---

## ✅ Quick Start Checklist

- [ ] Install Python 3.9+
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Place disaster data in `data/raw/disaster_events.csv`
- [ ] Run `01_data_exploration.ipynb`
- [ ] Review data quality summary
- [ ] Run `05_risk_modeling.ipynb` to fetch external data
- [ ] Explore use case demonstrations
- [ ] Generate reports for decision-makers
- [ ] Share findings with stakeholders

---

## 🎉 Success Metrics

**You'll know this project is successful when:**

✅ Ministers use your data to justify DRR budgets  
✅ You successfully apply for international climate finance  
✅ High-risk areas receive priority interventions  
✅ Your country reports Sendai Framework indicators  
✅ Evidence-based policies replace ad-hoc decisions  
✅ Disaster losses decrease over time  

---

**Remember:** Good data saves lives. Complete data enables smart decisions. This project transforms disaster records into decision-support tools for building resilience.

---

*Developed for disaster risk reduction practitioners, data analysts, and decision-makers committed to evidence-based resilience building.*
