# Disaster Risk Modeling System
**Transform disaster data into actionable insights for evidence-based decisions**

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/JuniorDieka/sendai-multi-hazard-risk-modeling)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Sendai Framework](https://img.shields.io/badge/Sendai-Framework-green.svg)](https://www.undrr.org/publication/sendai-framework-disaster-risk-reduction-2015-2030)

---

## 🎯 What This Does

This system helps **governments and decision-makers** turn disaster records into strategic insights for:

- 💰 **Budget Planning** - Justify disaster risk reduction investments with data
- 🌍 **Climate Finance** - Access international funding (Green Climate Fund, Adaptation Fund)
- 🚨 **Early Warning** - Predict high-risk periods and locations
- 📊 **Policy Making** - Make evidence-based decisions on land use and development

**Aligned with:** Sendai Framework 2015-2030 | SDG Indicators | UN Statistical Standards

---

## ⚡ The Problem & Solution

### ❌ Without This System
- "500 people died" → Cannot compare regions fairly
- "$10M in losses" → Cannot access climate finance
- Raw numbers → Cannot meet international reporting standards

### ✅ With This System
- "5 deaths per 100,000 population" → Sendai Framework compliant
- "0.5% of GDP lost" → Eligible for international funding
- Normalized indicators → Evidence-based prioritization

---

## 🚀 What It Does

### 📥 **Automated Data Enrichment**
- Fetches population & GDP data from World Bank/UN APIs
- Calculates Sendai Framework indicators automatically
- Handles missing data with smart interpolation

### 📊 **Comprehensive Analysis**
- **Frequency**: When do disasters happen? How often?
- **Severity**: Which events cause the most damage?
- **Location**: Where are the high-risk areas?
- **Trends**: Are disasters increasing or decreasing?
- **Predictions**: What to expect in the future (Machine Learning)

### 🎯 **Sendai Framework Indicators**
Calculates all required metrics:
- **Mortality** (deaths per 100,000 population)
- **Affected People** (affected per 100,000 population)
- **Economic Loss** (% of GDP)
- **Infrastructure Damage** (health, education facilities)

### 🤖 **Machine Learning Predictions**
- Forecast when disasters will occur
- Predict expected deaths and economic losses
- Identify high-risk locations
- Understand key risk factors

### 📈 **Decision-Support Outputs**
- Interactive dashboards and maps
- Budget allocation recommendations
- Climate finance application packages
- Risk rankings and priority lists

---

## 📁 Project Structure

```
📦 sendai-multi-hazard-risk-modeling/
├── 📂 data/
│   ├── raw/              # Your disaster data goes here
│   ├── processed/        # Cleaned data with indicators
│   └── external/         # Auto-fetched population/GDP data
├── 📓 notebooks/
│   ├── 01-06_*.ipynb     # Analysis notebooks
│   ├── 07_machine_learning.ipynb
│   └── USE_CASE_*.ipynb  # Decision-support demonstrations
├── 💻 src/               # Python modules (automated processing)
├── 📊 outputs/
│   ├── figures/          # Charts and maps
│   ├── reports/          # Excel/CSV reports
│   └── models/           # Trained ML models
└── ⚙️ config.py          # Settings and thresholds
```

---

## ⚡ Quick Start (5 Minutes)

### 1️⃣ Install
```bash
git clone https://github.com/JuniorDieka/sendai-multi-hazard-risk-modeling.git
cd sendai-multi-hazard-risk-modeling
pip install -r requirements.txt
```

### 2️⃣ Add Your Data
**Option A:** Use sample data (African countries 2010-2023)
```bash
cp data/raw/disaster_events_sample.csv data/raw/disaster_events.csv
```

**Option B:** Use your own data
- Place CSV file in `data/raw/disaster_events.csv`
- **DesInventar users**: Export and use directly - no changes needed!
- Minimum required: `year, hazardtype`
- Recommended: `deaths, affected, usdvalue, location`

### 3️⃣ Run Analysis
```bash
jupyter notebook
```
Open notebooks in order: `01` → `02` → ... → `07` → Use Cases

### 4️⃣ Get Results
- 📊 Reports in `outputs/reports/`
- 📈 Charts in `outputs/figures/`
- 🤖 ML models in `outputs/models/`

---

## 📖 How to Use

### 🔬 For Analysts & Technicians

**Step-by-Step Workflow:**
1. **Explore** → Check data quality and coverage
2. **Analyze** → Frequency, severity, spatial patterns
3. **Model** → Calculate Sendai indicators and risk scores
4. **Predict** → Train ML models for forecasting
5. **Report** → Generate outputs for decision-makers

**Quick Code Example:**
```python
from src.risk_metrics import add_sendai_indicators
from src.ml_models import DisasterFrequencyPredictor

# Calculate Sendai indicators
df = add_sendai_indicators(df)

# Train prediction model
model = DisasterFrequencyPredictor()
model.train(df)
model.save()
```

### 👔 For Decision-Makers

**4 Ready-to-Use Demonstrations:**

| Use Case | What You Get | Who Needs This |
|----------|--------------|----------------|
| **💰 Financing** | Budget priorities, ROI analysis, climate finance packages | Finance Ministry |
| **🚨 Preparedness** | Resource allocation, stockpile planning | Emergency Management |
| **⚠️ Early Warning** | Risk forecasts, seasonal patterns, alert thresholds | Disaster Management |
| **🏗️ Planning** | Land use zoning, infrastructure priorities | Planning Ministry |

---

## 💡 Why This Matters

### The Data Quality Difference

| Basic Data | With This System |
|------------|------------------|
| "500 deaths" | "5 deaths per 100,000 population" |
| "$10M lost" | "0.5% of GDP lost" |
| ❌ Cannot compare regions | ✅ Fair regional comparison |
| ❌ No climate finance access | ✅ Eligible for international funding |
| ❌ Ad-hoc decisions | ✅ Evidence-based prioritization |

### Real Impact

**For Government:**
- 💰 Access $XX million in climate finance
- 📊 Meet international reporting standards
- 🎯 Justify DRR budgets with ROI data (4:1 to 7:1 returns)

**For Citizens:**
- 🏥 Resources allocated to high-risk areas
- ⚡ Faster, more effective disaster response
- 🛡️ Proactive risk reduction saves lives


---

## 📚 Data Sources & Compatibility

**Your Disaster Data:**
- ✅ DesInventar exports (direct import, no changes needed)
- ✅ National disaster databases
- ✅ Emergency management records
- 📄 See `DATA_FORMAT_GUIDE.md` for details

**Automatic Enrichment:**
- 🌍 Population: World Bank / UN APIs
- 💰 GDP: World Bank API
- 🗺️ Geographic boundaries: Natural Earth

---

## 📊 What You Get

**Reports (Excel/CSV):**
- Risk scores by location
- Budget allocation priorities
- Climate finance application packages
- High-priority intervention areas

**Visualizations:**
- 📈 Interactive dashboards
- 🗺️ Geographic risk maps
- 📊 Trend analysis charts
- 🎯 Risk heatmaps

**Data Products:**
- Cleaned disaster data
- Sendai Framework indicators
- ML prediction models

---

## 💬 Communicating Results

**Key Messages by Audience:**

| Audience | Message |
|----------|----------|
| **Finance Minister** | "Disasters cost X% of GDP. Every $1 in DRR saves $4-7. We qualify for $XX million in climate finance." |
| **Health Minister** | "X deaths per 100,000 (Sendai target). High-risk areas need facility reinforcement." |
| **Planning Minister** | "Multi-hazard exposure mapped. Infrastructure should prioritize high-risk zones." |
| **Cabinet** | "Evidence-based DRR will save lives and money. We meet international standards." |

---

## 🔌 System Compatibility

**Works with:**
- ✅ DesInventar (90+ countries) - Direct import, no changes needed
- ✅ Sendai Framework Monitoring - Auto-calculates all indicators
- ✅ National Disaster Databases - Flexible column mapping

**Current Features:**
- ✅ Machine learning predictions
- ✅ Event frequency forecasting
- ✅ Impact severity prediction

**Planned Enhancements:**
- Real-time data integration
- Mobile data collection
- Multi-language support
- Climate scenario modeling

---

## 📚 References & Resources

**Framework Documentation:**
- [Sendai Framework 2015-2030](https://www.undrr.org/publication/sendai-framework-disaster-risk-reduction-2015-2030)
- [Technical Guidance on Indicators](https://www.undrr.org/publication/technical-guidance-monitoring-and-reporting-progress-achieving-global-targets-sendai)
- [SDG Indicators](https://unstats.un.org/sdgs/metadata/)

**Data Sources:**
- [World Bank Open Data](https://data.worldbank.org/)
- [UN Population Division](https://population.un.org/wpp/)
- [DesInventar](https://www.desinventar.net/)

---

## 🛠️ Customization

**Adapt to your context:**
1. Update country codes in `config.py`
2. Adjust severity thresholds for local conditions
3. Add region-specific hazard types
4. Translate outputs to local language

---

## 📄 License & Support

**License:** Open for public good - adapt and use freely for disaster risk reduction

**Support:**
- 📖 Review notebook documentation
- 💻 Check code comments in `src/`
- 📊 Examine sample data format
- 🌐 Consult Sendai Framework guidance

---

## ✅ Quick Checklist

**Setup (5 minutes):**
- [ ] Install Python 3.9+ and dependencies
- [ ] Add your disaster data CSV
- [ ] Run notebook 01 to check data quality

**Analysis (1-2 hours):**
- [ ] Run notebooks 01-07 in sequence
- [ ] Review generated reports and charts
- [ ] Explore use case demonstrations

**Action (ongoing):**
- [ ] Share findings with decision-makers
- [ ] Apply for climate finance
- [ ] Update data quarterly

---

## 🎯 Success Indicators

**You're succeeding when:**

✅ Ministers justify DRR budgets with your data  
✅ You access international climate finance  
✅ High-risk areas get priority interventions  
✅ Your country reports Sendai indicators  
✅ Evidence replaces ad-hoc decisions  
✅ Disaster losses decrease over time  

---

## 🌟 Bottom Line

**Good data saves lives. This system transforms disaster records into actionable insights for building resilience.**

*Built for disaster risk reduction practitioners, analysts, and decision-makers committed to evidence-based resilience.*

---

**Questions?** Check the notebooks, review `DATA_FORMAT_GUIDE.md`, or consult Sendai Framework guidance.
