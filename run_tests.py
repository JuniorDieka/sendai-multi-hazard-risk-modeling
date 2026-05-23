"""
Comprehensive test suite for the Sendai Multi-Hazard Risk Modeling project.

This script tests:
- Directory structure
- Required files
- Module imports
- Data loading
- Basic functionality

Run with: python run_tests.py
"""

import os
import sys
from pathlib import Path

def test_structure():
    """Test directory structure"""
    print("Testing directory structure...")
    required_dirs = [
        'data/raw', 
        'data/processed', 
        'data/external', 
        'notebooks', 
        'src', 
        'outputs/figures', 
        'outputs/reports',
        'outputs/models'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} MISSING")
            all_exist = False
    
    return all_exist

def test_files():
    """Test required files exist"""
    print("\nTesting required files...")
    required_files = [
        'config.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'TESTING_GUIDE.md',
        'src/__init__.py',
        'src/utils.py',
        'src/data_processing.py',
        'src/external_data.py',
        'src/risk_metrics.py',
        'src/analysis.py',
        'src/visualization.py',
        'src/policy_briefing.py',
        'data/raw/disaster_events_sample.csv'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} MISSING")
            all_exist = False
    
    return all_exist

def test_imports():
    """Test module imports"""
    print("\nTesting module imports...")
    
    try:
        import config
        print("  ✓ config")
    except Exception as e:
        print(f"  ✗ config - Error: {e}")
        return False
    
    try:
        from src import utils, data_processing, external_data, risk_metrics, analysis, visualization, policy_briefing
        print("  ✓ src.utils")
        print("  ✓ src.data_processing")
        print("  ✓ src.external_data")
        print("  ✓ src.risk_metrics")
        print("  ✓ src.analysis")
        print("  ✓ src.visualization")
        print("  ✓ src.policy_briefing")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_data():
    """Test data loading"""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        
        # Test sample data
        sample_file = Path('data/raw/disaster_events_sample.csv')
        if sample_file.exists():
            df = pd.read_csv(sample_file)
            print(f"  ✓ Loaded {len(df)} sample records")
            print(f"  ✓ Sample data has {len(df.columns)} columns")
        else:
            print("  ✗ Sample data file not found")
            return False
        
        # Check for processed data (optional)
        processed_file = Path('data/processed/disaster_events_processed.csv')
        if processed_file.exists():
            df_proc = pd.read_csv(processed_file)
            print(f"  ✓ Processed data exists ({len(df_proc)} records)")
        else:
            print("  ⚠ Processed data not yet created (run notebooks to generate)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data loading error: {e}")
        return False

def test_notebooks():
    """Test notebook files exist"""
    print("\nTesting notebook files...")
    
    required_notebooks = [
        'notebooks/01_data_exploration.ipynb',
        'notebooks/02_frequency_analysis.ipynb',
        'notebooks/05_risk_modeling.ipynb',
        'notebooks/06_dashboard.ipynb',
        'notebooks/USE_CASE_1_Financing.ipynb',
        'notebooks/USE_CASE_2_Preparedness.ipynb',
        'notebooks/USE_CASE_3_EarlyWarning.ipynb',
        'notebooks/USE_CASE_4_Planning.ipynb',
        'notebooks/GENERATE_POLICY_BRIEF.ipynb'
    ]
    
    all_exist = True
    for notebook in required_notebooks:
        if Path(notebook).exists():
            print(f"  ✓ {notebook}")
        else:
            print(f"  ✗ {notebook} MISSING")
            all_exist = False
    
    return all_exist

def test_config():
    """Test configuration values"""
    print("\nTesting configuration...")
    
    try:
        import config
        
        # Check key configuration exists
        assert hasattr(config, 'RAW_DATA_DIR'), "Missing RAW_DATA_DIR"
        assert hasattr(config, 'PROCESSED_DATA_DIR'), "Missing PROCESSED_DATA_DIR"
        assert hasattr(config, 'HAZARD_TYPES'), "Missing HAZARD_TYPES"
        assert hasattr(config, 'SENDAI_INDICATORS'), "Missing SENDAI_INDICATORS"
        assert hasattr(config, 'COUNTRY_ISO_CODES'), "Missing COUNTRY_ISO_CODES"
        
        print(f"  ✓ Configuration loaded successfully")
        print(f"  ✓ {len(config.HAZARD_TYPES)} hazard types configured")
        print(f"  ✓ {len(config.COUNTRY_ISO_CODES)} countries configured")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("SENDAI MULTI-HAZARD RISK MODELING - TEST SUITE")
    print("="*80)
    print()
    
    results = []
    
    # Run tests
    results.append(("Directory Structure", test_structure()))
    results.append(("Required Files", test_files()))
    results.append(("Module Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Data Loading", test_data()))
    results.append(("Notebook Files", test_notebooks()))
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready for use!")
        print("\nNext steps:")
        print("1. Copy sample data: copy data\\raw\\disaster_events_sample.csv data\\raw\\disaster_events.csv")
        print("2. Start Jupyter: jupyter notebook")
        print("3. Run notebooks in order: 01, 02, 05, USE_CASE_1, etc.")
        print("4. Generate policy brief: Run GENERATE_POLICY_BRIEF.ipynb")
    else:
        print("❌ SOME TESTS FAILED - Please fix issues above")
        print("\nCommon fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Check you're in the project root directory")
        print("- Verify all files were downloaded/created correctly")
    
    print("="*80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
