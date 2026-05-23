"""
Quick test script for ML models
Tests basic functionality before committing
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import PROCESSED_DATA_DIR, MODELS_DIR
from ml_models import (
    DisasterFrequencyPredictor,
    ImpactSeverityPredictor,
    RiskScorePredictor,
    train_all_models
)

def test_ml_models():
    """Test ML models with sample data"""
    
    print("="*80)
    print("TESTING MACHINE LEARNING MODELS")
    print("="*80)
    print()
    
    # Check if processed data exists
    data_file = PROCESSED_DATA_DIR / 'disaster_events_sendai.csv'
    
    if not data_file.exists():
        print("⚠ Processed data not found.")
        print("Please run notebooks/05_risk_modeling.ipynb first to generate processed data.")
        return False
    
    # Load data
    print("1. Loading data...")
    df = pd.read_csv(data_file)
    print(f"   ✓ Loaded {len(df)} events")
    print(f"   ✓ Time period: {df['year'].min()} - {df['year'].max()}")
    print()
    
    # Test 1: Frequency Predictor
    print("2. Testing Frequency Predictor...")
    try:
        freq_model = DisasterFrequencyPredictor(model_type='random_forest')
        freq_metrics = freq_model.train(df)
        print(f"   ✓ Model trained successfully")
        print(f"   ✓ MAE: {freq_metrics['mae']:.2f}")
        print(f"   ✓ R²: {freq_metrics['r2']:.3f}")
        
        # Test prediction
        prediction = freq_model.predict(2024, 6, 'FLOOD')
        print(f"   ✓ Prediction test: {prediction:.2f} events")
        
        # Test save/load
        freq_model.save('test_freq_model.pkl')
        print(f"   ✓ Model saved to {MODELS_DIR / 'test_freq_model.pkl'}")
        
        freq_model_loaded = DisasterFrequencyPredictor()
        freq_model_loaded.load('test_freq_model.pkl')
        print(f"   ✓ Model loaded successfully")
        print()
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False
    
    # Test 2: Impact Severity Predictor
    print("3. Testing Impact Severity Predictor...")
    try:
        if 'deaths' in df.columns and df['deaths'].notna().sum() > 10:
            impact_model = ImpactSeverityPredictor(target='deaths', model_type='random_forest')
            impact_metrics = impact_model.train(df)
            print(f"   ✓ Deaths predictor trained")
            print(f"   ✓ Test MAE: {impact_metrics['test_mae']:.2f}")
            print(f"   ✓ Test R²: {impact_metrics['test_r2']:.3f}")
            
            # Test prediction
            test_features = {
                'year': 2024,
                'month': 6,
                'month_sin': np.sin(2 * np.pi * 6 / 12),
                'month_cos': np.cos(2 * np.pi * 6 / 12),
                'duration': 3,
                'hazardtype': 'FLOOD'
            }
            prediction = impact_model.predict(test_features)
            print(f"   ✓ Prediction test: {prediction:.0f} deaths")
            
            # Test save/load
            impact_model.save('test_impact_model.pkl')
            print(f"   ✓ Model saved")
            print()
        else:
            print("   ⚠ Insufficient data for deaths prediction")
            print()
            
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False
    
    # Test 3: Risk Score Predictor
    print("4. Testing Risk Score Predictor...")
    try:
        if 'level2' in df.columns:
            risk_model = RiskScorePredictor(model_type='random_forest')
            risk_metrics = risk_model.train(df)
            print(f"   ✓ Risk score predictor trained")
            print(f"   ✓ MAE: {risk_metrics['mae']:.3f}")
            print(f"   ✓ R²: {risk_metrics['r2']:.3f}")
            
            # Test prediction
            prediction = risk_model.predict(
                event_count=5,
                total_deaths=100,
                total_affected=10000,
                total_loss=1000000,
                hazard_diversity=3
            )
            print(f"   ✓ Prediction test: {prediction:.3f} risk score")
            
            # Test save/load
            risk_model.save('test_risk_model.pkl')
            print(f"   ✓ Model saved")
            print()
        else:
            print("   ⚠ No location data available")
            print()
            
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False
    
    # Test 4: Train all models function
    print("5. Testing train_all_models() function...")
    try:
        results = train_all_models(df, save_models=False)
        print(f"   ✓ Trained {len(results)} models successfully")
        print()
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False
    
    # Cleanup test models
    print("6. Cleaning up test files...")
    try:
        for test_file in ['test_freq_model.pkl', 'test_impact_model.pkl', 'test_risk_model.pkl']:
            test_path = MODELS_DIR / test_file
            if test_path.exists():
                test_path.unlink()
        print("   ✓ Test files cleaned up")
        print()
    except Exception as e:
        print(f"   ⚠ Cleanup warning: {str(e)}")
        print()
    
    print("="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print()
    print("Machine learning models are working correctly.")
    print("You can now:")
    print("  1. Run notebooks/07_machine_learning.ipynb for full demonstration")
    print("  2. Commit the changes to GitHub")
    print()
    
    return True


if __name__ == "__main__":
    success = test_ml_models()
    sys.exit(0 if success else 1)
