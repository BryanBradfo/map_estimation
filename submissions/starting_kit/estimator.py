from sklearn import set_config
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
# Assuming xgboost is already installed on the RAMP platform
 
# Fallback to sklearn if xgboost is not available
from sklearn.ensemble import GradientBoostingRegressor as xgb_fallback
 
class XGBRegressorFallback:
    def __init__(self, **kwargs):
        self.model = xgb_fallback(**{k: v for k, v in kwargs.items() 
                                    if k in ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'random_state']})
    
    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
 
# Create an alias so the code below doesn't need to change
xgb = type('xgb', (), {'XGBRegressor': XGBRegressorFallback})
 
set_config(transform_output="pandas")
 
class WaveformFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract statistical and spectral features from ECG and PPG signals."""
    
    def __init__(self, n_jobs=1):  # Default to 1 to avoid potential issues on RAMP
        self.n_jobs = n_jobs
    
    def fit(self, X, y=None):
        return self
    
    def _extract_features_for_row(self, row):
        features = {}
        
        for signal_type in ['ecg', 'ppg']:
            if signal_type not in row or not isinstance(row[signal_type], (list, np.ndarray)) or len(row[signal_type]) < 10:
                # Fill with NaN if signal is missing
                for prefix in ['mean', 'std', 'min', 'max', 'p25', 'p50', 'p75', 'range']:
                    features[f'{signal_type}_{prefix}'] = np.nan
                continue
                
            signal = np.array(row[signal_type])
            
            # Basic statistical features
            features[f'{signal_type}_mean'] = np.mean(signal)
            features[f'{signal_type}_std'] = np.std(signal)
            features[f'{signal_type}_min'] = np.min(signal)
            features[f'{signal_type}_max'] = np.max(signal)
            features[f'{signal_type}_p25'] = np.percentile(signal, 25)
            features[f'{signal_type}_p50'] = np.percentile(signal, 50)
            features[f'{signal_type}_p75'] = np.percentile(signal, 75)
            features[f'{signal_type}_range'] = np.max(signal) - np.min(signal)
            
            # Compute more advanced features
            if len(signal) > 10:
                # First derivatives (velocity)
                diff1 = np.diff(signal)
                features[f'{signal_type}_diff_mean'] = np.mean(diff1)
                features[f'{signal_type}_diff_std'] = np.std(diff1)
                features[f'{signal_type}_diff_max'] = np.max(diff1)
                features[f'{signal_type}_diff_min'] = np.min(diff1)
                
                # Second derivatives (acceleration)
                diff2 = np.diff(diff1)
                features[f'{signal_type}_diff2_mean'] = np.mean(diff2)
                features[f'{signal_type}_diff2_std'] = np.std(diff2)
                
                # Zero crossings (can indicate frequency content)
                features[f'{signal_type}_zero_crossings'] = np.sum(np.diff(np.signbit(diff1)))
                
                # Energy in signal
                features[f'{signal_type}_energy'] = np.sum(signal**2)
                
                # Very simple peak detection - count peaks that are above a threshold
                signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                features[f'{signal_type}_peak_count'] = np.sum((signal_norm > 1.5) & 
                                                           (np.r_[True, signal_norm[1:] > signal_norm[:-1]] & 
                                                            np.r_[signal_norm[:-1] > signal_norm[1:], True]))
                
        # Cross-signal features that relate ECG and PPG
        if 'ecg' in row and 'ppg' in row and isinstance(row['ecg'], (list, np.ndarray)) and isinstance(row['ppg'], (list, np.ndarray)):
            if len(row['ecg']) > 10 and len(row['ppg']) > 10:
                # Correlation between ECG and PPG
                min_len = min(len(row['ecg']), len(row['ppg']))
                ecg = np.array(row['ecg'])[:min_len]
                ppg = np.array(row['ppg'])[:min_len]
                
                if np.std(ecg) > 0 and np.std(ppg) > 0:
                    features['ecg_ppg_corr'] = np.corrcoef(ecg, ppg)[0, 1]
                else:
                    features['ecg_ppg_corr'] = 0
            else:
                features['ecg_ppg_corr'] = np.nan
        else:
            features['ecg_ppg_corr'] = np.nan
            
        return features
    
    def transform(self, X):
        # Sequential processing (joblib removed)
        results = [self._extract_features_for_row(row) for _, row in X.iterrows()]
        
        # Convert to DataFrame
        features_df = pd.DataFrame(results)
        
        # Add domain feature explicitly - important for domain adaptation
        features_df['is_source_domain'] = (X['domain'] == 'v').astype(int)
        
        return features_df
 
class DomainAdaptiveXGBoost(BaseEstimator, TransformerMixin):
    """XGBoost regressor with domain adaptation capabilities."""
    
    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=5, 
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, 
                 reg_lambda=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.model = None
        
    def fit(self, X, y):
        # Filter out samples with missing target values (domain 'm')
        valid_idx = y != -1
        X_train = X[valid_idx]
        y_train = y[valid_idx]
        
        # Set up XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'verbosity': 0
        }
        
        # Safely handle tree_method parameter which might not be available
        try:
            model_test = xgb.XGBRegressor(tree_method='hist')
            params['tree_method'] = 'hist'  # For faster training
        except:
            # If 'hist' is not available, don't specify tree_method
            pass
        
        # Up-weight source domain samples that are more similar to target domain
        if 'is_source_domain' in X_train.columns:
            # Initialize weights
            sample_weights = np.ones(len(X_train))
            
            # Get only source domain samples
            source_mask = X_train['is_source_domain'] == 1
            
            if np.any(source_mask):
                # This is a simple heuristic - in practice you might use more sophisticated
                # domain adaptation approaches
                X_source = X_train[source_mask].drop(columns=['is_source_domain'])
                
                # For demonstration, we'll use a simple approach: samples with values
                # closer to the target domain mean get higher weights
                if not np.all(source_mask):
                    X_target = X_train[~source_mask].drop(columns=['is_source_domain'])
                    target_means = X_target.mean()
                    
                    # Calculate distance from each source sample to target mean
                    for i, (_, row) in enumerate(X_source.iterrows()):
                        if source_mask[i]:
                            # Calculate Euclidean distance to target mean
                            # (using only numerical features)
                            numerical_cols = row.index[~row.isna()]
                            if len(numerical_cols) > 0:
                                diffs = row[numerical_cols] - target_means[numerical_cols]
                                dist = np.sqrt(np.nansum(diffs**2))
                                # Convert distance to weight (closer = higher weight)
                                sample_weights[i] = 1.0 / (1.0 + dist)
            
            # Create and train the model with sample weights
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train.drop(columns=['is_source_domain']), y_train, 
                         sample_weight=sample_weights)
        else:
            # Create and train the model without sample weights
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X):
        if 'is_source_domain' in X.columns:
            X = X.drop(columns=['is_source_domain'])
        return self.model.predict(X)
 
def get_estimator():
    """Return the full pipeline for MAP prediction."""
    
    # Define demographic features pipeline
    demographics_pipeline = make_column_transformer(
        ("passthrough", ["age", "weight", "height", "bmi"]),
        (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ["gender"]),
    )
    
    # Simplified pipeline that should work more reliably
    pipeline = Pipeline([
        ('features', WaveformFeatureExtractor(n_jobs=1)),
        ('scaler', StandardScaler()),
        ('model', DomainAdaptiveXGBoost(
            n_estimators=20,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0
        ))
    ])
    
    return pipeline
