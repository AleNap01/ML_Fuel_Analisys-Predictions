import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class DrivingStylePredictor:
    def __init__(self, db_config, config=None):
        self.db_config = db_config
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')        # Configurazione di default ottimizzata per sessioni
        self.config = {
            'session_params': {
                'time_gap_minutes': 30,  # Gap per separare sessioni
                'min_session_duration': 5,  # Durata minima sessione (minuti)
                'min_session_points': 10,  # Punti minimi per sessione valida
                'speed_threshold': 0  # Velocità minima per considerare movimento
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'gradient_boosting': {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.1,
                'random_state': 42,
                'subsample': 0.8
            },
            'logistic_regression': {
                'random_state': 42,
                'max_iter': 1000,
                'C': 1.0,
                'solver': 'lbfgs',
                'penalty': 'l2',
                'class_weight': 'balanced'
            },
            'svm': {
                'random_state': 42,
                'probability': True,
                'C': 1.0,
                'kernel': 'rbf',
                'class_weight': 'balanced'
            },
            'feature_selection': {
                'k_features': 15,
                'method': 'mutual_info'
            },
            'training': {
                'cv_folds': 5,
                'test_size': 0.2,
                'random_state': 42
            }
        }
        if config:
            self.update_config(config)     
        self._initialize_feature_selector()
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.is_fitted = False
        
    def _initialize_feature_selector(self):
        if self.config['feature_selection']['method'] == 'mutual_info':
            self.feature_selector = SelectKBest(mutual_info_classif, k=self.config['feature_selection']['k_features'])
        elif self.config['feature_selection']['method'] == 'rfe':
            base_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFE(base_estimator, n_features_to_select=self.config['feature_selection']['k_features'])
        else:
            self.feature_selector = SelectKBest(f_classif, k=self.config['feature_selection']['k_features'])

    def update_config(self, new_config):
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        deep_update(self.config, new_config)
        self._initialize_feature_selector()

    def get_config(self):
        return self.config.copy()
    
    def connect_database(self):
        try:
            return psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['dbname'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
        except Exception as e:
            return e
        
    def load_data(self, query=None):
        connection = self.connect_database()
        if connection is None:
            return None
        if query is None:
            query = """
            SELECT 
                id, timestamp, ident, battery_voltage, can_engine_rpm, 
                can_pedal_brake_status, can_throttle_pedal_level, 
                engine_ignition_status, movement_status, position_altitude,
                position_direction, position_latitude, position_longitude, 
                position_satellites, position_speed, vehicle_mileage, 
                din_1, limite_velocita, tipo_strada, stile_guida
            FROM telemetry_temp
            WHERE stile_guida IS NOT NULL 
            AND din_1 = true 
            AND engine_ignition_status = true
            AND position_speed > 0
            AND movement_status = true
            AND can_engine_rpm BETWEEN 1 AND 8000
            AND position_latitude BETWEEN -90 AND 90
            AND position_longitude BETWEEN -180 AND 180
            AND position_altitude BETWEEN -500 AND 10000
            AND position_direction BETWEEN 1 AND 359
            AND position_satellites > 4
            AND vehicle_mileage < 2000000
            ORDER BY ident, timestamp
            """
        try:
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            records = cursor.fetchall()
            df = pd.DataFrame(records)
            cursor.close()
            connection.close()
            if not df.empty:
                print(f"Dati grezzi caricati: {len(df)} righe")
                # Converti timestamp se è stringa
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return self._create_driving_sessions(df)
            else:
                print("Nessun dato trovato")
                return None
        except Exception as e:
            print(f"Errore caricamento dati: {e}")
            if connection:
                connection.close()
            return None
        
    def load_batch_data(self, batch_size=10000, query=None):
        connection = self.connect_database()
        if connection is None:
            return None
        if query is None:
            query = """
            SELECT 
                id, timestamp, ident, battery_voltage, can_engine_rpm, 
                can_pedal_brake_status, can_throttle_pedal_level, 
                engine_ignition_status, movement_status, position_altitude,
                position_direction, position_latitude, position_longitude, 
                position_satellites, position_speed, vehicle_mileage, 
                din_1, limite_velocita, tipo_strada, stile_guida
            FROM telemetry_temp
            WHERE stile_guida IS NOT NULL 
            AND din_1 = true 
            AND engine_ignition_status = true
            AND position_speed > 0
            AND movement_status = true
            AND can_engine_rpm BETWEEN 1 AND 8000
            AND position_latitude BETWEEN -90 AND 90
            AND position_longitude BETWEEN -180 AND 180
            AND position_altitude BETWEEN -500 AND 10000
            AND position_direction BETWEEN 1 AND 359
            AND position_satellites > 4
            AND vehicle_mileage < 2000000
            ORDER BY ident, timestamp
            """
        try:
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            all_data = []
            while True:
                records = cursor.fetchmany(batch_size)
                if not records:
                    break
                all_data.extend(records)
                print(f"Caricato batch: {len(records)} righe (Totale: {len(all_data)})")
            cursor.close()
            connection.close()
            if all_data:
                df = pd.DataFrame(all_data)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"Dati grezzi totali: {len(df)} righe")
                return self._create_driving_sessions(df)
            else:
                return None
        except Exception as e:
            print(f"Errore caricamento batch: {e}")
            if connection:
                connection.close()
            return None
        
    def _create_driving_sessions(self, df):
        print("Creazione sessioni di guida...")
        sessions = []
        session_params = self.config['session_params']
        for vehicle_id in df['ident'].unique():
            vehicle_data = df[df['ident'] == vehicle_id].sort_values('timestamp').copy()
            vehicle_data['time_diff'] = vehicle_data['timestamp'].diff().dt.total_seconds() / 60
            session_starts = (vehicle_data['time_diff'] > session_params['time_gap_minutes']) | vehicle_data['time_diff'].isna()
            vehicle_data['session_id'] = session_starts.cumsum()
            for session_id in vehicle_data['session_id'].unique():
                session_data = vehicle_data[vehicle_data['session_id'] == session_id].copy()
                duration_minutes = (session_data['timestamp'].max() - session_data['timestamp'].min()).total_seconds() / 60
                if (len(session_data) >= session_params['min_session_points'] and 
                    duration_minutes >= session_params['min_session_duration'] and
                    session_data['position_speed'].max() >= session_params['speed_threshold']):
                    session_features = self._calculate_session_features(session_data)
                    if session_features:
                        sessions.append(session_features)
        if sessions:
            sessions_df = pd.DataFrame(sessions)
            print(f"Sessioni create: {len(sessions_df)} da {len(df)} record grezzi")
            return sessions_df
        else:
            print("Nessuna sessione valida creata")
            return None
        
    def _calculate_session_features(self, session_data):
        try:
            features = {
                'ident': session_data['ident'].iloc[0],
                'stile_guida': session_data['stile_guida'].mode().iloc[0] if not session_data['stile_guida'].mode().empty else session_data['stile_guida'].iloc[0],
                'session_duration': (session_data['timestamp'].max() - session_data['timestamp'].min()).total_seconds() / 60,
                'total_points': len(session_data)
            }
            def safe_numeric_conversion(series):
                try:
                    return pd.to_numeric(series, errors='coerce').fillna(0)
                except:
                    return pd.Series([0] * len(series))
            def safe_division(numerator, denominator, default=0):
                try:
                    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
                        return default
                    result = numerator / denominator
                    return result if np.isfinite(result) else default
                except:
                    return default
            def safe_mean(series, default=0):
                try:
                    if len(series) == 0:
                        return default
                    result = float(series.mean())
                    return result if np.isfinite(result) else default
                except:
                    return default
            def safe_std(series, default=0):
                try:
                    if len(series) <= 1:
                        return default
                    result = float(series.std())
                    return result if np.isfinite(result) else default
                except:
                    return default
            def safe_max(series, default=0):
                try:
                    if len(series) == 0:
                        return default
                    result = float(series.max())
                    return result if np.isfinite(result) else default
                except:
                    return default
            def safe_quantile(series, q, default=0):
                try:
                    if len(series) == 0:
                        return default
                    result = float(series.quantile(q))
                    return result if np.isfinite(result) else default
                except:
                    return default
            speed_data = safe_numeric_conversion(session_data['position_speed'])
            speed_data = speed_data[speed_data >= 0]  # Rimuovi velocità negative
            if len(speed_data) > 0:
                features.update({
                    'avg_speed': safe_mean(speed_data),
                    'max_speed': safe_max(speed_data),
                    'speed_std': safe_std(speed_data),
                    'speed_95_percentile': safe_quantile(speed_data, 0.95),
                    'time_above_50': safe_division((speed_data > 50).sum(), len(speed_data)) * 100
                })
            else:
                features.update({
                    'avg_speed': 0,
                    'max_speed': 0,
                    'speed_std': 0,
                    'speed_95_percentile': 0,
                    'time_above_50': 0
                })
            rpm_data = safe_numeric_conversion(session_data['can_engine_rpm'])
            rpm_data = rpm_data[(rpm_data >= 0) & (rpm_data <= 10000)]  # Filtra valori ragionevoli
            if len(rpm_data) > 0:
                features.update({
                    'avg_rpm': safe_mean(rpm_data),
                    'max_rpm': safe_max(rpm_data),
                    'rpm_std': safe_std(rpm_data),
                    'high_rpm_ratio': safe_division((rpm_data > 3000).sum(), len(rpm_data)) * 100
                })
            else:
                features.update({
                    'avg_rpm': 0,
                    'max_rpm': 0,
                    'rpm_std': 0,
                    'high_rpm_ratio': 0
                })
            if len(speed_data) > 1:
                try:
                    speed_changes = speed_data.diff().dropna()
                    speed_changes = speed_changes[np.isfinite(speed_changes)]  # Rimuovi inf/nan
                    positive_changes = speed_changes[speed_changes > 0]
                    negative_changes = speed_changes[speed_changes < 0]
                    features.update({
                        'avg_acceleration': safe_mean(positive_changes),
                        'avg_deceleration': abs(safe_mean(negative_changes)),
                        'harsh_acceleration_count': int((speed_changes > 5).sum()),
                        'harsh_braking_count': int((speed_changes < -5).sum())
                    })
                except:
                    features.update({
                        'avg_acceleration': 0,
                        'avg_deceleration': 0,
                        'harsh_acceleration_count': 0,
                        'harsh_braking_count': 0
                    })
            else:
                features.update({
                    'avg_acceleration': 0,
                    'avg_deceleration': 0,
                    'harsh_acceleration_count': 0,
                    'harsh_braking_count': 0
                })
            try:
                brake_data = safe_numeric_conversion(session_data['can_pedal_brake_status'])
                brake_data = brake_data[brake_data >= 0]  # Solo valori positivi
                features.update({
                    'brake_usage_ratio': safe_division((brake_data > 0).sum(), len(brake_data)) * 100,
                    'avg_brake_intensity': safe_mean(brake_data)
                })
            except:
                features.update({
                    'brake_usage_ratio': 0,
                    'avg_brake_intensity': 0
                })
            try:
                throttle_data = safe_numeric_conversion(session_data['can_throttle_pedal_level'])
                throttle_data = throttle_data[(throttle_data >= 0) & (throttle_data <= 100)]  # 0-100%
                features.update({
                    'avg_throttle': safe_mean(throttle_data),
                    'max_throttle': safe_max(throttle_data),
                    'aggressive_throttle_ratio': safe_division((throttle_data > 80).sum(), len(throttle_data)) * 100
                })
            except:
                features.update({
                    'avg_throttle': 0,
                    'max_throttle': 0,
                    'aggressive_throttle_ratio': 0
                })
            try:
                if len(speed_data) > 0 and len(rpm_data) > 0:
                    avg_speed = safe_mean(speed_data)
                    avg_rpm = safe_mean(rpm_data)
                    features['fuel_efficiency_score'] = safe_division(avg_speed, avg_rpm / 1000)
                else:
                    features['fuel_efficiency_score'] = 0
            except:
                features['fuel_efficiency_score'] = 0
            try:
                if 'limite_velocita' in session_data.columns:
                    limit_data = safe_numeric_conversion(session_data['limite_velocita'])
                    limit_data = limit_data[limit_data > 0]  # Solo limiti positivi
                    if len(limit_data) > 0 and len(speed_data) > 0:
                        min_len = min(len(speed_data), len(limit_data))
                        speed_vs_limit = speed_data.iloc[:min_len] / limit_data.iloc[:min_len]
                        speed_vs_limit = speed_vs_limit[np.isfinite(speed_vs_limit)]
                        features.update({
                            'avg_speed_vs_limit': safe_mean(speed_vs_limit),
                            'speeding_ratio': safe_division((speed_vs_limit > 1.1).sum(), len(speed_vs_limit)) * 100
                        })
                    else:
                        features.update({
                            'avg_speed_vs_limit': 0,
                            'speeding_ratio': 0
                        })
                else:
                    features.update({
                        'avg_speed_vs_limit': 0,
                        'speeding_ratio': 0
                    })
            except:
                features.update({
                    'avg_speed_vs_limit': 0,
                    'speeding_ratio': 0
                })
            try:
                speed_mean = safe_mean(speed_data)
                rpm_mean = safe_mean(rpm_data)
                features.update({
                    'speed_variability': safe_division(safe_std(speed_data), speed_mean),
                    'rpm_variability': safe_division(safe_std(rpm_data), rpm_mean)
                })
            except:
                features.update({
                    'speed_variability': 0,
                    'rpm_variability': 0
                })
            final_features = {}
            for key, value in features.items():
                try:
                    if key in ['ident', 'stile_guida']:
                        final_features[key] = value
                    else:
                        if pd.isna(value) or not np.isfinite(float(value)):
                            final_features[key] = 0.0
                        else:
                            final_features[key] = float(value)
                except:
                    final_features[key] = 0.0 if key not in ['ident', 'stile_guida'] else value
            return final_features
        except Exception as e:
            print(f"Errore calcolo features sessione: {e}")
            return None
        
    def get_data_statistics(self, df):
        if df is None or df.empty:
            return {'general_stats': {'total_records': 0}, 'style_distribution': [], 'session_analysis': {}}
        stats = {
            'general_stats': {
                'total_sessions': len(df),
                'unique_vehicles': df['ident'].nunique() if 'ident' in df.columns else 0,
                'unique_driving_styles': df['stile_guida'].nunique() if 'stile_guida' in df.columns else 0,
                'avg_session_duration': round(df['session_duration'].mean(), 2) if 'session_duration' in df.columns else 0,
                'avg_speed_overall': round(df['avg_speed'].mean(), 2) if 'avg_speed' in df.columns else 0
            },
            'style_distribution': [],
            'session_analysis': {
                'avg_points_per_session': round(df['total_points'].mean(), 0) if 'total_points' in df.columns else 0,
                'longest_session_minutes': round(df['session_duration'].max(), 2) if 'session_duration' in df.columns else 0,
                'shortest_session_minutes': round(df['session_duration'].min(), 2) if 'session_duration' in df.columns else 0
            }
        }
        if 'stile_guida' in df.columns:
            style_counts = df['stile_guida'].value_counts()
            total = len(df)
            for style, count in style_counts.items():
                stats['style_distribution'].append({
                    'stile_guida': str(style),
                    'count': int(count),
                    'percentage': round((count / total) * 100, 2)
                })
        return stats
    
    def preprocess_data(self, df):
        if df is None or df.empty:
            raise ValueError("DataFrame vuoto")
        data = df.copy()
        self.feature_names = [
            'session_duration', 'total_points', 'avg_speed', 'max_speed', 'speed_std',
            'speed_95_percentile', 'time_above_50', 'avg_rpm', 'max_rpm', 'rpm_std',
            'high_rpm_ratio', 'avg_acceleration', 'avg_deceleration', 'harsh_acceleration_count',
            'harsh_braking_count', 'brake_usage_ratio', 'avg_brake_intensity', 'avg_throttle',
            'max_throttle', 'aggressive_throttle_ratio', 'fuel_efficiency_score',
            'avg_speed_vs_limit', 'speeding_ratio', 'speed_variability', 'rpm_variability'
        ]
        available_features = [col for col in self.feature_names if col in data.columns]
        self.feature_names = available_features
        for col in ['avg_rpm', 'avg_throttle', 'max_throttle','rpm_std']:
            if col in df.columns:
                df[col] = np.sqrt(df[col]) * 0.5    
        if len(available_features) == 0:
            raise ValueError("Nessuna feature disponibile")
        X = data[available_features].fillna(0)
        y = data['stile_guida']
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        print(f"Features utilizzate: {len(available_features)}")
        print(f"Sessioni dopo preprocessing: {len(X)}")
        print(f"Classi target: {np.unique(y)}")
        return X, y
    
    def split_data(self, X, y, test_size=None, random_state=None):
        if test_size is None:
            test_size = self.config['training']['test_size']
        if random_state is None:
            random_state = self.config['training']['random_state']
        unique_classes = np.unique(y)
        min_class_count = min([np.sum(y == cls) for cls in unique_classes])
        if min_class_count < 2:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    def get_model_instance(self, model_name):
        model_map = {
            'Random Forest': RandomForestClassifier(**self.config['random_forest']),
            'Gradient Boosting': GradientBoostingClassifier(**self.config['gradient_boosting']),
            'Logistic Regression': LogisticRegression(**self.config['logistic_regression']),
            'SVM': SVC(**self.config['svm'])
        }
        return model_map.get(model_name)
    from sklearn.preprocessing import LabelEncoder

    def train_models(self, X_train, y_train, selected_models):
        print("\n=== ADDESTRAMENTO MODELLI SU SESSIONI ===")
        try:
            # Fitting del LabelEncoder
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_train)

            # Preprocessing
            X_train_imputed = self.imputer.fit_transform(X_train)
            X_train_scaled = self.scaler.fit_transform(X_train_imputed)
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_encoded)

            cv_scores = {}
            cv_folds = self.config['training']['cv_folds']

            for model_name in selected_models:
                try:
                    model = self.get_model_instance(model_name)
                    if model is None:
                        continue
                    # Valutazione e addestramento
                    scores = cross_val_score(model, X_train_selected, y_encoded, cv=cv_folds, scoring='f1_weighted')
                    cv_scores[model_name] = scores.mean()
                    model.fit(X_train_selected, y_encoded)
                    self.models[model_name] = model
                    print(f"{model_name} - Score: {cv_scores[model_name]:.4f}")
                except Exception as e:
                    print(f"Errore {model_name}: {e}")
                    continue

            if cv_scores:
                best_model_name = max(cv_scores, key=cv_scores.get)
                self.best_model = self.models[best_model_name]
                self.is_fitted = True
                print(f"\nMiglior modello: {best_model_name} (Score: {cv_scores[best_model_name]:.4f})")
            else:
                raise ValueError("Nessun modello addestrato")

            return self.models
        except Exception as e:
            print(f"Errore addestramento: {e}")
            raise e

    def evaluate_model(self, X_test, y_test):
        if not self.is_fitted:
            raise ValueError("Nessun modello addestrato")
        try:
            X_test_imputed = self.imputer.transform(X_test)
            X_test_scaled = self.scaler.transform(X_test_imputed)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            y_pred = self.best_model.predict(X_test_selected)
            y_pred_proba = self.best_model.predict_proba(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            print(f"\nRisultati valutazione su sessioni:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        except Exception as e:
            print(f"Errore valutazione: {e}")
            raise e
        
    def feature_importance(self):
        if not self.is_fitted or not hasattr(self.best_model, 'feature_importances_'):
            return None
        try:
            if hasattr(self.feature_selector, 'get_support'):
                selected_mask = self.feature_selector.get_support()
                selected_features = [self.feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
            else:
                selected_features = self.feature_names[:len(self.best_model.feature_importances_)]
            importances = self.best_model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return feature_imp
        except Exception as e:
            print(f"Errore feature importance: {e}")
            return None
