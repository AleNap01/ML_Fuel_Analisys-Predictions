import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Configurazione della pagina
st.set_page_config(
    page_title="⛽ Fuel Consumption Predictor",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Codice CSS per lo stile
st.markdown("""
<style>
  :root {
    /* Colori testo default (dark mode) */
    --text-color: white;
    --text-color-light: #f0f0f0;
    --text-color-success: white;
    --text-color-warning: white;
  }
  @media (prefers-color-scheme: light) {
    :root {
      /* Colori testo per tema chiaro */
      --text-color: #111111;
      --text-color-light: #333333;
      --text-color-success: #2f6627;
      --text-color-warning: #8b0000;
    }
  }

  .main > div {padding-top: 2rem;}
  
  .stMetric {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: var(--text-color);
      padding: 1.2rem;
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.15);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255,255,255,0.1);
  }
  
  .stTabs [data-baseweb="tab-list"] {
      gap: 1rem;
      justify-content: center;
      margin-bottom: 1rem;
  }

  .stTabs [data-baseweb="tab"] {
      background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
      color: var(--text-color-light);
      border-radius: 12px;
      padding: 0.8rem 1.5rem;
      border: 1px solid rgba(255, 255, 255, 0.1);
      font-weight: 600;
      backdrop-filter: blur(6px);
      -webkit-backdrop-filter: blur(6px);
      transition: all 0.3s ease;
      box-shadow: 0 4px 10px rgba(0,0,0,0.2);
  }

  .stTabs [data-baseweb="tab"]:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
  }

  .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: var(--text-color);
      box-shadow: 0 6px 24px rgba(102, 126, 234, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.15);
  }
  
  .success-box {
      background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
      padding: 1rem;
      border-radius: 12px;
      color: var(--text-color-success);
      font-weight: 600;
      text-align: center;
      margin: 1rem 0;
  }
  
  .info-card {
      background: linear-gradient(135deg, rgba(102,126,234,0.25) 0%, rgba(118,75,162,0.25) 100%);
      padding: 1.5rem;
      border-radius: 20px;
      color: var(--text-color-light);
      margin: 1.5rem 0;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      transition: transform 0.3s ease, box-shadow 0.3s ease;
  }

  .info-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
  }


</style>
""", unsafe_allow_html=True)

# Creazione classe per la predizione del carburante
class FuelPredictorApp:
    def __init__(self):
        self._init_session_state()
        
    def _init_session_state(self):
        defaults = {
            'connection': None, 'data': None, 'models': {}, 'best_model': None,
            'X_test': None, 'y_test': None, 'results': {}, 'trained': False,
            'fuel_delta_stats': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
    def run(self):
        # Header stilizzato
        st.markdown("""
        <div class="info-card">
            <h1 style="margin: 0; font-size: 2.5rem;">⛽ Fuel Consumption Predictor</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Advanced Machine Learning Dashboard for Fuel Consumption Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        self._setup_sidebar()
        
        if not st.session_state.connection:
            st.markdown("""
            <style>
            :root {
                --warning-text-color: white;
            }
            @media (prefers-color-scheme: light) {
                :root {
                --warning-text-color: #111111;
                }
            }
            </style>

            <div style="text-align: center; padding: 3rem; border-radius: 16px; margin: 2rem 0; color: var(--warning-text-color);">
                <h2 style="margin-bottom: 1rem;">⚠️ Database Connection Required</h2>
                <p style="font-size: 1.1rem;">Please connect to the database using the sidebar to begin</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🤖 ML Training", "📈 Results & Analysis"])
        
        with tab1:
            self._data_overview_page()
        with tab2:
            self._training_page()
        with tab3:
            self._results_page()

        # Quick Actions 
        self._show_quick_actions()
            
    def _setup_sidebar(self):
        with st.sidebar:
            st.markdown("## ⚙️ Configuration Panel")
            
            
            with st.expander("🔗 Database Connection", expanded=True):
                db_config = {
                    'host': st.text_input("🌐 Host", ""),
                    'port': st.text_input("🔌 Port", ""),
                    'dbname': st.text_input("🗄️ Database", ""),
                    'user': st.text_input("👤 User", ""),
                    'password': st.text_input("🔐 Password", "", type="password")

                }
                
                if st.button("🔌 Connect to Database", type="primary", use_container_width=True):
                    self._connect_database(db_config)
            
            # Stato connessione
            if st.session_state.connection:
                st.success("✅ Database Connected Successfully!")
            else:
                st.warning("❌ Not connected to database")
                
    def _connect_database(self, db_config):
        try:
            conn = psycopg2.connect(
                host=db_config['host'], port=db_config['port'],
                dbname=db_config['dbname'], user=db_config['user']
            )
            st.session_state.connection = conn
            st.success("✅ Connected!")
        except Exception as e:
            st.error(f"❌ Connection Error")
            st.session_state.connection = None

         # Caricamento dati del database       
    def _data_overview_page(self):
        st.markdown("### 📊 Data Loading & Exploration")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            vehicle_id = st.text_input("🚗 Vehicle ID", "352625697549655")
            batch_size = st.slider("📦 Max Records", 1000, 50000, 10000)
            
        with col2:
            if st.button("🔄 Load Data", type="primary", use_container_width=True):
                self._load_data(vehicle_id, batch_size)
        
        if st.session_state.data is not None:
            self._display_data_overview()
    
    def _load_data(self, vehicle_id, batch_size):
        with st.spinner("🔄 Loading data from database..."):
            try:
                query = f"""
                SELECT * FROM telemetry_temp
                WHERE ident = '{vehicle_id}' AND can_fuel_consumed IS NOT NULL
                    AND din_1 = true AND engine_ignition_status = true
                    AND position_speed > 0 AND movement_status = true
                    AND can_engine_rpm BETWEEN 1 AND 8000
                    AND position_latitude BETWEEN -90 AND 90
                    AND position_longitude BETWEEN -180 AND 180
                    AND position_satellites > 4
                    AND can_vehicle_mileage < 2000000
                LIMIT {batch_size};
                """
                
                df = pd.read_sql_query(query, st.session_state.connection)
                
                if df.empty:
                    st.error("❌ No data found!")
                    return
                
                # Preprocessing
                df = df.sort_values('timestamp')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df['fuel_delta'] = df['can_fuel_consumed'].diff()
                df = df[df['fuel_delta'] > 0]
                df['fuel_delta_smooth'] = df['fuel_delta'].ewm(span=20, adjust=False).mean()
                
                # Conserva statistiche
                st.session_state.fuel_delta_stats = {
                    'mean': df['fuel_delta_smooth'].mean(),
                    'std': df['fuel_delta_smooth'].std(),
                    'q25': df['fuel_delta_smooth'].quantile(0.25),
                    'q75': df['fuel_delta_smooth'].quantile(0.75)
                }
                
                st.session_state.data = df
                st.balloons()
                st.success(f"✅ Successfully loaded {len(df):,} records")
                
            except Exception as e:
                st.error(f"❌ Loading Error: {str(e)}")
    
    def _display_data_overview(self):
        data = st.session_state.data
        
        # Display delle metriche del dataset
        st.markdown("### 📈 Dataset Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = [
            ("📊", "Records", f"{len(data):,}"),
            ("⛽", "Avg Fuel", f"{data['fuel_delta'].mean():.4f}"),
            ("🚗", "Avg Speed", f"{data['position_speed'].mean():.1f} km/h"),
            ("🔄", "Avg RPM", f"{data['can_engine_rpm'].mean():.0f}"),
            ("🔋", "Avg Voltage", f"{data['battery_voltage'].mean():.1f}V")
        ]
        
        for col, (icon, label, value) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.metric(f"{icon} {label}", value)
        
        # Visualizzazione dei dati
        st.markdown("### 📊 Data Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_fuel = go.Figure()
            fig_fuel.add_trace(go.Scatter(
                x=data.index, y=data['fuel_delta'],
                mode='lines', name='Original', line=dict(width=1, color='lightblue')
            ))
            fig_fuel.add_trace(go.Scatter(
                x=data.index, y=data['fuel_delta_smooth'],
                mode='lines', name='Smoothed', line=dict(width=2, color='#667eea')
            ))
            fig_fuel.update_layout(
                title="⛽ Fuel Consumption Over Time",
                xaxis_title="Timestamp", yaxis_title="Fuel Delta",
                hovermode='x unified', height=450, title_font_size=16, title_x=0.5
            )
            st.plotly_chart(fig_fuel, use_container_width=True)
        
        with col2:
            sample_data = data.sample(min(1000, len(data)))
            fig_scatter = px.scatter(
                sample_data, x='position_speed', y='fuel_delta_smooth',
                title="🚗 Speed vs Fuel Consumption",
                color='can_engine_rpm', color_continuous_scale='viridis'
            )
            fig_scatter.update_layout(height=450, title_font_size=16, title_x=0.5)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Distribuzione grafica consumo carburante
        fig_hist = px.histogram(
            data, x='fuel_delta_smooth', nbins=50,
            title="📈 Fuel Consumption Distribution",
            color_discrete_sequence=['#764ba2']
        )
        fig_hist.update_layout(title_font_size=16, title_x=0.5)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Data Sample Preview
        st.markdown("### 🔍 Data Sample Preview")
        display_cols = ['battery_voltage', 'can_engine_rpm', 'position_speed', 'fuel_delta', 'fuel_delta_smooth']
        st.dataframe(data[display_cols].head(10), use_container_width=True)
            
    def _training_page(self):
        st.markdown("### 🤖 Machine Learning Training Center")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please load data first from the Data Explorer tab")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🎯 Model Selection")
            models_to_train = {
                "🌲 Random Forest": st.checkbox("Random Forest", True),
                "🚀 Gradient Boosting": st.checkbox("Gradient Boosting", True),
                "📊 Linear Regression": st.checkbox("Linear Regression", True),
                "⚡ XGBoost": st.checkbox("XGBoost", False)
            }
            
            st.markdown("#### ⚙️ Training Configuration")
            test_size = st.slider("📊 Test Size", 0.1, 0.4, 0.2)
            random_state = st.number_input("🎲 Random State", 1, 1000, 42)
            
        with col2:
            st.markdown("#### 📊 Training Status")
            status_color = "🟢" if st.session_state.trained else "🔴"
            st.markdown(f"""
            <div style="background: {'linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%)' if st.session_state.trained else 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)'}; 
                        padding: 1rem; border-radius: 12px; color: white; text-align: center;">
                <h4 style="margin: 0;">{status_color} Status: {'Models Trained' if st.session_state.trained else 'Ready to Train'}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"📊 Available Records: {len(st.session_state.data):,}")
        
        if st.button("🚀 Start Training Process", type="primary", use_container_width=True):
            selected_models = [name.split(' ', 1)[1] for name, selected in models_to_train.items() if selected]
            
            if not selected_models:
                st.error("❌ Please select at least one model to train")
                return
                
            self._train_models(selected_models, test_size, int(random_state))
        
    def _train_models(self, selected_models, test_size, random_state):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Preprocessing data...")
            progress_bar.progress(20)
            
            # Preparazione delle features
            feature_cols = ['battery_voltage', 'can_engine_rpm', 'can_throttle_pedal_level', 'position_speed', 'vehicle_mileage']
            target_col = 'fuel_delta_smooth'
            
            df_model = st.session_state.data[feature_cols + [target_col]].dropna()
            df_model = df_model.copy()
            df_model['vehicle_mileage'] = df_model['vehicle_mileage'] / 10000
            
            X = df_model[feature_cols]
            y = df_model[target_col]
            
            status_text.text("🎯 Splitting data...")
            progress_bar.progress(40)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            st.session_state.X_test, st.session_state.y_test = X_test, y_test
            
            status_text.text("🤖 Training models...")
            progress_bar.progress(60)
            
            # Definizione dei modelli
            model_dict = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
                "Gradient Boosting": HistGradientBoostingRegressor(random_state=random_state),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            }
            
            # Allenamento dei modelli
            results = {}
            models = {}
            
            for name in selected_models:
                if name in model_dict:
                    model = model_dict[name]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {"mse": mse, "r2": r2}
                    models[name] = model
            
            st.session_state.results = results
            st.session_state.models = models
            self._select_best_model()
            st.session_state.trained = True
            
            progress_bar.progress(100)
            status_text.text("✅ Training completed!")
            
            st.markdown(f"""
            <div class="success-box">
                🎉 Successfully trained {len(models)} models!
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    def _select_best_model(self):
        if not st.session_state.results:
            return
            
        # Seleziona il modello migliore in base alla regolarizzazione R²
        best_model_name = max(st.session_state.results, key=lambda x: st.session_state.results[x]['r2'])
        st.session_state.best_model = {
            'name': best_model_name,
            'model': st.session_state.models[best_model_name]
        }
    
    def _results_page(self):
        st.markdown("### 📈 Results & Performance Analysis")
        
        if not st.session_state.trained:
            st.warning("⚠️ Please train models first in the ML Training tab")
            return
        
        # Metriche finali
        results_df = pd.DataFrame(st.session_state.results).T
        results_df = results_df.round(4)
        
        st.markdown("#### 🎯 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        best_r2 = results_df['r2'].max()
        best_mse = results_df['mse'].min()
        avg_r2 = results_df['r2'].mean()
        avg_mse = results_df['mse'].mean()
        
        metrics_data = [
            ("🏆", "Best R²", best_r2, "#667eea"),
            ("🎯", "Best MSE", best_mse, "#764ba2"),
            ("📊", "Avg R²", avg_r2, "#56ab2f"),
            ("📈", "Avg MSE", avg_mse, "#ff9a9e")
        ]
        
        for col, (icon, name, value, color) in zip([col1, col2, col3, col4], metrics_data):
            with col:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color}80 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
                    <h3 style="margin: 0; font-size: 1.8rem;">{icon}</h3>
                    <h4 style="margin: 0.5rem 0 0 0;">{name}</h4>
                    <h2 style="margin: 0.5rem 0 0 0; font-size: 1.5rem;">{value:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Comparazione dei modelli
        st.markdown("#### 🏆 Model Performance Comparison")
        st.dataframe(results_df, use_container_width=True)
        
        if st.session_state.best_model:
            st.success(f"🏆 Best Model: **{st.session_state.best_model['name']}**")
        
        # Visualizzazione dei risultati
        col1, col2 = st.columns(2)
        
        with col1:
            fig_r2 = px.bar(
                x=list(st.session_state.results.keys()),
                y=[v['r2'] for v in st.session_state.results.values()],
                title="📈 R² Score Comparison (Higher is Better)",
                color=list(st.session_state.results.keys()),
                color_discrete_sequence=['#667eea', '#764ba2', '#56ab2f', '#ff9a9e']
            )
            fig_r2.update_layout(showlegend=False, height=400, title_font_size=16, title_x=0.5)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            fig_mse = px.bar(
                x=list(st.session_state.results.keys()),
                y=[v['mse'] for v in st.session_state.results.values()],
                title="📉 MSE Comparison (Lower is Better)",
                color=list(st.session_state.results.keys()),
                color_discrete_sequence=['#ff9a9e', '#56ab2f', '#764ba2', '#667eea']
            )
            fig_mse.update_layout(showlegend=False, height=400, title_font_size=16, title_x=0.5)
            st.plotly_chart(fig_mse, use_container_width=True)
        
        # Importanza delle features
        if st.session_state.best_model and hasattr(st.session_state.best_model['model'], 'feature_importances_'):
            st.markdown("#### 🔝 Feature Importance Analysis")
            feature_names = ['battery_voltage', 'can_engine_rpm', 'can_throttle_pedal_level', 'position_speed', 'vehicle_mileage']
            
            importances = st.session_state.best_model['model'].feature_importances_
            feat_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            fig_imp = px.bar(
                feat_imp_df, x='importance', y='feature', orientation='h',
                title="🔝 Feature Importance", color='importance',
                color_continuous_scale='viridis'
            )
            fig_imp.update_layout(height=400, title_font_size=16, title_x=0.5)
            st.plotly_chart(fig_imp, use_container_width=True)

    def _show_quick_actions(self):
        st.markdown("---")
        st.markdown("### 🛠️ Quick Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear Data", type="secondary", use_container_width=True):
                st.session_state.data = None
                st.success("✅ Data cleared successfully!")
                st.rerun()

        with col2:
            if st.button("🔄 Reset Models", type="secondary", use_container_width=True):
                st.session_state.trained = False
                st.session_state.results = {}
                st.session_state.models = {}
                st.success("✅ Models reset successfully!")
                st.rerun()

        with col3:
            if st.button("🗑️ Reset Everything", type="secondary", use_container_width=True):
                st.session_state.clear()
                st.success("✅ Everything reset successfully!")
                st.rerun()

# Avvio per l'applicazione
if __name__ == "__main__":
    app = FuelPredictorApp()
    app.run()