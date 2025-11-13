import streamlit as st
import pandas as pd
import plotly.express as px
from DrivingStylePredictor import DrivingStylePredictor

st.set_page_config(
    page_title="🚗 AI Driving Style Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# css
st.markdown("""
<style>
  .main > div {padding-top: 2rem;}
  
  /* --- Modalità scura (default) --- */
  .stMetric {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
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
      color: #f0f0f0;
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
      color: white;
      box-shadow: 0 6px 24px rgba(102, 126, 234, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.15);
  }

  .sidebar .stSelectbox, .sidebar .stSlider {
      margin-bottom: 1rem;
  }
  
  .success-box {
      background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
      padding: 1rem;
      border-radius: 12px;
      color: white;
      font-weight: 600;
      text-align: center;
      margin: 1rem 0;
  }
  
  .info-card {
      background: linear-gradient(135deg, rgba(102,126,234,0.25) 0%, rgba(118,75,162,0.25) 100%);
      padding: 1.5rem;
      border-radius: 20px;
      color: #f0f0f0;
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

  /* --- Modalità chiara --- */
  @media (prefers-color-scheme: light) {
    .stMetric {
      background: linear-gradient(135deg, #a3bffa 0%, #b497ce 100%);
      color: #222222;
      box-shadow: 0 8px 24px rgba(0,0,0,0.1);
      border: 1px solid rgba(0, 0, 0, 0.15);
      backdrop-filter: none;
    }
    .stTabs [data-baseweb="tab"] {
      background: linear-gradient(135deg, rgba(163,191,250,0.3), rgba(180,151,206,0.3));
      color: #222222;
      border: 1px solid rgba(0, 0, 0, 0.15);
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
      backdrop-filter: none;
      -webkit-backdrop-filter: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
      box-shadow: 0 8px 20px rgba(163, 191, 250, 0.3);
    }
    .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg, #a3bffa, #b497ce);
      color: #222222;
      box-shadow: 0 6px 20px rgba(163, 191, 250, 0.4);
      border: 1px solid rgba(0, 0, 0, 0.2);
    }
    .success-box {
      background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%);
      color: #222222;
    }
    .info-card {
      background: linear-gradient(135deg, rgba(163,191,250,0.35) 0%, rgba(180,151,206,0.35) 100%);
      color: #222222;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
      border: 1px solid rgba(0, 0, 0, 0.15);
      backdrop-filter: none;
      -webkit-backdrop-filter: none;
    }
    .info-card:hover {
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
    }
  }
</style>
""", unsafe_allow_html=True)

# Iinizilizazione dello stato della sessione
defaults = {
    'predictor': None, 'data': None, 'trained': False, 'results': None,
    'config': None, 'X_test': None, 'y_test': None, 'feature_importance': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Header
st.markdown("""
<div class="info-card">
    <h1 style="margin: 0; font-size: 2.5rem;">🚗 AI Driving Style Predictor</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        Advanced Machine Learning Dashboard for Automotive Behavior Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")
    

    # Database Connection
    with st.expander("🔗 Database Connection", expanded=True):
        db_config = {
            'host': st.text_input("🌐 Host", ""),
            'port': st.text_input("🔌 Port", ""),
            'dbname': st.text_input("🗄️ Database", ""),
            'user': st.text_input("👤 User", ""),
            'password': st.text_input("🔐 Password", "", type="password")
        }

        if st.button("🔌 Connect to Database",key='dashboard', type="primary", use_container_width=True):
            try:
                predictor = DrivingStylePredictor(db_config) 
                conn = predictor.connect_database()
                if conn:
                    conn.close()
                    st.session_state.predictor = predictor  
                    st.session_state.config = predictor.get_config()
                    st.success("✅ Database Connected Successfully!")
                else:
                    st.error("❌ Failed to connect: No connection returned.")
            except Exception as e:
                st.error(f"❌ Connection Error")

    # Advanced Settings
    if st.session_state.predictor:
        with st.expander("🎛️ Advanced Settings", expanded=False):
            # parametri di sessione
            st.markdown("**📊 Session Parameters**")
            time_gap = st.slider("⏱️ Time Gap (min)", 5, 120, st.session_state.config['session_params']['time_gap_minutes'])
            min_duration = st.slider("⏰ Min Duration (min)", 1, 30, st.session_state.config['session_params']['min_session_duration'])
            min_points = st.slider("📍 Min Points", 3, 50, st.session_state.config['session_params']['min_session_points'])

            # selectione delle feature
            st.markdown("**🎯 Feature Selection**")
            k_features = st.slider("🔝 Top K Features", 5, 25, st.session_state.config['feature_selection']['k_features'])
            selection_method = st.selectbox("🔬 Selection Method", 
                ['mutual_info', 'f_classif', 'rfe'], 
                index=['mutual_info', 'f_classif', 'rfe'].index(st.session_state.config['feature_selection']['method'])
            )

            # parametri di training
            st.markdown("**🎓 Training Parameters**")
            test_size = st.slider("📊 Test Size", 0.1, 0.5, st.session_state.config['training']['test_size'])
            cv_folds = st.selectbox("🔄 CV Folds", [3, 5, 10], index=[3, 5, 10].index(st.session_state.config['training']['cv_folds']))

            st.markdown("**🤖 Model Parameters**")

            # Random Forest
            st.markdown("### 🌲 Random Forest")
            rf_estimators = st.slider("Trees", 50, 500, st.session_state.config['random_forest']['n_estimators'])
            rf_depth = st.slider("Max Depth", 3, 20, st.session_state.config['random_forest']['max_depth'])
            rf_min_samples = st.slider("Min Samples Split", 2, 10, 2)

            # Gradient Boosting
            st.markdown("### 🚀 Gradient Boosting")
            gb_estimators = st.slider("Estimators", 50, 300, st.session_state.config['gradient_boosting']['n_estimators'])
            gb_lr = st.slider("Learning Rate", 0.01, 0.3, st.session_state.config['gradient_boosting']['learning_rate'])
            gb_depth = st.slider("Max Depth", 3, 8, 3)

            # Logistic Regression
            st.markdown("### 📊 Logistic Regression")
            lr_c = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
            lr_solver = st.selectbox("Solver", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
            lr_max_iter = st.slider("Max Iterations", 100, 1000, 500)

            # SVM
            st.markdown("### 🎯 Support Vector Machine")
            svm_c = st.slider("C Parameter", 0.1, 10.0, 1.0)
            svm_kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'])
            svm_gamma = st.selectbox("Gamma", ['scale', 'auto'])

            # Update button
            if st.button("💾 Update Configuration", type="primary", use_container_width=True):
                new_config = {
                    'session_params': {
                        'time_gap_minutes': time_gap,
                        'min_session_duration': min_duration,
                        'min_session_points': min_points
                    },
                    'feature_selection': {
                        'k_features': k_features,
                        'method': selection_method
                    },
                    'training': {
                        'test_size': test_size,
                        'cv_folds': cv_folds
                    },
                    'random_forest': {
                        'n_estimators': rf_estimators,
                        'max_depth': rf_depth,
                        'min_samples_split': rf_min_samples
                    },
                    'gradient_boosting': {
                        'n_estimators': gb_estimators,
                        'learning_rate': gb_lr,
                        'max_depth': gb_depth
                    },
                    'logistic_regression': {
                        'C': lr_c,
                        'solver': lr_solver,
                        'max_iter': lr_max_iter
                    },
                    'svm': {
                        'C': svm_c,
                        'kernel': svm_kernel,
                        'gamma': svm_gamma
                    }
                }
                st.session_state.predictor.update_config(new_config)
                st.session_state.config = st.session_state.predictor.get_config()
                st.success("✅ Configuration Updated Successfully!")

# Main Content
if not st.session_state.predictor:
    st.markdown("""
    <style>
    :root {
        --text-color: white;
        --background-color: transparent;
    }
    @media (prefers-color-scheme: light) {
        :root {
        --text-color: #111111;
        }
    }
    </style>

    <div style="
        text-align: center; 
        padding: 3rem; 
        border-radius: 16px; 
        margin: 2rem 0;
        color: var(--text-color);
    ">
    <h2 style="margin-bottom: 1rem;">⚠️ Database Connection Required</h2>
    <p style="font-size: 1.1rem;">Please connect to the database using the sidebar to begin</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🤖 ML Training", "📈 Results & Analysis", "📋 System Info"])

# Tab 1: caricamento e esplorazione dei dati
with tab1:
    st.markdown("### 📊 Data Loading & Exploration")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        batch_size = st.slider("📦 Batch Size", 1000, 100000, 15000)
    with col2:
        if st.button("🔄 Load Data", type="primary", use_container_width=True):
            with st.spinner("🔄 Loading data from database..."):
                try:
                    st.session_state.data = st.session_state.predictor.load_batch_data(batch_size)
                    if st.session_state.data is not None:
                        st.success(f"✅ Successfully loaded {len(st.session_state.data):,} sessions")
                    else:
                        st.error("❌ No data loaded from database")
                except Exception as e:
                    st.error(f"❌ Loading Error: {e}")
    
    
    if st.session_state.data is not None:
        stats = st.session_state.predictor.get_data_statistics(st.session_state.data)
        
        # visualizzazione delle statistiche
        st.markdown("### 📈 Dataset Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = [
            ("📊", "Sessions", f"{len(st.session_state.data):,}"),
            ("🚗", "Vehicles", f"{stats['general_stats'].get('unique_vehicles', 0):,}"),
            ("🎯", "Styles", f"{stats['general_stats'].get('unique_driving_styles', 0)}"),
            ("⚡", "Avg Speed", f"{stats['general_stats'].get('avg_speed_overall', 0):.1f} km/h"),
            ("⏱️", "Avg Duration", f"{stats['general_stats'].get('avg_session_duration', 0):.1f} min")
        ]
        
        for col, (icon, label, value) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.metric(f"{icon} {label}", value)

        if stats['style_distribution']:
            st.markdown("### 📊 Data Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                style_df = pd.DataFrame(stats['style_distribution'])
                fig = px.pie(style_df, values='count', names='stile_guida', 
                           title="🎯 Driving Style Distribution",
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(showlegend=True, height=450, 
                                title_font_size=16, title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'avg_speed' in st.session_state.data.columns and 'stile_guida' in st.session_state.data.columns:
                    fig = px.box(st.session_state.data, x='stile_guida', y='avg_speed',
                               title="📈 Speed Distribution by Driving Style",
                               color='stile_guida',
                               color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_layout(height=450, title_font_size=16, title_x=0.5)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔍 Data Sample Preview")
        display_cols = ['avg_speed', 'max_speed', 'avg_rpm', 'session_duration', 'total_points', 'stile_guida']
        available_cols = [col for col in display_cols if col in st.session_state.data.columns]
        
        if available_cols:
            styled_df = st.session_state.data[available_cols].head(10).style.format({
                'avg_speed': '{:.1f}',
                'max_speed': '{:.1f}',
                'avg_rpm': '{:.0f}',
                'session_duration': '{:.1f}',
                'total_points': '{:.0f}'
            })
            st.dataframe(styled_df, use_container_width=True)

# Tab 2: training dei modelli
with tab2:
    st.markdown("### 🤖 Machine Learning Training Center")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from the Data Explorer tab")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🎯 Model Selection")
            models = {
                "🌲 Random Forest": st.checkbox("Random Forest", True),
                "🚀 Gradient Boosting": st.checkbox("Gradient Boosting", True), 
                "📊 Logistic Regression": st.checkbox("Logistic Regression", True),
                "🎯 SVM": st.checkbox("Support Vector Machine", False)
            }
            
            
        
        with col2:
            st.markdown("#### 📊 Training Status")
            status_color = "🟢" if st.session_state.trained else "🔴"
            st.markdown(f"""
            <div style="background: {'linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%)' if st.session_state.trained else 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)'}; 
                        padding: 1rem; border-radius: 12px; color: white; text-align: center;">
                <h4 style="margin: 0;">{status_color} Status: {'Model Trained' if st.session_state.trained else 'Ready to Train'}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.data is not None:
                st.info(f"📊 Available Sessions: {len(st.session_state.data):,}")
                st.info(f"🎯 Unique Styles: {st.session_state.data['stile_guida'].nunique()}")
        
        if st.button("🚀 Start Training Process", type="primary", use_container_width=True):
            selected = [name.split(' ', 1)[1] for name, sel in models.items() if sel]
            
            if not selected:
                st.error("❌ Please select at least one model to train")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 Preprocessing data...")
                    progress_bar.progress(20)
                    
                    X, y = st.session_state.predictor.preprocess_data(st.session_state.data)
                    X_train, X_test, y_train, y_test = st.session_state.predictor.split_data(X, y)
                    
                    status_text.text("🎯 Splitting data...")
                    progress_bar.progress(40)
                    
                    st.session_state.X_test, st.session_state.y_test = X_test, y_test
                    
                    status_text.text("🤖 Training models...")
                    progress_bar.progress(60)
                    
                    models_trained = st.session_state.predictor.train_models(X_train, y_train, selected)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Training completed!")
                    
                    st.session_state.trained = True
                    
                    st.markdown(f"""
                    <div class="success-box">
                        🎉 Successfully trained {len(models_trained)} models!
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                    progress_bar.empty()
                    status_text.empty()

# Tab 3: risultati e analisi delle prestazioni
with tab3:
    st.markdown("### 📈 Results & Performance Analysis")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train models first in the ML Training tab")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📊 Run Model Evaluation", type="primary", use_container_width=True):
                with st.spinner("🔄 Evaluating model performance..."):
                    try:
                        results = st.session_state.predictor.evaluate_model(
                            st.session_state.X_test, st.session_state.y_test
                        )
                        st.session_state.results = results
                        
                        # Get feature importance
                        st.session_state.feature_importance = st.session_state.predictor.feature_importance()
                        
                        st.success("✅ Evaluation completed successfully!")
                    except Exception as e:
                        st.error(f"❌ Evaluation Error: {e}")
        
        with col2:
            if st.button("🔄 Refresh Analysis", type="secondary", use_container_width=True):
                st.rerun()
        
        if st.session_state.results:
            st.markdown("#### 🎯 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            metrics_data = [
                ("🎯", "Accuracy", st.session_state.results['accuracy'], "#667eea"),
                ("🔍", "Precision", st.session_state.results['precision'], "#764ba2"),
                ("📊", "Recall", st.session_state.results['recall'], "#56ab2f"),
                ("⚖️", "F1-Score", st.session_state.results['f1_score'], "#ff9a9e")
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
            
            if st.session_state.feature_importance is not None:
                st.markdown("#### 🔝 Feature Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    top_features = st.session_state.feature_importance.head(10)
                    fig = px.bar(top_features, x='importance', y='feature', 
                               orientation='h', 
                               title="🔝 Top 10 Most Important Features",
                               color='importance',
                               color_continuous_scale='viridis')
                    fig.update_layout(height=500, title_font_size=16, title_x=0.5)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if len(st.session_state.predictor.models) > 1:
                        model_scores = {}
                        for name, model in st.session_state.predictor.models.items():
                            try:
                                X_test_processed = st.session_state.predictor.feature_selector.transform(
                                    st.session_state.predictor.scaler.transform(
                                        st.session_state.predictor.imputer.transform(st.session_state.X_test)
                                    )
                                )
                                pred = model.predict(X_test_processed)
                                from sklearn.metrics import f1_score
                                score = f1_score(st.session_state.y_test, pred, average='weighted')
                                model_scores[name] = score
                            except:
                                pass
                        
                        if model_scores:
                            scores_df = pd.DataFrame(list(model_scores.items()), columns=['Model', 'F1_Score'])
                            fig = px.bar(scores_df, x='Model', y='F1_Score',
                                       title="🏆 Model Performance Comparison",
                                       color='F1_Score',
                                       color_continuous_scale='blues')
                            fig.update_layout(height=500, title_font_size=16, title_x=0.5)
                            st.plotly_chart(fig, use_container_width=True)

# Tab 4: configurazione e stato del sistema 
with tab4:
    st.markdown("### 📋 System Information & Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Current Configuration")
        if st.session_state.config:
            config_display = {
                "⏱️ Session Gap": f"{st.session_state.config['session_params']['time_gap_minutes']} min",
                "⏰ Min Duration": f"{st.session_state.config['session_params']['min_session_duration']} min",
                "📍 Min Points": f"{st.session_state.config['session_params']['min_session_points']}",
                "📊 Test Size": f"{st.session_state.config['training']['test_size']:.1%}",
                "🔄 CV Folds": f"{st.session_state.config['training']['cv_folds']}",
                "🎯 Feature Selection": f"{st.session_state.config['feature_selection']['method']}",
                "🔝 Top K Features": f"{st.session_state.config['feature_selection']['k_features']}"
            }
            
            for key, value in config_display.items():
                st.markdown(f"**{key}**: {value}")
    
    with col2:
        st.markdown("#### 🤖 Model Status")
        if st.session_state.predictor and st.session_state.predictor.models:
            for model_name in st.session_state.predictor.models.keys():
                st.markdown(f"✅ **{model_name}** - Ready")
        else:
            st.markdown("❌ **No models trained yet**")
        
        st.markdown("#### 📊 Data Status")
        if st.session_state.data is not None:
            st.markdown(f"📊 **Loaded Sessions**: {len(st.session_state.data):,}")
            st.markdown(f"🎯 **Unique Styles**: {st.session_state.data['stile_guida'].nunique()}")
        else:
            st.markdown("❌ **No data loaded**")
        
        if st.session_state.trained:
            st.markdown("🎯 **System Status**: Ready for Analysis")
        else:
            st.markdown("⚠️ **System Status**: Waiting for Training")

#footer
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
        st.session_state.results = None
        if st.session_state.predictor:
            st.session_state.predictor.models = {}
        st.success("✅ Models reset successfully!")
        st.rerun()

with col3:
    if st.button("🗑️ Reset Everything", type="secondary", use_container_width=True):
        st.session_state.clear()
        st.success("✅ Everything reset successfully!")
        st.rerun()

