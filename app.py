"""
Energy Consumption Forecasting Dashboard
Dự báo nhu cầu năng lượng - UCI Household Power Consumption
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import sys
import json
from datetime import datetime, timedelta

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-title {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #3B82F6;
    }
    .best-model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .insight-card {
        background: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

# ==================== XÁC ĐỊNH ĐƯỜNG DẪN ====================
@st.cache_resource
def get_project_root():
    """Xác định project root linh hoạt"""
    cwd = os.getcwd()
    if os.path.basename(cwd) == "scripts":
        return os.path.abspath("..")
    elif os.path.basename(cwd) == "notebooks":
        return os.path.abspath("..")
    else:
        return cwd

PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ==================== DATA LOADER ====================
@st.cache_data
def load_all_data():
    """Load tất cả dữ liệu cần thiết cho dashboard"""
    data = {}
    warnings = []
    
    try:
        # 1. Load model comparison
        comp_path = os.path.join(DATA_DIR, "final_model_comparison.csv")
        if os.path.exists(comp_path):
            data['comparison'] = pd.read_csv(comp_path)
        else:
            warnings.append("Không tìm thấy final_model_comparison.csv")
        
        # 2. Load best model info
        best_path = os.path.join(DATA_DIR, "best_model_info.json")
        if os.path.exists(best_path):
            with open(best_path, 'r') as f:
                data['best_model'] = json.load(f)
        
        # 3. Load results từ các models
        results_files = {
            'baseline': 'baseline_results.csv',
            'arima': 'arima_results.csv',
            'ets': 'ets_results.csv'
        }
        
        data['results'] = {}
        for model, filename in results_files.items():
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                data['results'][model] = df
            else:
                warnings.append(f"Không tìm thấy {filename}")
        
        # 4. Load residuals
        residuals_files = {
            'baseline': 'baseline_residuals.csv',
            'arima': 'arima_residuals.csv',
            'ets': 'ets_residuals.csv'
        }
        
        data['residuals'] = {}
        for model, filename in residuals_files.items():
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(file_path):
                data['residuals'][model] = pd.read_csv(file_path)
        
        # 5. Load feature data để lấy thông tin thêm
        feature_path = os.path.join(DATA_DIR, "feature_engineered_data.parquet")
        if os.path.exists(feature_path):
            data['features'] = pd.read_parquet(feature_path)
        
        # 6. Load anomalies
        anomaly_path = os.path.join(DATA_DIR, "anomalies_detected.csv")
        if os.path.exists(anomaly_path):
            data['anomalies'] = pd.read_csv(anomaly_path)
        
    except Exception as e:
        warnings.append(f"Lỗi khi load dữ liệu: {str(e)}")
    
    data['warnings'] = warnings
    return data

# ==================== SIDEBAR ====================
def render_sidebar(data):
    """Render sidebar với thông tin tổng quan"""
    with st.sidebar:
        st.markdown("## ⚡ Energy Forecast")
        st.markdown("### Dashboard - Nhóm XX")
        
        st.markdown("---")
        
        # Navigation
        section = st.radio(
            "📌 Chọn phần:",
            [
                "🏠 Tổng quan",
                "📊 So sánh mô hình",
                "📈 Dự báo chi tiết",
                "🔍 Phân tích lỗi",
                "📉 Phân tích phần dư",
                "⚠️ Anomaly Detection"
            ]
        )
        
        st.markdown("---")
        
        # Thông tin dữ liệu
        if 'features' in data:
            st.markdown("### 📊 Thông tin dữ liệu")
            df = data['features']
            st.metric("Tổng số mẫu", f"{df.shape[0]:,}")
            st.metric("Khoảng thời gian", f"{df.index.min().strftime('%Y-%m')} → {df.index.max().strftime('%Y-%m')}")
            st.metric("Số features", df.shape[1])
        
        # Model tốt nhất
        if 'best_model' in data:
            st.markdown("### 🏆 Best Model")
            best = data['best_model']
            st.info(f"**{best['best_model']}**")
            st.metric("RMSE", f"{best['rmse']:.4f}")
            st.metric("MAE", f"{best['mae']:.4f}")
            st.metric("SMAPE", f"{best['smape']:.2f}%")
        
        # Warnings
        if data.get('warnings'):
            with st.expander("⚠️ Cảnh báo"):
                for w in data['warnings']:
                    st.warning(w)
        
        st.markdown("---")
        st.markdown("#### 👥 Thành viên nhóm")
        st.markdown("""
        - Thành viên 1
        - Thành viên 2
        - Thành viên 3
        - Thành viên 4
        - Thành viên 5
        """)
        
        return section

# ==================== TRANG TỔNG QUAN ====================
def render_overview(data):
    """Trang tổng quan project"""
    st.markdown('<h1 class="main-title">⚡ Dự báo nhu cầu năng lượng</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Giới thiệu project
        
        **Bộ dữ liệu:** UCI Household Power Consumption
        
        **Mục tiêu:** Dự báo công suất tiêu thụ điện (Global Active Power)
        
        **Pipeline:**
        1. **Tiền xử lý & EDA** - Làm sạch, xử lý missing, phân tích khám phá
        2. **Anomaly Detection** - Phát hiện điểm bất thường
        3. **Feature Engineering** - Tạo lag, rolling, time features
        4. **Baseline Model** - Seasonal Naïve
        5. **ARIMA** - Mô hình thống kê kinh điển
        6. **ETS (Holt-Winters)** - Mô hình xu hướng và mùa vụ
        7. **Đánh giá & So sánh** - Chọn mô hình tốt nhất
        """)
    
    with col2:
        if 'comparison' in data:
            st.markdown("### 📊 Tổng kết mô hình")
            comp_df = data['comparison'].copy()
            comp_df['RMSE'] = comp_df['RMSE'].round(4)
            comp_df['MAE'] = comp_df['MAE'].round(4)
            comp_df['SMAPE'] = comp_df['SMAPE'].round(2)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # Metrics overview
    if 'results' in data and data['results']:
        st.markdown('<h2 class="section-title">📈 Tổng quan dự báo</h2>', unsafe_allow_html=True)
        
        # Lấy dữ liệu mẫu
        first_model = list(data['results'].keys())[0]
        sample_df = data['results'][first_model].iloc[:168]  # 7 ngày đầu
        
        fig = go.Figure()
        
        # Thực tế
        fig.add_trace(go.Scatter(
            x=sample_df['timestamp'],
            y=sample_df['actual'],
            mode='lines',
            name='Thực tế',
            line=dict(color='black', width=2)
        ))
        
        # Dự báo các model
        colors = {'baseline': 'blue', 'arima': 'red', 'ets': 'green'}
        for model in data['results'].keys():
            model_df = data['results'][model].iloc[:168]
            pred_col = f"{model}_pred" if model != 'baseline' else 'baseline_pred'
            if pred_col in model_df.columns:
                fig.add_trace(go.Scatter(
                    x=model_df['timestamp'],
                    y=model_df[pred_col],
                    mode='lines',
                    name=model.capitalize(),
                    line=dict(color=colors.get(model, 'gray'), dash='dash')
                ))
        
        fig.update_layout(
            title='So sánh dự báo các mô hình - 7 ngày đầu test',
            xaxis_title='Thời gian',
            yaxis_title='Global Active Power (kW)',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== TRANG SO SÁNH MÔ HÌNH ====================
def render_model_comparison(data):
    """Trang so sánh chi tiết các mô hình"""
    st.markdown('<h2 class="section-title">📊 So sánh chi tiết các mô hình</h2>', unsafe_allow_html=True)
    
    if 'comparison' not in data:
        st.warning("Không có dữ liệu so sánh")
        return
    
    comp_df = data['comparison']
    
    # Metrics cards
    cols = st.columns(len(comp_df))
    for i, (_, row) in enumerate(comp_df.iterrows()):
        with cols[i]:
            if row['Model'] == data.get('best_model', {}).get('best_model', ''):
                st.markdown(f"### 🏆 {row['Model']}")
            else:
                st.markdown(f"### {row['Model']}")
            st.metric("MAE", f"{row['MAE']:.4f}")
            st.metric("RMSE", f"{row['RMSE']:.4f}")
            st.metric("SMAPE", f"{row['SMAPE']:.2f}%")
    
    # Biểu đồ so sánh
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📈 Radar Chart", "📉 Improvement"])
    
    with tab1:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('MAE', 'RMSE', 'SMAPE (%)'),
            shared_yaxes=False
        )
        
        models = comp_df['Model'].tolist()
        colors = ['#3B82F6' if m != data.get('best_model', {}).get('best_model', '') else '#10B981' for m in models]
        
        # MAE
        fig.add_trace(
            go.Bar(x=models, y=comp_df['MAE'], marker_color=colors, text=comp_df['MAE'].round(4), textposition='outside'),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=models, y=comp_df['RMSE'], marker_color=colors, text=comp_df['RMSE'].round(4), textposition='outside'),
            row=1, col=2
        )
        
        # SMAPE
        fig.add_trace(
            go.Bar(x=models, y=comp_df['SMAPE'], marker_color=colors, text=comp_df['SMAPE'].round(2), textposition='outside'),
            row=1, col=3
        )
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_yaxes(title_text="MAE", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_yaxes(title_text="SMAPE (%)", row=1, col=3)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Radar chart
        categories = ['MAE', 'RMSE', 'SMAPE']
        
        fig = go.Figure()
        
        for _, row in comp_df.iterrows():
            # Chuẩn hóa về 0-1 (giá trị càng nhỏ càng tốt)
            values = []
            for cat in categories:
                max_val = comp_df[cat].max()
                min_val = comp_df[cat].min()
                # Đảo ngược để giá trị tốt nhất ở rìa ngoài
                norm_val = 1 - (row[cat] - min_val) / (max_val - min_val + 1e-10)
                values.append(norm_val)
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            height=500,
            title="So sánh các mô hình (chuẩn hóa - càng ra ngoài càng tốt)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'best_model' in data:
            best = data['best_model']['best_model']
            best_rmse = data['best_model']['rmse']
            
            st.markdown(f"### 📈 Mức cải thiện so với Baseline")
            
            baseline_rmse = comp_df[comp_df['Model'] == 'Baseline']['RMSE'].values[0]
            arima_rmse = comp_df[comp_df['Model'] == 'Arima']['RMSE'].values[0] if 'Arima' in comp_df['Model'].values else None
            ets_rmse = comp_df[comp_df['Model'] == 'Ets']['RMSE'].values[0] if 'Ets' in comp_df['Model'].values else None
            
            improvement_data = []
            if arima_rmse:
                imp = ((baseline_rmse - arima_rmse) / baseline_rmse) * 100
                improvement_data.append({'Model': 'ARIMA', 'Improvement': imp})
            if ets_rmse:
                imp = ((baseline_rmse - ets_rmse) / baseline_rmse) * 100
                improvement_data.append({'Model': 'ETS', 'Improvement': imp})
            
            if improvement_data:
                imp_df = pd.DataFrame(improvement_data)
                
                fig = px.bar(imp_df, x='Model', y='Improvement', 
                           title='Mức cải thiện so với Baseline (%)',
                           text_auto='.1f',
                           color='Improvement',
                           color_continuous_scale='Greens')
                
                fig.update_traces(textposition='outside')
                fig.update_layout(yaxis_title="Cải thiện (%)", height=400)
                
                st.plotly_chart(fig, use_container_width=True)

# ==================== TRANG DỰ BÁO CHI TIẾT ====================
def render_forecast_detail(data):
    """Trang xem chi tiết dự báo"""
    st.markdown('<h2 class="section-title">📈 Dự báo chi tiết</h2>', unsafe_allow_html=True)
    
    if not data.get('results'):
        st.warning("Không có dữ liệu dự báo")
        return
    
    # Chọn model
    available_models = list(data['results'].keys())
    selected_models = st.multiselect(
        "Chọn mô hình để hiển thị:",
        available_models,
        default=available_models[:min(2, len(available_models))]
    )
    
    # Chọn khoảng thời gian
    first_model = available_models[0]
    min_date = data['results'][first_model]['timestamp'].min()
    max_date = data['results'][first_model]['timestamp'].max()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Từ ngày:", min_date.date())
    with col2:
        end_date = st.date_input("Đến ngày:", max_date.date())
    
    # Filter data
    filtered_data = {}
    for model in selected_models:
        df = data['results'][model]
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        filtered_data[model] = df[mask]
    
    # Vẽ biểu đồ
    fig = go.Figure()
    
    # Thực tế
    first_filtered = filtered_data[selected_models[0]]
    fig.add_trace(go.Scatter(
        x=first_filtered['timestamp'],
        y=first_filtered['actual'],
        mode='lines',
        name='Thực tế',
        line=dict(color='black', width=2)
    ))
    
    # Dự báo
    colors = {'baseline': 'blue', 'arima': 'red', 'ets': 'green'}
    for model in selected_models:
        df = filtered_data[model]
        pred_col = f"{model}_pred" if model != 'baseline' else 'baseline_pred'
        if pred_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[pred_col],
                mode='lines',
                name=model.capitalize(),
                line=dict(color=colors.get(model, 'gray'), dash='dash', width=1.5)
            ))
    
    fig.update_layout(
        title=f'Dự báo từ {start_date} đến {end_date}',
        xaxis_title='Thời gian',
        yaxis_title='Global Active Power (kW)',
        hovermode='x unified',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị bảng dữ liệu
    with st.expander("📋 Xem dữ liệu chi tiết"):
        display_df = first_filtered[['timestamp', 'actual']].copy()
        for model in selected_models:
            df = filtered_data[model]
            pred_col = f"{model}_pred" if model != 'baseline' else 'baseline_pred'
            if pred_col in df.columns:
                display_df[model.capitalize()] = df[pred_col].values
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ==================== TRANG PHÂN TÍCH LỖI ====================
def render_error_analysis(data):
    """Trang phân tích lỗi dự báo"""
    st.markdown('<h2 class="section-title">🔍 Phân tích lỗi dự báo</h2>', unsafe_allow_html=True)
    
    if not data.get('results'):
        st.warning("Không có dữ liệu")
        return
    
    # Chọn model để phân tích
    models = list(data['results'].keys())
    selected_model = st.selectbox("Chọn mô hình để phân tích:", models, format_func=lambda x: x.capitalize())
    
    # Lấy dữ liệu
    df = data['results'][selected_model].copy()
    pred_col = f"{selected_model}_pred" if selected_model != 'baseline' else 'baseline_pred'
    
    # Tính lỗi
    df['error'] = df['actual'] - df[pred_col]
    df['abs_error'] = np.abs(df['error'])
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    tab1, tab2, tab3 = st.tabs(["📊 Theo giờ", "📅 Theo thứ", "📆 Theo tháng"])
    
    with tab1:
        # Lỗi theo giờ
        hourly_error = df.groupby('hour')['abs_error'].mean().reset_index()
        
        fig = px.bar(hourly_error, x='hour', y='abs_error',
                    title='Lỗi trung bình theo giờ trong ngày',
                    labels={'hour': 'Giờ', 'abs_error': 'MAE'},
                    text_auto='.3f')
        
        fig.update_traces(marker_color='#3B82F6')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Thống kê
        best_hour = hourly_error.loc[hourly_error['abs_error'].idxmin(), 'hour']
        worst_hour = hourly_error.loc[hourly_error['abs_error'].idxmax(), 'hour']
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"✅ Giờ dự báo tốt nhất: **{best_hour:.0f}h** (MAE = {hourly_error['abs_error'].min():.4f})")
        with col2:
            st.error(f"⚠️ Giờ dự báo tệ nhất: **{worst_hour:.0f}h** (MAE = {hourly_error['abs_error'].max():.4f})")
    
    with tab2:
        # Lỗi theo thứ
        df['dow_name'] = df['dayofweek'].map(lambda x: dow_labels[x])
        dow_error = df.groupby('dow_name')['abs_error'].mean().reindex(dow_labels).reset_index()
        
        fig = px.bar(dow_error, x='dow_name', y='abs_error',
                    title='Lỗi trung bình theo thứ',
                    labels={'dow_name': 'Thứ', 'abs_error': 'MAE'},
                    text_auto='.3f',
                    color='abs_error',
                    color_continuous_scale='Reds')
        
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Lỗi theo tháng
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df['month_name'] = df['month'].map(lambda x: month_labels[x-1])
        month_error = df.groupby('month_name')['abs_error'].mean().reindex(month_labels).reset_index()
        
        fig = px.line(month_error, x='month_name', y='abs_error',
                     title='Lỗi trung bình theo tháng',
                     labels={'month_name': 'Tháng', 'abs_error': 'MAE'},
                     markers=True)
        
        fig.update_traces(line=dict(color='red', width=2), marker=dict(size=8))
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== TRANG PHÂN TÍCH PHẦN DƯ ====================
def render_residual_analysis(data):
    """Trang phân tích phần dư"""
    st.markdown('<h2 class="section-title">📉 Phân tích phần dư (Residuals)</h2>', unsafe_allow_html=True)
    
    if not data.get('residuals'):
        st.warning("Không có dữ liệu phần dư")
        return
    
    # Chọn model
    models = list(data['residuals'].keys())
    selected_model = st.selectbox("Chọn mô hình:", models, format_func=lambda x: x.capitalize())
    
    residuals_df = data['residuals'][selected_model]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{residuals_df['residual'].mean():.4f}")
    with col2:
        st.metric("Std", f"{residuals_df['residual'].std():.4f}")
    with col3:
        st.metric("Skewness", f"{residuals_df['residual'].skew():.4f}")
    with col4:
        st.metric("Kurtosis", f"{residuals_df['residual'].kurtosis():.4f}")
    
    tab1, tab2, tab3 = st.tabs(["📈 Residuals over time", "📊 Distribution", "📉 Q-Q Plot"])
    
    with tab1:
        fig = px.line(residuals_df, x='timestamp', y='residual',
                     title=f'Phần dư theo thời gian - {selected_model.capitalize()}',
                     labels={'timestamp': 'Thời gian', 'residual': 'Residuals'})
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.histogram(residuals_df, x='residual', nbins=50,
                          title=f'Phân phối phần dư - {selected_model.capitalize()}',
                          labels={'residual': 'Residuals', 'count': 'Tần suất'},
                          marginal='box')
        
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Q-Q plot
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(8, 8))
        stats.probplot(residuals_df['residual'], dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot - {selected_model.capitalize()}')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# ==================== TRANG ANOMALY DETECTION ====================
def render_anomaly_detection(data):
    """Trang phát hiện bất thường"""
    st.markdown('<h2 class="section-title">⚠️ Phát hiện bất thường (Anomaly Detection)</h2>', unsafe_allow_html=True)
    
    if 'anomalies' not in data:
        st.warning("Không có dữ liệu anomaly")
        return
    
    anomalies = data['anomalies']
    
    st.markdown(f"### 📊 Tổng quan")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số điểm bất thường", f"{len(anomalies):,}")
    with col2:
        if 'features' in data:
            total = len(data['features'])
            st.metric("Tỷ lệ", f"{(len(anomalies)/total)*100:.4f}%")
    with col3:
        st.metric("Giá trị trung bình", f"{anomalies['Global_active_power'].mean():.2f} kW")
    
    # Biểu đồ phân phối anomaly theo thời gian
    anomalies['hour'] = pd.to_datetime(anomalies['Datetime']).dt.hour if 'Datetime' in anomalies.columns else anomalies.index.hour if hasattr(anomalies.index, 'hour') else 0
    
    fig = px.histogram(anomalies, x='hour', nbins=24,
                      title='Phân phối anomaly theo giờ',
                      labels={'hour': 'Giờ', 'count': 'Số lượng anomaly'})
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị bảng anomalies
    with st.expander("📋 Danh sách các điểm bất thường"):
        st.dataframe(anomalies, use_container_width=True)

# ==================== MAIN APP ====================
def main():
    """Main application"""
    
    # Load data
    with st.spinner("🔄 Đang tải dữ liệu..."):
        data = load_all_data()
    
    # Render sidebar
    section = render_sidebar(data)
    
    # Main content
    if section == "🏠 Tổng quan":
        render_overview(data)
    elif section == "📊 So sánh mô hình":
        render_model_comparison(data)
    elif section == "📈 Dự báo chi tiết":
        render_forecast_detail(data)
    elif section == "🔍 Phân tích lỗi":
        render_error_analysis(data)
    elif section == "📉 Phân tích phần dư":
        render_residual_analysis(data)
    elif section == "⚠️ Anomaly Detection":
        render_anomaly_detection(data)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>⚡ <strong>Energy Consumption Forecasting Dashboard - Nhóm XX</strong></p>
            <p>📊 Pipeline: Preprocessing → Anomaly Detection → Feature Engineering → Baseline → ARIMA → ETS → Evaluation</p>
            <p>⏰ Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()