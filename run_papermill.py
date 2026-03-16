import papermill as pm
import os
from pathlib import Path
import time

# ==================== CONFIGURATION ====================
# Tạo thư mục lưu kết quả runs
os.makedirs("notebooks/runs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# Kernel name (thường là 'python3' hoặc 'base')
KERNEL_NAME = "python3"

# Thời gian bắt đầu
start_time = time.time()

print("="*60)
print("🚀 BẮT ĐẦU PIPELINE DỰ BÁO NĂNG LƯỢNG")
print("="*60)

# ==================== NOTEBOOK 1: PREPROCESSING AND EDA ====================
print("\n📊 [1/7] Đang chạy: 01_preprocessing_and_eda.ipynb...")
try:
    pm.execute_notebook(
        "notebooks/01_preprocessing_and_eda.ipynb",
        "notebooks/runs/01_preprocessing_and_eda_run.ipynb",
        parameters=dict(
            # Đường dẫn file dữ liệu gốc
            DATA_PATH="data/raw/household_power_consumption.txt",
            
            # Phương pháp xử lý missing values
            HANDLE_MISSING="interpolate",
            
            # Thư mục lưu dữ liệu đã xử lý
            OUTPUT_DIR="data/processed",
            
            # Tên file output
            OUTPUT_FILENAME="cleaned_data.parquet",
            
            # Bật/tắt các biểu đồ
            PLOT_TIME_SERIES=True,
            PLOT_DISTRIBUTIONS=True,
            PLOT_SEASONAL=True,
            PLOT_CORRELATION=True,
            PLOT_ANOMALY_PREVIEW=True
        ),
        kernel_name=KERNEL_NAME
    )
    print("✅ Hoàn thành notebook 1")
except Exception as e:
    print(f"❌ Lỗi notebook 1: {e}")

# ==================== NOTEBOOK 2: ANOMALY DETECTION ====================
print("\n🔍 [2/7] Đang chạy: 02_anomaly_detection.ipynb...")
try:
    pm.execute_notebook(
        "notebooks/02_anomaly_detection.ipynb",
        "notebooks/runs/02_anomaly_detection_run.ipynb",
        parameters=dict(
            # Đường dẫn dữ liệu đã làm sạch
            CLEANED_DATA_PATH="data/processed/cleaned_data.parquet",
            
            # Thư mục lưu kết quả
            OUTPUT_DIR="data/processed",
            
            # Tên file output
            ANOMALY_OUTPUT_FILENAME="anomalies_detected.csv",
            DATA_WITH_ANOMALY_FILENAME="data_with_anomaly_flags.parquet",
            
            # Phương pháp phát hiện
            DETECTION_METHOD="both",
            ZSCORE_THRESHOLD=3,
            TARGET_COLUMN="Global_active_power",
            
            # Bật/tắt biểu đồ
            PLOT_ANOMALY_TIMESERIES=True,
            PLOT_ANOMALY_DISTRIBUTION=True,
            PLOT_ANOMALY_STATS=True
        ),
        kernel_name=KERNEL_NAME
    )
    print("✅ Hoàn thành notebook 2")
except Exception as e:
    print(f"❌ Lỗi notebook 2: {e}")

# ==================== NOTEBOOK 3: FEATURE ENGINEERING ====================
print("\n⚙️ [3/7] Đang chạy: 03_feature_engineering.ipynb...")
try:
    pm.execute_notebook(
        "notebooks/03_feature_engineering.ipynb",
        "notebooks/runs/03_feature_engineering_run.ipynb",
        parameters=dict(
            # Đường dẫn dữ liệu đã có flags anomaly
            INPUT_DATA_PATH="data/processed/data_with_anomaly_flags.parquet",
            
            # Thư mục lưu kết quả
            OUTPUT_DIR="data/processed",
            
            # Tên file output
            FEATURE_OUTPUT_FILENAME="feature_engineered_data.parquet",
            
            # Cột mục tiêu
            TARGET_COLUMN="Global_active_power",
            
            # Tham số cho features
            LAG_HOURS=[1, 2, 3, 6, 12, 24, 48],
            ROLLING_WINDOWS=[6, 12, 24, 48],
            
            # Bật/tắt tạo features
            CREATE_TIME_FEATURES=True,
            CREATE_LAG_FEATURES=True,
            CREATE_ROLLING_FEATURES=True,
            
            # Bật/tắt biểu đồ
            PLOT_FEATURE_CORRELATION=True,
            PLOT_FEATURE_DISTRIBUTION=True,
            PLOT_TARGET_VS_FEATURES=True
        ),
        kernel_name=KERNEL_NAME
    )
    print("✅ Hoàn thành notebook 3")
except Exception as e:
    print(f"❌ Lỗi notebook 3: {e}")

# ==================== NOTEBOOK 4: BASELINE FORECASTING ====================
print("\n📈 [4/7] Đang chạy: 04_baseline_forecasting.ipynb...")
try:
    pm.execute_notebook(
        "notebooks/04_baseline_forecasting.ipynb",
        "notebooks/runs/04_baseline_forecasting_run.ipynb",
        parameters=dict(
            # Đường dẫn dữ liệu đã feature engineering
            INPUT_DATA_PATH="data/processed/feature_engineered_data.parquet",
            
            # Thư mục lưu kết quả
            OUTPUT_DIR="data/processed",
            
            # Tên file output
            BASELINE_RESULTS_FILENAME="baseline_results.csv",
            BASELINE_METRICS_FILENAME="baseline_metrics.csv",
            
            # Cột mục tiêu
            TARGET_COLUMN="Global_active_power",
            
            # Tham số mô hình
            TEST_SIZE=0.2,
            SEASONAL_PERIOD=24,
            
            # Bật/tắt biểu đồ
            PLOT_TRAIN_TEST_SPLIT=True,
            PLOT_BASELINE_FORECAST=True,
            PLOT_RESIDUALS=True
        ),
        kernel_name=KERNEL_NAME
    )
    print("✅ Hoàn thành notebook 4")
except Exception as e:
    print(f"❌ Lỗi notebook 4: {e}")

# ==================== NOTEBOOK 5: ARIMA FORECASTING ====================
print("\n📉 [5/7] Đang chạy: 05_arima_forecasting.ipynb...")
try:
    pm.execute_notebook(
        "notebooks/05_arima_forecasting.ipynb",
        "notebooks/runs/05_arima_forecasting_run.ipynb",
        parameters=dict(
            # Đường dẫn dữ liệu đã feature engineering
            INPUT_DATA_PATH="data/processed/feature_engineered_data.parquet",
            
            # Thư mục lưu kết quả
            OUTPUT_DIR="data/processed",
            
            # Tên file output
            ARIMA_RESULTS_FILENAME="arima_results.csv",
            ARIMA_METRICS_FILENAME="arima_metrics.csv",
            ARIMA_MODEL_INFO_FILENAME="arima_model_info.json",
            
            # Cột mục tiêu
            TARGET_COLUMN="Global_active_power",
            
            # Tham số mô hình
            TEST_SIZE=0.2,
            USE_AUTO_ARIMA=True,
            FIXED_ARIMA_ORDER=(1, 1, 1),
            MAX_P=5,
            MAX_D=2,
            MAX_Q=5,
            SEASONAL=True,
            SEASONAL_PERIOD=24,
            
            # Bật/tắt biểu đồ
            PLOT_ACF_PACF=True,
            PLOT_ARIMA_FORECAST=True,
            PLOT_RESIDUALS=True,
            PLOT_DIAGNOSTICS=True
        ),
        kernel_name=KERNEL_NAME
    )
    print("✅ Hoàn thành notebook 5")
except Exception as e:
    print(f"❌ Lỗi notebook 5: {e}")

# ==================== NOTEBOOK 6: ETS FORECASTING ====================
print("\n📊 [6/7] Đang chạy: 06_ets_forecasting.ipynb...")
try:
    pm.execute_notebook(
        "notebooks/06_ets_forecasting.ipynb",
        "notebooks/runs/06_ets_forecasting_run.ipynb",
        parameters=dict(
            # Đường dẫn dữ liệu đã feature engineering
            INPUT_DATA_PATH="data/processed/feature_engineered_data.parquet",
            
            # Thư mục lưu kết quả
            OUTPUT_DIR="data/processed",
            
            # Tên file output
            ETS_RESULTS_FILENAME="ets_results.csv",
            ETS_METRICS_FILENAME="ets_metrics.csv",
            ETS_MODEL_INFO_FILENAME="ets_model_info.json",
            
            # Cột mục tiêu
            TARGET_COLUMN="Global_active_power",
            
            # Tham số mô hình
            TEST_SIZE=0.2,
            SEASONAL_PERIOD=24,
            TREND_COMPONENT=True,
            SEASONAL_COMPONENT=True,
            DAMPED_TREND=False,
            INITIALIZATION_METHOD='estimated',
            
            # Bật/tắt biểu đồ
            PLOT_ETS_FORECAST=True,
            PLOT_RESIDUALS=True,
            PLOT_COMPONENTS=True,
            PLOT_COMPARISON=True
        ),
        kernel_name=KERNEL_NAME
    )
    print("✅ Hoàn thành notebook 6")
except Exception as e:
    print(f"❌ Lỗi notebook 6: {e}")

# ==================== NOTEBOOK 7: EVALUATION AND INTERPRETATION ====================
print("\n📋 [7/7] Đang chạy: 07_evaluation_and_interpretation.ipynb...")
try:
    pm.execute_notebook(
        "notebooks/07_evaluation_and_interpretation.ipynb",
        "notebooks/runs/07_evaluation_and_interpretation_run.ipynb",
        parameters=dict(
            # Thư mục chứa kết quả
            OUTPUT_DIR="data/processed",
            
            # Đường dẫn đến kết quả các mô hình
            BASELINE_RESULTS_PATH="data/processed/baseline_results.csv",
            BASELINE_METRICS_PATH="data/processed/baseline_metrics.csv",
            ARIMA_RESULTS_PATH="data/processed/arima_results.csv",
            ARIMA_METRICS_PATH="data/processed/arima_metrics.csv",
            ETS_RESULTS_PATH="data/processed/ets_results.csv",
            ETS_METRICS_PATH="data/processed/ets_metrics.csv",
            
            # Đường dẫn dữ liệu gốc
            ORIGINAL_DATA_PATH="data/processed/feature_engineered_data.parquet",
            
            # Tên file output tổng hợp
            FINAL_COMPARISON_FILENAME="final_model_comparison.csv",
            FINAL_REPORT_FILENAME="final_report.txt",
            BEST_MODEL_FILENAME="best_model_info.json",
            
            # Bật/tắt biểu đồ
            PLOT_FINAL_COMPARISON=True,
            PLOT_ERROR_DISTRIBUTION=True,
            PLOT_RESIDUALS_ALL=True,
            PLOT_SEASONAL_ERROR=True,
            PLOT_INSIGHTS=True
        ),
        kernel_name=KERNEL_NAME
    )
    print("✅ Hoàn thành notebook 7")
except Exception as e:
    print(f"❌ Lỗi notebook 7: {e}")

# ==================== TỔNG KẾT ====================
end_time = time.time()
total_time = end_time - start_time
minutes = int(total_time // 60)
seconds = int(total_time % 60)

print("\n" + "="*60)
print("🎉 PIPELINE HOÀN THÀNH!")
print("="*60)
print(f"\n⏱️ Tổng thời gian chạy: {minutes} phút {seconds} giây")
print("\n📁 Kết quả được lưu tại:")
print("   - Notebooks runs: notebooks/runs/")
print("   - Dữ liệu xử lý: data/processed/")
print("   - Biểu đồ: outputs/figures/")
print("   - Báo cáo: outputs/reports/")
print("\n📊 File kết quả chính:")
print("   - final_model_comparison.csv: So sánh các mô hình")
print("   - best_model_info.json: Thông tin mô hình tốt nhất")
print("   - final_report.txt: Báo cáo tổng kết")
print("\n🚀 Chạy dashboard: streamlit run app.py")
print("="*60)