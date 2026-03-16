"""
Energy Forecasting Library
Thư viện chứa tất cả các hàm xử lý cho bài toán dự báo năng lượng
UCI Household Power Consumption Dataset

THÔNG TIN DATASET (THEO UCI):
================================
- Địa điểm: Sceaux (cách Paris 7km, Pháp)
- Thời gian: Tháng 12/2006 → Tháng 11/2010 (47 tháng)
- Tổng số phép đo: 2.075.259 dòng
- Tần suất: 1 phút/lần
- Missing values: ~1.25% (được biểu thị bằng ;; trong file)
- Ngày đặc biệt: 28/04/2007 (có nhiều missing values)

CÔNG THỨC QUAN TRỌNG:
================================
active_energy_not_submetered = (global_active_power * 1000 / 60) 
                                - sub_metering_1 
                                - sub_metering_2 
                                - sub_metering_3

Giải thích:
- global_active_power: kW (cần đổi sang W: *1000)
- Chia 60 vì đo mỗi phút (tính ra Wh/phút)
- Kết quả: năng lượng tiêu thụ bởi các thiết bị không được đo bởi sub-meters

Các lớp chính:
- DataLoader: Đọc dữ liệu
- Preprocessor: Tiền xử lý
- AnomalyDetector: Phát hiện bất thường
- FeatureEngineer: Tạo đặc trưng
- ForecastingModels: Các mô hình dự báo
- Evaluator: Đánh giá mô hình
- Utils: Các hàm tiện ích
- VisualizationHelper: Vẽ biểu đồ (tùy chọn)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import os
import json
warnings.filterwarnings('ignore')

# ==================== 1. DATA LOADER ====================

class DataLoader:
    """Xử lý đọc dữ liệu từ các nguồn khác nhau"""
    
    @staticmethod
    def load_raw_data(file_path: str) -> pd.DataFrame:
        """
        Đọc dữ liệu thô từ file txt của UCI
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn đến file dữ liệu
            
        Returns:
        --------
        pd.DataFrame
            DataFrame chứa dữ liệu thô
        """
        try:
            # File UCI có dấu ; làm separator, giá trị thiếu là '?'
            df = pd.read_csv(
                file_path, 
                sep=';', 
                na_values=['?', '', 'NA', 'null'],
                keep_default_na=True,
                low_memory=False
            )
            print(f"✅ Đã đọc {len(df):,} dòng từ {file_path}")
            return df
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {file_path}")
            return None
        except Exception as e:
            print(f"❌ Lỗi khi đọc file: {e}")
            return None
    
    @staticmethod
    def load_processed_data(file_path: str) -> pd.DataFrame:
        """
        Đọc dữ liệu đã xử lý (parquet hoặc csv)
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn đến file dữ liệu đã xử lý
            
        Returns:
        --------
        pd.DataFrame
            DataFrame chứa dữ liệu đã xử lý
        """
        path = Path(file_path)
        
        try:
            if not path.exists():
                print(f"❌ Không tìm thấy file: {file_path}")
                return None
                
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
                print(f"✅ Đã đọc {len(df):,} dòng từ {path.name}")
            else:
                df = pd.read_csv(path)
                # Thử parse datetime nếu có cột timestamp
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"✅ Đã đọc {len(df):,} dòng từ {path.name}")
            
            return df
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {file_path}: {e}")
            return None
    
    @staticmethod
    def get_dataset_info() -> dict:
        """
        Trả về thông tin chi tiết về dataset theo mô tả UCI
        
        Returns:
        --------
        dict
            Dictionary chứa thông tin dataset
        """
        info = {
            'source': 'UCI Machine Learning Repository',
            'location': 'Sceaux, Pháp (cách Paris 7km)',
            'period': '12/2006 - 11/2010',
            'months': 47,
            'total_measurements': 2075259,
            'missing_percentage': 1.25,
            'special_missing_date': '28/04/2007',
            'frequency': '1 phút',
            'description': 'Đo lường tiêu thụ điện hộ gia đình',
            'formula': 'active_energy_not_submetered = (global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3)'
        }
        
        print("\n" + "="*60)
        print("📊 THÔNG TIN DATASET UCI HOUSEHOLD POWER CONSUMPTION")
        print("="*60)
        for key, value in info.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        print("="*60 + "\n")
        
        return info


# ==================== 2. PREPROCESSOR ====================

class Preprocessor:
    """Tiền xử lý dữ liệu"""
    
    @staticmethod
    def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Kết hợp cột Date và Time thành datetime index
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa cột Date và Time
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với datetime index
        """
        if 'Date' in df.columns and 'Time' in df.columns:
            try:
                df['Datetime'] = pd.to_datetime(
                    df['Date'] + ' ' + df['Time'], 
                    format='%d/%m/%Y %H:%M:%S',
                    errors='coerce'
                )
                df = df.set_index('Datetime').drop(['Date', 'Time'], axis=1)
                df = df.sort_index()
                print(f"✅ Đã parse datetime: {df.index.min()} → {df.index.max()}")
                print(f"📊 Tổng số: {len(df):,} dòng")
            except Exception as e:
                print(f"❌ Lỗi parse datetime: {e}")
        else:
            print("⚠️ Không tìm thấy cột Date và Time")
        
        return df
    
    @staticmethod
    def convert_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi các cột về kiểu số
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần chuyển đổi
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với các cột đã chuyển về numeric
        """
        for col in df.columns:
            if col not in ['Date', 'Time', 'Datetime']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("✅ Đã chuyển đổi các cột về kiểu số")
        return df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Xử lý giá trị thiếu
        Dataset có ~1.25% missing values theo mô tả UCI
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần xử lý
        method : str
            Phương pháp xử lý: 'interpolate', 'ffill', 'bfill', 'drop', 'mean'
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã xử lý missing values
        """
        original_len = len(df)
        total_cells = original_len * len(df.columns)
        missing_before = df.isna().sum().sum()
        missing_percent = (missing_before / total_cells) * 100
        
        print("\n" + "="*60)
        print("📊 PHÂN TÍCH MISSING VALUES")
        print("="*60)
        print(f"   - Tổng số ô: {total_cells:,}")
        print(f"   - Missing trước xử lý: {missing_before:,} ô")
        print(f"   - Tỷ lệ missing: {missing_percent:.4f}%")
        print(f"   - Theo UCI: ~1.25%")
        print("-" * 60)
        
        if missing_before == 0:
            print("✅ Không có missing values")
            return df
        
        # Thống kê theo cột
        missing_by_col = df.isna().sum()
        missing_cols = missing_by_col[missing_by_col > 0]
        
        print(f"\n📌 Missing theo cột:")
        for col, count in missing_cols.items():
            col_percent = (count / len(df)) * 100
            print(f"   - {col}: {count:,} dòng ({col_percent:.4f}%)")
        
        # Xử lý missing
        if method == 'interpolate':
            # Nội suy theo thời gian
            df = df.interpolate(method='time', limit_direction='both', limit=10)
        elif method == 'ffill':
            df = df.ffill()
        elif method == 'bfill':
            df = df.bfill()
        elif method == 'drop':
            df = df.dropna()
        elif method == 'mean':
            df = df.fillna(df.mean())
        else:
            print(f"⚠️ Phương pháp {method} không được hỗ trợ, dùng interpolate")
            df = df.interpolate(method='time', limit_direction='both')
        
        missing_after = df.isna().sum().sum()
        print(f"\n✅ KẾT QUẢ XỬ LÝ:")
        print(f"   - Missing sau xử lý: {missing_after:,} ô")
        print(f"   - Đã xử lý: {missing_before - missing_after:,} ô")
        print(f"   - Phương pháp: {method}")
        print("="*60 + "\n")
        
        return df
    
    @staticmethod
    def resample_data(df: pd.DataFrame, freq: str = 'H') -> pd.DataFrame:
        """
        Resample dữ liệu theo tần suất mong muốn
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame với datetime index
        freq : str
            Tần suất resample: 'H' (giờ), 'D' (ngày), 'W' (tuần)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã resample
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            print("❌ Index phải là DatetimeIndex để resample")
            return df
        
        # Resample và lấy mean
        df_resampled = df.resample(freq).mean()
        
        print(f"✅ Đã resample từ {freq} sang {freq}: {len(df):,} → {len(df_resampled):,} dòng")
        
        return df_resampled
    
    @staticmethod
    def calculate_active_energy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính năng lượng hoạt động không được đo bởi sub-meters
        Công thức: (global_active_power * 1000 / 60) - sub_metering_1 - sub_metering_2 - sub_metering_3
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa các cột: Global_active_power, Sub_metering_1, Sub_metering_2, Sub_metering_3
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với cột active_energy_not_submetered
        """
        required_cols = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        
        # Kiểm tra các cột cần thiết
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Thiếu các cột: {missing_cols}")
            return df
        
        # Tính toán theo công thức
        df['active_energy_not_submetered'] = (
            df['Global_active_power'] * 1000 / 60 - 
            df['Sub_metering_1'] - 
            df['Sub_metering_2'] - 
            df['Sub_metering_3']
        )
        
        # Đảm bảo không âm (có thể có sai số đo)
        df['active_energy_not_submetered'] = df['active_energy_not_submetered'].clip(lower=0)
        
        print("\n" + "="*60)
        print("⚡ ĐÃ TÍNH ACTIVE ENERGY NOT SUBMETERED")
        print("="*60)
        print("Công thức: (global_active_power * 1000 / 60) - sub1 - sub2 - sub3")
        print(f"\n📊 Thống kê:")
        print(f"   - Mean: {df['active_energy_not_submetered'].mean():.4f} Wh")
        print(f"   - Std: {df['active_energy_not_submetered'].std():.4f} Wh")
        print(f"   - Min: {df['active_energy_not_submetered'].min():.4f} Wh")
        print(f"   - Max: {df['active_energy_not_submetered'].max():.4f} Wh")
        print(f"   - % zero: {(df['active_energy_not_submetered'] == 0).mean()*100:.2f}%")
        print("="*60 + "\n")
        
        return df
    
    @staticmethod
    def check_special_date(df: pd.DataFrame, date: str = '2007-04-28') -> pd.DataFrame:
        """
        Kiểm tra dữ liệu ngày đặc biệt (có missing values)
        Theo UCI: ngày 28/04/2007 có nhiều missing values
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame với datetime index
        date : str
            Ngày cần kiểm tra (mặc định: 28/04/2007)
            
        Returns:
        --------
        pd.DataFrame
            Dữ liệu của ngày đặc biệt
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            print("⚠️ Index không phải DatetimeIndex")
            return pd.DataFrame()
        
        # Lọc theo ngày
        df_date = df[df.index.strftime('%Y-%m-%d') == date]
        
        if len(df_date) == 0:
            print(f"\n📌 Không có dữ liệu ngày {date}")
        else:
            print(f"\n" + "="*60)
            print(f"📅 KIỂM TRA NGÀY ĐẶC BIỆT: {date}")
            print("="*60)
            print(f"   - Số dòng: {len(df_date)} (theo lý thuyết: 1440 dòng cho 24h)")
            print(f"   - Thời gian: {df_date.index.min()} → {df_date.index.max()}")
            
            # Kiểm tra missing values
            missing = df_date.isna().sum()
            missing_cols = missing[missing > 0]
            
            if len(missing_cols) > 0:
                print(f"\n   ❌ Missing values trong ngày này:")
                for col, count in missing_cols.items():
                    print(f"      - {col}: {count} dòng ({count/len(df_date)*100:.2f}%)")
            else:
                print(f"\n   ✅ Không có missing values trong ngày này")
            
            print("="*60 + "\n")
        
        return df_date


# ==================== 3. ANOMALY DETECTOR ====================

class AnomalyDetector:
    """Phát hiện bất thường trong dữ liệu"""
    
    @staticmethod
    def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.Series:
        """
        Phát hiện outlier bằng Z-score
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        column : str
            Tên cột cần phát hiện outlier
        threshold : float
            Ngưỡng Z-score (mặc định: 3)
            
        Returns:
        --------
        pd.Series
            Boolean mask đánh dấu outlier
        """
        mean = df[column].mean()
        std = df[column].std()
        
        if std == 0:
            return pd.Series(False, index=df.index)
        
        z_scores = np.abs((df[column] - mean) / std)
        return z_scores > threshold
    
    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
        """
        Phát hiện outlier bằng IQR (Interquartile Range)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        column : str
            Tên cột cần phát hiện outlier
            
        Returns:
        --------
        pd.Series
            Boolean mask đánh dấu outlier
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    @staticmethod
    def mark_anomalies(df: pd.DataFrame, column: str, method: str = 'both', 
                       zscore_threshold: float = 3) -> pd.DataFrame:
        """
        Đánh dấu các điểm bất thường
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        column : str
            Tên cột cần phát hiện anomaly
        method : str
            Phương pháp: 'zscore', 'iqr', 'both'
        zscore_threshold : float
            Ngưỡng cho Z-score
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với các cột đánh dấu anomaly
        """
        result = df.copy()
        
        if method in ['zscore', 'both']:
            result['is_anomaly_zscore'] = AnomalyDetector.detect_outliers_zscore(
                df, column, zscore_threshold
            )
        
        if method in ['iqr', 'both']:
            result['is_anomaly_iqr'] = AnomalyDetector.detect_outliers_iqr(df, column)
        
        if method == 'both':
            result['is_anomaly'] = result['is_anomaly_zscore'] | result['is_anomaly_iqr']
        elif method == 'zscore':
            result['is_anomaly'] = result['is_anomaly_zscore']
        elif method == 'iqr':
            result['is_anomaly'] = result['is_anomaly_iqr']
        
        n_anomalies = result['is_anomaly'].sum()
        print(f"✅ Phát hiện {n_anomalies} điểm bất thường ({(n_anomalies/len(df))*100:.4f}%)")
        
        return result


# ==================== 4. FEATURE ENGINEERING ====================

class FeatureEngineer:
    """Tạo đặc trưng cho time series"""
    
    @staticmethod
    def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các đặc trưng thời gian từ datetime index
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame với datetime index
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với các time features mới
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            print("⚠️ Index không phải DatetimeIndex, bỏ qua time features")
            return df
        
        result = df.copy()
        
        # Các đặc trưng cơ bản
        result['hour'] = result.index.hour
        result['day'] = result.index.day
        result['month'] = result.index.month
        result['year'] = result.index.year
        result['dayofweek'] = result.index.dayofweek
        result['quarter'] = result.index.quarter
        result['dayofyear'] = result.index.dayofyear
        result['weekofyear'] = result.index.isocalendar().week.astype(int)
        
        # Boolean features
        result['is_weekend'] = (result['dayofweek'] >= 5).astype(int)
        result['is_month_start'] = (result.index.is_month_start).astype(int)
        result['is_month_end'] = (result.index.is_month_end).astype(int)
        
        # Cyclical encoding cho hour và month
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['month_sin'] = np.sin(2 * np.pi * (result['month'] - 1) / 12)
        result['month_cos'] = np.cos(2 * np.pi * (result['month'] - 1) / 12)
        
        return result
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
        """
        Tạo lag features
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        column : str
            Tên cột cần tạo lag
        lags : list
            Danh sách các lag cần tạo (ví dụ: [1, 2, 3, 24])
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với các lag features
        """
        result = df.copy()
        
        for lag in lags:
            result[f'{column}_lag_{lag}'] = result[column].shift(lag)
        
        return result
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, column: str, windows: list) -> pd.DataFrame:
        """
        Tạo rolling statistics
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        column : str
            Tên cột cần tính rolling
        windows : list
            Danh sách các window size
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với các rolling features
        """
        result = df.copy()
        
        for window in windows:
            result[f'{column}_roll_mean_{window}'] = result[column].rolling(window=window).mean()
            result[f'{column}_roll_std_{window}'] = result[column].rolling(window=window).std()
            result[f'{column}_roll_min_{window}'] = result[column].rolling(window=window).min()
            result[f'{column}_roll_max_{window}'] = result[column].rolling(window=window).max()
        
        return result


# ==================== 5. FORECASTING MODELS ====================

class ForecastingModels:
    """Các mô hình dự báo chuỗi thời gian"""
    
    @staticmethod
    def split_time_series(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
        """
        Chia train/test theo thời gian (không random)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        target_col : str
            Tên cột mục tiêu
        test_size : float
            Tỷ lệ dữ liệu test (0.2 = 20%)
            
        Returns:
        --------
        tuple
            (y_train, y_test, X_train, X_test)
        """
        n = len(df)
        split_idx = int(n * (1 - test_size))
        
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        
        y_train = train[target_col]
        y_test = test[target_col]
        
        # Bỏ target column khỏi features
        X_train = train.drop(columns=[target_col]) if target_col in train.columns else train
        X_test = test.drop(columns=[target_col]) if target_col in test.columns else test
        
        print(f"✅ Train: {len(y_train):,} samples ({train.index[0]} → {train.index[-1]})")
        print(f"✅ Test: {len(y_test):,} samples ({test.index[0]} → {test.index[-1]})")
        
        return y_train, y_test, X_train, X_test
    
    @staticmethod
    def seasonal_naive(y_train: pd.Series, y_test: pd.Series, seasonal_period: int = 24) -> pd.Series:
        """
        Mô hình Seasonal Naïve
        Dự báo = giá trị cùng thời điểm của chu kỳ trước
        
        Parameters:
        -----------
        y_train : pd.Series
            Dữ liệu train
        y_test : pd.Series
            Dữ liệu test (chỉ lấy index)
        seasonal_period : int
            Chu kỳ mùa vụ (24 cho dữ liệu giờ)
            
        Returns:
        --------
        pd.Series
            Dự báo cho tập test
        """
        predictions = []
        
        for i in range(len(y_test)):
            if i < seasonal_period and len(y_train) >= seasonal_period:
                # Lấy giá trị từ seasonal_period cuối của train
                pred = y_train.iloc[-seasonal_period + i]
            elif i >= seasonal_period:
                # Dùng giá trị đã dự báo trước đó
                pred = predictions[i - seasonal_period]
            else:
                # Fallback: dùng giá trị cuối cùng của train
                pred = y_train.iloc[-1]
            
            predictions.append(pred)
        
        return pd.Series(predictions, index=y_test.index)
    
    @staticmethod
    def arima_forecast(y_train: pd.Series, y_test: pd.Series, order: tuple = None):
        """
        Dự báo bằng ARIMA
        
        Parameters:
        -----------
        y_train : pd.Series
            Dữ liệu train
        y_test : pd.Series
            Dữ liệu test (chỉ lấy index)
        order : tuple
            Bậc ARIMA (p,d,q). Nếu None thì dùng (1,1,1)
            
        Returns:
        --------
        tuple
            (fitted_model, predictions)
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            if order is None:
                order = (1, 1, 1)
            
            # Huấn luyện mô hình
            model = ARIMA(y_train, order=order)
            fitted_model = model.fit()
            
            # Dự báo
            predictions = fitted_model.forecast(steps=len(y_test))
            
            return fitted_model, predictions
            
        except ImportError:
            print("⚠️ statsmodels chưa được cài đặt. Chạy: pip install statsmodels")
            return None, pd.Series([np.nan] * len(y_test), index=y_test.index)
        except Exception as e:
            print(f"❌ Lỗi khi chạy ARIMA: {e}")
            return None, pd.Series([np.nan] * len(y_test), index=y_test.index)
    
    @staticmethod
    def ets_forecast(y_train: pd.Series, y_test: pd.Series, seasonal_periods: int = 24,
                     trend: bool = True, seasonal: bool = True, 
                     damped_trend: bool = False, initialization_method: str = 'estimated'):
        """
        Dự báo bằng ETS (Holt-Winters)
        
        Parameters:
        -----------
        y_train : pd.Series
            Dữ liệu train
        y_test : pd.Series
            Dữ liệu test (chỉ lấy index)
        seasonal_periods : int
            Chu kỳ mùa vụ
        trend : bool
            Có thành phần trend không
        seasonal : bool
            Có thành phần mùa vụ không
        damped_trend : bool
            Có dập tắt trend không
        initialization_method : str
            Phương pháp khởi tạo: 'estimated', 'heuristic'
            
        Returns:
        --------
        tuple
            (fitted_model, predictions)
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Xác định loại model
            if trend and seasonal:
                model = ExponentialSmoothing(
                    y_train,
                    seasonal_periods=seasonal_periods,
                    trend='add',
                    seasonal='add',
                    damped_trend=damped_trend,
                    initialization_method=initialization_method
                )
            elif trend and not seasonal:
                model = ExponentialSmoothing(
                    y_train,
                    trend='add',
                    damped_trend=damped_trend,
                    initialization_method=initialization_method
                )
            elif not trend and seasonal:
                model = ExponentialSmoothing(
                    y_train,
                    seasonal_periods=seasonal_periods,
                    seasonal='add',
                    initialization_method=initialization_method
                )
            else:
                model = ExponentialSmoothing(
                    y_train,
                    initialization_method=initialization_method
                )
            
            # Huấn luyện
            fitted_model = model.fit()
            
            # Dự báo
            predictions = fitted_model.forecast(steps=len(y_test))
            
            return fitted_model, predictions
            
        except ImportError:
            print("⚠️ statsmodels chưa được cài đặt. Chạy: pip install statsmodels")
            return None, pd.Series([np.nan] * len(y_test), index=y_test.index)
        except Exception as e:
            print(f"❌ Lỗi khi chạy ETS: {e}")
            return None, pd.Series([np.nan] * len(y_test), index=y_test.index)


# ==================== 6. EVALUATOR ====================

class Evaluator:
    """Đánh giá mô hình dự báo"""
    
    @staticmethod
    def calculate_mae(y_true: np.array, y_pred: np.array) -> float:
        """
        Mean Absolute Error
        
        Parameters:
        -----------
        y_true : np.array
            Giá trị thực tế
        y_pred : np.array
            Giá trị dự báo
            
        Returns:
        --------
        float
            MAE
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def calculate_rmse(y_true: np.array, y_pred: np.array) -> float:
        """
        Root Mean Square Error
        
        Parameters:
        -----------
        y_true : np.array
            Giá trị thực tế
        y_pred : np.array
            Giá trị dự báo
            
        Returns:
        --------
        float
            RMSE
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def calculate_smape(y_true: np.array, y_pred: np.array) -> float:
        """
        Symmetric Mean Absolute Percentage Error
        
        Parameters:
        -----------
        y_true : np.array
            Giá trị thực tế
        y_pred : np.array
            Giá trị dự báo
            
        Returns:
        --------
        float
            SMAPE (%)
        """
        denominator = (np.abs(y_true) + np.abs(y_pred))
        # Tránh chia cho 0
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / denominator)
        return smape
    
    @staticmethod
    def calculate_all_metrics(y_true: np.array, y_pred: np.array, model_name: str = '') -> dict:
        """
        Tính tất cả các metrics
        
        Parameters:
        -----------
        y_true : np.array
            Giá trị thực tế
        y_pred : np.array
            Giá trị dự báo
        model_name : str
            Tên mô hình
            
        Returns:
        --------
        dict
            Dictionary chứa các metrics
        """
        return {
            'model': model_name,
            'mae': Evaluator.calculate_mae(y_true, y_pred),
            'rmse': Evaluator.calculate_rmse(y_true, y_pred),
            'smape': Evaluator.calculate_smape(y_true, y_pred)
        }
    
    @staticmethod
    def residual_analysis(y_true: np.array, y_pred: np.array) -> dict:
        """
        Phân tích phần dư
        
        Parameters:
        -----------
        y_true : np.array
            Giá trị thực tế
        y_pred : np.array
            Giá trị dự báo
            
        Returns:
        --------
        dict
            Dictionary chứa thông tin phần dư
        """
        residuals = y_true - y_pred
        
        return {
            'residuals': residuals,
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis(),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q1': np.percentile(residuals, 25),
            'q2': np.percentile(residuals, 50),
            'q3': np.percentile(residuals, 75)
        }
    
    @staticmethod
    def compare_models(results_dict: dict) -> pd.DataFrame:
        """
        So sánh nhiều mô hình
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary chứa kết quả của các mô hình
            {model_name: {'mae': ..., 'rmse': ..., 'smape': ...}}
            
        Returns:
        --------
        pd.DataFrame
            DataFrame so sánh các mô hình
        """
        comparison = []
        
        for model_name, metrics in results_dict.items():
            comparison.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'SMAPE': metrics['smape']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('RMSE')
        
        return df_comparison


# ==================== 7. UTILITIES ====================

class Utils:
    """Các hàm tiện ích"""
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, path: str, index: bool = True):
        """
        Lưu DataFrame ra file
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần lưu
        path : str
            Đường dẫn file (hỗ trợ .csv, .parquet)
        index : bool
            Có lưu index không
        """
        path_obj = Path(path)
        
        # Tạo thư mục nếu chưa có
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if path_obj.suffix == '.parquet':
                df.to_parquet(path_obj, index=index)
            else:
                # Mặc định lưu csv
                if not path_obj.suffix:
                    path_obj = path_obj.with_suffix('.csv')
                df.to_csv(path_obj, index=index)
            
            print(f"✅ Đã lưu {len(df):,} dòng vào {path_obj}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu file {path_obj}: {e}")
    
    @staticmethod
    def print_section(title: str, char: str = "=", width: int = 60):
        """
        In tiêu đề section đẹp mắt
        
        Parameters:
        -----------
        title : str
            Tiêu đề
        char : str
            Ký tự trang trí
        width : int
            Độ rộng
        """
        print(f"\n{char*width}")
        print(f"{title.center(width)}")
        print(f"{char*width}\n")
    
    @staticmethod
    def get_basic_stats(df: pd.DataFrame, column: str) -> dict:
        """
        Lấy thống kê cơ bản của một cột
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        column : str
            Tên cột
            
        Returns:
        --------
        dict
            Dictionary chứa các thống kê
        """
        if column not in df.columns:
            print(f"❌ Không tìm thấy cột {column}")
            return {}
        
        return {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max(),
            'q1': df[column].quantile(0.25),
            'q3': df[column].quantile(0.75),
            'missing': df[column].isna().sum()
        }
    
    @staticmethod
    def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Đảm bảo DataFrame có datetime index
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần kiểm tra
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với datetime index
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Thử chuyển đổi từ cột timestamp
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)
        elif 'Datetime' in df.columns:
            df = df.set_index('Datetime')
            df.index = pd.to_datetime(df.index)
        elif 'date' in df.columns:
            df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
        
        return df
    
    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Load file cấu hình JSON
        
        Parameters:
        -----------
        config_path : str
            Đường dẫn file config
            
        Returns:
        --------
        dict
            Dictionary chứa cấu hình
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Đã load config từ {config_path}")
            return config
        except Exception as e:
            print(f"❌ Lỗi khi load config: {e}")
            return {}
    
    @staticmethod
    def validate_uci_dataset(df: pd.DataFrame) -> dict:
        """
        Kiểm tra dữ liệu có đúng với mô tả UCI không
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần kiểm tra
            
        Returns:
        --------
        dict
            Kết quả kiểm tra
        """
        print("\n" + "="*60)
        print("🔍 KIỂM TRA DỮ LIỆU SO VỚI MÔ TẢ UCI")
        print("="*60)
        
        results = {}
        
        # Kiểm tra số dòng
        expected_rows = 2075259
        actual_rows = len(df)
        results['rows_match'] = actual_rows == expected_rows
        print(f"\n📊 Số dòng:")
        print(f"   - Theo UCI: {expected_rows:,}")
        print(f"   - Thực tế: {actual_rows:,}")
        print(f"   - Kết quả: {'✅' if results['rows_match'] else '⚠️'} {'Khớp' if results['rows_match'] else f'Chênh lệch {abs(actual_rows - expected_rows):,} dòng'}")
        
        # Kiểm tra khoảng thời gian
        if isinstance(df.index, pd.DatetimeIndex):
            start_date = df.index.min()
            end_date = df.index.max()
            
            print(f"\n📅 Khoảng thời gian:")
            print(f"   - Theo UCI: 12/2006 → 11/2010")
            print(f"   - Thực tế: {start_date.strftime('%m/%Y')} → {end_date.strftime('%m/%Y')}")
            
        # Kiểm tra missing values
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        results['missing_percent'] = missing_pct
        print(f"\n⚠️ Missing values:")
        print(f"   - Theo UCI: ~1.25%")
        print(f"   - Thực tế: {missing_pct:.4f}%")
        
        # Kiểm tra các cột bắt buộc
        required_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        results['all_columns_present'] = len(missing_cols) == 0
        
        print(f"\n📌 Các cột:")
        if missing_cols:
            print(f"   ❌ Thiếu: {missing_cols}")
        else:
            print(f"   ✅ Đầy đủ 7 cột theo UCI")
        
        print("="*60 + "\n")
        
        return results


# ==================== 8. VISUALIZATION HELPER ====================

class VisualizationHelper:
    """Helper cho việc vẽ biểu đồ (có thể dùng trong notebooks)"""
    
    @staticmethod
    def plot_time_series(df: pd.DataFrame, column: str, title: str = None, 
                         figsize: tuple = (15, 5)):
        """
        Vẽ biểu đồ chuỗi thời gian
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame với datetime index
        column : str
            Tên cột cần vẽ
        title : str
            Tiêu đề
        figsize : tuple
            Kích thước figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        df[column].plot(ax=ax, color='blue', linewidth=0.8)
        ax.set_xlabel('Thời gian')
        ax.set_ylabel(column)
        ax.set_title(title or f'Biểu đồ {column} theo thời gian')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_forecast_comparison(actual: pd.Series, forecasts: dict, 
                                  title: str = None, figsize: tuple = (15, 5)):
        """
        So sánh dự báo của nhiều mô hình
        
        Parameters:
        -----------
        actual : pd.Series
            Giá trị thực tế
        forecasts : dict
            Dictionary {model_name: predicted_series}
        title : str
            Tiêu đề
        figsize : tuple
            Kích thước figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Vẽ thực tế
        actual.plot(ax=ax, color='black', linewidth=2, label='Thực tế')
        
        # Vẽ dự báo
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (name, pred) in enumerate(forecasts.items()):
            pred.plot(ax=ax, color=colors[i % len(colors)], 
                     linestyle='--', linewidth=1.5, label=name)
        
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Giá trị')
        ax.set_title(title or 'So sánh dự báo các mô hình')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax


# ==================== MAIN - KHI CHẠY TRỰC TIẾP ====================

if __name__ == "__main__":
    """
    Chạy file này trực tiếp để xem thông tin dataset
    """
    print("\n" + "="*60)
    print("⚡ ENERGY FORECASTING LIBRARY - UCI DATASET INFO")
    print("="*60)
    
    # Lấy thông tin dataset
    DataLoader.get_dataset_info()
    
    print("\n📌 Các class có sẵn:")
    print("   - DataLoader: Đọc dữ liệu")
    print("   - Preprocessor: Tiền xử lý")
    print("   - AnomalyDetector: Phát hiện bất thường")
    print("   - FeatureEngineer: Tạo đặc trưng")
    print("   - ForecastingModels: Các mô hình dự báo")
    print("   - Evaluator: Đánh giá mô hình")
    print("   - Utils: Tiện ích")
    print("   - VisualizationHelper: Vẽ biểu đồ")
    
    print("\n✅ Thư viện đã sẵn sàng sử dụng!")