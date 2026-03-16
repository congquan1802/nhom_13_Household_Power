"""
Simple ARIMA and ETS for Energy Forecasting
Tự viết - Siêu nhẹ, không cần statsmodels
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class SimpleARIMA:
    """
    ARIMA đơn giản tự viết - chỉ dùng numpy và sklearn
    
    Parameters:
    -----------
    order : tuple
        (p, d, q) - p: AR bậc, d: sai phân, q: MA bậc
    seasonal_order : tuple
        (P, D, Q, m) - m: chu kỳ mùa vụ
    """
    
    def __init__(self, order=(1,1,1), seasonal_order=(0,0,0,24)):
        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.m = seasonal_order
        self.model = None
        self.last_values = None
        self.seasonal_components = None
        self.trend = None
        self.level = None
        
    def _difference(self, series, d=1):
        """Tính sai phân bậc d"""
        diff = series.copy()
        for _ in range(d):
            diff = diff.diff().dropna()
        return diff
    
    def _inverse_difference(self, diff_series, original, d=1):
        """Khôi phục từ sai phân (đơn giản)"""
        if d == 0:
            return diff_series
        # Cộng dồn để khôi phục
        last = original.iloc[-1]
        result = [last]
        for val in diff_series:
            result.append(result[-1] + val)
        return np.array(result[1:])
    
    def _create_lag_features(self, y, max_lag):
        """Tạo lag features cho AR"""
        X, y_target = [], []
        for i in range(max_lag, len(y)):
            X.append(y.iloc[i-max_lag:i].values)
            y_target.append(y.iloc[i])
        return np.array(X), np.array(y_target)
    
    def fit(self, y):
        """
        Huấn luyện mô hình ARIMA đơn giản
        
        Parameters:
        -----------
        y : pd.Series
            Dữ liệu huấn luyện
        """
        self.y_original = y
        y_work = y.copy()
        
        # 1. Sai phân theo bậc d
        for _ in range(self.d):
            y_work = y_work.diff().dropna()
        
        # 2. Xử lý mùa vụ nếu có
        if self.P > 0 and self.m > 0:
            # Tách seasonal component đơn giản
            self.seasonal_components = []
            for i in range(self.m):
                if len(y_work) > i:
                    seasonal_idx = range(i, len(y_work), self.m)
                    if len(seasonal_idx) > 0:
                        seasonal_mean = y_work.iloc[seasonal_idx].mean()
                        self.seasonal_components.append(seasonal_mean)
                    else:
                        self.seasonal_components.append(0)
            
            # Loại bỏ seasonal
            for i in range(len(y_work)):
                y_work.iloc[i] = y_work.iloc[i] - self.seasonal_components[i % self.m]
        
        # 3. AR component
        if self.p > 0:
            X, y_target = self._create_lag_features(y_work, self.p)
            
            if len(X) > 0:
                self.model = LinearRegression()
                self.model.fit(X, y_target)
                self.last_values = y_work.iloc[-self.p:].values
            else:
                self.model = None
                self.last_values = [y_work.iloc[-1]]
        else:
            self.model = None
            self.last_values = [y_work.mean()] if len(y_work) > 0 else [0]
        
        # 4. Lưu thông tin trend/level
        self.level = y_work.mean()
        if len(y_work) > 1:
            self.trend = (y_work.iloc[-1] - y_work.iloc[0]) / len(y_work)
        else:
            self.trend = 0
            
        return self
    
    def predict(self, steps):
        """
        Dự báo steps bước tiếp theo
        
        Parameters:
        -----------
        steps : int
            Số bước cần dự báo
            
        Returns:
        --------
        np.array
            Giá trị dự báo
        """
        predictions = []
        last_p = list(self.last_values) if self.last_values is not None else [0]
        
        # Dự báo trên dữ liệu đã sai phân
        for i in range(steps):
            if self.p > 0 and self.model is not None:
                # Dùng AR model
                if len(last_p) >= self.p:
                    X_pred = np.array([last_p[-self.p:]]).reshape(1, -1)
                    pred = self.model.predict(X_pred)[0]
                else:
                    pred = self.level
            else:
                # Fallback: dùng level + trend
                pred = self.level + self.trend * (i + 1)
            
            predictions.append(pred)
            last_p.append(pred)
        
        predictions = np.array(predictions)
        
        # Thêm seasonal component nếu có
        if self.P > 0 and self.m > 0 and self.seasonal_components is not None:
            for i in range(len(predictions)):
                predictions[i] += self.seasonal_components[i % self.m]
        
        # Khôi phục từ sai phân
        for _ in range(self.d):
            last_original = self.y_original.iloc[-1]
            predictions = np.cumsum(predictions) + last_original
        
        return predictions


class SimpleETS:
    """
    ETS (Holt-Winters) đơn giản tự viết
    
    Parameters:
    -----------
    seasonal_periods : int
        Chu kỳ mùa vụ (mặc định: 24 cho dữ liệu giờ)
    trend : str
        'add' hoặc None
    seasonal : str
        'add' hoặc None
    """
    
    def __init__(self, seasonal_periods=24, trend='add', seasonal='add'):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.level = None
        self.trend_component = None
        self.seasonal_components = None
        
    def fit(self, y):
        """
        Huấn luyện mô hình ETS đơn giản
        """
        y_values = y.values
        n = len(y_values)
        
        # 1. Tính seasonal components
        if self.seasonal == 'add' and self.seasonal_periods > 0:
            n_seasons = n // self.seasonal_periods
            self.seasonal_components = []
            
            for i in range(self.seasonal_periods):
                idx = list(range(i, n, self.seasonal_periods))[:n_seasons]
                if len(idx) > 0:
                    seasonal_mean = np.mean([y_values[j] for j in idx])
                else:
                    seasonal_mean = 0
                self.seasonal_components.append(seasonal_mean)
            
            # Chuẩn hóa để tổng = 0
            seasonal_mean_all = np.mean(self.seasonal_components)
            self.seasonal_components = [s - seasonal_mean_all for s in self.seasonal_components]
            
            # Loại bỏ seasonal để tính trend
            y_deseasonal = []
            for i in range(n):
                y_deseasonal.append(y_values[i] - self.seasonal_components[i % self.seasonal_periods])
            y_deseasonal = np.array(y_deseasonal)
        else:
            self.seasonal_components = None
            y_deseasonal = y_values
        
        # 2. Tính trend
        x = np.arange(len(y_deseasonal))
        if self.trend == 'add':
            # Linear regression để tìm trend
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y_deseasonal, rcond=None)[0]
            self.trend_component = slope
            self.level = intercept
        else:
            self.trend_component = 0
            self.level = np.mean(y_deseasonal)
        
        return self
    
    def predict(self, steps):
        """
        Dự báo steps bước tiếp theo
        """
        predictions = []
        
        for i in range(steps):
            # Level + trend
            if self.trend == 'add':
                pred = self.level + self.trend_component * (i + 1)
            else:
                pred = self.level
            
            # Thêm seasonal
            if self.seasonal == 'add' and self.seasonal_components is not None:
                pred += self.seasonal_components[i % self.seasonal_periods]
            
            predictions.append(pred)
        
        return np.array(predictions)


# ==================== EVALUATION FUNCTIONS ====================

def calculate_mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / denominator)