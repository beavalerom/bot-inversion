import pandas as pd
import numpy as np
from scipy.stats import linregress
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def calculate_indicators(df, target_return=0.025, target_days=5):
    """
    Calcula indicadores técnicos avanzados con enfoque en calidad de señales.
    
    Args:
        df: DataFrame con datos OHLCV
        target_return: Retorno objetivo (2.5% por defecto)
        target_days: Días hacia adelante (5 días para más estabilidad)
    """
    data = df.copy()
    
    # ============================================
    # 1. INDICADORES DE TENDENCIA (MUY IMPORTANTES)
    # ============================================
    
    # EMAs - Más sensibles que SMAs
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    
    # Distancias a EMAs (normalizadas)
    data['Dist_EMA_9'] = (data['Close'] - data['EMA_9']) / data['Close']
    data['Dist_EMA_21'] = (data['Close'] - data['EMA_21']) / data['Close']
    data['Dist_EMA_50'] = (data['Close'] - data['EMA_50']) / data['Close']
    
    # Golden/Death Cross (señal muy potente)
    data['EMA_Cross_Short'] = (data['EMA_9'] > data['EMA_21']).astype(int)
    data['EMA_Cross_Long'] = (data['EMA_50'] > data['EMA_200']).astype(int)
    
    # Cambio en los cruces (momento del cruce)
    data['EMA_Cross_Short_Change'] = data['EMA_Cross_Short'].diff()
    data['EMA_Cross_Long_Change'] = data['EMA_Cross_Long'].diff()
    
    # ADX - Fuerza de la tendencia (usa librería 'ta')
    adx_indicator = ADXIndicator(high=data['High'], low=data['Low'], 
                                  close=data['Close'], window=14)
    data['ADX'] = adx_indicator.adx()
    data['ADX_pos'] = adx_indicator.adx_pos()
    data['ADX_neg'] = adx_indicator.adx_neg()
    data['Trend_Strength'] = (data['ADX'] > 25).astype(int)  # Tendencia fuerte
    
    # CCI - Commodity Channel Index
    cci = CCIIndicator(high=data['High'], low=data['Low'], 
                       close=data['Close'], window=20)
    data['CCI'] = cci.cci()
    data['CCI_Signal'] = np.where(data['CCI'] > 100, 1, 
                                   np.where(data['CCI'] < -100, -1, 0))
    
    # ============================================
    # 2. INDICADORES DE MOMENTUM
    # ============================================
    
    # RSI mejorado
    rsi = RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi.rsi()
    
    # Zonas de RSI (más granulares)
    data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
    data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)
    data['RSI_Neutral'] = ((data['RSI'] >= 40) & (data['RSI'] <= 60)).astype(int)
    
    # RSI Divergence (aproximación simple)
    data['RSI_Slope'] = data['RSI'].diff(5)
    data['Price_Slope'] = data['Close'].pct_change(5)
    data['RSI_Divergence'] = np.where(
        (data['Price_Slope'] < 0) & (data['RSI_Slope'] > 0), 1,  # Divergencia alcista
        np.where((data['Price_Slope'] > 0) & (data['RSI_Slope'] < 0), -1, 0)  # Divergencia bajista
    )
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], 
                                  close=data['Close'], window=14, smooth_window=3)
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    data['Stoch_Oversold'] = (data['Stoch_K'] < 20).astype(int)
    data['Stoch_Overbought'] = (data['Stoch_K'] > 80).astype(int)
    
    # MACD
    macd = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    
    # Cruce de MACD
    data['MACD_Cross'] = (data['MACD'] > data['MACD_Signal']).astype(int)
    data['MACD_Cross_Change'] = data['MACD_Cross'].diff()
    
    # Momentum multi-periodo
    data['ROC_5'] = data['Close'].pct_change(5)  # Rate of Change
    data['ROC_10'] = data['Close'].pct_change(10)
    data['ROC_20'] = data['Close'].pct_change(20)
    
    # ============================================
    # 3. VOLATILIDAD Y BANDAS
    # ============================================
    
    # Bollinger Bands
    bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['BB_Mid'] = bb.bollinger_mavg()
    data['BB_Width'] = bb.bollinger_wband()
    data['BB_Pct'] = bb.bollinger_pband()
    
    # Posición en las bandas
    data['BB_Position'] = np.where(data['Close'] > data['BB_High'], 1,
                                    np.where(data['Close'] < data['BB_Low'], -1, 0))
    
    # ATR - Volatilidad real
    atr = AverageTrueRange(high=data['High'], low=data['Low'], 
                           close=data['Close'], window=14)
    data['ATR'] = atr.average_true_range()
    data['ATR_Pct'] = data['ATR'] / data['Close']
    
    # Volatilidad histórica
    data['Volatility_20'] = data['Close'].rolling(window=20).std() / data['Close']
    data['Volatility_50'] = data['Close'].rolling(window=50).std() / data['Close']
    data['Vol_Ratio'] = data['Volatility_20'] / data['Volatility_50'].replace(0, 1)
    
    # ============================================
    # 4. VOLUMEN (CRÍTICO PARA CRIPTO)
    # ============================================
    
    if 'Volume' in data.columns:
        # Volumen normalizado
        data['Vol_SMA_20'] = data['Volume'].rolling(window=20).mean()
        data['Vol_Ratio'] = data['Volume'] / data['Vol_SMA_20'].replace(0, 1)
        data['Vol_Spike'] = (data['Vol_Ratio'] > 2.0).astype(int)  # Pico de volumen
        
        # OBV - On Balance Volume
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        data['OBV'] = obv
        data['OBV_SMA'] = data['OBV'].rolling(window=20).mean()
        data['OBV_Trend'] = (data['OBV'] > data['OBV_SMA']).astype(int)
        
        # VWAP (Volume Weighted Average Price) - aproximación
        data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        data['Price_vs_VWAP'] = (data['Close'] - data['VWAP']) / data['VWAP']
    else:
        data['Vol_Ratio'] = 1
        data['Vol_Spike'] = 0
        data['OBV_Trend'] = 0
        data['Price_vs_VWAP'] = 0
    
    # ============================================
    # 5. PATRONES DE PRECIO
    # ============================================
    
    # Rangos de precio
    data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Close_Open_Range'] = (data['Close'] - data['Open']) / data['Open']
    
    # Máximos y mínimos recientes
    data['Highest_20'] = data['High'].rolling(window=20).max()
    data['Lowest_20'] = data['Low'].rolling(window=20).min()
    data['Price_Position'] = (data['Close'] - data['Lowest_20']) / (data['Highest_20'] - data['Lowest_20']).replace(0, 1)
    
    # Soporte y resistencia (aproximación)
    data['Near_High'] = (data['Close'] >= data['Highest_20'] * 0.98).astype(int)
    data['Near_Low'] = (data['Close'] <= data['Lowest_20'] * 1.02).astype(int)
    
    # ============================================
    # 6. FEATURES TEMPORALES
    # ============================================
    
    if 'Date' in data.columns:
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
        data['DayOfMonth'] = data['Date'].dt.day
        data['IsMonthStart'] = (data['DayOfMonth'] <= 5).astype(int)
        data['IsMonthEnd'] = (data['DayOfMonth'] >= 25).astype(int)
    else:
        data['DayOfWeek'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
        data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
        data['IsMonthStart'] = 0
        data['IsMonthEnd'] = 0
    
    # ============================================
    # 7. FEATURES COMBINADAS (SEÑALES COMPUESTAS)
    # ============================================
    
    # Señal alcista combinada
    data['Bullish_Signal'] = (
        (data['EMA_Cross_Short'] == 1) & 
        (data['RSI'] > 40) & 
        (data['RSI'] < 70) &
        (data['MACD_Cross'] == 1) &
        (data['ADX'] > 20)
    ).astype(int)
    
    # Señal bajista combinada
    data['Bearish_Signal'] = (
        (data['EMA_Cross_Short'] == 0) & 
        (data['RSI'] > 70) &
        (data['MACD_Cross'] == 0)
    ).astype(int)
    
    # Consolidación (mercado lateral)
    data['Consolidation'] = (
        (data['ADX'] < 20) &
        (data['BB_Width'] < data['BB_Width'].rolling(50).mean())
    ).astype(int)
    
    # ============================================
    # 8. LAGS (MEMORIA TEMPORAL)
    # ============================================
    
    lag_features = [
        'RSI', 'MACD_Diff', 'ADX', 'Stoch_K', 'CCI',
        'ROC_5', 'Vol_Ratio', 'ATR_Pct', 'BB_Pct'
    ]
    
    for col in lag_features:
        if col in data.columns:
            data[f'{col}_lag1'] = data[col].shift(1)
            data[f'{col}_lag2'] = data[col].shift(2)
            data[f'{col}_lag3'] = data[col].shift(3)
    
    # ============================================
    # 9. TARGET (MÁS ESTRICTO)
    # ============================================
    
    # Calculamos retorno futuro
    future_close = data['Close'].shift(-target_days)
    data['Future_Return'] = (future_close - data['Close']) / data['Close']
    
    # Target: Solo marcamos 1 si sube significativamente y de forma sostenida
    # Verificamos que NO caiga más del 1% en el camino
    max_drawdown = data['Close'].rolling(window=target_days).min().shift(-target_days)
    drawdown = (max_drawdown - data['Close']) / data['Close']
    
    data['Target'] = (
        (data['Future_Return'] > target_return) &  # Sube lo suficiente
        (drawdown > -0.015)  # No cae más del 1.5% en el camino
    ).astype(int)
    
    # Target adicional para stop-loss
    data['Will_Drop'] = (data['Future_Return'] < -0.04).astype(int)
    
    # Limpieza
    data = data.drop(columns=['Future_Return'], errors='ignore')
    
    return data


def get_feature_columns():
    """Devuelve las features más importantes (optimizadas)."""
    return [
        # Tendencia (PESO ALTO)
        'Dist_EMA_9', 'Dist_EMA_21', 'Dist_EMA_50',
        'EMA_Cross_Short', 'EMA_Cross_Long',
        'EMA_Cross_Short_Change', 'EMA_Cross_Long_Change',
        'ADX', 'ADX_pos', 'ADX_neg', 'Trend_Strength',
        'CCI', 'CCI_Signal',
        
        # Momentum
        'RSI', 'RSI_Oversold', 'RSI_Overbought', 'RSI_Neutral',
        'RSI_Divergence', 'RSI_Slope',
        'Stoch_K', 'Stoch_D', 'Stoch_Oversold', 'Stoch_Overbought',
        'MACD', 'MACD_Signal', 'MACD_Diff', 'MACD_Cross', 'MACD_Cross_Change',
        'ROC_5', 'ROC_10', 'ROC_20',
        
        # Volatilidad
        'BB_Width', 'BB_Pct', 'BB_Position',
        'ATR_Pct', 'Volatility_20', 'Vol_Ratio',
        
        # Volumen
        'Vol_Ratio', 'Vol_Spike', 'OBV_Trend', 'Price_vs_VWAP',
        
        # Patrones
        'High_Low_Range', 'Price_Position', 'Near_High', 'Near_Low',
        
        # Temporal
        'DayOfWeek', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
        
        # Señales combinadas
        'Bullish_Signal', 'Bearish_Signal', 'Consolidation',
        
        # Lags (memoria)
        'RSI_lag1', 'RSI_lag2', 'RSI_lag3',
        'MACD_Diff_lag1', 'MACD_Diff_lag2', 'MACD_Diff_lag3',
        'ADX_lag1', 'ADX_lag2', 'ADX_lag3',
        'Stoch_K_lag1', 'Stoch_K_lag2', 'Stoch_K_lag3',
        'CCI_lag1', 'CCI_lag2', 'CCI_lag3',
        'ROC_5_lag1', 'ROC_5_lag2', 'ROC_5_lag3',
        'Vol_Ratio_lag1', 'Vol_Ratio_lag2', 'Vol_Ratio_lag3',
        'ATR_Pct_lag1', 'ATR_Pct_lag2', 'ATR_Pct_lag3',
        'BB_Pct_lag1', 'BB_Pct_lag2', 'BB_Pct_lag3',
    ]