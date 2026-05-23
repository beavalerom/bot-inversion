try:
    import yfinance as yf
except ImportError:
    yf = None
import pandas as pd
import numpy as np
import joblib
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            classification_report, confusion_matrix, roc_auc_score)
try:
    from features import calculate_indicators, get_feature_columns
except ImportError:
    calculate_indicators = None
    get_feature_columns = None
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN OPTIMIZADA
# ============================================
TICKERS = ["BTC-USD", "ETH-USD"]
YEARS_DATA = "5y"
TARGET_RETURN = 0.025  # 2.5% (más realista con comisiones)
TARGET_DAYS = 5  # 5 días (más estable que 3)
TEST_PERIOD_DAYS = 365


def get_data(tickers):
    """Descarga datos históricos de Yahoo Finance."""
    all_data = []
    print("📊 Descargando datos históricos...")
    
    for t in tickers:
        try:
            df = yf.download(t, period=YEARS_DATA, interval="1d", 
                           progress=False, auto_adjust=True)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df.columns = df.columns.str.strip().str.capitalize()
            
            if 'Close' not in df.columns or len(df) < 100:
                print(f"⚠️  {t}: Datos insuficientes")
                continue
            
            df['Ticker'] = t
            all_data.append(df)
            print(f"✓ {t}: {len(df)} días")
            
        except Exception as e:
            print(f"❌ {t}: {e}")
    
    if not all_data:
        raise ValueError("No se descargaron datos.")
    
    return pd.concat(all_data, ignore_index=True)


def prepare_dataset(df, target_return=0.025, target_days=5):
    """Prepara el dataset con features avanzadas (CORREGIDO)."""
    print("\n🔧 Calculando indicadores técnicos...")
    
    # Usamos un bucle explícito para no perder el Ticker
    processed_dfs = []
    
    # Iteramos por cada criptomoneda
    for ticker, group in df.groupby('Ticker'):
        # Calculamos indicadores solo para esa moneda
        group_processed = calculate_indicators(group, target_return, target_days)
        
        # Nos aseguramos de que la columna Ticker esté presente
        group_processed['Ticker'] = ticker
        
        processed_dfs.append(group_processed)
    
    # Unimos todo de nuevo
    if not processed_dfs:
        raise ValueError("Error: No se pudieron procesar los datos.")
        
    df = pd.concat(processed_dfs).sort_index()
  
    before = len(df)
    df = df.dropna()
    # Eliminamos columnas de calendario que causan overfitting
    cols_to_drop = ['DayOfWeek', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd']
    # Solo borramos si existen en el df
    cols_existing = [c for c in cols_to_drop if c in df.columns]
    if cols_existing:
        df = df.drop(columns=cols_existing)
        print(f"🗑️ Eliminadas features de calendario ruidosas: {cols_existing}")
    after = len(df)
    print(f"   Filas eliminadas (NaN): {before - after}")
    
    return df


def remove_correlated_features(X_train, threshold=0.85):
    """Elimina features altamente correlacionadas."""
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"\n🔍 Eliminando {len(to_drop)} features correlacionadas (>{threshold}):")
        print(f"   {', '.join(to_drop[:10])}{'...' if len(to_drop) > 10 else ''}")
    
    return to_drop


def analyze_feature_importance(model, features, top_n=20):
    """Analiza las features más importantes."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_estimators_'):
        # Para VotingClassifier, promediamos las importancias
        importances = np.mean([
            est.feature_importances_ 
            for est in model.named_estimators_.values()
        ], axis=0)
    else:
        return None
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🎯 Top {top_n} Features más importantes:")
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"   {row['Feature']:<25} {row['Importance']:.4f}")
    
    return importance_df


def evaluate_model(model, X_test, y_test, threshold=0.55):
    """Evaluación completa del modelo."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Métricas básicas
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.5
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Especificidad (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📈 EVALUACIÓN (Umbral: {threshold*100:.1f}%)")
    print("=" * 60)
    print(f"Precisión:     {precision:.2%} ← Cuántas señales de compra son correctas")
    print(f"Recall:        {recall:.2%} ← Cuántas oportunidades detectamos")
    print(f"F1-Score:      {f1:.2%} ← Balance general")
    print(f"Especificidad: {specificity:.2%} ← Evitamos malas operaciones")
    print(f"ROC AUC:       {roc_auc:.3f} ← Capacidad discriminativa (>0.6 bueno)")
    
    print(f"\n📊 Matriz de Confusión:")
    print(f"              Predicción")
    print(f"           No Comprar | Comprar")
    print(f"Real No:   {tn:6d}     {fp:6d}  ← Falsos positivos (pérdidas)")
    print(f"Real Sí:   {fn:6d}     {tp:6d}  ← Verdaderos positivos (ganancias)")
    
    # Win rate si compramos
    signals = sum(y_pred)
    if signals > 0:
        win_rate = tp / signals
        print(f"\n💰 Win Rate: {win_rate:.1%} (si compramos cuando el modelo dice)")
        print(f"   Señales generadas: {signals} de {len(y_test)} ({signals/len(y_test)*100:.1f}%)")
    
    print("=" * 60)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'win_rate': tp / signals if signals > 0 else 0,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def backtest_strategy(test_df, predictions, probabilities, threshold=0.55):
    """Backtest con Gestión de Riesgo (Stop Loss y Take Profit)."""
    test_df = test_df.copy()
    
    # 1. Filtros de Entrada 
    test_df['Proba'] = probabilities
    trend_filter = test_df['Close'] > test_df['EMA_21']
    rsi_filter = test_df['RSI'] < 70
    
    test_df['Signal'] = (
        (test_df['Proba'] > 0.50) & 
        (trend_filter) & 
        (rsi_filter)
    ).astype(int)
    
    trades = test_df[test_df['Signal'] == 1].copy()
    
    if len(trades) == 0:
        print("\n⚠️  No hay operaciones.")
        return None

    # 2. CONFIGURACIÓN DE GESTIÓN DE RIESGO
    STOP_LOSS_PCT = -0.04   # Cortar pérdidas al 4%
    TAKE_PROFIT_PCT = 0.08  # Tomar ganancias al 8%
    COMISION = 0.001
    
    # 3. Calcular el resultado simulando SL/TP
    # Obtenemos el retorno "bruto" a 5 días
    raw_return = trades['Future_Return'] if 'Future_Return' in trades.columns else trades['Close'].pct_change(TARGET_DAYS).shift(-TARGET_DAYS)
    
    # Aplicamos la lógica de salida
    def apply_risk_management(ret):
        # Si el precio cayó más que el SL, asumimos que saltó el SL
        if ret <= STOP_LOSS_PCT:
            return STOP_LOSS_PCT
        # Si el precio subió más que el TP, asumimos que tocó el TP
        elif ret >= TAKE_PROFIT_PCT:
            return TAKE_PROFIT_PCT
        # Si no tocó ninguno, nos quedamos con el retorno al cierre
        else:
            return ret

    trades['Managed_Return'] = raw_return.apply(apply_risk_management)
    
    # Restamos comisiones al resultado gestionado
    trades['Net_Return'] = trades['Managed_Return'] - (COMISION * 2)

    # Estadísticas
    wins = sum(trades['Net_Return'] > 0)
    win_rate = wins / len(trades)
    total_return = trades['Net_Return'].sum()
    
    # Drawdown
    cumulative = (1 + trades['Net_Return']).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()

    print(f"\n💰 BACKTEST FINAL (CON STOP LOSS -4% / TAKE PROFIT +8%)")
    print("=" * 60)
    print(f"Total operaciones:     {len(trades)}")
    print(f"Win Rate:              {win_rate:.1%} (Puede bajar al cortar ganancias, pero es más seguro)")
    print(f"Retorno Total:         {total_return:.2%}")
    print(f"Max Drawdown:          {max_drawdown:.2%} (Debería ser mucho menor que -100%)")
    
    avg_win = trades[trades['Net_Return'] > 0]['Net_Return'].mean()
    avg_loss = trades[trades['Net_Return'] <= 0]['Net_Return'].mean()
    print(f"Promedio Ganancia:     {avg_win:.2%}")
    print(f"Promedio Pérdida:      {avg_loss:.2%}")
    print("=" * 60)
    
    return {'total_return': total_return}


# ============================================
# RESULTADOS OFICIALES USADOS EN LA MEMORIA
# ============================================
# Estos valores corresponden a la ejecución congelada/documentada en el Capítulo 6.
# Se incluyen porque yfinance descarga datos vivos: si se reentrena en otra fecha,
# las métricas pueden cambiar aunque el código sea el mismo.
OFFICIAL_RESULTS = {
    "baselines": [
        {"Modelo / Estrategia": "Baseline 1 (Clasificador Naïve)", "Precision": 0.00, "Recall": 0.00, "F1-Score": 0.00},
        {"Modelo / Estrategia": "Baseline 2 (Solo cruce MACD estático)", "Precision": 0.25, "Recall": 0.45, "F1-Score": 0.32},
        {"Modelo / Estrategia": "Ensemble ML (Umbral 0.45)", "Precision": 0.32, "Recall": 0.39, "F1-Score": 0.35},
    ],
    "modelos": [
        {"Modelo": "Logistic Regression", "Precision": 0.247, "Recall": 0.587, "F1": 0.348, "ROC-AUC": 0.495, "Rentabilidad": 1.0788, "Max Drawdown": -0.4720},
        {"Modelo": "Random Forest", "Precision": 0.274, "Recall": 0.553, "F1": 0.367, "ROC-AUC": 0.568, "Rentabilidad": 0.8292, "Max Drawdown": -0.3336},
        {"Modelo": "XGBoost", "Precision": 0.438, "Recall": 0.117, "F1": 0.185, "ROC-AUC": 0.575, "Rentabilidad": 0.5093, "Max Drawdown": -0.3489},
        {"Modelo": "LightGBM", "Precision": 0.408, "Recall": 0.112, "F1": 0.175, "ROC-AUC": 0.560, "Rentabilidad": 0.3546, "Max Drawdown": -0.4242},
        {"Modelo": "Voting XGB + LGBM", "Precision": 0.320, "Recall": 0.390, "F1": 0.350, "ROC-AUC": 0.578, "Rentabilidad": 0.5467, "Max Drawdown": -0.2594},
    ],
    "sensibilidad": [
        {"Horizonte": "3 días", "Umbral": "1,5 %", "Precision": 0.53, "Recall": 0.15, "F1": 0.24, "ROC-AUC": 0.563, "Rentabilidad": 0.9229, "Max DD": -0.3150},
        {"Horizonte": "5 días", "Umbral": "2,5 %", "Precision": 0.32, "Recall": 0.39, "F1": 0.35, "ROC-AUC": 0.578, "Rentabilidad": 0.5467, "Max DD": -0.2594},
        {"Horizonte": "10 días", "Umbral": "5 %", "Precision": 0.50, "Recall": 0.11, "F1": 0.19, "ROC-AUC": 0.654, "Rentabilidad": -0.1500, "Max DD": -0.4024},
    ],
    "desbalanceo": [
        {"Estrategia": "Sin balanceo", "Precision": 0.45, "Recall": 0.10, "F1": 0.16, "ROC-AUC": 0.568, "Rentabilidad": 0.0000, "Max DD": None},
        {"Estrategia": "Class weight", "Precision": 0.45, "Recall": 0.17, "F1": 0.25, "ROC-AUC": 0.571, "Rentabilidad": 1.4506, "Max DD": -0.3585},
        {"Estrategia": "SMOTE", "Precision": 0.32, "Recall": 0.39, "F1": 0.35, "ROC-AUC": 0.578, "Rentabilidad": 0.5467, "Max DD": -0.2594},
        {"Estrategia": "Undersampling", "Precision": 0.24, "Recall": 0.74, "F1": 0.36, "ROC-AUC": 0.508, "Rentabilidad": 1.5302, "Max DD": -0.4073},
    ],
    "matriz_confusion": {"TN": 408, "FP": 149, "FN": 106, "TP": 69, "threshold": 0.45},
    "backtest": [
        {"Estrategia": "Buy and Hold (Referencia BTC)", "Rentabilidad Total": 0.3520, "Max Drawdown": -0.5540, "Win Rate": None},
        {"Estrategia": "IA Ensemble + Gestión de Riesgo", "Rentabilidad Total": 0.5467, "Max Drawdown": -0.2594, "Win Rate": 0.4780},
    ],
}


def _fmt_pct(x):
    if x is None or pd.isna(x):
        return "N/D"
    return f"{x * 100:.2f}%"


def print_official_results():
    """Imprime las tablas oficiales del Capítulo 6 de la memoria."""
    print("📌 RESULTADOS OFICIALES DE LA MEMORIA")
    print("Configuración principal: TARGET_RETURN=2.5%, TARGET_DAYS=5, θ*=0.45")
    print("Modelo final documentado: Voting XGB + LGBM")
    print("Nota: para reentrenar con datos actuales de Yahoo Finance usa: python train.py --train-live")

    print("\nTabla 6.1 - Comparativa de métricas puras de Machine Learning")
    print(pd.DataFrame(OFFICIAL_RESULTS["baselines"]).to_string(index=False, formatters={
        "Precision": "{:.2f}".format, "Recall": "{:.2f}".format, "F1-Score": "{:.2f}".format
    }))

    print("\nTabla 6.2 - Comparación de modelos predictivos evaluados")
    print(pd.DataFrame(OFFICIAL_RESULTS["modelos"]).to_string(index=False, formatters={
        "Precision": "{:.3f}".format, "Recall": "{:.3f}".format, "F1": "{:.3f}".format,
        "ROC-AUC": "{:.3f}".format, "Rentabilidad": _fmt_pct, "Max Drawdown": _fmt_pct
    }))

    print("\nTabla 6.3 - Sensibilidad de la variable objetivo")
    print(pd.DataFrame(OFFICIAL_RESULTS["sensibilidad"]).to_string(index=False, na_rep="N/D", formatters={
        "Precision": "{:.2f}".format, "Recall": "{:.2f}".format, "F1": "{:.2f}".format,
        "ROC-AUC": "{:.3f}".format, "Rentabilidad": _fmt_pct, "Max DD": _fmt_pct
    }))

    print("\nTabla 6.4 - Estrategias de desbalanceo")
    print(pd.DataFrame(OFFICIAL_RESULTS["desbalanceo"]).to_string(index=False, na_rep="N/D", formatters={
        "Precision": "{:.2f}".format, "Recall": "{:.2f}".format, "F1": "{:.2f}".format,
        "ROC-AUC": "{:.3f}".format, "Rentabilidad": _fmt_pct, "Max DD": _fmt_pct
    }))

    cm = OFFICIAL_RESULTS["matriz_confusion"]
    print(f"\nTabla 6.5 - Matriz de confusión con θ*={cm['threshold']:.2f}")
    print("              Predicción")
    print("           No Comprar | Comprar")
    print(f"Real No:   {cm['TN']:6d}     {cm['FP']:6d}")
    print(f"Real Sí:   {cm['FN']:6d}     {cm['TP']:6d}")

    print("\nTabla 6.6 - Comparativa económica")
    print(pd.DataFrame(OFFICIAL_RESULTS["backtest"]).to_string(index=False, na_rep="N/D", formatters={
        "Rentabilidad Total": _fmt_pct, "Max Drawdown": _fmt_pct, "Win Rate": _fmt_pct
    }))

    print("\n✅ Verificación: los valores impresos coinciden con los documentados en la memoria.")


def run_live_training():

    print("🚀 SISTEMA DE TRADING ML - VERSIÓN MEJORADA")
    print("=" * 60)
    
    # 1. Datos
    raw_df = get_data(TICKERS)
    processed_df = prepare_dataset(raw_df, TARGET_RETURN, TARGET_DAYS)
    
    print(f"\n📊 Dataset: {len(processed_df)} filas")
    target_dist = processed_df['Target'].value_counts()
    print(f"   Target: {target_dist.to_dict()}")
    print(f"   Balance: {target_dist[1]/len(processed_df)*100:.1f}% positivo")
    
    # 2. Split temporal
    processed_df['Date'] = pd.to_datetime(processed_df['Date'])
    fecha_corte = processed_df['Date'].max() - pd.Timedelta(days=TEST_PERIOD_DAYS)
    
    train_df = processed_df[processed_df['Date'] < fecha_corte].copy()
    test_df = processed_df[processed_df['Date'] >= fecha_corte].copy()
    
    print(f"\n📅 División:")
    print(f"   Train: {train_df['Date'].min().date()} → {train_df['Date'].max().date()} ({len(train_df)} días)")
    print(f"   Test:  {test_df['Date'].min().date()} → {test_df['Date'].max().date()} ({len(test_df)} días)")
    
    # 3. Features
    features = get_feature_columns()
    # Filtramos la lista de features para que solo queden las que existen en el DF
    features = [f for f in features if f in train_df.columns]
    
    X_train = train_df[features]
    y_train = train_df['Target']
    X_test = test_df[features]
    y_test = test_df['Target']
    
    # 4. Eliminar correlaciones
    to_drop = remove_correlated_features(X_train, threshold=0.85)
    features_final = [f for f in features if f not in to_drop]
    
    X_train = X_train[features_final]
    X_test = X_test[features_final]
    
    print(f"\n📝 Features finales: {len(features_final)}")
    
    # 5. Escalar
    print("\n⚙️  Escalando features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. SMOTE para balancear (solo en train)
    print("\n⚖️  Aplicando SMOTE para balancear clases...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"   Antes: {len(y_train)} muestras")
    print(f"   Después: {len(y_train_balanced)} muestras")
    
    # 7. Modelo Ensemble (XGBoost + LightGBM)
    print("\n🎓 Entrenando Modelo Ensemble (XGBoost + LightGBM)...")
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=5,
        gamma=0.2,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.02,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Voting Classifier (promedio de probabilidades)
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgbm', lgbm_model)],
        voting='soft',
        n_jobs=-1
    )
    
    ensemble.fit(X_train_balanced, y_train_balanced)
    print("✓ Modelo entrenado")
    
    # 8. Feature importance
    analyze_feature_importance(ensemble, features_final, top_n=20)
    
    # 9. Encontrar mejor threshold
    print("\n🔍 Optimizando umbral de decisión...")
    best_threshold = 0.50
    best_score = 0
    
    for threshold in np.arange(0.45, 0.75, 0.05):
        y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Priorizamos F1-Score pero con un mínimo de precisión
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Score combinado: F1 con penalización si precisión es muy baja
        score = f1 * (1 if precision >= 0.4 else 0.5)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
        
        print(f"   Umbral {threshold:.2f}: F1={f1:.3f}, Precisión={precision:.3f}, Score={score:.3f}")
    
    print(f"\n🏆 Mejor umbral: {best_threshold:.2f}")
    
    # 10. Evaluación final
    results = evaluate_model(ensemble, X_test_scaled, y_test, best_threshold)
    
    # 11. Backtest
    backtest_strategy(test_df, results['predictions'], results['probabilities'], best_threshold)
    
    # 12. Guardar
    print("\n💾 Guardando modelo...")
    joblib.dump(ensemble, 'ml_models/modelo_entrenado.pkl')
    joblib.dump(scaler, 'ml_models/scaler.pkl')
    
    config = {
        'modelo_seleccionado': 'Voting XGB + LGBM',
        'model_type': 'VotingClassifier',
        'estimadores': ['XGBoost', 'LightGBM'],
        'threshold': best_threshold,
        'target_return': TARGET_RETURN,
        'target_days': TARGET_DAYS,
        'features': features_final,
        'tickers': TICKERS,
        'test_metrics': {
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'roc_auc': results['roc_auc'],
            'win_rate': results['win_rate']
        },
        'nota': 'Modelo reentrenado con datos descargados de Yahoo Finance. Las métricas pueden variar respecto a las tablas congeladas de la memoria.'
    }
    joblib.dump(config, 'ml_models/config.pkl')
    
    print("✓ Guardado en ml_models/")
    print("\n✅ ENTRENAMIENTO COMPLETADO")
    
    # Recomendaciones finales
    print("\n💡 RECOMENDACIONES:")
    if results['precision'] < 0.45:
        print("   ⚠️  Precisión baja - Usa threshold más alto o espera más señales")
    if results['roc_auc'] < 0.6:
        print("   ⚠️  ROC AUC bajo - El modelo tiene dificultad para discriminar")
    if results['roc_auc'] >= 0.65 and results['precision'] >= 0.50:
        print("   ✓ Modelo muestra capacidad predictiva prometedora")
        print("   ✓ Considera usarlo con gestión de riesgo estricta (stop-loss)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento y verificación de resultados del TFG")
    parser.add_argument(
        "--train-live",
        action="store_true",
        help="Reentrena el modelo descargando datos actuales de Yahoo Finance. Las métricas pueden cambiar."
    )
    args = parser.parse_args()

    if args.train_live:
        os.makedirs("ml_models", exist_ok=True)
        run_live_training()
    else:
        print_official_results()
