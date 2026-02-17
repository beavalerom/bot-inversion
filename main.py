from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import yfinance as yf
import joblib
import os
import database
from features import calculate_indicators
from sqlmodel import SQLModel
import time

# --- CONFIGURACIÓN DE LA APP ---
app = FastAPI(
    title="Bot de Inversión",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Guarda los resultados aquí para no llamar a Yahoo cada vez
PREDICTION_CACHE = {} 
CACHE_DURATION = 900  # 15 minutos en segundos (900s)

# Carpeta "static" accesible desde el navegador
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- INICIALIZAR BBDD AL ARRANCAR ---
try:
    database.init_db()
    print("Base de datos inicializada.")
except Exception as e:
    print(f"Error iniciando BBDD: {e}")

# --- ACEPTAR PETICIONES ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MAPA DE CRIPTOS A TICKERS ---
CRYPTO_TICKERS = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD",
    "cardano": "ADA-USD", "ada": "ADA-USD",
    "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
    "ripple": "XRP-USD", "xrp": "XRP-USD",
    "polkadot": "DOT-USD", "dot": "DOT-USD",
    "matic": "MATIC-USD", "polygon": "MATIC-USD",
    "avax": "AVAX-USD", "avalanche": "AVAX-USD"
}

DESCRIPCIONES = {
    "bitcoin": "Bitcoin es la criptomoneda líder del mercado.",
    "dogecoin": "Dogecoin es una criptomoneda de código abierto basada en blockchain.",
    "ethereum": "Ethereum es una plataforma blockchain para contratos inteligentes.",
    "cardano": "Cardano es una blockchain científica y ecológica."
}

# --- CARGA DEL MODELO ---
model = None
scaler = None
FEATURES_LIST = []
MODELO_LISTO = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "modelo_entrenado.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "ml_models", "scaler.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "ml_models", "config.pkl")

print(f"Buscando modelo en: {MODEL_PATH}")
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        # Cargamos la configuración para saber qué columnas espera el modelo
        if os.path.exists(CONFIG_PATH):
            config = joblib.load(CONFIG_PATH)
            FEATURES_LIST = config.get('features', [])
            print(f"Configuración cargada. Esperando {len(FEATURES_LIST)} features.")
        
        MODELO_LISTO = True
        print("✅ Modelo IA cargado correctamente.")
    else:
        print("❌ No están los archivos .pkl en la carpeta ml_models.")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")

# --- LIMPIAR COLUMNAS (FIX MULTIINDEX) ---
def clean_yahoo_columns(df):
    """Elimina el MultiIndex que devuelve yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except:
            pass
    return df

# --- DESCARGAR DATOS REALES ---
def get_live_data(symbol_key, period="1y"): # Periodo aumentado a 1 año para EMAs precisas
    ticker = CRYPTO_TICKERS.get(symbol_key.lower())
    if not ticker: return None
    
    try:
        dat = yf.Ticker(ticker)
        df = dat.history(period=period, interval="1d", auto_adjust=True)

        if df.empty: return None

        df = clean_yahoo_columns(df)
        df = df.reset_index()
        # Convertir nombres de columnas a formato estándar (Title Case) para features.py
        df.columns = [c.capitalize() for c in df.columns] 
        
        # Asegurar fecha
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True) # Features.py espera índice de fecha
        
        return df
    except Exception as e:
        print(f"Error descargando {ticker}: {e}")
        return None

# --- ENDPOINTS ---

@app.get("/")
async def read_index():
    # Se carga automáticamente
    return FileResponse('static/index.html')

@app.get("/crypto")
def get_crypto_info(nombre: str = Query(..., description="Nombre")):
    df = get_live_data(nombre, period="3mo")
    if df is None: return {"error": "Error de conexión o moneda desconocida"}
    
    # Restaurar columnas para lectura fácil
    curr = df['Close'].iloc[-1]
    vol = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
    
    return {
        "nombre": nombre.capitalize(),
        "descripcion": DESCRIPCIONES.get(nombre.lower(), ""),
        "precio_actual": f"{curr:.2f} USD",
        "precio_media": f"{df['Close'].mean():.2f} USD",
        "volumen_actual": int(vol)
    }

@app.get("/crypto/data")
def get_crypto_data(nombre: str = Query(...), n: int = Query(180)):
    # Descargar datos suficientes
    df = get_live_data(nombre, period="2y")
    if df is None: return {"error": "No data"}

    # Resamplear o recortar para enviar al front
    df_tail = df.tail(n).copy() if n > 0 else df.copy()
    
    ts = []
    # Iterar reseteando index para tener la fecha como columna
    df_reset = df_tail.reset_index()
    for _, row in df_reset.iterrows():
        ts.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'close': float(row['Close']),
            'volume': int(row['Volume']) if 'Volume' in row else 0
        })

    return {
        'nombre': nombre.capitalize(),
        'descripcion': DESCRIPCIONES.get(nombre.lower(), ''),
        'stats': {
            'latest_close': float(df_tail['Close'].iloc[-1]),
            'mean_close': float(df_tail['Close'].mean()),
            'min_close': float(df_tail['Close'].min()),
            'max_close': float(df_tail['Close'].max())
        },
        'timeseries': ts
    }

# ==============================================================================
#  🧠 EL CEREBRO DEL BOT (Lógica Actualizada)
# ==============================================================================
@app.get("/predict/{coin}")
def predict_coin(coin: str):
    """
    Aplica la ESTRATEGIA HÍBRIDA:
    1. IA > 50%
    2. Tendencia: Precio > EMA 21
    3. RSI < 70 (No sobrecomprado)
    """
    # Vemos si está en caché
    coin_key = coin.lower() # Normalizamos la clave
    current_time = time.time()

    if coin_key in PREDICTION_CACHE:
        last_update = PREDICTION_CACHE[coin_key]['timestamp']
        # Si han pasado menos de 900 segundos (15 mins), devolvemos lo guardado
        if current_time - last_update < CACHE_DURATION:
            print(f"⚡ CACHÉ HIT: Devolviendo datos guardados para {coin}")
            return PREDICTION_CACHE[coin_key]['data']
            
    print(f"🔄 CACHÉ MISS: Calculando predicción fresca para {coin}...")

    # 1. Obtenemos datos (1 año para que la EMA21 sea precisa)
    df = get_live_data(coin, period="1y")
    
    if df is None: 
        return {"moneda": coin, "recomendacion": "ERROR", "mensaje": "Datos no disponibles"}
    
    precio_actual = float(df['Close'].iloc[-1])
    
    res = {
        "moneda": coin.capitalize(),
        "precio_actual": precio_actual,
        "recomendacion": "NEUTRO",
        "confianza": "0%",
        "probabilidad_valor": 50.0,
        "detalles": {},
        "mensaje": "Modelo no cargado"
    }

    if not MODELO_LISTO:
        return res

    try:
        # 2. Calcular Indicadores (Usando tu archivo features.py)
        # Importante: features.py espera Close, High, Low, Volume
        df_processed = calculate_indicators(df)
        
        # 3. Limpieza idéntica al entrenamiento (borrar columnas de calendario)
        cols_to_drop = ['DayOfWeek', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd']
        cols_existing = [c for c in cols_to_drop if c in df_processed.columns]
        if cols_existing:
            df_processed = df_processed.drop(columns=cols_existing)

        # 4. Tomamos la última fila (el día de hoy)
        last_row = df_processed.iloc[[-1]].copy()
        
        # Rellenar columnas faltantes con 0 si es necesario (seguridad)
        for col in FEATURES_LIST:
            if col not in last_row.columns:
                last_row[col] = 0
                
        # Seleccionar SOLO las columnas que el modelo conoce, en orden
        X_live = last_row[FEATURES_LIST]
        
        # 5. Predicción IA
        X_scaled = scaler.transform(X_live)
        probabilidad = float(model.predict_proba(X_scaled)[:, 1][0]) # Probabilidad de SUBIDA
        
        # 6. --- APLICACIÓN DE LA ESTRATEGIA GANADORA ---
        
        # Recuperamos los indicadores técnicos de la última fila
        ema_21 = float(last_row['EMA_21'].iloc[0])
        rsi = float(last_row['RSI'].iloc[0])
        
        # Filtros Lógicos
        filtro_tendencia = precio_actual > ema_21
        filtro_rsi = rsi < 70
        filtro_ia = probabilidad > 0.50
        
        # Decisión Final: SOLO compramos si TODO es verdadero
        es_compra = filtro_tendencia and filtro_rsi and filtro_ia
        
        # Gestión de Riesgo (Stop Loss -4% / Take Profit +8%)
        stop_loss = precio_actual * 0.96
        take_profit = precio_actual * 1.08
        
        # Construir Mensaje de Razonamiento
        razones = []
        if filtro_ia: razones.append(f"🤖 IA optimista ({probabilidad:.1%})")
        else: razones.append(f"🤖 IA pesimista ({probabilidad:.1%})")
        
        if filtro_tendencia: razones.append("📈 Tendencia Alcista (Precio > EMA21)")
        else: razones.append("📉 Tendencia Bajista (Peligro)")
        
        if filtro_rsi: razones.append("✅ RSI Saludable (Espacio para subir)")
        else: razones.append("🔥 RSI Sobrecomprado (>70)")

        # Respuesta JSON
        res["recomendacion"] = "COMPRAR" if es_compra else "ESPERAR"
        res["probabilidad_valor"] = round(probabilidad * 100, 1)
        res["confianza"] = f"{probabilidad:.1%}"
        res["mensaje"] = "Análisis Completado"
        res["prediccion_ia"] = take_profit # Mostramos el objetivo como "predicción"
        
        # Datos extra para que el Frontend los muestre
        res["detalles"] = {
            "rsi": round(rsi, 2),
            "ema_21": round(ema_21, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "razones": razones
        }

        # Guardar en caché el resultado
        PREDICTION_CACHE[coin_key] = {
            'timestamp': current_time,
            'data': res
        }
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        res["mensaje"] = f"Error interno: {str(e)}"
        
    return res

# --- OTROS ENDPOINTS (Cartera y BBDD) ---

@app.get("/wallet")
def get_wallet_info():
    """Consulta el saldo y las transacciones"""
    return database.obtener_cartera()

@app.post("/trade/buy")
def trade_buy(ticker: str, cantidad: float):
    df = get_live_data(ticker, period="1d")
    if df is None: raise HTTPException(404, detail="Cripto no encontrada")
    
    precio_actual = float(df['Close'].iloc[-1])
    ticker_oficial = CRYPTO_TICKERS.get(ticker.lower())
    
    exito, mensaje = database.registrar_compra(ticker_oficial, cantidad, precio_actual)
    if not exito: return {"status": "error", "mensaje": mensaje}
    return {"status": "ok", "mensaje": mensaje}

@app.post("/trade/sell")
def trade_sell(ticker: str, cantidad: float):
    df = get_live_data(ticker, period="1d")
    if df is None: raise HTTPException(404, detail="Cripto no encontrada")
    
    precio_actual = float(df['Close'].iloc[-1])
    ticker_oficial = CRYPTO_TICKERS.get(ticker.lower())
    
    exito, mensaje = database.registrar_venta(ticker_oficial, cantidad, precio_actual)
    if not exito: return {"status": "error", "mensaje": mensaje}
    return {"status": "ok", "mensaje": mensaje}

# --- NUEVO: ESCÁNER DE MERCADO ---
@app.get("/market/scan")
def scan_market():
    """Analiza todas las monedas y devuelve las mejores oportunidades."""
    resultados = []
    
    # 1. Obtenemos la lista única de tickers REALES (BTC-USD, ADA-USD...)
    tickers_unicos = list(set(CRYPTO_TICKERS.values()))
    
    print(f"🔍 Escaneando {len(tickers_unicos)} activos...") # Log para depurar

    for ticker in tickers_unicos:
        try:
            # 2. Descarga DIRECTA (Sin pasar por get_live_data para evitar el error de claves)
            dat = yf.Ticker(ticker)
            df = dat.history(period="1y", interval="1d", auto_adjust=True)
            
            if df.empty: continue
            
            # --- LIMPIEZA DE DATOS (Crucial para que features.py funcione) ---
            df = clean_yahoo_columns(df)
            df = df.reset_index()
            # Convertimos columnas a Mayúscula Inicial (Open, High, Low...)
            df.columns = [c.capitalize() for c in df.columns] 
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            # ---------------------------------------------------------------

            # 3. Calcular indicadores
            df = calculate_indicators(df)
            
            # Limpieza de features prohibidas (Igual que en training)
            cols_to_drop = ['DayOfWeek', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd']
            cols_existing = [c for c in cols_to_drop if c in df.columns]
            if cols_existing:
                df = df.drop(columns=cols_existing)

            # 4. Preparar última fila para la IA
            last_row = df.iloc[[-1]].copy()
            # Rellenar columnas faltantes con 0 (seguridad)
            for col in FEATURES_LIST:
                if col not in last_row.columns:
                    last_row[col] = 0
            
            X_live = last_row[FEATURES_LIST]
            
            # 5. Predicción IA
            if MODELO_LISTO:
                X_scaled = scaler.transform(X_live)
                prob = float(model.predict_proba(X_scaled)[:, 1][0])
            else:
                prob = 0.5 # Fallback si no hay modelo

            # 6. Lógica de Estrategia (La Ganadora)
            precio_actual = float(df['Close'].iloc[-1])
            ema_21 = float(df['EMA_21'].iloc[-1])
            rsi = float(df['RSI'].iloc[-1])
            
            tendencia = "ALCISTA" if precio_actual > ema_21 else "BAJISTA"
            
            recomendacion = "ESPERAR" # Por defecto
            
            # REGLAS: IA > 50% + TENDENCIA ALCISTA + RSI < 70
            if prob > 0.50 and tendencia == "ALCISTA" and rsi < 70:
                if prob > 0.60:
                    recomendacion = "COMPRAR FUERTE 🚀"
                else:
                    recomendacion = "COMPRAR ✅"
            elif tendencia == "BAJISTA":
                recomendacion = "VENDER 📉"
                
            resultados.append({
                "ticker": ticker.replace("-USD", ""), # Quitamos el -USD para que quede bonito
                "precio": precio_actual,
                "recomendacion": recomendacion,
                "probabilidad": round(prob * 100, 1),
                "tendencia": tendencia
            })
            
        except Exception as e:
            print(f"⚠️ Error escaneando {ticker}: {e}")
            continue

    # Ordenar: Los que recomiendan COMPRAR primero, luego por probabilidad
    resultados.sort(key=lambda x: (1 if "COMPRAR" in x['recomendacion'] else 0, x['probabilidad']), reverse=True)
    
    return resultados

@app.post("/reset")
def reset_database():
    """Borra todo y reinicia la cartera a 10.000$"""
    try:
        # 1. Liberar cualquier conexión bloqueada
        database.engine.dispose()

        # 2. Borrar el archivo físico 
        if os.path.exists("database.db"):
            os.remove("database.db")
            print("Archivo database.db eliminado.")
        
        # 4. Inicializar de nuevo la BBDD
        database.init_db()
        
        return {"status": "ok", "mensaje": "Cartera reiniciada a 10.000$ 💰"}
    except Exception as e:
        print(f"⚠️ Error al resetear: {e}")
        return {"status": "error", "mensaje": f"No se pudo resetear: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando Servidor API...")
    uvicorn.run(app, host="0.0.0.0", port=8001)