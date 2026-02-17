from sqlmodel import SQLModel, Field, Session, select, create_engine
from datetime import datetime
from typing import Optional

# --- 1. MODELOS DE DATOS (TABLAS) ---
class Wallet(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    saldo_usd: float = Field(default=10000.0)  # Saldo inicial ficticio

class Transaccion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    fecha: datetime = Field(default_factory=datetime.now)
    ticker: str
    accion: str  # "COMPRA" o "VENTA"
    cantidad: float
    precio_unitario: float
    total_usd: float

# --- 2. CONEXIÓN ---
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url)

# --- 3. FUNCIONES DE LÓGICA DE NEGOCIO ---

def init_db():
    """Crea las tablas y la cartera inicial si no existen."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        wallet = session.exec(select(Wallet)).first()
        if not wallet:
            # Crear cartera con 10k por defecto
            session.add(Wallet(saldo_usd=10000.0))
            session.commit()

def obtener_cartera():
    """Devuelve el saldo, las últimas transacciones y el PORTFOLIO acumulado."""
    with Session(engine) as session:
        wallet = session.exec(select(Wallet)).first()
        if not wallet: return {"saldo": 0, "historial": [], "portfolio": {}}
        # A) Calcular cuánto tienes de cada moneda
        # Necesitamos leer TODAS las transacciones para sumar compras y restar ventas
        all_txs = session.exec(select(Transaccion)).all()
        
        portfolio = {}
        
        for tx in all_txs:
            coin = tx.ticker.lower() # Usamos minúsculas para agrupar (bitcoin, Bitcoin -> bitcoin)
            
            if coin not in portfolio:
                portfolio[coin] = 0.0
            
            if tx.accion == "COMPRA":
                portfolio[coin] += tx.cantidad
            elif tx.accion == "VENTA":
                portfolio[coin] -= tx.cantidad
        
        # Filtramos monedas con saldo 0 (o casi 0 por decimales)
        portfolio_limpio = {k: v for k, v in portfolio.items() if v > 0.00001}

        # B) Obtener historial reciente para mostrar en lista
        recent_txs = session.exec(select(Transaccion).order_by(Transaccion.id.desc()).limit(10)).all()
        
        return {
            "saldo": wallet.saldo_usd, 
            "historial": recent_txs,
            "portfolio": portfolio_limpio 
        }

def registrar_compra(ticker: str, cantidad: float, precio_actual: float):
    """Intenta realizar una compra. Devuelve (Exito: bool, Mensaje: str)"""
    coste_total = precio_actual * cantidad
    
    with Session(engine) as session:
        wallet = session.exec(select(Wallet)).first()
        
        if wallet.saldo_usd < coste_total:
            return False, f"Saldo insuficiente. Tienes {wallet.saldo_usd:.2f}$, necesitas {coste_total:.2f}$"
        
        # Restar saldo
        wallet.saldo_usd -= coste_total
        
        # Crear transacción
        tx = Transaccion(
            ticker=ticker.upper(),
            accion="COMPRA",
            cantidad=cantidad,
            precio_unitario=precio_actual,
            total_usd=coste_total
        )
        
        session.add(wallet)
        session.add(tx)
        session.commit()
        session.refresh(wallet)
        
        return True, f"Compra exitosa. Nuevo saldo: {wallet.saldo_usd:.2f}$"

def registrar_venta(ticker: str, cantidad: float, precio_actual: float):
    """Venta segura usando SQLModel y verificando fondos"""
    ingreso_total = precio_actual * cantidad
    ticker_lower = ticker.lower()

    # 1. Verificar si tenemos la moneda (calculando desde el historial)
    datos_actuales = obtener_cartera()
    cantidad_en_cartera = datos_actuales["portfolio"].get(ticker_lower, 0.0)
    if cantidad_en_cartera < cantidad:
        return False, f"No tienes suficiente {ticker.upper()}. Tienes: {cantidad_en_cartera:.4f}"
 
    # 2. Proceder a la venta
    with Session(engine) as session:
        wallet = session.exec(select(Wallet)).first()
        
        # Sumar saldo
        wallet.saldo_usd += ingreso_total
        
        # Registrar transacción
        tx = Transaccion(
            ticker=ticker.upper(),
            accion="VENTA",
            cantidad=cantidad,
            precio_unitario=precio_actual,
            total_usd=ingreso_total
        )
        
        session.add(wallet)
        session.add(tx)
        session.commit()
        
        return True, f"Venta exitosa. Recuperado: {ingreso_total:.2f}$"