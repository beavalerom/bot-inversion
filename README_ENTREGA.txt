ENTREGA DEL CÓDIGO FUENTE - TFG

Estructura:
- main.py: backend FastAPI y endpoints de la aplicación.
- database.py: modelos SQLModel y operaciones de cartera/transacciones.
- features.py: ingeniería de características e indicadores técnicos.
- train.py: script de entrenamiento y verificación de resultados.
- verificar_resultados_memoria.py: imprime los resultados oficiales del Capítulo 6.
- static/: frontend HTML, CSS y JavaScript.
- ml_models/: modelo, scaler y configuración serializados.

Uso recomendado para comprobar los números de la memoria:
    python train.py

Equivalente:
    python verificar_resultados_memoria.py

Reentrenamiento con datos actuales de Yahoo Finance:
    python train.py --train-live

Nota importante:
Los datos de Yahoo Finance son vivos. Si se reentrena en otra fecha, las métricas pueden variar.
Por eso el modo normal de train.py imprime la ejecución congelada/documentada en la memoria.
