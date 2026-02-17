// --- CONFIGURACIÓN GLOBAL ---
const API_PORT = "8001"; 
const API_URL = `http://127.0.0.1:${API_PORT}`;

// MAPEO DE MONEDAS
const TICKER_MAP = {
    "bitcoin": "btc", "btc": "btc",
    "ethereum": "eth", "eth": "eth",
    "solana": "sol", "sol": "sol",
    "cardano": "ada", "ada": "ada",
    "dogecoin": "doge", "doge": "doge",
    "ripple": "xrp", "xrp": "xrp",
    "polkadot": "dot", "dot": "dot",
    "matic": "matic", "polygon": "matic",
    "avax": "avax", "avalanche": "avax"
};

// --- FUNCION DE RESET (PARA EL BOTÓN) ---
async function resetDatabase() {
  if (!confirm("⚠️ ¿Estás seguro? Esto borrará todo tu dinero y reiniciará la cuenta a $10,000.")) {
    return;
  }

  createMessage('🗑️ Reiniciando sistema...', 'bot');
  try {
    const res = await fetch(`${API_URL}/reset`, { method: 'POST' });
    const data = await res.json();
    
    if (data.status === 'ok') {
      createMessage(`✅ ${data.mensaje}`, 'bot');
      updateWallet(); // Pone el saldo a 10.000 visualmente
    } else {
      createMessage(`❌ Error: ${data.mensaje}`, 'bot');
    }
  } catch (err) {
    console.error(err);
    createMessage(`Error de conexión: ${err.message}`, 'bot');
  }
}

// --- FUNCIONES DE CARTERA Y COMPRA ---

async function updateWallet() {
  try {
    const res = await fetch(`${API_URL}/wallet`);
    const data = await res.json();
    const balanceElem = document.getElementById('wallet-balance');
    if (balanceElem) {
      const saldo = data.saldo_usd !== undefined ? data.saldo_usd : data.saldo;
      balanceElem.textContent = `${saldo.toFixed(2)} USD`;
    }
  } catch (err) {
    console.error("Error cartera:", err);
  }
}

async function ejecutarCompra(ticker, cantidad) {
  createMessage(`⏳ Comprando ${cantidad} ${ticker.toUpperCase()}...`, 'bot');
  try {
    const params = new URLSearchParams({ ticker: ticker, cantidad: cantidad });
    const res = await fetch(`${API_URL}/trade/buy?${params}`, { method: 'POST' });
    const data = await res.json();
    
    if (data.status === 'error') {
      createMessage(`❌ Error: ${data.mensaje}`, 'bot');
    } else {
      createMessage(`✅ ¡COMPRA REALIZADA! ${data.mensaje}`, 'bot');
      updateWallet(); 
    }
  } catch (err) {
    createMessage(`Error de red: ${err.message}`, 'bot');
  }
}

async function ejecutarVenta(ticker, cantidad) {
  createMessage(`💸 Vendiendo ${cantidad} ${ticker.toUpperCase()}...`, 'bot');
  try {
    const params = new URLSearchParams({ ticker: ticker, cantidad: cantidad });
    const res = await fetch(`${API_URL}/trade/sell?${params}`, { method: 'POST' });
    const data = await res.json();
    
    if (data.status === 'ok') {
       createMessage(`✅ VENTA EXITOSA. Has recuperado dinero.`, 'bot');
       updateWallet();
    } else {
       createMessage(`❌ Error: ${data.mensaje}`, 'bot');
    }
  } catch (err) {
    createMessage(`Error: ${err.message}`, 'bot');
  }
}

async function verCartera() {
  createMessage('📂 Consultando tu cartera...', 'bot');
  try {
    const res = await fetch(`${API_URL}/wallet`);
    const data = await res.json();
    const portfolio = data.portfolio; 
    
    const unificados = {};
    for (const [key, amount] of Object.entries(portfolio)) {
        if (amount > 0) {
            let nombre = key.toLowerCase();
            if (TICKER_MAP[nombre]) nombre = TICKER_MAP[nombre]; 
            if (!unificados[nombre]) unificados[nombre] = 0;
            unificados[nombre] += amount;
        }
    }

    if (Object.keys(unificados).length === 0) {
      createMessage('Tu cartera está vacía. ¡Empieza a invertir!', 'bot');
      return;
    }

    let html = `
      <div style="margin-top:10px;">
        <p><strong>Tus Activos:</strong></p>
        <table style="width:100%; border-collapse: collapse; font-size:0.9em;">
          <tr style="border-bottom: 1px solid #ccc;">
            <th style="text-align:left; padding:5px;">Moneda</th>
            <th style="text-align:right; padding:5px;">Cantidad</th>
          </tr>
    `;

    for (const [coin, amount] of Object.entries(unificados)) {
        if (amount > 0) {
            html += `
              <tr style="border-bottom: 1px solid #eee;">
                <td style="padding:5px; text-transform:uppercase; font-weight:bold;">${coin}</td>
                <td style="padding:5px; text-align:right;">${amount.toFixed(4)}</td>
              </tr>
            `;
        }
    }

    html += `</table></div>`;
    const saldo = data.saldo_usd !== undefined ? data.saldo_usd : data.saldo;
    html += `<p style="margin-top:10px; font-size:0.85em;">💰 Saldo libre: <strong>${saldo.toFixed(2)} $</strong></p>`;

    createMessage(html, 'bot', true);
  } catch (err) {
    createMessage('❌ Error al obtener la cartera.', 'bot');
  }
}

function createMessage(text, sender = 'user', asHTML = false) {
  const chat = document.getElementById('chat');
  const wrapper = document.createElement('div');
  wrapper.className = `message ${sender}`;
  const avatar = document.createElement('div');
  avatar.className = `avatar ${sender}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (asHTML) bubble.innerHTML = text;
  else bubble.textContent = text;
  if (sender === 'bot') {
    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
  } else {
    wrapper.appendChild(bubble);
    wrapper.appendChild(avatar);
  }
  if(chat) {
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
  }
  return wrapper;
}

// --- LÓGICA PRINCIPAL ---

document.addEventListener('DOMContentLoaded', () => {
  const chat = document.getElementById('chat');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('user-input');

  updateWallet(); 
  createMessage('Hola 👋 Soy tu Broker IA. Escribe una moneda (ej: "Solana", "BTC") para analizar.', 'bot');

  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    const userMsg = input.value.trim();
    if (!userMsg) return;

    createMessage(userMsg, 'user');
    input.value = '';
    
    const lower = userMsg.toLowerCase();
    
    // Detección de moneda
    let found = null;
    const palabras = lower.split(" ");
    for (let p of palabras) {
        const limpia = p.replace(/[^a-z0-9]/g, '');
        if (TICKER_MAP[limpia]) {
            found = TICKER_MAP[limpia];
            break;
        }
    }

    // --- CASO A: ESCÁNER DE MERCADO (LISTA) ---
    if (lower.includes('mejor') || lower.includes('recomienda') || lower.includes('top')) {
        createMessage('🕵️‍♂️ Escaneando el mercado con IA...', 'bot');
        try {
            const res = await fetch(`${API_URL}/market/scan`);
            const oportunidades = await res.json();
            
            let html = `
                <div style="margin-top:10px;">
                    <p><strong>📊 Top Oportunidades IA:</strong></p>
                    <table style="width:100%; border-collapse: separate; border-spacing: 0 5px; font-size:0.85em;">
            `;
            
            oportunidades.forEach(op => {
                let color = '#666'; 
                let icon = '⏸️';
                if (op.recomendacion.includes('COMPRAR')) { color = '#10b981'; icon = '🚀'; }
                else if (op.recomendacion.includes('VENDER')) { color = '#ef4444'; icon = '📉'; }
                
                html += `
                  <tr style="background:white; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                    <td style="padding:8px; font-weight:bold;">${op.ticker}</td>
                    <td style="padding:8px;">$${op.precio.toFixed(2)}</td>
                    <td style="padding:8px; color:${color}; font-weight:bold;">${icon} ${op.probabilidad}%</td>
                    <td style="padding:8px; text-align:right;">
                        <button onclick="document.getElementById('user-input').value='${op.ticker}'; document.getElementById('chat-form').requestSubmit();" 
                                style="background:${color}; border:none; color:white; border-radius:4px; cursor:pointer; padding:2px 6px;">Ver</button>
                    </td>
                  </tr>`;
            });

            html += `</table></div>`;
            createMessage(html, 'bot', true);
        } catch (err) {
            console.error(err);
            createMessage('❌ Error al escanear el mercado.', 'bot');
        }
        return; 
    }

    // --- CASO B: COMPARACIÓN (EJ: "BTC vs ETH") ---
    if (lower.includes('vs') || lower.includes('compar')) {
        // Intentar extraer las dos monedas
        // Buscamos todas las palabras que sean tickers válidos
        const palabras = lower.split(" ");
        const encontrados = [];
        
        for (let p of palabras) {
            const limpia = p.replace(/[^a-z0-9]/g, '');
            if (TICKER_MAP[limpia]) {
                // Guardamos el ticker normalizado (ej: 'btc')
                encontrados.push(TICKER_MAP[limpia]);
            }
        }
        
        // Eliminamos duplicados (ej: si escribe "btc vs bitcoin")
        const unicos = [...new Set(encontrados)];

        if (unicos.length < 2) {
            createMessage('⚠️ Para comparar, necesito dos monedas. Ej: "Bitcoin vs Ethereum"', 'bot');
            return;
        }

        const c1 = unicos[0];
        const c2 = unicos[1];
        
        createMessage(`⚖️ Comparando <strong>${c1.toUpperCase()}</strong> vs <strong>${c2.toUpperCase()}</strong>...`, 'bot', true);

        try {
            // 1. Pedir datos de ambas
            const [res1, res2, pred1, pred2] = await Promise.all([
                fetch(`${API_URL}/crypto/data?nombre=${c1}&n=180`),
                fetch(`${API_URL}/crypto/data?nombre=${c2}&n=180`),
                fetch(`${API_URL}/predict/${c1}`),
                fetch(`${API_URL}/predict/${c2}`)
            ]);

            const data1 = await res1.json();
            const data2 = await res2.json();
            const p1 = await pred1.json();
            const p2 = await pred2.json();

            // 2. Calcular Rendimiento % (quien ha subido más en 6 meses)
            const var1 = ((data1.stats.latest_close - data1.timeseries[0].close) / data1.timeseries[0].close) * 100;
            const var2 = ((data2.stats.latest_close - data2.timeseries[0].close) / data2.timeseries[0].close) * 100;
            
            const ganador = var1 > var2 ? c1 : c2;

            // 3. Generar HTML de Comparación
            const cid = 'comp-chart-' + Date.now();
            
            const html = `
              <div class="compare-card" style="background:white; padding:10px; border-radius:8px; border:1px solid #eee;">
                <div class="compare-title" style="text-align:center; font-weight:bold; color:#2a5bd7; margin-bottom:10px;">
                    ${data1.nombre} vs ${data2.nombre}
                </div>
                
                <table class="corr-table" style="width:100%; text-align:center; font-size:0.9em; border-collapse:collapse;">
                  <thead style="background:#f3f4f6;">
                    <tr>
                      <th style="padding:5px;">Métrica</th>
                      <th style="padding:5px;">${c1.toUpperCase()}</th>
                      <th style="padding:5px;">${c2.toUpperCase()}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style="padding:5px; font-weight:bold;">Precio</td>
                      <td>$${data1.stats.latest_close.toFixed(2)}</td>
                      <td>$${data2.stats.latest_close.toFixed(2)}</td> 
                    </tr>
                    <tr>
                      <td style="padding:5px; font-weight:bold;">IA</td>
                      <td style="color:${p1.recomendacion === 'COMPRAR' ? '#10b981' : '#ef4444'}; font-weight:bold;">${p1.recomendacion}</td>
                      <td style="color:${p2.recomendacion === 'COMPRAR' ? '#10b981' : '#ef4444'}; font-weight:bold;">${p2.recomendacion}</td>
                    </tr>
                    <tr>
                      <td style="padding:5px; font-weight:bold;">Rendimiento (6m)</td>
                      <td style="color:${var1 >= 0 ? '#10b981' : '#ef4444'}">${var1.toFixed(1)}%</td>
                      <td style="color:${var2 >= 0 ? '#10b981' : '#ef4444'}">${var2.toFixed(1)}%</td>
                    </tr>
                  </tbody>
                </table>

                <div class="compare-stats" style="text-align:center; margin-top:15px; padding:8px; background:#e0f2fe; border-radius:6px; color:#0369a1; font-weight:bold;">
                  🏆 Ganador Histórico: ${ganador.toUpperCase()}
                </div>

                <div style="position: relative; height:200px; width:100%; margin-top:10px;">
                    <canvas id="${cid}"></canvas>
                </div>
              </div>
            `;
            
            const wrapper = createMessage(html, 'bot', true);

            // 4. Gráfico Comparativo (Normalizado a %)
            setTimeout(() => {
                const canvas = document.getElementById(cid);
                if(canvas && window.Chart) {
                    const base1 = data1.timeseries[0].close;
                    const base2 = data2.timeseries[0].close;
                    // Convertimos precios a porcentaje de crecimiento desde el día 0
                    const chartData1 = data1.timeseries.map(t => ((t.close - base1)/base1)*100);
                    const chartData2 = data2.timeseries.map(t => ((t.close - base2)/base2)*100);

                    new Chart(canvas.getContext('2d'), {
                        type: 'line',
                        data: {
                            labels: data1.timeseries.map(t => t.date),
                            datasets: [
                                { label: c1.toUpperCase() + ' %', data: chartData1, borderColor: '#2a5bd7', borderWidth: 2, pointRadius: 0 },
                                { label: c2.toUpperCase() + ' %', data: chartData2, borderColor: '#ef4444', borderWidth: 2, pointRadius: 0 }
                            ]
                        },
                        options: { 
                            responsive: true, 
                            maintainAspectRatio: false,
                            plugins: { legend: { display: true, position:'bottom' } }, 
                            scales: { y: { ticks: { callback: v => v + '%' } } } 
                        }
                    });
                }
            }, 100);

        } catch (e) {
            console.error("Error en comparación:", e);
            createMessage('❌ Error al comparar. Revisa que ambas monedas existan.', 'bot');
        }
        return; // Salimos para no ejecutar el análisis individual
    }

    // --- CASO C: ANÁLISIS INDIVIDUAL ---
    if (found) {
      try {
        // 1. Pedir datos básicos
        const res = await fetch(`${API_URL}/crypto/data?nombre=${found}&n=180`);
        const data = await res.json();
        
        if (data.error) {
          createMessage(`❌ Error: ${data.error}`, 'bot');
          return;
        }

        // 2. Pedir PREDICCIÓN IA
        let predictionHtml = '';
        try {
            const predRes = await fetch(`${API_URL}/predict/${found}`);
            if (predRes.ok) {
                const predData = await predRes.json();
                
                let color = '#f59e0b';
                let icono = '⚖️';
                if (predData.recomendacion === 'COMPRAR') { color = '#10b981'; icono = '🚀'; }
                if (predData.recomendacion === 'VENDER') { color = '#ef4444'; icono = '📉'; }

                let razonesHtml = "";
                if (predData.detalles && predData.detalles.razones) {
                    razonesHtml = `<ul style="margin:5px 0; padding-left:15px; font-size:0.85em; color:#555;">` + 
                                  predData.detalles.razones.map(r => `<li>${r}</li>`).join("") + 
                                  `</ul>`;
                }

                const sl = predData.detalles?.stop_loss ? predData.detalles.stop_loss.toFixed(2) : '?';
                const tp = predData.detalles?.take_profit ? predData.detalles.take_profit.toFixed(2) : '?';
                const prob = predData.probabilidad_valor || 50;

                predictionHtml = `
                    <div style="margin: 15px 0; padding: 12px; background: white; border-left: 4px solid ${color}; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="margin:0; color:${color}; font-size:1.1em;">${icono} ${predData.recomendacion}</h3>
                            <span style="font-weight:bold; font-size:0.9em; color:#666;">IA: ${prob}%</span>
                        </div>
                        <div style="margin-top:8px;">
                            <strong>🧠 Análisis Técnico:</strong>
                            ${razonesHtml}
                        </div>
                        <div style="margin-top:10px; padding:8px; background:#f8fafc; border-radius:6px; display:flex; justify-content:space-between; font-size:0.9em;">
                            <span style="color:#ef4444;">🛑 Stop: <strong>$${sl}</strong></span>
                            <span style="color:#10b981;">🎯 Meta: <strong>$${tp}</strong></span>
                        </div>
                    </div>`;
            }
        } catch (e) { console.log("IA no disponible", e); }

        // --- PREPARAR DATOS DEL GRÁFICO ---
        const prices = data.timeseries.map(t => t.close);
        const dates = data.timeseries.map(t => t.date);
        
        // Color del gráfico basado en si subió o bajó hoy
        const priceToday = prices[prices.length - 1];
        const priceYesterday = prices[prices.length - 2];
        const change24h = ((priceToday - priceYesterday) / priceYesterday) * 100;
        const color24h = change24h >= 0 ? '#10b981' : '#ef4444';
        const symbol24h = change24h >= 0 ? '▲' : '▼';
        
        // ID único para el gráfico
        const cid = 'chart-' + Date.now() + Math.random().toString(36).substr(2, 5);

        // Panel de Compra
        const inputId = 'qty-' + Date.now();
        const tradingPanel = `
          <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;">
            <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                <input type="number" id="${inputId}" value="0.5" step="any" style="flex:1; padding: 8px; border: 1px solid #ccc; border-radius: 5px; text-align: center;">
                <div onclick="ejecutarCompra('${found}', document.getElementById('${inputId}').value)" 
                     class="cool-btn" style="flex:1; background:#10b981; cursor:pointer; text-align:center; padding:8px; border-radius:5px; color:white;">COMPRAR</div>
                <div onclick="ejecutarVenta('${found}', document.getElementById('${inputId}').value)" 
                     class="cool-btn" style="flex:1; background:#ef4444; cursor:pointer; text-align:center; padding:8px; border-radius:5px; color:white;">VENDER</div>
            </div>
          </div>
        `;

        // HTML FINAL
        const html = `
          <div class="crypto-info">
            <div class="crypto-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div>
                    <div class="crypto-title" style="margin:0; font-size:1.4em; font-weight:bold;">${data.nombre}</div>
                    <div style="font-size:0.9em; color:#666;">${data.descripcion || 'Criptoactivo'}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.2em; font-weight:bold;">$${priceToday.toFixed(2)}</div>
                    <div style="font-size:0.9em; font-weight:bold; color:${color24h};">
                        ${symbol24h} ${change24h.toFixed(2)}%
                    </div>
                </div>
            </div>

            <div class="crypto-stats" style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:5px; margin-bottom:10px; background:#f9fafb; padding:8px; border-radius:8px; text-align:center; font-size:0.85em;">
              <div>
                <small style="color:#888;">Mínimo</small><br><strong>$${data.stats.min_close?.toFixed(2)}</strong>
              </div>
              <div>
                <small style="color:#888;">Media</small><br><strong>$${data.stats.mean_close?.toFixed(2)}</strong>
              </div>
              <div>
                <small style="color:#888;">Máximo</small><br><strong>$${data.stats.max_close?.toFixed(2)}</strong>
              </div>
            </div>
            
            ${predictionHtml}
            
            <div style="position: relative; height:200px; width:100%;">
                <canvas id="${cid}"></canvas>
            </div>
            
            ${tradingPanel} 
          </div>`;

        // Renderizar mensaje
        const wrapper = createMessage(html, 'bot', true);

        // --- LÓGICA DE GRÁFICO (RESTAURADA) ---
        setTimeout(() => {
            const canvas = document.getElementById(cid);
            if (canvas && window.Chart) {
                // Color de la línea según la tendencia GLOBAL (Inicio vs Final)
                // Igual que en tu versión original
                const tendenciaGlobal = prices[prices.length - 1] >= prices[0] ? '#10b981' : '#ef4444';

                new Chart(canvas.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: dates,
                        datasets: [{
                            label: 'Precio',
                            data: prices,
                            borderColor: tendenciaGlobal, // Color original
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.1,
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: { 
                            x: { display: false },
                            y: { display: true } // Eje Y visible para ver precios
                        }
                    }
                });
            } else {
                console.error("No se pudo cargar el gráfico (Canvas o Chart.js faltante)");
            }
        }, 100); // Pequeño retraso para asegurar que el HTML existe

      } catch (err) {
        console.error(err);
        createMessage('Error obteniendo datos.', 'bot');
      }
      return;
    }

    createMessage('No te he entendido. Prueba con "Solana", "BTC" o "Recomienda".', 'bot');
  });
});