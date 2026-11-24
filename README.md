# Trabajo aplicado T1 – Síntesis de un arreglo lineal de antenas con WOA

**Asignatura:** Optimización · **Autor:** Octavio Bedoya Vélez · **Entrega:** 25-nov-2026
**Algoritmo:** Whale Optimization Algorithm (WOA) · **D = 10** · **Presupuesto:** 3 000 evaluaciones por corrida

Minimiza el nivel de lóbulos laterales (SLL) de un arreglo lineal *broadside* de 20 elementos isotrópicos
(d = λ/2) ajustando 10 amplitudes de excitación simétricas, con la restricción de ancho de haz FNBW ≤ 17°:

`f(a) = SLL(a) [dB] + 100 · max(0, FNBW(a) − 17°)`, con `a_n ∈ [0, 1]`, n = 1…10.

El informe completo está en [`informe/Informe_T1.pdf`](informe/Informe_T1.pdf) y la presentación en [`presentacion/Presentacion_T1_WOA.pptx`](presentacion/Presentacion_T1_WOA.pptx) ([PDF](presentacion/Presentacion_T1_WOA.pdf)).

---

## Estructura del repositorio

```
optimizacion_T1_grupo4/
├── README.md
├── informe/
│   └── Informe_T1.pdf               # informe final
├── presentacion/
│   ├── Presentacion_T1_WOA.pptx     # diapositivas editables (con notas del orador)
│   └── Presentacion_T1_WOA.pdf      # versión lista para proyectar
└── python/
    ├── antena.py               # modelo y función objetivo (con contador de evaluaciones)
    ├── woa.py                  # algoritmo WOA
    ├── busqueda_aleatoria.py   # línea base
    ├── protocolo.py            # protocolo experimental (30 corridas, Wilcoxon, sensibilidad)
    ├── validacion.py           # validación V1, V2 y V3
    ├── verificar_modelo.py     # 10 comprobaciones del modelo
    ├── v4_nec.py               # V4 (opcional): dipolos reales con NEC2
    ├── main.py                 # ejecuta todo (excepto V4)
    ├── requirements.txt
    └── resultados/             # NO se versiona: se genera al ejecutar main.py
```

## Código fuente: qué hace cada archivo

| Componente que pide la guía | Archivo | Contenido |
|---|---|---|
| **Función objetivo** | `antena.py` | Factor de arreglo, SLL, FNBW, `f_antena`, ancho a −3 dB, directividad y la clase `Objetivo`, que **cuenta las evaluaciones** y guarda el historial de convergencia. |
| **Algoritmo** | `woa.py` | WOA (Mirjalili y Lewis, 2016) con parámetro `b` y opción `vectorial` (A y C vectoriales o escalares). |
| **Línea base** | `busqueda_aleatoria.py` | Búsqueda aleatoria pura: 3 000 puntos uniformes en la caja. |
| **Protocolo experimental** | `protocolo.py` | 30 corridas, tabla estadística, convergencia ±1σ, Wilcoxon, sensibilidad de `b` (5 niveles × 30) y variante escalar/vectorial. |
| **Validación** | `validacion.py` | V1 (Dolph–Chebyshev y uniforme), V2 (malla 0,005°, −3 dB, directividad), V3 (5 bits, Monte Carlo ±5 %, fallas). |
| **Verificación del modelo** | `verificar_modelo.py` | 10 comprobaciones contra los valores de la guía. Si falla alguna, `main.py` se detiene. |
| **V4 (opcional)** | `v4_nec.py` | Genera los `.nec` de 20 dipolos y los simula con `nec2c`. |
| **Orquestador** | `main.py` | Ejecuta la verificación, el protocolo y la validación V1–V3 en ese orden. |

---

## Cómo reproducir todos los resultados

### Requisitos
- Python ≥ 3.10
- `numpy`, `scipy`, `matplotlib` (versiones mínimas en `python/requirements.txt`)

### Pasos
```bash
cd python
pip install -r requirements.txt
python main.py
```

`main.py` ejecuta, en este orden:

1. `verificar_modelo.py`: comprobaciones del modelo.
2. `protocolo.py`: 30 corridas de WOA y de búsqueda aleatoria, Wilcoxon, sensibilidad de `b` y variante escalar. Crea `resultados/` y guarda `mejor_solucion_WOA.json`.
3. `validacion.py`: V1, V2 y V3 con la mejor solución del paso 2. Escribe en `resultados/validacion/`.

La ejecución completa tarda unos minutos. La carpeta `resultados/` **no está en el repositorio**: se crea sola
al ejecutar `main.py`.

> Ejecute los comandos **desde la carpeta `python/`**. Si ve `Could not open requirements file`, está en la carpeta equivocada.

### Semillas (reproducibilidad)

| Qué | Semilla | Dónde |
|---|---|---|
| 30 corridas de WOA | 1, 2, …, 30 (la corrida *i* usa la semilla *i*) | `protocolo.py` (`SEMILLAS = range(1, 31)`) |
| 30 corridas de búsqueda aleatoria | 1, 2, …, 30 (las mismas, para el Wilcoxon pareado) | `protocolo.py` |
| Sensibilidad de `b` (5 niveles × 30 corridas) | 1, 2, …, 30 en cada nivel | `protocolo.py` |
| Monte Carlo de ±5 % (200 realizaciones) | 12345 | `validacion.py` (`SEMILLA_MC`) |

Los generadores usan `numpy.random.default_rng(semilla)`, por lo que los resultados se repiten exactamente en una misma
versión de NumPy. Con otra versión pueden cambiar los decimales: use siempre los valores de **su** ejecución.

### V4 (opcional): dipolos reales con NEC2
Ejecute V4 **después** de `main.py`, porque lee `resultados/mejor_solucion_WOA.json`:
```bash
cd python
python v4_nec.py
```
- Siempre genera los `.nec` en `resultados/V4/`; se pueden abrir en 4NEC2 (File → Open).
- Solo los simula si `nec2c` está instalado.
- `nec2c` no tiene versión oficial para Windows. En Windows se puede usar **WSL (Ubuntu)**:
  ```bash
  wsl -d Ubuntu
  sudo apt update && sudo apt install -y nec2c python3-numpy python3-scipy python3-matplotlib
  cd "/mnt/c/<ruta>/optimizacion_T1_grupo4/python"
  python3 v4_nec.py
  ```
- `nec2c` limita el largo de los nombres de archivo, por eso el script lo ejecuta dentro de la carpeta con nombres relativos
  (funciona aunque la ruta del proyecto sea larga).

---

## Dónde está cada resultado

Todo se escribe en `python/resultados/` (ver las secciones del informe):

| Resultado | Archivos | Sección del informe |
|---|---|---|
| Tabla estadística (mejor, mediana, media, desv. est., peor, tiempo) | `tabla_estadistica.md/.csv` | 4.1 |
| Curva de convergencia (media ±1σ) | `fig_convergencia.png`, `curvas_*.csv` | 4.2 |
| Wilcoxon frente a búsqueda aleatoria | `wilcoxon.txt` | 4.3 |
| Sensibilidad de `b` | `tabla_sensibilidad_b.md`, `fig_sensibilidad_boxplot.png`, `fig_sensibilidad_convergencia.png`, `kruskal_sensibilidad.txt` | 4.4 |
| Variante A y C escalar / vectorial | `tabla_variante_A_C.md`, `variante_A_C.txt` | 4.5 |
| Mejor solución | `mejor_solucion_WOA.json`, `soluciones_WOA.csv` | 5 |
| V1: Dolph–Chebyshev y uniforme | `validacion/V1_*`, `fig_V1_*` | 5 (V1) |
| V2: malla fina, −3 dB, directividad | `validacion/V2_*`, `fig_V2_primer_nulo.png` | 5 (V2) |
| V3a: cuantización de 5 bits | `validacion/V3a_*` | 5 (V3a) |
| V3b: Monte Carlo ±5 % | `validacion/V3b_*`, `fig_V3b_montecarlo.png` | 5 (V3b) |
| V3c: falla de un elemento | `validacion/V3c_*`, `fig_V3c_fallas.png` | 5 (V3c) |
| V4: acoplamiento mutuo | `V4/V4_tabla.md`, `V4/fig_V4_acoplamiento.png`, `V4/*.nec`, `V4/*.out` | 5 (V4) |

---

## Decisiones de implementación

1. **Presupuesto = 3 000 evaluaciones totales, contando la inicialización** (contador dentro de la función objetivo): 30 ballenas + 99 iteraciones × 30. La búsqueda aleatoria usa 3 000 puntos.
2. **A y C son vectores** (una componente aleatoria por dimensión), como dice el artículo; |A| < 1 se evalúa componente a componente. p y l son escalares por ballena. La variante escalar converge prematuramente (ver `tabla_variante_A_C.md`).
3. **X\*** es la mejor solución encontrada hasta el momento; actualización síncrona; las soluciones se recortan a [0, 1].
4. **Primer nulo** = primera muestra donde |AF| deja de decrecer. Se analiza θ ∈ [90°, 180°] porque el patrón es simétrico. Si todas las a_n = 0, f = 10³ (decisión propia).
5. **Estadística:** desviación estándar muestral (N − 1). Wilcoxon de rangos con signo **pareado por semilla**, unilateral (H₁: WOA < azar), α = 0,05.
6. **V3a:** 32 niveles k/31 tras normalizar al máximo. **V3b:** ε ~ U(−5 %, +5 %) independiente por a_n, 200 realizaciones. **V3c:** a_n = 0 (cada a_n corresponde a **un par** de elementos simétricos).
7. **V2:** directividad por integración trapezoidal con malla de 0,005°; ancho a −3 dB como el cruce de −3,0103 dB con interpolación lineal.
8. **V4:** f = 299,792458 MHz (λ = 1 m), 20 dipolos de 0,47 m a lo largo de x, apilados en z con d = 0,5 m, fuentes de tensión con amplitud a_n, corte φ = 90° (plano H).

## Resultados principales

| Indicador | Valor |
|---|---|
| WOA (b = 1): mejor / mediana | −29,25 dB / −27,98 dB |
| Búsqueda aleatoria: mejor / mediana | −23,90 dB / −19,03 dB |
| Wilcoxon (WOA < azar) | p ≈ 9,3×10⁻¹⁰ (WOA gana 30/30) |
| Óptimo teórico (Dolph–Chebyshev) | −30,00 dB |
| Sensibilidad a `b` | no significativa (Kruskal–Wallis p = 0,40) |
| V4: degradación por acoplamiento mutuo | +0,80 dB (WOA) |

## Referencias
- Mirjalili, S. y Lewis, A. (2016). The whale optimization algorithm. *Advances in Engineering Software, 95*, 51–67.
- Dolph, C. L. (1946). A current distribution for broadside arrays which optimizes the relationship between beam width and side-lobe level. *Proceedings of the IRE, 34*(6), 335–348.
- Balanis, C. A. (2016). *Antenna Theory: Analysis and Design* (4.ª ed.). Wiley.
