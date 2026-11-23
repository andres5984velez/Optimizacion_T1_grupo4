# Trabajo aplicado T1 – Síntesis de un arreglo lineal de antenas (WOA)

Grupo: Octavio Bedoya Vélez y Jonathan Constaín Guerrero · Asignatura Optimización · Entrega: 25-nov-2026
Algoritmo: Whale Optimization Algorithm (WOA) · D = 10 · Presupuesto: 3 000 evaluaciones por corrida

## Cómo reproducir TODO (semillas 1–30; Monte Carlo con semilla 12345)
```
cd python
pip install -r requirements.txt        # numpy, scipy, matplotlib  (Python >= 3.10)
python main.py                         # verificación del modelo + protocolo (sec. 3) + validación V1-V3  (~30 s)
python v4_nec.py                       # OPCIONAL (V4): archivos .nec + simulación NEC2 si hay `nec2c`
```
Todo queda en `python/resultados/`. Los resultados que vienen en el zip son la salida de referencia;
si en su máquina cambian los decimales (otra versión de NumPy), usen SIEMPRE los suyos en el informe.

## Archivos
| Archivo | Qué hace |
|---|---|
| `antena.py` | Modelo: factor de arreglo, SLL, FNBW, función objetivo, ancho a −3 dB, directividad, **contador de evaluaciones** (`Objetivo`) |
| `woa.py` | WOA desde el artículo (parámetro `b`, opción `vectorial`) |
| `busqueda_aleatoria.py` | Línea base (3 000 puntos uniformes) |
| `protocolo.py` | 30 corridas, tabla estadística, convergencia ±1σ, Wilcoxon, sensibilidad de `b` (5 niveles × 30), variante escalar/vectorial |
| `validacion.py` | V1 (Dolph–Chebyshev y uniforme), V2 (malla 0.005°, −3 dB, directividad), V3 (5 bits, Monte Carlo ±5 %, fallas) |
| `verificar_modelo.py` | 10 comprobaciones del modelo contra los valores de la propia guía |
| `v4_nec.py` | (Opcional) dipolos reales con acoplamiento mutuo (NEC2 / 4NEC2) |
| `main.py` | Ejecuta todo lo anterior (excepto V4) |
