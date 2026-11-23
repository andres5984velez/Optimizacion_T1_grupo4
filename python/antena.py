"""
antena.py -- Modelo del problema T1: sintesis de un arreglo lineal de antenas.

Arreglo lineal broadside de 2M = 20 elementos isotropicos, d = lambda/2 (k*d = pi),
fase uniforme y excitaciones simetricas respecto al centro (a_n, n = 1..10; a_1 en los
elementos centrales):

    AF(theta) = 2 * sum_{n=1}^{10} a_n * cos[ (n - 1/2) * k*d * cos(theta) ]

Patron evaluado en theta in [0, 180] grados con paso 0.05 (3601 puntos), normalizado a su
maximo (dB).  El lobulo principal (centrado en 90 grados) va desde 90 hasta el primer minimo
local; FNBW = 2*(theta_nulo - 90).  SLL = maximo del patron normalizado fuera del lobulo
principal.

    f(a) = SLL(a)[dB] + 100 * max(0, FNBW(a) - 17)

Notas de implementacion
  * Como a_n >= 0, el maximo del patron esta en 90 grados y AF(180 - theta) = AF(theta)
    (el coseno es par), asi que basta analizar theta in [90, 180]; el nulo del otro lado
    es simetrico.
  * Primer minimo local = primera muestra donde |AF| deja de decrecer (diferencia > 0).
    Si no existe (p. ej. un solo par de elementos), el lobulo ocupa todo el rango:
    FNBW = 180 y el "lobulo lateral" se toma como la ultima muestra.
  * Si todas las amplitudes son ~0 el patron no se puede normalizar: se devuelve una
    penalizacion fija de 1e3 (decision propia; el enunciado no la define).
"""
import numpy as np

_trapecio = getattr(np, "trapezoid", None) or np.trapz   # compatible con NumPy 1.x y 2.x

N_ELEM = 10                 # variables a_1..a_10
KD = np.pi                  # k*d con d = lambda/2
PASO_OPT = 0.05             # paso angular de la optimizacion (grados)
FNBW_MAX = 17.0             # restriccion de ancho de haz (grados)
PENAL_FNBW = 100.0          # penalizacion estatica por grado excedido
PENAL_NULO = 1e3            # f cuando el patron es identicamente cero
LB = np.zeros(N_ELEM)
UB = np.ones(N_ELEM)

_CACHE = {}


def malla(paso=PASO_OPT):
    """(theta en grados, M) con M[k, n-1] = cos((n-1/2)*KD*cos(theta_k)). Se cachea por paso."""
    clave = round(paso, 6)
    if clave not in _CACHE:
        n = int(round(180.0 / paso)) + 1
        th = np.linspace(0.0, 180.0, n)
        k = np.arange(1, N_ELEM + 1)
        M = np.cos((k - 0.5) * KD * np.cos(np.radians(th))[:, None])
        _CACHE[clave] = (th, M)
    return _CACHE[clave]


def factor_arreglo(a, paso=PASO_OPT):
    """|AF(theta)| (sin normalizar) sobre la malla; devuelve (theta, |AF|)."""
    th, M = malla(paso)
    return th, np.abs(2.0 * (M @ np.asarray(a, float)))


def patron_db(a, paso=PASO_OPT, piso=-100.0):
    """Patron normalizado en dB (con piso para evitar -inf).  Devuelve (theta, dB)."""
    th, af = factor_arreglo(a, paso)
    pico = af.max()
    if pico < 1e-12:
        return th, np.full_like(th, piso)
    return th, np.maximum(20.0 * np.log10(np.maximum(af / pico, 1e-12)), piso)


def metricas_af(af, paso):
    """(SLL en dB, FNBW en grados) a partir de |AF| muestreado en theta = 0..180 con `paso` grados
    (sirve tambien para patrones que vienen de otra herramienta, p. ej. NEC2 en la validacion V4)."""
    pico = af.max()
    if pico < 1e-12:
        return np.nan, np.nan
    i90 = (len(af) - 1) // 2
    mitad = af[i90:]                          # theta = 90 .. 180
    pos = np.flatnonzero(np.diff(mitad) > 0)  # primera muestra donde |AF| vuelve a subir
    i_nulo = int(pos[0]) if pos.size else len(mitad) - 1
    fnbw = 2.0 * i_nulo * paso
    sll = 20.0 * np.log10(max(mitad[i_nulo:].max() / pico, 1e-12))
    return sll, fnbw


def metricas(a, paso=PASO_OPT):
    """Devuelve (SLL en dB, FNBW en grados) del arreglo con amplitudes a."""
    return metricas_af(factor_arreglo(a, paso)[1], paso)


def f_antena(a, paso=PASO_OPT):
    """Funcion objetivo del problema: SLL + 100*max(0, FNBW - 17).  a: vector de 10 reales."""
    sll, fnbw = metricas(a, paso)
    if np.isnan(sll):
        return PENAL_NULO
    return float(sll + PENAL_FNBW * max(0.0, fnbw - FNBW_MAX))


def ancho_haz_3db(a, paso=0.005):
    """Ancho de haz a -3 dB (grados): cruces del patron en dB con -3.0103 dB alrededor de 90.
    Usa interpolacion lineal entre muestras."""
    th, db = patron_db(a, paso)
    i90 = (len(th) - 1) // 2
    nivel = -10.0 * np.log10(2.0)             # -3.0103 dB (potencia mitad)
    lado = db[i90:]
    j = int(np.flatnonzero(lado < nivel)[0])  # primera muestra por debajo del nivel
    t0, t1 = th[i90 + j - 1], th[i90 + j]
    y0, y1 = lado[j - 1], lado[j]
    t_cruce = t0 + (nivel - y0) * (t1 - t0) / (y1 - y0)
    return 2.0 * (t_cruce - 90.0)


def directividad(a, paso=0.005):
    """D = 2 / integral_0^pi |AF_norm|^2 sen(theta) dtheta (integracion trapezoidal)."""
    th, af = factor_arreglo(a, paso)
    af = af / af.max()
    t = np.radians(th)
    return float(2.0 / _trapecio(af ** 2 * np.sin(t), t))


class Objetivo:
    """Envuelve a f con un CONTADOR de evaluaciones (como exige la guia, seccion 3.1).

    Guarda ademas el mejor valor encontrado tras cada evaluacion (historial de longitud
    `presupuesto`), que es la curva de convergencia frente al numero de evaluaciones.
    Lanza un error si se intenta exceder el presupuesto.
    """

    def __init__(self, presupuesto=3000, f=f_antena):
        self.f = f
        self.presupuesto = presupuesto
        self.n = 0
        self.mejor_f = np.inf
        self.mejor_x = None
        self.historial = np.full(presupuesto, np.nan)

    def __call__(self, x):
        if self.n >= self.presupuesto:
            raise RuntimeError("Se excedio el presupuesto de evaluaciones")
        x = np.asarray(x, float)
        val = self.f(x)
        if val < self.mejor_f:
            self.mejor_f, self.mejor_x = val, x.copy()
        self.historial[self.n] = self.mejor_f
        self.n += 1
        return val
