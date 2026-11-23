"""
busqueda_aleatoria.py -- Linea base: busqueda aleatoria pura (seccion 3.2 de la guia).
3000 puntos uniformes en la caja [lb, ub]; se guarda el mejor.
"""
import numpy as np


def busqueda_aleatoria(f, lb, ub, semilla, presupuesto=3000):
    rng = np.random.default_rng(semilla)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    P = lb + rng.random((presupuesto, lb.size)) * (ub - lb)
    mejor_x, mejor_f = None, np.inf
    for x in P:
        v = f(x)
        if v < mejor_f:
            mejor_x, mejor_f = x.copy(), v
    return mejor_x, float(mejor_f)
