"""
woa.py -- Whale Optimization Algorithm (Mirjalili & Lewis, 2016), implementado desde el articulo.

Ecuaciones (nombres segun el articulo; verifique la numeracion con su PDF):
    A = 2*a*r - a ;  C = 2*r2 ;  a decrece linealmente de 2 a 0   (r, r2: vectores aleatorios en [0,1]^D)
    Cerco (p < 0.5 y |A| < 1):     D = |C*X* - X| ;  X(t+1) = X* - A*D
    Busqueda (p < 0.5 y |A| >= 1): D = |C*X_rand - X| ; X(t+1) = X_rand - A*D
    Espiral (p >= 0.5):            D' = |X* - X| ;  X(t+1) = D' * exp(b*l) * cos(2*pi*l) + X*
    p ~ U(0,1) (50 % cerco/busqueda, 50 % espiral); l ~ U(-1,1); b = constante de la espiral.

Decisiones de implementacion (documentarlas en el informe):
  * A y C son VECTORES (una componente aleatoria por dimension), como dice el articulo ("r es un
    vector aleatorio en [0,1]"); el criterio |A| < 1 se evalua componente a componente.
    La opcion vectorial=False usa un escalar por ballena (variante frecuente en la practica);
    en este problema converge prematuramente (ver experimento adicional en protocolo.py).
  * p y l son escalares: uno por ballena e iteracion.
  * X* es la mejor solucion encontrada hasta el momento.
  * Actualizacion sincrona: todas las ballenas se mueven con la poblacion de la iteracion anterior;
    X_rand es una ballena elegida al azar de esa poblacion.
  * Presupuesto TOTAL de evaluaciones = 3000, contando la inicializacion:
    n_pob + n_iter*n_pob = 3000  ->  n_pob = 30, n_iter = 99.
  * Las ballenas que salen de [lb, ub] se recortan a los limites.
"""
import numpy as np


def woa(f, lb, ub, semilla, n_pob=30, presupuesto=3000, b=1.0, vectorial=True):
    """f: funcion objetivo con contador (recibe un vector, devuelve un escalar).
    Devuelve (mejor_x, mejor_f)."""
    rng = np.random.default_rng(semilla)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    n_iter = (presupuesto - n_pob) // n_pob
    forma = (n_pob, D) if vectorial else (n_pob, 1)        # A y C: vectores o escalares por ballena

    X = lb + rng.random((n_pob, D)) * (ub - lb)
    fX = np.array([f(x) for x in X])                       # n_pob evaluaciones (inicializacion)
    i = int(np.argmin(fX))
    X_best, f_best = X[i].copy(), float(fX[i])

    for t in range(n_iter):
        a = 2.0 * (1.0 - t / n_iter)                       # a: 2 -> 0 (linealmente)
        A = 2.0 * a * rng.random(forma) - a
        C = 2.0 * rng.random(forma)
        p = rng.random(n_pob)[:, None]
        l = rng.uniform(-1.0, 1.0, n_pob)[:, None]
        X_rand = X[rng.integers(0, n_pob, n_pob)]          # ballena aleatoria de la poblacion

        busca = (p < 0.5) & (np.abs(A) >= 1.0)             # exploracion global (|A| >= 1)
        X_ref = np.where(busca, X_rand, X_best)            # ballena aleatoria o mejor solucion
        X_cerco = X_ref - A * np.abs(C * X_ref - X)        # cerco / busqueda de presa

        X_espiral = np.abs(X_best - X) * np.exp(b * l) * np.cos(2.0 * np.pi * l) + X_best

        X = np.clip(np.where(p < 0.5, X_cerco, X_espiral), lb, ub)
        fX = np.array([f(x) for x in X])                   # n_pob evaluaciones
        j = int(np.argmin(fX))
        if fX[j] < f_best:
            X_best, f_best = X[j].copy(), float(fX[j])
    return X_best, f_best
