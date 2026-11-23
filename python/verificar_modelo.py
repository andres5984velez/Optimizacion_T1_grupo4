"""
verificar_modelo.py -- Comprobaciones del modelo contra los valores que da la propia guia
(V1) y contra propiedades matematicas. Se ejecuta al inicio de main.py.
"""
import warnings
import numpy as np
from scipy.signal.windows import chebwin

from antena import (metricas, f_antena, directividad, factor_arreglo, malla, Objetivo, LB, UB)
from woa import woa


def amplitudes_chebyshev(sll_db=30.0, n_total=20):
    """Dolph-Chebyshev: a_1..a_10 (centro -> borde), normalizadas a su maximo.
    scipy.signal.windows.chebwin(20, 30), como sugiere la guia (Balanis da el mismo resultado)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # aviso irrelevante de scipy para atenuaciones < 45 dB
        w = chebwin(n_total, sll_db)
    a = w[n_total // 2:]
    return a / a.max()


def verificar():
    ok = True

    def chequeo(nombre, condicion, detalle=""):
        nonlocal ok
        print(f"  [{'OK' if condicion else 'FALLA'}] {nombre} {detalle}")
        ok &= bool(condicion)

    print("Verificacion del modelo:")
    sll_u, fnbw_u = metricas(np.ones(10))
    chequeo("Uniforme: SLL = -13.2 dB (guia)", abs(sll_u + 13.2) < 0.05, f"-> {sll_u:.3f} dB")
    chequeo("Uniforme: FNBW = 11.5 grados (guia)", abs(fnbw_u - 11.5) < 0.06, f"-> {fnbw_u:.2f}")
    a_ch = amplitudes_chebyshev()
    sll_c, fnbw_c = metricas(a_ch)
    chequeo("Chebyshev 30 dB: SLL = -30 dB (guia)", abs(sll_c + 30) < 0.01, f"-> {sll_c:.3f} dB")
    chequeo("Chebyshev 30 dB: FNBW = 17.0 grados (guia)", abs(fnbw_c - 17.0) < 0.06, f"-> {fnbw_c:.2f}")
    chequeo("f(Chebyshev) = -30 (sin penalizacion)", abs(f_antena(a_ch) + 30) < 0.01, f"-> {f_antena(a_ch):.3f}")
    chequeo("Directividad del uniforme = 20 (N elementos)", abs(directividad(np.ones(10)) - 20) < 0.01,
            f"-> {directividad(np.ones(10)):.3f}")
    th, af = factor_arreglo(np.random.default_rng(0).random(10))
    chequeo("Simetria AF(theta) = AF(180 - theta)", np.allclose(af, af[::-1], atol=1e-9))
    chequeo("Escala invariante: f(a) = f(3a)", abs(f_antena(np.arange(1, 11) / 10) - f_antena(3 * np.arange(1, 11) / 10)) < 1e-9)
    obj = Objetivo(3000); woa(obj, LB, UB, 1)
    chequeo("Contador: WOA usa exactamente 3000 evaluaciones", obj.n == 3000, f"-> {obj.n}")
    o1 = Objetivo(3000); o2 = Objetivo(3000); woa(o1, LB, UB, 7); woa(o2, LB, UB, 7)
    chequeo("Reproducibilidad: misma semilla, mismo resultado", o1.mejor_f == o2.mejor_f)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verificar() else 1)
