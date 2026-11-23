"""
protocolo.py -- Protocolo experimental comun (secciones 3.2 y 3.3 de la guia).

  * 30 corridas de WOA (semillas 1..30), presupuesto 3000 evaluaciones contadas en la funcion objetivo
  * linea base: busqueda aleatoria pura (3000 puntos, 30 corridas, mismas semillas)
  * tabla estadistica (mejor, mediana, media, desv. est., peor, tiempo medio por corrida)
  * curva de convergencia media +-1 desv. est. frente al numero de evaluaciones
  * prueba de rangos con signo de Wilcoxon (alpha = 0.05), emparejando corridas por semilla
  * sensibilidad del parametro b de la espiral: 0.5, 1.0, 1.5, 2.0, 3.0 (30 corridas por nivel)
"""
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from antena import Objetivo, LB, UB
from woa import woa
from busqueda_aleatoria import busqueda_aleatoria

SEMILLAS = list(range(1, 31))
PRESUPUESTO = 3000
N_POB = 30
B_BASE = 1.0
NIVELES_B = [0.5, 1.0, 1.5, 2.0, 3.0]
ALFA = 0.05


def correr_30(algoritmo, **kw):
    """30 corridas independientes. Devuelve dict con finales, historiales (30x3000), x y tiempos."""
    finales, hist, xs, tiempos = [], [], [], []
    for s in SEMILLAS:
        obj = Objetivo(PRESUPUESTO)
        t0 = time.perf_counter()
        x, fx = algoritmo(obj, LB, UB, s, **kw)
        tiempos.append(time.perf_counter() - t0)
        assert obj.n == PRESUPUESTO, "el presupuesto no se uso exactamente"
        assert abs(obj.historial[-1] - fx) < 1e-12
        finales.append(fx); hist.append(obj.historial.copy()); xs.append(x)
    return dict(finales=np.array(finales), hist=np.array(hist), xs=np.array(xs), tiempos=np.array(tiempos))


def fila(nombre, r):
    f = r["finales"]
    return dict(Algoritmo=nombre, Mejor=f.min(), Mediana=np.median(f), Media=f.mean(),
                Desv_est=f.std(ddof=1), Peor=f.max(), Tiempo_medio_s=r["tiempos"].mean())


def guardar_tabla(dir_res, base, filas, cols):
    cab = [c for c, _ in cols]
    txt = [[fmt.format(fl[c]) for c, fmt in cols] for fl in filas]
    with open(os.path.join(dir_res, base + ".csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(cab) + "\n" + "\n".join(",".join(t) for t in txt) + "\n")
    with open(os.path.join(dir_res, base + ".md"), "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(cab) + " |\n|" + "---|" * len(cab) + "\n")
        fh.write("\n".join("| " + " | ".join(t) + " |" for t in txt) + "\n")
    print("\n" + " | ".join(cab))
    for t in txt:
        print(" | ".join(t))


COLS = [("Algoritmo", "{}"), ("Mejor", "{:.4f}"), ("Mediana", "{:.4f}"), ("Media", "{:.4f}"),
        ("Desv_est", "{:.4f}"), ("Peor", "{:.4f}"), ("Tiempo_medio_s", "{:.3f}")]


def banda(ax, hist, etiqueta, color):
    ev = np.arange(1, hist.shape[1] + 1)
    m, sd = hist.mean(0), hist.std(0, ddof=1)
    ax.plot(ev, m, color=color, lw=2, label=etiqueta)
    ax.fill_between(ev, m - sd, m + sd, color=color, alpha=0.2)


def run(dir_res):
    os.makedirs(dir_res, exist_ok=True)

    # ------------------------------------------------ 30 corridas de WOA y de la linea base
    woa_r = correr_30(woa, n_pob=N_POB, presupuesto=PRESUPUESTO, b=B_BASE)
    rs_r = correr_30(busqueda_aleatoria, presupuesto=PRESUPUESTO)
    np.savetxt(os.path.join(dir_res, "curvas_WOA.csv"), woa_r["hist"], delimiter=",", fmt="%.8g")
    np.savetxt(os.path.join(dir_res, "curvas_BusquedaAleatoria.csv"), rs_r["hist"], delimiter=",", fmt="%.8g")
    np.savetxt(os.path.join(dir_res, "soluciones_WOA.csv"), woa_r["xs"], delimiter=",", fmt="%.10f",
               header="a1..a10 de cada corrida (semillas 1..30)")

    print("\n=== TABLA ESTADISTICA (valor final de f, 30 corridas, 3000 evaluaciones) ===")
    guardar_tabla(dir_res, "tabla_estadistica", [fila("WOA (b=1)", woa_r), fila("Busqueda aleatoria", rs_r)], COLS)

    # mejor solucion del WOA (insumo de la validacion)
    i = int(np.argmin(woa_r["finales"]))
    mejor = dict(semilla=SEMILLAS[i], f=float(woa_r["finales"][i]), a=woa_r["xs"][i].tolist())
    with open(os.path.join(dir_res, "mejor_solucion_WOA.json"), "w") as fh:
        json.dump(mejor, fh, indent=2)
    print(f"\nMejor corrida WOA: semilla {mejor['semilla']}, f = {mejor['f']:.4f}")

    # ------------------------------------------------ convergencia (media +- 1 desv. est.)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, logx in zip(axs, (False, True)):
        banda(ax, woa_r["hist"], "WOA (b = 1)", "tab:blue")
        banda(ax, rs_r["hist"], "Busqueda aleatoria", "tab:gray")
        ax.axhline(-30, color="r", ls="--", lw=1, label="SLL optimo teorico (Chebyshev) = -30 dB")
        ax.set_xlabel("Numero de evaluaciones de f"); ax.set_ylabel("Mejor f encontrado")
        ax.grid(True, alpha=0.3)
        if logx:
            ax.set_xscale("log"); ax.set_title("Misma curva, eje x logaritmico")
        else:
            ax.set_title("Convergencia: media +- 1 desv. est. (30 corridas)")
    axs[0].legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(dir_res, "fig_convergencia.png"), dpi=150); plt.close(fig)

    # ------------------------------------------------ Wilcoxon (pares por semilla)
    fw, fr = woa_r["finales"], rs_r["finales"]
    menor = stats.wilcoxon(fw, fr, alternative="less")
    dos = stats.wilcoxon(fw, fr, alternative="two-sided")
    victorias = int(np.sum(fw < fr))
    texto = (f"Prueba de rangos con signo de Wilcoxon (pares por semilla, n = {len(fw)}, alfa = {ALFA})\n"
             f"H0: la mediana de (f_WOA - f_aleatoria) es 0.   Minimizacion: H1 = WOA da f menor.\n"
             f"  Unilateral (WOA < azar): estadistico W = {menor.statistic:.1f},  p = {menor.pvalue:.4g}  -> "
             f"{'se rechaza H0' if menor.pvalue < ALFA else 'NO se rechaza H0'}\n"
             f"  Bilateral:               estadistico W = {dos.statistic:.1f},  p = {dos.pvalue:.4g}\n"
             f"Corridas en que WOA supera a la busqueda aleatoria: {victorias}/{len(fw)}\n"
             f"Mediana WOA = {np.median(fw):.4f}   Mediana aleatoria = {np.median(fr):.4f}   "
             f"Diferencia de medianas = {np.median(fw) - np.median(fr):.4f}\n")
    with open(os.path.join(dir_res, "wilcoxon.txt"), "w", encoding="utf-8") as fh:
        fh.write(texto)
    print("\n=== WILCOXON ===\n" + texto)

    # ------------------------------------------------ experimento adicional: A y C vectoriales vs escalares
    esc_r = correr_30(woa, n_pob=N_POB, presupuesto=PRESUPUESTO, b=B_BASE, vectorial=False)
    wvar = stats.wilcoxon(woa_r["finales"], esc_r["finales"], alternative="less")
    print("\n=== EXPERIMENTO ADICIONAL: A y C vectoriales (principal) vs escalares por ballena ===")
    guardar_tabla(dir_res, "tabla_variante_A_C", [fila("WOA vectorial (principal)", woa_r),
                                                   fila("WOA escalar (variante)", esc_r)], COLS)
    t_var = (f"Wilcoxon pareado (vectorial < escalar): W = {wvar.statistic:.1f}, p = {wvar.pvalue:.4g}\n"
             f"Mediana vectorial = {np.median(woa_r['finales']):.4f}; mediana escalar = {np.median(esc_r['finales']):.4f}\n")
    with open(os.path.join(dir_res, "variante_A_C.txt"), "w", encoding="utf-8") as fh:
        fh.write(t_var)
    print(t_var)

    # ------------------------------------------------ sensibilidad de b
    print("=== SENSIBILIDAD DEL PARAMETRO b (30 corridas por nivel) ===")
    sens, filas = {}, []
    for b in NIVELES_B:
        r = woa_r if b == B_BASE else correr_30(woa, n_pob=N_POB, presupuesto=PRESUPUESTO, b=b)
        sens[b] = r
        filas.append(fila(f"WOA b={b}", r))
        np.savetxt(os.path.join(dir_res, f"curvas_WOA_b_{b}.csv"), r["hist"], delimiter=",", fmt="%.8g")
    guardar_tabla(dir_res, "tabla_sensibilidad_b", filas, COLS)
    kw = stats.kruskal(*[sens[b]["finales"] for b in NIVELES_B])
    with open(os.path.join(dir_res, "kruskal_sensibilidad.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Kruskal-Wallis entre los 5 niveles de b: H = {kw.statistic:.3f}, p = {kw.pvalue:.4g}\n")
    print(f"\nKruskal-Wallis entre niveles de b: H = {kw.statistic:.3f}, p = {kw.pvalue:.4g}")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.boxplot([sens[b]["finales"] for b in NIVELES_B], showmeans=True)
    ax.set_xticklabels([str(b) for b in NIVELES_B])
    ax.set_xlabel("Constante de la espiral b"); ax.set_ylabel("Valor final de f (30 corridas)")
    ax.set_title("Sensibilidad de WOA al parametro b"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(dir_res, "fig_sensibilidad_boxplot.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for b in NIVELES_B:
        ax.plot(np.arange(1, PRESUPUESTO + 1), sens[b]["hist"].mean(0), lw=1.8, label=f"b = {b}")
    ax.set_xscale("log"); ax.set_xlabel("Numero de evaluaciones de f (log)"); ax.set_ylabel("Mejor f medio")
    ax.set_title("Convergencia media de WOA por nivel de b"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(dir_res, "fig_sensibilidad_convergencia.png"), dpi=150); plt.close(fig)

    return woa_r, mejor
