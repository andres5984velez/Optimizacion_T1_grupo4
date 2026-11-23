"""
validacion.py -- Validacion de la solucion encontrada por WOA (niveles V1, V2 y V3 de la guia).

  V1 Referencia        : Dolph-Chebyshev (optimo para FNBW = 17 grados) y arreglo uniforme
  V2 Verificacion      : patron recalculado con malla 10 veces mas fina (0.005 grados), ancho de
                         haz a -3 dB y directividad por integracion numerica
  V3 Robustez          : (a) cuantizacion a 5 bits, (b) Monte Carlo +-5 % (200), (c) falla de a_n = 0
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from antena import (metricas, f_antena, patron_db, factor_arreglo, ancho_haz_3db, directividad,
                    FNBW_MAX, PASO_OPT)
from verificar_modelo import amplitudes_chebyshev

PASO_FINO = 0.005
N_MC = 200
ERROR_MC = 0.05
SEMILLA_MC = 12345
BITS = 5


def f_desde(sll, fnbw):
    return sll + 100.0 * max(0.0, fnbw - FNBW_MAX)


def tabla(dir_v, base, cab, filas):
    with open(os.path.join(dir_v, base + ".csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(cab) + "\n" + "\n".join(",".join(map(str, f)) for f in filas) + "\n")
    with open(os.path.join(dir_v, base + ".md"), "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(cab) + " |\n|" + "---|" * len(cab) + "\n")
        fh.write("\n".join("| " + " | ".join(map(str, f)) + " |" for f in filas) + "\n")
    print(f"\n[{base}]\n" + " | ".join(cab))
    for f in filas:
        print(" | ".join(map(str, f)))


def cuantizar(a, bits=BITS):
    """Atenuadores de `bits` bits: 2**bits niveles uniformes en [0, 1] (k/(2**bits-1)),
    tras normalizar a la amplitud maxima."""
    niveles = 2 ** bits - 1
    return np.round(a / a.max() * niveles) / niveles


def run(dir_res, mejor):
    dir_v = os.path.join(dir_res, "validacion")
    os.makedirs(dir_v, exist_ok=True)
    a_woa = np.array(mejor["a"]); a_woa = a_woa / a_woa.max()
    disenos = {"WOA (mejor corrida)": a_woa, "Chebyshev -30 dB": amplitudes_chebyshev(), "Uniforme": np.ones(10)}
    colores = {"WOA (mejor corrida)": "tab:blue", "Chebyshev -30 dB": "tab:red", "Uniforme": "tab:gray"}

    # ============================================================ V1 - referencia
    print("\n================ V1 - REFERENCIA (Dolph-Chebyshev y uniforme) ================")
    filas = []
    for nom, a in disenos.items():
        sll, fnbw = metricas(a)
        filas.append([nom, f"{sll:.3f}", f"{fnbw:.2f}", f"{f_antena(a):.3f}"])
    tabla(dir_v, "V1_tabla_comparacion", ["Diseno", "SLL (dB)", "FNBW (grados)", "f"], filas)
    sll_w = float(filas[0][1])
    print(f"Brecha de SLL respecto al optimo de Chebyshev (-30 dB): {sll_w + 30:.3f} dB")

    a_ch = disenos["Chebyshev -30 dB"]
    filas = [[n + 1, f"{a_woa[n]:.4f}", f"{a_ch[n]:.4f}", f"{a_woa[n] - a_ch[n]:+.4f}"] for n in range(10)]
    tabla(dir_v, "V1_amplitudes", ["n", "a_n WOA (norm.)", "a_n Chebyshev (norm.)", "diferencia"], filas)
    print(f"RMS de la diferencia de amplitudes = {np.sqrt(np.mean((a_woa - a_ch) ** 2)):.4f}; "
          f"max |dif| = {np.max(np.abs(a_woa - a_ch)):.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    n = np.arange(1, 11); w = 0.27
    for k, (nom, a) in enumerate(disenos.items()):
        ax.bar(n + (k - 1) * w, a, w, label=nom, color=colores[nom])
    ax.set_xlabel("n (1 = elementos centrales)"); ax.set_ylabel("Amplitud normalizada a_n")
    ax.set_title("Distribucion de amplitudes"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(dir_v, "fig_V1_amplitudes.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (x0, x1) in zip(axs, [(0, 180), (80, 100)]):
        for nom, a in disenos.items():
            th, db = patron_db(a)
            ax.plot(th, db, color=colores[nom], label=nom, lw=1.4)
        ax.axhline(-30, color="k", ls=":", lw=1); ax.set_xlim(x0, x1); ax.set_ylim(-60, 2)
        ax.set_xlabel("theta (grados)"); ax.set_ylabel("Patron normalizado (dB)"); ax.grid(True, alpha=0.3)
    axs[0].set_title("Patron de radiacion"); axs[1].set_title("Zoom al lobulo principal"); axs[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(dir_v, "fig_V1_patrones.png"), dpi=150); plt.close(fig)

    # ============================================================ V2 - verificacion independiente
    print("\n================ V2 - MALLA FINA, ANCHO A -3 dB Y DIRECTIVIDAD ================")
    filas = []
    for nom, a in disenos.items():
        s1, w1 = metricas(a, PASO_OPT); s2, w2 = metricas(a, PASO_FINO)
        d = directividad(a, PASO_FINO)
        filas.append([nom, f"{s1:.3f}", f"{s2:.3f}", f"{w1:.2f}", f"{w2:.3f}",
                      f"{f_desde(s2, w2):.3f}", f"{ancho_haz_3db(a):.3f}", f"{d:.3f}", f"{10 * np.log10(d):.2f}"])
    tabla(dir_v, "V2_tabla_malla_fina",
          ["Diseno", "SLL 0.05 (dB)", "SLL 0.005 (dB)", "FNBW 0.05 (gr)", "FNBW 0.005 (gr)",
           "f con malla fina", "Ancho a -3 dB (gr)", "Directividad D", "D (dBi)"], filas)

    # las 30 soluciones del WOA en malla gruesa vs fina
    xs = np.loadtxt(os.path.join(dir_res, "soluciones_WOA.csv"), delimiter=",")
    g = np.array([metricas(a, PASO_OPT) for a in xs]); fi = np.array([metricas(a, PASO_FINO) for a in xs])
    exceso = fi[:, 1] - FNBW_MAX
    resumen = (f"Las 30 soluciones de WOA evaluadas con malla fina (0.005 grados):\n"
               f"  dif. de SLL (fina - gruesa): media = {np.mean(fi[:, 0] - g[:, 0]):+.4f} dB, "
               f"max |dif| = {np.max(np.abs(fi[:, 0] - g[:, 0])):.4f} dB\n"
               f"  soluciones cuyo FNBW supera 17 grados en malla fina: {int(np.sum(exceso > 1e-9))}/30 "
               f"(exceso maximo = {max(0.0, exceso.max()):.3f} grados)\n")
    with open(os.path.join(dir_v, "V2_30_soluciones.txt"), "w", encoding="utf-8") as fh:
        fh.write(resumen)
    np.savetxt(os.path.join(dir_v, "V2_30_soluciones.csv"), np.c_[g, fi], delimiter=",", fmt="%.5f",
               header="SLL_0.05,FNBW_0.05,SLL_0.005,FNBW_0.005")
    print(resumen)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for paso, est, nom in [(PASO_OPT, "o-", "malla 0.05 (optimizacion)"), (PASO_FINO, "-", "malla 0.005 (verificacion)")]:
        th, af = factor_arreglo(a_woa, paso); af = af / af.max()
        m = (th >= 96) & (th <= 101)
        ax.plot(th[m], af[m], est, ms=3, lw=1.2, label=nom)
    ax.axvline(90 + FNBW_MAX / 2, color="r", ls="--", lw=1, label="limite FNBW = 17 grados")
    ax.set_xlabel("theta (grados)"); ax.set_ylabel("|AF| normalizado")
    ax.set_title("WOA: primer nulo con ambas mallas"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(dir_v, "fig_V2_primer_nulo.png"), dpi=150); plt.close(fig)

    # ============================================================ V3 - robustez
    print("\n================ V3 - ROBUSTEZ ================")
    # (a) cuantizacion
    filas = []
    for nom in ("WOA (mejor corrida)", "Chebyshev -30 dB"):
        a = disenos[nom]; aq = cuantizar(a)
        s0, w0 = metricas(a); sq, wq = metricas(aq)
        filas.append([nom, f"{s0:.3f}", f"{sq:.3f}", f"{sq - s0:+.3f}", f"{w0:.2f}", f"{wq:.2f}",
                      f"{np.max(np.abs(a - aq)):.4f}"])
    tabla(dir_v, "V3a_cuantizacion_5bits",
          ["Diseno", "SLL ideal (dB)", "SLL 5 bits (dB)", "Delta SLL", "FNBW ideal", "FNBW 5 bits", "max error amplitud"], filas)

    # (b) Monte Carlo: a_n*(1+eps), eps ~ U(-5 %, +5 %), 200 realizaciones, 10 amplitudes independientes
    rng = np.random.default_rng(SEMILLA_MC)
    eps = rng.uniform(-ERROR_MC, ERROR_MC, size=(N_MC, 10))
    filas, mc = [], {}
    for nom in ("WOA (mejor corrida)", "Chebyshev -30 dB"):
        a = disenos[nom]
        res = np.array([metricas(a * (1 + e)) for e in eps])
        mc[nom] = res
        s0 = metricas(a)[0]
        filas.append([nom, f"{s0:.3f}", f"{res[:, 0].mean():.3f}", f"{res[:, 0].std(ddof=1):.3f}",
                      f"{res[:, 0].min():.3f}", f"{res[:, 0].max():.3f}",
                      f"{100 * np.mean(res[:, 1] > FNBW_MAX + 1e-9):.1f}"])
    tabla(dir_v, "V3b_montecarlo_5pct",
          ["Diseno", "SLL nominal", "SLL medio", "Desv. est.", "SLL mejor", "SLL peor", "% con FNBW>17"], filas)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, (nom, res) in zip(axs, mc.items()):
        ax.hist(res[:, 0], bins=20, color=colores[nom], alpha=0.8)
        ax.axvline(metricas(disenos[nom])[0], color="k", ls="--", label="nominal")
        ax.set_title(nom); ax.set_xlabel("SLL (dB)"); ax.set_ylabel("Realizaciones"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Monte Carlo: errores de amplitud uniformes +-5 % ({N_MC} realizaciones)")
    fig.tight_layout(); fig.savefig(os.path.join(dir_v, "fig_V3b_montecarlo.png"), dpi=150); plt.close(fig)

    # (c) falla de cada a_n (a_n = 0, uno a la vez)
    filas, dsll = [], {}
    for nom in ("WOA (mejor corrida)", "Chebyshev -30 dB"):
        a = disenos[nom]; s0 = metricas(a)[0]; dsll[nom] = []
        for n in range(10):
            af = a.copy(); af[n] = 0.0
            s, w = metricas(af)
            dsll[nom].append(s - s0)
            filas.append([nom, n + 1, f"{s:.3f}", f"{s - s0:+.3f}", f"{w:.2f}"])
    tabla(dir_v, "V3c_fallas", ["Diseno", "a_n anulado", "SLL (dB)", "Delta SLL (dB)", "FNBW (gr)"], filas)
    for nom, d in dsll.items():
        k = int(np.argmax(d))
        print(f"Elemento mas critico en {nom}: a_{k + 1} (empeora el SLL en {d[k]:+.2f} dB)")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, (nom, d) in enumerate(dsll.items()):
        ax.bar(np.arange(1, 11) + (k - 0.5) * 0.4, d, 0.4, label=nom, color=colores[nom])
    ax.set_xlabel("a_n anulado"); ax.set_ylabel("Delta SLL (dB)"); ax.set_title("Falla de un a_n")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(dir_v, "fig_V3c_fallas.png"), dpi=150); plt.close(fig)
