"""
v4_nec.py -- (OPCIONAL, bonificacion V4) Efecto del acoplamiento mutuo con dipolos reales.

Genera los archivos .nec del arreglo de 20 dipolos de media onda excitados con las amplitudes
optimizadas por WOA (y, como referencia, Chebyshev y uniforme). Los .nec se abren directamente en
4NEC2 (File > Open). Si `nec2c` esta instalado (mismo motor NEC-2), tambien los simula y calcula el
SLL/FNBW del patron con acoplamiento, comparandolo con el modelo isotropico.

Geometria: f = 299.792458 MHz (lambda = 1 m); dipolos paralelos al eje x, apilados a lo largo del eje z
con d = 0.5 m, longitud 0.47 m, radio 1 mm, 21 segmentos, espacio libre.  Excitacion por fuentes de
TENSION con amplitud a_n (fase 0).  El corte phi = 90 grados (plano yz) es el plano H de los
dipolos (omnidireccional): ahi el patron del dipolo es plano y se compara con el factor de arreglo.

Uso:   python v4_nec.py        (despues de main.py; escribe en resultados/V4/)
"""
import json
import os
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from antena import metricas_af, factor_arreglo
from verificar_modelo import amplitudes_chebyshev

F_MHZ = 299.792458
D_ELEM = 0.5
L_DIP = 0.47
RADIO = 0.001
N_SEG = 21
PASO = 0.1                      # paso angular del patron NEC (grados)


def escribir_nec(a, ruta, titulo):
    L = ["CM " + titulo, f"CM f = {F_MHZ} MHz (lambda = 1 m), d = {D_ELEM} m, dipolos de {L_DIP} m a lo largo de x",
         "CM Excitacion por tension con amplitud a_n (n = 1 en los elementos centrales)", "CE"]
    for m in range(1, 21):
        z = (m - 10.5) * D_ELEM
        L.append(f"GW {m},{N_SEG},{-L_DIP / 2:.5f},0,{z:.5f},{L_DIP / 2:.5f},0,{z:.5f},{RADIO}")
    L.append("GE 0")
    for m in range(1, 21):
        n = int(abs(m - 10.5) + 0.5)
        L.append(f"EX 0,{m},{N_SEG // 2 + 1},0,{a[n - 1]:.6f},0")
    L.append(f"FR 0,1,0,0,{F_MHZ},0")
    L.append(f"RP 0,{int(round(180 / PASO)) + 1},1,1000,0,90,{PASO},0")
    L.append("EN")
    with open(ruta, "w") as fh:
        fh.write("\n".join(L) + "\n")


def simular(ruta_nec):
    """Ejecuta nec2c; devuelve la ruta del .out o None si nec2c no esta disponible."""
    if shutil.which("nec2c") is None:
        return None
    ruta_out = ruta_nec[:-4] + ".out"
    subprocess.run(["nec2c", "-i", ruta_nec, "-o", ruta_out], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ruta_out


def leer_patron(ruta_out):
    """(theta, ganancia total en dB) del bloque RADIATION PATTERNS."""
    th, g = [], []
    en_bloque = False
    for linea in open(ruta_out, errors="ignore"):
        if "RADIATION PATTERNS" in linea:
            en_bloque = True
            continue
        if en_bloque:
            t = linea.split()
            try:
                th.append(float(t[0])); g.append(float(t[4]))
            except (ValueError, IndexError):
                continue
    return np.array(th), np.array(g)


def run(dir_res):
    dir_v = os.path.join(dir_res, "V4")
    os.makedirs(dir_v, exist_ok=True)
    a_woa = np.array(json.load(open(os.path.join(dir_res, "mejor_solucion_WOA.json")))["a"])
    disenos = {"WOA": a_woa / a_woa.max(), "Chebyshev": amplitudes_chebyshev(), "Uniforme": np.ones(10)}
    for nom, a in disenos.items():
        escribir_nec(a, os.path.join(dir_v, f"arreglo_{nom}.nec"), f"Arreglo 20 dipolos - {nom}")
    print(f"Archivos .nec escritos en {dir_v}")
    if shutil.which("nec2c") is None:
        print("nec2c no esta instalado: abra los .nec en 4NEC2 (File > Open) y mida el SLL del corte phi = 90.")
        return
    filas, patrones = [], {}
    for nom, a in disenos.items():
        out = simular(os.path.join(dir_v, f"arreglo_{nom}.nec"))
        th, g = leer_patron(out)
        af_nec = 10.0 ** ((g - g.max()) / 20.0)                 # amplitud normalizada del patron NEC
        s_nec, w_nec = metricas_af(af_nec, PASO)
        s_iso, w_iso = metricas_af(factor_arreglo(a, PASO)[1], PASO)
        patrones[nom] = (th, g - g.max(), factor_arreglo(a, PASO)[1])
        filas.append([nom, f"{s_iso:.2f}", f"{s_nec:.2f}", f"{s_nec - s_iso:+.2f}", f"{w_iso:.1f}", f"{w_nec:.1f}",
                      f"{g.max():.2f}"])
    cab = ["Diseno", "SLL isotropico (dB)", "SLL con dipolos+acoplamiento (dB)", "Degradacion (dB)",
           "FNBW isotropico", "FNBW NEC", "Ganancia max NEC (dBi)"]
    with open(os.path.join(dir_v, "V4_tabla.md"), "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(cab) + " |\n|" + "---|" * len(cab) + "\n")
        fh.write("\n".join("| " + " | ".join(f) + " |" for f in filas) + "\n")
    print("\n" + " | ".join(cab))
    for f in filas:
        print(" | ".join(f))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, nom in zip(axs, ("WOA", "Chebyshev")):
        th, gn, af = patrones[nom]
        ax.plot(th, 20 * np.log10(np.maximum(af / af.max(), 1e-6)), label="modelo isotropico")
        ax.plot(th, gn, label="NEC2: dipolos + acoplamiento mutuo")
        ax.set_ylim(-60, 2); ax.set_xlabel("theta (grados)"); ax.set_title(nom); ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("Patron normalizado (dB)"); axs[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(dir_v, "fig_V4_acoplamiento.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    run(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados"))
