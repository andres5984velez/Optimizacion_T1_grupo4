"""
main.py -- Ejecuta TODO el trabajo: verificacion del modelo, protocolo (seccion 3) y validacion V1-V3.
Uso:   python main.py        (resultados en ./resultados/)
Todo es reproducible: semillas 1..30 (corridas) y 12345 (Monte Carlo de V3).
"""
import os
import sys

from verificar_modelo import verificar
import protocolo
import validacion

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")

if __name__ == "__main__":
    if not verificar():
        sys.exit("El modelo no pasa las verificaciones: revisar antena.py")
    woa_r, mejor = protocolo.run(DIR)
    validacion.run(DIR, mejor)
    print("\nListo. Resultados en:", DIR)
