"""
Script de prueba para KGeoMIP — k-Partición de Mínima Información.

Valida:
  1. KGeoMIP corre correctamente para k=2, 3, 4 sobre N3A y N4A.
  2. La partición atómica (k=n) da la mayor pérdida.
  3. Los tiempos de ejecución son razonables.

Nota sobre semántica:
    KGeoMIP usa semántica IIT estricta: cada parte Si ve solo su propio presente Si_t.
    GeoMIP usa un corte asimétrico diferente. Los valores de φ para k=2 son distintos
    (IIT estricto ≠ corte asimétrico de GeoMIP), lo cual se explica en el reporte.

Ejecutar desde GeoMIP/src/Method2_Dynamic_Programming_Reformulation/:
    uv run exec_kgeomip.py
"""

from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.kgeomip import KGeoMIP, stirling2

import numpy as np
from pathlib import Path

GEOMIP_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = GEOMIP_ROOT / "data" / "samples"


def cargar_tpm(nombre: str) -> np.ndarray:
    return np.genfromtxt(SAMPLES_DIR / nombre, delimiter=",")


def sep(titulo: str):
    print(f"\n{'═'*60}")
    print(f"  {titulo}")
    print('═'*60)


def main():
    aplicacion.profiler_habilitado = False
    aplicacion.pagina_sample_network = "A"

    # ── Referencia: GeoMIP original ────────────────────────────────────────
    sep("GeoMIP original (referencia, corte asimétrico)")
    tpm_3 = cargar_tpm("N3A.csv")
    est = "000"; cond = alc = mec = "111"
    sol_geo = GeometricSIA(Manager(est)).aplicar_estrategia(cond, alc, mec, tpm_3)
    print(sol_geo)
    print(f"  → φ GeoMIP (corte asimétrico) = {sol_geo.perdida:.4f}")

    # ── N3A: KGeoMIP k=2,3 ────────────────────────────────────────────────
    sep("N3A · KGeoMIP k=2 (IIT estricto)")
    print(f"  S(3,2) = {stirling2(3,2)} bi-particiones\n")
    sol_k2 = KGeoMIP(Manager(est), k=2).aplicar_estrategia(cond, alc, mec, tpm_3)
    print(sol_k2)

    sep("N3A · KGeoMIP k=3")
    print(f"  S(3,3) = {stirling2(3,3)} (solo la partición atómica {{A}}{{B}}{{C}})\n")
    sol_k3 = KGeoMIP(Manager(est), k=3).aplicar_estrategia(cond, alc, mec, tpm_3)
    print(sol_k3)

    print(f"\n  φ GeoMIP (ref)  = {sol_geo.perdida:.4f}  (corte asimétrico)")
    print(f"  φ KGeoMIP k=2   = {sol_k2.perdida:.4f}  (IIT estricto)")
    print(f"  φ KGeoMIP k=3   = {sol_k3.perdida:.4f}  (IIT estricto)")
    print(f"  Nota: φ puede crecer con k (no hay monotonía garantizada con esta métrica)")

    # ── N4A: KGeoMIP k=2,3,4 ─────────────────────────────────────────────
    sep("N4A · KGeoMIP k=2, k=3, k=4")
    tpm_4 = cargar_tpm("N4A.csv")
    e4 = "0000"; c4 = a4 = m4 = "1111"
    print(f"  S(4,2)={stirling2(4,2)}, S(4,3)={stirling2(4,3)}, S(4,4)={stirling2(4,4)}\n")

    phis = {}
    for k_val in [2, 3, 4]:
        sol = KGeoMIP(Manager(e4), k=k_val).aplicar_estrategia(c4, a4, m4, tpm_4)
        phis[k_val] = sol.perdida
        print(sol)

    sep("Resumen")
    print(f"  N3A  GeoMIP:  φ = {sol_geo.perdida:.4f}  (corte asimétrico, referencia)")
    print(f"  N3A  k=2:     φ = {sol_k2.perdida:.4f}  t = {sol_k2.tiempo_ejecucion:.4f}s")
    print(f"  N3A  k=3:     φ = {sol_k3.perdida:.4f}  t = {sol_k3.tiempo_ejecucion:.4f}s")
    for k_val in [2, 3, 4]:
        print(f"  N4A  k={k_val}:     φ = {phis[k_val]:.4f}")
    print()


if __name__ == "__main__":
    main()