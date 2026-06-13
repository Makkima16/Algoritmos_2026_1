# -*- coding: utf-8 -*-
"""
Métricas de calidad de una k-partición a partir de sus distribuciones.

Entradas (de Solution / JSON de resultados):
  S = distribucion_subsistema  (sistema original)
  P = distribucion_particion   (sistema tras la partición)
  phi = perdida_phi
  particion = string con la notación de grupos (KQNodes o KGeoMIP)

Las 4 familias de métricas responden:
  1. ¿Qué tan adecuada es la pérdida Φ frente al sistema original?  (pérdida relativa)
  2. ¿Cuánto crece / se desvía la distribución?                     (crecimiento)
  3. ¿Qué columnas/nodos conservan mejor su información?            (cohesión)
  4. (Comparativa KQNodes vs KGeoMIP — se arma en main.py cruzando resultados)
"""

import re
import math

EPS = 1e-12

# Umbrales cualitativos para la pérdida relativa.
UMBRAL_PHI_REL_BAJA = 0.10
UMBRAL_PHI_REL_MEDIA = 0.30

# Tolerancia para considerar que una columna "cambió".
EPS_CAMBIO = 1e-6


def _as_list(x):
    if x is None:
        return []
    return [float(v) for v in x]


def _label_order_key(lbl: str) -> int:
    """Valor base-26 para ordenar etiquetas de nodo (A<B<...<Z<AA<...)."""
    v = 0
    for c in lbl:
        v = v * 26 + (ord(c) - ord("A") + 1)
    return v


def parse_grupos(particion: str) -> list[list[str]]:
    """
    Extrae los grupos de la primera línea (futuros/efecto, en MAYÚSCULAS) de la
    notación de partición. Soporta ambos formatos:
      KQNodes : "⎛ I ⎞⎛ A,B,C ⎞\n⎝ i ⎠⎝ a,b,c ⎠"
      KGeoMIP : "| E || A,B,C |\n| e || a,b,c |"
    """
    if not particion:
        return []
    primera = particion.strip().splitlines()[0]
    chunks = re.split(r"[⎛⎞⎝⎠|]+", primera)
    grupos: list[list[str]] = []
    for ch in chunks:
        labels = [t.strip() for t in ch.split(",") if t.strip()]
        labels = [t for t in labels if re.fullmatch(r"[A-Z]+", t)]
        if labels:
            grupos.append(labels)
    return grupos


def perdida_relativa(S, P, phi) -> dict:
    masa = sum(S) if S else 0.0
    phi_rel = float(phi) / max(masa, EPS)
    if phi_rel < UMBRAL_PHI_REL_BAJA:
        banda = "baja"
    elif phi_rel < UMBRAL_PHI_REL_MEDIA:
        banda = "media"
    else:
        banda = "alta"
    return {
        "phi": float(phi),
        "masa_subsistema": masa,
        "phi_relativo": phi_rel,
        "banda": banda,
    }


def crecimiento_distribucion(S, P) -> dict:
    n = min(len(S), len(P))
    deltas = [abs(P[i] - S[i]) for i in range(n)]
    delta_l1 = sum(deltas)
    max_drift = max(deltas) if deltas else 0.0
    cambiadas = sum(1 for d in deltas if d > EPS_CAMBIO)
    suma_s = sum(S)
    suma_p = sum(P)
    return {
        "delta_l1": delta_l1,
        "max_drift": max_drift,
        "columnas_cambiadas": cambiadas,
        "n_columnas": n,
        "pct_columnas_cambiadas": (cambiadas / n) if n else 0.0,
        "masa_subsistema": suma_s,
        "masa_particion": suma_p,
        "crecimiento_masa": suma_p - suma_s,
    }


def cohesion_por_columna(S, P, particion) -> dict:
    n = min(len(S), len(P))
    deltas = [abs(P[i] - S[i]) for i in range(n)]
    max_d = max(deltas) if deltas else 0.0
    coh = [1.0 - (deltas[i] / (max_d + EPS)) for i in range(n)]

    grupos = parse_grupos(particion)
    labels_sorted = sorted({l for g in grupos for l in g}, key=_label_order_key)
    col_por_label = {}
    if len(labels_sorted) == n and n > 0:
        col_por_label = {l: i for i, l in enumerate(labels_sorted)}

    label_por_col = {i: l for l, i in col_por_label.items()}
    columnas = []
    for i in range(n):
        columnas.append({
            "columna": i,
            "label": label_por_col.get(i),
            "delta": deltas[i],
            "subsistema": S[i],
            "particion": P[i],
            "cohesion": coh[i],
        })

    # Cohesión agregada por grupo de la partición.
    grupos_info = []
    for g in grupos:
        idxs = [col_por_label[l] for l in g if l in col_por_label]
        if idxs:
            grupos_info.append({
                "labels": g,
                "cohesion_media": sum(coh[i] for i in idxs) / len(idxs),
                "delta_total": sum(deltas[i] for i in idxs),
                "tamano": len(g),
            })

    return {
        "columnas": columnas,
        "grupos": grupos_info,
        "cohesion_media_global": (sum(coh) / n) if n else 0.0,
    }


def calcular_metricas(distribucion_subsistema, distribucion_particion, perdida_phi, particion) -> dict:
    """Calcula las 3 familias por-resultado (la comparativa se hace aparte)."""
    S = _as_list(distribucion_subsistema)
    P = _as_list(distribucion_particion)
    phi = float(perdida_phi) if perdida_phi is not None else 0.0
    return {
        "perdida_relativa": perdida_relativa(S, P, phi),
        "crecimiento": crecimiento_distribucion(S, P),
        "cohesion": cohesion_por_columna(S, P, particion or ""),
    }
