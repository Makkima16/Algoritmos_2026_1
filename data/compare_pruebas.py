# -*- coding: utf-8 -*-
"""
Benchmark comparativo KGeoMIP vs KQNodes sobre las Pruebas de AYDA 2026-1.

Lee los CSV de Pruebas en data/Pruebas/ (notación de letras A-Z), carga las
TPMs de samples_binary/ y samples_no_binary/, ejecuta ambos motores para cada
combinación (N, k, prueba) y genera gráficas estilo Manual Técnico (Fig. 5 y Fig. 6):
  - Escalabilidad: tiempo medio vs N en escala log, una gráfica por k.
  - Pérdida vs k: δ_k medio vs k, una gráfica por red.

El worker (_worker_motor.py) se invoca como subproceso por cada lote (N, motor, k)
para evitar el conflicto de importación entre KGeoMIP y KQNodes (ambos definen `src`).

Uso:
    .venv/Scripts/python.exe data/compare_pruebas.py [opciones]

Opciones:
    --nets  N10,N15,N20    Redes a ejecutar (por defecto: N10,N15,N20).
    --ks    2,3            Valores de k a correr (por defecto: 2,3).
    --max-pruebas 10       Máximo de pruebas por red (por defecto: todas).
    --output dir           Carpeta de salida (por defecto: data/results_compare/).
    --no-vacio             Prohíbe mecanismo ∅ (por defecto se permite).
    --figures-only         Solo regenera las gráficas desde el CSV existente.
"""

import argparse
import contextlib
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Rutas base ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
WORKER = DATA_ROOT / "_worker_motor.py"
PYTHON = sys.executable

RESULT_SENTINEL = "@@RESULT@@"

# ── Catálogo de redes ────────────────────────────────────────────────────────
# Cada entrada: n, ruta al TPM, CSV de pruebas, estado_inicial, candidato (todo-unos).
NETWORKS = {
    "N10": {
        "n": 10,
        "tpm": DATA_ROOT / "samples_binary" / "N10A.csv",
        "pruebas": DATA_ROOT / "Pruebas" / "Pruebas_N10.csv",
        "estado": "1" + "0" * 9,
    },
    "N15": {
        "n": 15,
        "tpm": DATA_ROOT / "samples_no_binary" / "N15B.csv",
        "pruebas": DATA_ROOT / "Pruebas" / "Pruebas_N15.csv",
        "estado": "1" + "0" * 14,
    },
    "N20": {
        "n": 20,
        "tpm": DATA_ROOT / "samples_binary" / "N20A.csv",
        "pruebas": DATA_ROOT / "Pruebas" / "Pruebas_N20.csv",
        "estado": "1" + "0" * 19,
    },
    "N22": {
        "n": 22,
        "tpm": DATA_ROOT / "samples_binary" / "N22A.csv",
        "pruebas": DATA_ROOT / "Pruebas" / "Pruebas_N22.csv",
        "estado": "1" + "0" * 21,
    },
    "N25": {
        "n": 25,
        "tpm": DATA_ROOT / "samples_binary" / "N25A.csv",
        "pruebas": DATA_ROOT / "Pruebas" / "Pruebas_N25.csv",
        "estado": "1" + "0" * 24,
    },
}

ENGINE_LABEL = {"geomip": "KGeoMIP", "qnodes": "KQNodes"}
ENGINE_COLOR  = {"geomip": "#1F77B4", "qnodes": "#ED7D31"}
ENGINE_MARKER = {"geomip": "o", "qnodes": "s"}


# ── Utilidades ───────────────────────────────────────────────────────────────

def etiquetas_a_bits(etiquetas: str, n: int) -> str:
    """Convierte 'ABDEG' a máscara binaria de longitud n (A=bit 0, B=bit 1, …)."""
    activos = set(etiquetas.strip().upper())
    return "".join("1" if chr(65 + i) in activos else "0" for i in range(n))


def cargar_pruebas(path: Path, n: int, max_pruebas: int | None = None) -> list[dict]:
    """
    Lee el CSV de Pruebas (cabecera en fila 1, notación letras) y devuelve
    una lista de dicts {alcance, mecanismo} en formato binario de longitud n.
    Omite filas donde algún campo está vacío o produce máscara todo-ceros.
    """
    pruebas = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # saltar cabecera
        for row in reader:
            if len(row) < 3:
                continue
            alc_lbl = row[1].strip()
            mec_lbl = row[2].strip()
            if not alc_lbl or not mec_lbl:
                continue
            alc = etiquetas_a_bits(alc_lbl, n)
            mec = etiquetas_a_bits(mec_lbl, n)
            if "1" not in alc or "1" not in mec:
                continue
            pruebas.append({"alcance": alc, "mecanismo": mec,
                            "alcance_lbl": alc_lbl, "mecanismo_lbl": mec_lbl})
            if max_pruebas and len(pruebas) >= max_pruebas:
                break
    return pruebas


def correr_lote(engine: str, net_cfg: dict, k: int, pruebas: list[dict],
                permitir_vacio: bool) -> list[dict]:
    """
    Invoca _worker_motor.py como subproceso para un lote (motor, N, k).
    Devuelve lista de dicts con los resultados de cada prueba.
    """
    n = net_cfg["n"]
    candidato = "1" * n
    estado = net_cfg["estado"]
    tpm_path = net_cfg["tpm"]

    cfg = {
        "engine": engine,
        "root": str(REPO_ROOT / ("KGeoMIP" if engine == "geomip" else "KQNodes")),
        "tpm": str(tpm_path),
        "estado": estado,
        "candidato": candidato,
        "permitir_presente_vacio": permitir_vacio,
        "k": k,
        "tests": [{"row": i + 1, "alcance": p["alcance"], "mecanismo": p["mecanismo"]}
                  for i, p in enumerate(pruebas)],
    }

    resultados: list[dict] = []
    boot_time = 0.0

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(cfg, tf, ensure_ascii=False)
        lote_path = tf.name

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            [PYTHON, str(WORKER), lote_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        for linea in proc.stdout:
            linea = linea.strip()
            if not linea.startswith(RESULT_SENTINEL):
                continue
            payload = json.loads(linea[len(RESULT_SENTINEL):])
            if "ready" in payload:
                boot_time = time.time() - t0
                continue
            if "fatal" in payload:
                print(f"    [FATAL] {payload['fatal']}")
                continue
            if not payload.get("ok"):
                print(f"    [ERROR fila {payload.get('row')}] {payload.get('error')}")
                continue
            row_idx = payload["row"] - 1
            p = pruebas[row_idx]
            t_search = float(payload.get("tiempo", 0.0))
            t_prep = float(payload.get("prep", 0.0))
            resultados.append({
                "engine": engine,
                "n": n,
                "k": k,
                "prueba": row_idx + 1,
                "alcance_lbl": p["alcance_lbl"],
                "mecanismo_lbl": p["mecanismo_lbl"],
                "perdida": float(payload["perdida"]),
                "t_search": t_search,
                "t_prep": t_prep,
                "t_total": t_search + t_prep,
            })
            status = "OK"
            print(f"    [{status}] prueba {payload['row']:>3}  "
                  f"δ={float(payload['perdida']):.4f}  "
                  f"t={t_search + t_prep:.3f}s  "
                  f"({len(resultados)}/{len(pruebas)})")
        proc.wait()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(lote_path)

    print(f"    → lote completado: {len(resultados)}/{len(pruebas)}  "
          f"warmup={boot_time:.2f}s  total={time.time() - t0:.2f}s")
    return resultados


# ── Orquestación principal ───────────────────────────────────────────────────

def correr_todo(nets: list[str], ks: list[int], max_pruebas: int | None,
                permitir_vacio: bool) -> list[dict]:
    """Itera (red, motor, k) y devuelve todos los resultados como lista de dicts."""
    todos: list[dict] = []
    for net in nets:
        cfg = NETWORKS[net]
        if not cfg["tpm"].exists():
            print(f"[SKIP] {net}: TPM no encontrada en {cfg['tpm']}")
            continue
        pruebas = cargar_pruebas(cfg["pruebas"], cfg["n"], max_pruebas)
        if not pruebas:
            print(f"[SKIP] {net}: sin pruebas válidas.")
            continue
        print(f"\n{'=' * 60}\n{net}  (n={cfg['n']}, {len(pruebas)} prueba(s))")
        for engine in ("geomip", "qnodes"):
            for k in ks:
                etiqueta = f"{net} · {ENGINE_LABEL[engine]} · k={k}"
                print(f"\n  [{etiqueta}]")
                t0 = time.time()
                resultados = correr_lote(engine, cfg, k, pruebas, permitir_vacio)
                todos.extend(resultados)
    return todos


# ── Persistencia de resultados ───────────────────────────────────────────────

def guardar_csv(resultados: list[dict], path: Path, append: bool = False) -> None:
    if not resultados:
        return
    campos = list(resultados[0].keys())
    previos: list[dict] = []
    if append and path.exists():
        previos = cargar_csv(path)
        # Deduplicar por clave compuesta; el nuevo resultado prevalece.
        clave = lambda r: (r["engine"], r["n"], r["k"], r["prueba"])
        existentes = {clave(r) for r in resultados}
        previos = [r for r in previos if clave(r) not in existentes]
    todos = previos + resultados
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(todos)
    print(f"\n  → CSV guardado: {path} ({len(todos)} registros)")


def cargar_csv(path: Path) -> list[dict]:
    registros = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["n"] = int(row["n"])
            row["k"] = int(row["k"])
            row["prueba"] = int(row["prueba"])
            row["perdida"] = float(row["perdida"])
            row["t_search"] = float(row["t_search"])
            row["t_prep"] = float(row["t_prep"])
            row["t_total"] = float(row["t_total"])
            registros.append(row)
    return registros


# ── Generación de gráficas (estilo Fig. 5 y Fig. 6 del manual) ───────────────

def _agrupar(registros: list[dict], engine: str, k: int) -> dict[int, list[float]]:
    """Devuelve {n: [t_total, …]} para un motor y k dados."""
    grupos: dict[int, list[float]] = {}
    for r in registros:
        if r["engine"] == engine and r["k"] == k:
            grupos.setdefault(r["n"], []).append(r["t_total"])
    return grupos


def _agrupar_perdida(registros: list[dict], engine: str, n: int) -> dict[int, list[float]]:
    """Devuelve {k: [perdida, …]} para un motor y N dados."""
    grupos: dict[int, list[float]] = {}
    for r in registros:
        if r["engine"] == engine and r["n"] == n:
            grupos.setdefault(r["k"], []).append(r["perdida"])
    return grupos


def fig_escalabilidad(registros: list[dict], out_dir: Path, k: int) -> None:
    """
    Genera gráfica estilo Fig. 5: tiempo medio vs N (escala log) para k dado.
    Una curva por motor; barras de error = desv. estándar entre pruebas.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f"Escalabilidad — k={k}", fontsize=12)
    ax.set_xlabel("Número de nodos N")
    ax.set_ylabel("Tiempo por prueba (s)")
    ax.set_yscale("log")

    for engine in ("geomip", "qnodes"):
        grupos = _agrupar(registros, engine, k)
        if not grupos:
            continue
        ns_ord = sorted(grupos)
        medias = [float(np.mean(grupos[n])) for n in ns_ord]
        stds   = [float(np.std(grupos[n])) if len(grupos[n]) > 1 else 0.0 for n in ns_ord]
        ax.errorbar(
            ns_ord, medias, yerr=stds,
            label=ENGINE_LABEL[engine],
            color=ENGINE_COLOR[engine],
            marker=ENGINE_MARKER[engine],
            capsize=4,
            linewidth=1.5,
        )

    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    ruta = out_dir / f"comp_scalability_k{k}.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  → {ruta.name}")


def fig_perdida_vs_k(registros: list[dict], out_dir: Path, net: str) -> None:
    """
    Genera gráfica estilo Fig. 6: δ_k medio vs k para una red dada.
    Una curva por motor; barras de error = desv. estándar entre pruebas.
    """
    cfg = NETWORKS.get(net)
    if cfg is None:
        return
    n = cfg["n"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title(f"Pérdida vs k — {net} (n={n})", fontsize=12)
    ax.set_xlabel("k (número de partes)")
    ax.set_ylabel("Pérdida δ_k (EMD)")

    for engine in ("geomip", "qnodes"):
        grupos = _agrupar_perdida(registros, engine, n)
        if not grupos:
            continue
        ks_ord = sorted(grupos)
        medias = [float(np.mean(grupos[k])) for k in ks_ord]
        stds   = [float(np.std(grupos[k])) if len(grupos[k]) > 1 else 0.0 for k in ks_ord]
        ax.errorbar(
            ks_ord, medias, yerr=stds,
            label=ENGINE_LABEL[engine],
            color=ENGINE_COLOR[engine],
            marker=ENGINE_MARKER[engine],
            capsize=4,
            linewidth=1.5,
        )

    ax.set_xticks(sorted({r["k"] for r in registros if r["n"] == n}))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    ruta = out_dir / f"comp_loss_vs_k_{net}.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  → {ruta.name}")


def generar_figuras(registros: list[dict], out_dir: Path,
                    ks: list[int], nets: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generando gráficas en {out_dir} …")
    for k in ks:
        fig_escalabilidad(registros, out_dir, k)
    for net in nets:
        fig_perdida_vs_k(registros, out_dir, net)


# ── Punto de entrada ─────────────────────────────────────────────────────────

def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Benchmark KGeoMIP vs KQNodes con las Pruebas de AYDA 2026-1."
    )
    ap.add_argument("--nets", default="N10,N15,N20,N22",
                    help="Redes separadas por coma (ej. N10,N15,N20,N22). Por defecto: N10,N15,N20,N22.")
    ap.add_argument("--ks", default="3,4,5",
                    help="Valores de k separados por coma. Por defecto: 2,3.")
    ap.add_argument("--max-pruebas", type=int, default=None,
                    help="Máximo de pruebas por red. Por defecto: todas.")
    ap.add_argument("--output", default=str(DATA_ROOT / "results_compare"),
                    help="Carpeta de salida. Por defecto: data/results_compare/.")
    ap.add_argument("--no-vacio", dest="vacio", action="store_false", default=True,
                    help="Prohíbe mecanismo ∅ (las pérdidas de KGeoMIP se inflan).")
    ap.add_argument("--figures-only", action="store_true",
                    help="Solo regenera gráficas desde el CSV existente.")
    ap.add_argument("--append", action="store_true",
                    help="Agrega resultados al CSV existente (sin sobreescribir los previos).")
    args = ap.parse_args()

    nets = [n.strip() for n in args.nets.split(",") if n.strip()]
    ks   = [int(k) for k in args.ks.split(",") if k.strip()]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"

    # ── Validaciones ──────────────────────────────────────────────────────────
    invalidas = [n for n in nets if n not in NETWORKS]
    if invalidas:
        sys.exit(f"Redes desconocidas: {invalidas}. Disponibles: {list(NETWORKS)}")

    print("=" * 60)
    print(f"  Benchmark KGeoMIP vs KQNodes")
    print(f"  Redes  : {nets}")
    print(f"  k      : {ks}")
    print(f"  Pruebas: {'todas' if args.max_pruebas is None else args.max_pruebas}")
    print(f"  Salida : {out_dir}")
    print(f"  ∅ mec. : {'permitido' if args.vacio else 'prohibido'}")
    print("=" * 60)

    # ── Ejecutar o cargar ─────────────────────────────────────────────────────
    if args.figures_only:
        if not csv_path.exists():
            sys.exit(f"--figures-only pero no existe {csv_path}")
        print(f"\n  Cargando resultados de {csv_path} …")
        registros = cargar_csv(csv_path)
        print(f"  {len(registros)} registros cargados.")
    else:
        t_inicio = time.time()
        registros = correr_todo(nets, ks, args.max_pruebas, args.vacio)
        guardar_csv(registros, csv_path, append=args.append)
        print(f"\n  Tiempo total: {time.time() - t_inicio:.1f} s")

    if not registros:
        print("  Sin resultados — no se generan gráficas.")
        return

    generar_figuras(registros, out_dir / "figures", ks, nets)
    print("\n  ¡Listo!")


if __name__ == "__main__":
    main()
