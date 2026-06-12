# -*- coding: utf-8 -*-
"""
Backend FastAPI del dashboard QNodes & GeoMIP.

- Mantiene dos workers persistentes (uno por algoritmo), precargados al arranque
  ("ya en ejecución, anclados a modo manual": una corrida por petición).
- Expone endpoints REST para ejecutar corridas manuales y por bloque (SSE),
  explorar resultados guardados y calcular métricas.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import metrics as M
import results_reader as R
from block_writer import escribir_xlsx_bloque, formatear_tiempo
from workers import pool, ALGO_ROOTS

app = FastAPI(title="QNodes & GeoMIP Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # Deja ambos motores "ya en ejecución".
    pool.precargar()


@app.on_event("shutdown")
def _shutdown():
    pool.shutdown()


# ── Modelos de petición ────────────────────────────────────────────────────

class RunReq(BaseModel):
    algo: str
    ruta_tpm: str
    estado: str
    candidato: str
    alcance: str
    mecanismo: str
    k: int | None = None
    permitir_presente_vacio: bool = False


class BlockReq(BaseModel):
    algo: str
    ruta_tpm: str
    ruta_csv: str
    estado: str
    candidato: str
    n: int
    k: int | None = None
    permitir_presente_vacio: bool = False


class MetricReq(BaseModel):
    distribucion_subsistema: list[float]
    distribucion_particion: list[float]
    perdida_phi: float
    particion: str = ""


# ── Salud y catálogos ──────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"workers": pool.health()}


@app.get("/api/datasets")
def datasets(algo: str):
    _validar_algo(algo)
    return {"datasets": R.listar_datasets(algo)}


@app.get("/api/pruebas")
def pruebas(algo: str, n: int | None = None):
    _validar_algo(algo)
    archivos = R.listar_pruebas(algo, n)
    return {"pruebas": archivos}


@app.get("/api/pruebas/contenido")
def pruebas_contenido(ruta: str):
    return {"filas": R.leer_csv_pruebas(ruta)}


# ── Ejecución manual (una corrida) ─────────────────────────────────────────

@app.post("/api/run")
def run(req: RunReq):
    _validar_algo(req.algo)
    worker = pool.get(req.algo)
    resp = worker.run(req.model_dump())
    if not resp.get("ok"):
        raise HTTPException(status_code=500, detail=resp.get("error", "Error desconocido"))
    resp["metricas"] = M.calcular_metricas(
        resp["distribucion_subsistema"],
        resp["distribucion_particion"],
        resp["perdida_phi"],
        resp["particion"],
    )
    return resp


# ── Ejecución por bloque (SSE / streaming) ─────────────────────────────────

@app.post("/api/block")
def block(req: BlockReq):
    _validar_algo(req.algo)
    worker = pool.get(req.algo)
    filas_csv = R.leer_csv_pruebas(req.ruta_csv)

    root = ALGO_ROOTS[req.algo]
    sub = "con_estado_vacio" if req.permitir_presente_vacio else "sin_estado_vacio"
    k_nombre = str(req.k) if req.k is not None else "All"
    fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefijo = "results_KQNodes" if req.algo == "qnodes" else "results_kGeoMIP"
    ruta_salida = root / "results" / "block" / sub / f"{prefijo}_N{req.n}_k{k_nombre}_{fecha}.xlsx"

    def _evento(tipo: str, payload: dict) -> str:
        return f"data: {json.dumps({'tipo': tipo, **payload}, ensure_ascii=False)}\n\n"

    def gen():
        total = len(filas_csv)
        yield _evento("inicio", {"total": total, "salida": str(ruta_salida)})
        resultados = []
        t0 = time.time()

        for idx, fila in enumerate(filas_csv, start=1):
            num = fila["num"]
            alcance_raw = fila["alcance"]
            mecanismo_raw = fila["mecanismo"]

            base = {"num": num, "alcance": alcance_raw, "mecanismo": mecanismo_raw}

            if not alcance_raw or not mecanismo_raw:
                base["error"] = "Alcance o mecanismo vacíos en el CSV"
                resultados.append(base)
                yield _evento("fila", {"idx": idx, "total": total, **base})
                continue

            bits_alcance = R.letras_a_binario(alcance_raw, req.n)
            bits_mecanismo = R.letras_a_binario(mecanismo_raw, req.n)

            pet = {
                "algo": req.algo,
                "ruta_tpm": req.ruta_tpm,
                "estado": req.estado,
                "candidato": req.candidato,
                "alcance": bits_alcance,
                "mecanismo": bits_mecanismo,
                "k": req.k,
                "permitir_presente_vacio": req.permitir_presente_vacio,
            }
            resp = worker.run(pet)

            if not resp.get("ok"):
                base["error"] = resp.get("error", "Error desconocido")
                resultados.append(base)
                yield _evento("fila", {"idx": idx, "total": total, **base})
                continue

            tiempo_seg = float(resp.get("tiempo_total_segundos", 0.0))
            fila_res = {
                **base,
                "particion": resp["particion"],
                "perdida": float(resp["perdida_phi"]),
                "tiempo_fmt": formatear_tiempo(tiempo_seg),
                "tiempo_seg": tiempo_seg,
                "error": None,
            }
            resultados.append(fila_res)
            yield _evento("fila", {"idx": idx, "total": total, **fila_res})

        tiempo_total = time.time() - t0
        try:
            escribir_xlsx_bloque(ruta_salida, resultados, tiempo_total)
        except Exception as e:  # noqa: BLE001
            yield _evento("error", {"mensaje": f"No se pudo escribir el XLSX: {e}"})

        exitosas = sum(1 for r in resultados if not r.get("error"))
        yield _evento("fin", {
            "total": total,
            "exitosas": exitosas,
            "con_error": total - exitosas,
            "tiempo_total": tiempo_total,
            "tiempo_total_fmt": formatear_tiempo(tiempo_total),
            "salida": str(ruta_salida),
        })

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Resultados guardados ───────────────────────────────────────────────────

@app.get("/api/results")
def results(algo: str, tipo: str):
    _validar_algo(algo)
    if tipo not in ("manual", "block"):
        raise HTTPException(status_code=400, detail="tipo debe ser 'manual' o 'block'")
    return {"resultados": R.listar_resultados(algo, tipo)}


@app.get("/api/results/detail")
def results_detail(ruta: str):
    try:
        detalle = R.leer_detalle(ruta)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Si es un JSON manual con distribuciones, adjuntar métricas.
    if detalle["tipo"] == "manual":
        d = detalle["data"]
        if "distribucion_subsistema" in d and "distribucion_particion" in d:
            detalle["metricas"] = M.calcular_metricas(
                d.get("distribucion_subsistema"),
                d.get("distribucion_particion"),
                d.get("perdida_phi", 0.0),
                d.get("particion", ""),
            )
    return detalle


@app.post("/api/metrics")
def calc_metrics(req: MetricReq):
    return M.calcular_metricas(
        req.distribucion_subsistema,
        req.distribucion_particion,
        req.perdida_phi,
        req.particion,
    )


# ── Comparativa QNodes vs GeoMIP (resultados manuales) ─────────────────────

@app.get("/api/comparativa")
def comparativa():
    """Cruza los JSON manuales de ambos algoritmos por (alcance, mecanismo, candidato, k)."""
    indice: dict[tuple, dict] = {}
    for algo in ("qnodes", "geomip"):
        for meta in R.listar_resultados(algo, "manual"):
            try:
                d = R.leer_detalle(meta["ruta"])["data"]
            except Exception:
                continue
            clave = (
                d.get("sistema_candidato"),
                d.get("alcance"),
                d.get("mecanismo"),
                str(d.get("k_solicitado")),
            )
            entrada = indice.setdefault(clave, {
                "candidato": d.get("sistema_candidato"),
                "alcance": d.get("alcance"),
                "mecanismo": d.get("mecanismo"),
                "k": d.get("k_solicitado"),
            })
            S = d.get("distribucion_subsistema") or []
            masa = sum(float(x) for x in S) or 1.0
            phi = float(d.get("perdida_phi", 0.0))
            entrada[algo] = {
                "phi": phi,
                "phi_relativo": phi / masa,
                "tiempo": float(d.get("tiempo_total_segundos", 0.0)),
                "archivo": meta["archivo"],
            }
    pares = [v for v in indice.values() if "qnodes" in v and "geomip" in v]
    return {"comparaciones": pares, "n_pares": len(pares)}


# ── Utilidades ─────────────────────────────────────────────────────────────

def _validar_algo(algo: str):
    if algo not in ALGO_ROOTS:
        raise HTTPException(status_code=400, detail=f"Algoritmo inválido: {algo}")
