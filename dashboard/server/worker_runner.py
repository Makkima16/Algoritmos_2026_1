# -*- coding: utf-8 -*-
"""
Worker persistente para UN algoritmo (qnodes | geomip).

Materializa el requisito "KQNodes y KGeoMIP deben estar ya en proceso de ejecución,
anclados a modo manual": es un proceso vivo que, por cada línea JSON recibida en
stdin, ejecuta UNA sola corrida (un `aplicar_estrategia`) y responde con UNA línea
en stdout.

Aislamiento de `src`: KQNodes y KGeoMIP tienen ambos un paquete top-level llamado
`src`. Por eso cada uno corre en su propio intérprete, con `cwd`/`sys.path`
apuntando a su raíz. Este script recibe (algo, raiz) por argumentos.

Protocolo (texto, una línea por mensaje):
  - El worker imprime en stdout líneas de protocolo prefijadas con SENTINEL.
  - Toda la salida ruidosa del algoritmo (prints, loggers, colorama) se redirige
    a stderr para no contaminar el canal de protocolo. Además el backend filtra
    por el prefijo SENTINEL, de modo que cualquier fuga residual se ignora.
"""

import sys
import os
import io
import json
import re
import contextlib
import traceback
from pathlib import Path

SENTINEL = "___GUI_MSG___"

ALGO = sys.argv[1] if len(sys.argv) > 1 else ""
ROOT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path.cwd()

# Asegurar que `import src...` resuelva al paquete del algoritmo correcto.
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402

# Caché de TPMs por ruta (precarga / reutilización entre corridas).
_TPM_CACHE: dict = {}


def emit(obj: dict) -> None:
    """Escribe un mensaje de protocolo en stdout (canal limpio)."""
    sys.stdout.write(SENTINEL + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _serializar_solucion(sol) -> dict:
    """Extrae atributos de Solution SIN llamar str(sol) (dispararía voz pyttsx3)."""
    def _lista(x):
        return x.tolist() if hasattr(x, "tolist") else x

    return {
        "estrategia": getattr(sol, "estrategia", ALGO),
        "perdida_phi": float(sol.perdida),
        "distribucion_subsistema": _lista(sol.distribucion_subsistema),
        "distribucion_particion": _lista(sol.distribucion_particion),
        "particion": str(sol.particion),
        "tiempo_total_segundos": float(getattr(sol, "tiempo_ejecucion", 0.0)),
        "tiempo_preparacion_segundos": float(getattr(sol, "tiempo_preparacion", 0.0)),
    }


# ── Inicialización específica por algoritmo ────────────────────────────────

run = None  # función run(req) -> Solution

if ALGO == "qnodes":
    with contextlib.redirect_stdout(sys.stderr):
        from src.models.base.application import aplicacion
        from src.strategies.q_nodes import DynamicPartition
        try:
            from src.middlewares.profile import gestor_perfilado
        except Exception:
            gestor_perfilado = None
        try:
            aplicacion.desactivar_profiling()
        except Exception:
            pass
        if gestor_perfilado is not None:
            try:
                gestor_perfilado.enabled = False
            except Exception:
                pass

    def _cargar_tpm_qnodes(ruta: str):
        if ruta not in _TPM_CACHE:
            _TPM_CACHE[ruta] = np.genfromtxt(ruta, delimiter=",")
        return _TPM_CACHE[ruta]

    def run(req):  # noqa: F811
        tpm = _cargar_tpm_qnodes(req["ruta_tpm"])
        analizador = DynamicPartition(tpm)
        return analizador.aplicar_estrategia(
            estado_inicial=req["estado"],
            condicion=req["candidato"],
            alcance=req["alcance"],
            mecanismo=req["mecanismo"],
            k=req.get("k"),
            permitir_presente_vacio=req.get("permitir_presente_vacio", False),
        )

elif ALGO == "geomip":
    with contextlib.redirect_stdout(sys.stderr):
        from src.models.base.application import aplicacion
        from src.controllers.manager import Manager
        from src.controllers.strategies.kgeomip import KGeoMIP
        from src.lazy_tpm import cargar_tpm as _cargar_tpm_geomip
        try:
            aplicacion.desactivar_profiling()
        except Exception:
            pass

    def _set_pagina(nombre_archivo: str) -> None:
        """Ajusta pagina_sample_network al sufijo del archivo (N10A -> 'A')."""
        m = re.match(r"^N\d+([A-Za-z]*)", Path(nombre_archivo).stem)
        if m and m.group(1):
            try:
                aplicacion.pagina_sample_network = m.group(1)
            except Exception:
                pass

    def _cargar_tpm_geo(ruta: str):
        if ruta not in _TPM_CACHE:
            _TPM_CACHE[ruta] = _cargar_tpm_geomip(Path(ruta))
        return _TPM_CACHE[ruta]

    def run(req):  # noqa: F811
        _set_pagina(req["ruta_tpm"])
        tpm = _cargar_tpm_geo(req["ruta_tpm"])
        manager = Manager(req["estado"])
        inst = KGeoMIP(manager)
        return inst.aplicar_estrategia(
            condicion=req["candidato"],
            alcance=req["alcance"],
            mecanismo=req["mecanismo"],
            tpm=tpm,
            k=req.get("k"),
            permitir_presente_vacio=req.get("permitir_presente_vacio", False),
        )

else:
    emit({"ready": False, "error": f"Algoritmo desconocido: {ALGO!r}"})
    sys.exit(1)


# ── Bucle principal: una corrida por línea ─────────────────────────────────

emit({"ready": True, "algo": ALGO})

for linea in sys.stdin:
    linea = linea.strip()
    if not linea:
        continue
    try:
        req = json.loads(linea)
    except Exception as e:
        emit({"ok": False, "error": f"JSON inválido: {e}"})
        continue

    if req.get("cmd") == "ping":
        emit({"ok": True, "pong": True, "algo": ALGO})
        continue

    try:
        with contextlib.redirect_stdout(sys.stderr):
            solucion = run(req)
        emit({"ok": True, **_serializar_solucion(solucion)})
    except Exception as e:
        emit({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()[-2000:],
        })
