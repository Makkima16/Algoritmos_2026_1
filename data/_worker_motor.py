# -*- coding: utf-8 -*-
"""
Worker de un solo lote (motor, N, k) para la suite de pruebas 2026-1.

Se ejecuta como SUBPROCESO desde run_suite_2026.py. Como KGeoMIP y KQNodes definen
ambos un paquete `src`, no pueden convivir en el mismo proceso: cada worker importa
UN solo motor, ajustando sys.path/cwd a la raíz del motor correspondiente.

Lee un JSON de entrada con el lote a ejecutar y emite por stdout, una línea por
prueba, un registro JSON precedido por el sentinela RESULT_SENTINEL. Todo el ruido
(prints/logs internos de los motores) se silencia redirigiendo stdout/stderr a
devnull durante el cálculo, de modo que el canal de resultados quede limpio.

Uso:
    python _worker_motor.py <ruta_lote.json>

Formato del JSON de entrada:
{
  "engine":    "geomip" | "qnodes",
  "root":      "<ruta absoluta a la raíz del motor>",
  "tpm":       "<ruta absoluta al CSV de la TPM>",
  "estado":    "<bits del estado inicial, longitud N>",
  "candidato": "<bits del sistema candidato, longitud N>",
  "permitir_presente_vacio": false,
  "k":         3,
  "tests":     [ {"row": 6, "alcance": "<bits>", "mecanismo": "<bits>"}, ... ]
}
"""

import os
import sys
import json
import io
import contextlib
import logging
import time

RESULT_SENTINEL = "@@RESULT@@"


def _emit(obj: dict) -> None:
    """Escribe un registro de resultado al stdout real, con sentinela y flush."""
    sys.__stdout__.write(RESULT_SENTINEL + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


@contextlib.contextmanager
def _silenciar():
    """Silencia stdout, stderr y logging durante el bloque (motores son ruidosos)."""
    nivel_previo = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    devnull = open(os.devnull, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()
        logging.disable(nivel_previo)


def main() -> None:
    # Los motores formatean particiones con caracteres unicode (⎛, ⎝, …) y algunos
    # handlers de logging escriben a la consola (cp1252 en Windows). Forzamos UTF-8
    # con reemplazo para que ningún write accidental tumbe la prueba.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ruta_lote = sys.argv[1]
    with open(ruta_lote, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    engine = cfg["engine"]
    root = cfg["root"]
    tpm_path = cfg["tpm"]
    estado = cfg["estado"]
    candidato = cfg["candidato"]
    k = cfg["k"]
    permitir_vacio = cfg.get("permitir_presente_vacio", False)
    tests = cfg["tests"]

    # Ajustar el entorno de importación a la raíz del motor.
    sys.path.insert(0, root)
    os.chdir(root)

    # ── Cargar motor + TPM ────────────────────────────────────────────────
    import numpy as np

    if engine == "geomip":
        from src.models.base.application import aplicacion
        aplicacion.profiler_habilitado = False
        from src.controllers.manager import Manager
        from src.controllers.strategies.kgeomip import KGeoMIP, warmup_motor
        from src.models.core.lazy_tpm import cargar_tpm

        from src.models.base.sia import _CANDIDATO_CACHE
        from src.models.core.system import System

        with _silenciar():
            tpm = cargar_tpm(tpm_path)
            manager = Manager(estado)
            motor = KGeoMIP(manager)
            # Arranque del motor, igual que el modo bloque de exec_kgeomip.py, para que
            # la PRIMERA prueba no pague costes únicos del lote:
            #   1. warmup_motor(): Numba JIT del kernel de tabla de costos + pool de hilos.
            warmup_motor()
            #   2. Pre-caché del candidato condicionado. Construir y condicionar el System
            #      completo es O(N·2^N) (varios s en N grande) y solo depende de tpm/estado/
            #      condicion —idénticos en todo el lote—, así que sia_preparar_subsistema lo
            #      cachea en la 1ª prueba y lo reutiliza. Hacerlo aquí mueve ese coste al
            #      boot (contabilizado como arranque) en vez de inflar la prueba 1.
            _est_ini = np.array(list(estado), dtype=np.int8)
            _dims_cond = np.array(
                [i for i, b in enumerate(candidato) if b == "0"], dtype=np.int8
            )
            _ckey = (id(tpm), tuple(_est_ini.tolist()), candidato)
            if _ckey not in _CANDIDATO_CACHE:
                _CANDIDATO_CACHE[_ckey] = System(tpm, _est_ini).condicionar(_dims_cond)

        def _resolver(alcance_bits, mecanismo_bits):
            return motor.aplicar_estrategia(
                condicion=candidato,
                alcance=alcance_bits,
                mecanismo=mecanismo_bits,
                tpm=tpm,
                k=k,
                permitir_presente_vacio=permitir_vacio,
            )

    elif engine == "qnodes":
        from src.models.base.application import aplicacion
        aplicacion.desactivar_profiling()
        from src.strategies.q_nodes import DynamicPartition
        from src.middlewares.profile import gestor_perfilado
        gestor_perfilado.enabled = False

        with _silenciar():
            tpm = np.genfromtxt(tpm_path, delimiter=",")
            motor = DynamicPartition(tpm)

        def _resolver(alcance_bits, mecanismo_bits):
            return motor.aplicar_estrategia(
                estado_inicial=estado,
                condicion=candidato,
                alcance=alcance_bits,
                mecanismo=mecanismo_bits,
                k=k,
                permitir_presente_vacio=permitir_vacio,
            )
    else:
        _emit({"fatal": f"motor desconocido: {engine}"})
        return

    _emit({"ready": True, "n_tests": len(tests)})

    # ── Bucle de pruebas ──────────────────────────────────────────────────
    for t in tests:
        row = t["row"]
        alcance_bits = t["alcance"]
        mecanismo_bits = t["mecanismo"]
        t0 = time.time()
        try:
            with _silenciar():
                solucion = _resolver(alcance_bits, mecanismo_bits)
            t_total = float(
                getattr(solucion, "tiempo_ejecucion",
                        getattr(solucion, "tiempo_total", time.time() - t0))
            )
            t_prep = float(getattr(solucion, "tiempo_preparacion", 0.0))
            # tiempo "de la prueba" = sólo búsqueda. En KGeoMIP tiempo_ejecucion
            # incluye la preparación, así que se le resta; en KQNodes ya la excluye.
            if engine == "geomip":
                t_search = max(0.0, t_total - t_prep)
            else:
                t_search = t_total
            _emit({
                "row": row,
                "ok": True,
                "particion": str(solucion.particion),
                "perdida": float(solucion.perdida),
                "tiempo": t_search,
                "prep": t_prep,
            })
        except Exception as e:  # noqa: BLE001  — cualquier fallo se reporta y se sigue
            _emit({"row": row, "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
