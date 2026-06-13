# -*- coding: utf-8 -*-
"""
Gestor de los dos procesos worker persistentes (KQNodes y KGeoMIP).

Cada worker es un subprocess de `worker_runner.py` lanzado con el `cwd` de su
algoritmo. El pool los mantiene vivos (precargados) desde el arranque del backend
y serializa el acceso con un lock por worker (un `aplicar_estrategia` a la vez =
modo manual). Si un worker muere, se re-spawnea en la siguiente petición.
"""

import os
import sys
import json
import threading
import collections
import subprocess
from pathlib import Path

SENTINEL = "___GUI_MSG___"

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parents[1]  # dashboard/server -> dashboard -> repo
WORKER_SCRIPT = SERVER_DIR / "worker_runner.py"

ALGO_ROOTS = {
    "qnodes": REPO_ROOT / "KQNodes",
    "geomip": REPO_ROOT / "KGeoMIP",
}


class Worker:
    """Un proceso worker persistente para un algoritmo."""

    def __init__(self, algo: str):
        if algo not in ALGO_ROOTS:
            raise ValueError(f"Algoritmo desconocido: {algo!r}")
        self.algo = algo
        self.root = ALGO_ROOTS[algo]
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.ready = False
        self.last_error: str | None = None
        self.stderr_tail: collections.deque = collections.deque(maxlen=300)
        self.start()

    # ── ciclo de vida ──────────────────────────────────────────────────────

    def start(self) -> None:
        self.ready = False
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"

        self.proc = subprocess.Popen(
            [sys.executable, "-X", "utf8", str(WORKER_SCRIPT), self.algo, str(self.root)],
            cwd=str(self.root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        try:
            msg = self._read_until_msg()
        except Exception as e:
            self.last_error = (
                f"El worker '{self.algo}' no inició.\n"
                f"{e}\n--- stderr ---\n" + "\n".join(self.stderr_tail)
            )
            return

        if msg.get("ready"):
            self.ready = True
            self.last_error = None
        else:
            self.last_error = msg.get("error") or "El worker no reportó readiness."

    def _drain_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self.stderr_tail.append(line.rstrip("\n"))

    def _read_until_msg(self) -> dict:
        """Lee stdout hasta encontrar una línea de protocolo (prefijo SENTINEL)."""
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if line.startswith(SENTINEL):
                return json.loads(line[len(SENTINEL):].strip())
            # Líneas no-protocolo (fugas) se ignoran.
        raise RuntimeError(
            f"El worker '{self.algo}' cerró stdout inesperadamente.\n"
            "--- stderr ---\n" + "\n".join(self.stderr_tail)
        )

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ── ejecución ──────────────────────────────────────────────────────────

    def run(self, req: dict) -> dict:
        """Despacha UNA corrida y devuelve la respuesta serializada del worker."""
        with self.lock:
            if not self.is_alive():
                self.start()
            if not self.is_alive():
                return {"ok": False, "error": self.last_error or "Worker caído."}
            try:
                assert self.proc is not None and self.proc.stdin is not None
                self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
                return self._read_until_msg()
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"Fallo de comunicación con worker '{self.algo}': {e}",
                    "trace": "\n".join(self.stderr_tail),
                }

    def status(self) -> dict:
        return {
            "algo": self.algo,
            "alive": self.is_alive(),
            "ready": self.ready,
            "last_error": self.last_error,
        }


class WorkerPool:
    """Mantiene un worker por algoritmo, creados perezosa o eagermente."""

    def __init__(self):
        self._workers: dict[str, Worker] = {}
        self._lock = threading.Lock()

    def get(self, algo: str) -> Worker:
        with self._lock:
            w = self._workers.get(algo)
            if w is None:
                w = Worker(algo)
                self._workers[algo] = w
            return w

    def precargar(self) -> None:
        """Arranca ambos workers (los deja 'ya en ejecución')."""
        for algo in ALGO_ROOTS:
            try:
                self.get(algo)
            except Exception as e:  # noqa: BLE001
                print(f"[WorkerPool] No se pudo precargar '{algo}': {e}", file=sys.stderr)

    def health(self) -> dict:
        estados = {}
        for algo in ALGO_ROOTS:
            w = self._workers.get(algo)
            estados[algo] = w.status() if w else {"algo": algo, "alive": False, "ready": False, "last_error": None}
        return estados

    def shutdown(self) -> None:
        for w in self._workers.values():
            if w.proc and w.is_alive():
                try:
                    w.proc.terminate()
                except Exception:
                    pass


pool = WorkerPool()
