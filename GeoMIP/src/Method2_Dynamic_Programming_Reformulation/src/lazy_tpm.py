# -*- coding: utf-8 -*-
"""
lazy_tpm.py — Carga lazy de TPM para sistemas grandes (N ≥ 20)

Reemplaza np.genfromtxt() con un lector por chunks que nunca
carga el archivo completo en memoria.

Uso básico:
    from lazy_tpm import LazyTPM

    tpm = LazyTPM("ruta/N25A.csv")
    print(tpm.info())

    # Iterar chunk a chunk
    for chunk in tpm.chunks():
        procesar(chunk)          # np.ndarray (chunk_size x N)

    # Acceder a filas específicas (por índice de estado)
    fila = tpm.get_estado(estado_idx=1048576)  # np.ndarray (N,)

    # Marginalizar una columna (nodo) sin cargar todo
    dist = tpm.marginal_nodo(nodo_idx=3)       # np.ndarray (2^N,)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────

# Tamaño de chunk por defecto según N (potencias de 2)
_CHUNK_TABLE = {
    range(0,  17): 256,       # N ≤ 16  →  256 estados
    range(17, 21): 512,       # N 17–20 →  512 estados
    range(21, 24): 1_024,     # N 21–23 → 1024 estados
    range(24, 27): 2_048,     # N 24–26 → 2048 estados
}
_CHUNK_DEFAULT = 4_096        # N ≥ 27


def _chunk_size_para(n: int) -> int:
    for rango, size in _CHUNK_TABLE.items():
        if n in rango:
            return size
    return _CHUNK_DEFAULT


# ─────────────────────────────────────────────────────────────
#  Clase principal
# ─────────────────────────────────────────────────────────────

class LazyTPM:
    """
    Lector lazy de TPM en formato CSV (estado-nodo).

    No carga el archivo en RAM. Lee únicamente las filas
    necesarias en el momento en que se solicitan.

    Attributes
    ----------
    ruta        : Path al archivo CSV
    n_nodos     : número de variables del sistema (columnas)
    n_estados   : total de filas (2^N esperado)
    chunk_size  : filas por bloque de lectura
    """

    def __init__(self, ruta: str | Path, chunk_size: Optional[int] = None):
        self.ruta = Path(ruta)
        if not self.ruta.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.ruta}")

        # Inferir N desde la primera fila (muy rápido)
        primera = pd.read_csv(self.ruta, header=None, nrows=1)
        self.n_nodos: int = primera.shape[1]

        # Contar filas sin cargar el archivo (lectura de texto pura)
        print(f"  Contando filas de {self.ruta.name}...", end=" ", flush=True)
        self.n_estados: int = self._contar_filas()
        print(f"{self.n_estados:,} estados")

        n_esperado = 2 ** self.n_nodos
        if self.n_estados != n_esperado:
            print(f"  ⚠ Advertencia: se esperaban {n_esperado:,} filas "
                  f"para N={self.n_nodos}, hay {self.n_estados:,}.")

        self.chunk_size: int = chunk_size or _chunk_size_para(self.n_nodos)
        self.n_chunks: int = math.ceil(self.n_estados / self.chunk_size)

    # ── Metadatos ────────────────────────────────────────────

    def info(self) -> str:
        tam_gb = (self.n_estados * self.n_nodos * 8) / (1024 ** 3)
        return (
            f"\n{'─'*50}\n"
            f"  Archivo   : {self.ruta.name}\n"
            f"  N nodos   : {self.n_nodos}\n"
            f"  Estados   : {self.n_estados:,}  (2^{self.n_nodos})\n"
            f"  Tamaño    : {tam_gb:.2f} GB (si se cargara completo)\n"
            f"  Chunk size: {self.chunk_size:,} estados\n"
            f"  N chunks  : {self.n_chunks:,}\n"
            f"{'─'*50}"
        )

    # ── Iteración por chunks ─────────────────────────────────

    def chunks(self, inicio: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """
        Generador que yields (chunk_idx, array) de forma lazy.
        Nunca más de `chunk_size` filas en RAM.

        Parameters
        ----------
        inicio : índice de chunk desde donde empezar (default 0)

        Yields
        ------
        (chunk_idx, np.ndarray de shape (chunk_size, n_nodos))
        """
        skip = inicio * self.chunk_size
        chunk_idx = inicio

        for df in pd.read_csv(
            self.ruta,
            header=None,
            chunksize=self.chunk_size,
            skiprows=range(0, skip + 1) if skip > 0 else None,
            dtype=np.float32,      # float32 en vez de float64: mitad de RAM
        ):
            yield chunk_idx, df.to_numpy()
            chunk_idx += 1

    # ── Acceso a estado individual ───────────────────────────

    def get_estado(self, estado_idx: int) -> np.ndarray:
        """
        Lee UNA fila (un estado) del CSV sin cargar nada más.
        Útil para acceder al estado inicial sin leer todo el archivo.

        Parameters
        ----------
        estado_idx : índice entero del estado (0-based)

        Returns
        -------
        np.ndarray de shape (n_nodos,)
        """
        if not (0 <= estado_idx < self.n_estados):
            raise IndexError(f"Estado {estado_idx} fuera de rango [0, {self.n_estados-1}]")

        df = pd.read_csv(
            self.ruta,
            header=None,
            skiprows=estado_idx,
            nrows=1,
            dtype=np.float32,
        )
        return df.to_numpy().flatten()

    # ── Marginalización por nodo ─────────────────────────────

    def marginal_nodo(self, nodo_idx: int) -> np.ndarray:
        """
        Extrae la columna de un nodo específico procesando chunk a chunk.
        Equivale a tpm[:, nodo_idx] pero sin cargar la matriz completa.

        Parameters
        ----------
        nodo_idx : índice de columna (0-based)

        Returns
        -------
        np.ndarray de shape (n_estados,) con las probabilidades del nodo
        """
        if not (0 <= nodo_idx < self.n_nodos):
            raise IndexError(f"Nodo {nodo_idx} fuera de rango [0, {self.n_nodos-1}]")

        resultado = np.empty(self.n_estados, dtype=np.float32)
        for chunk_idx, arr in self.chunks():
            inicio = chunk_idx * self.chunk_size
            fin = inicio + arr.shape[0]
            resultado[inicio:fin] = arr[:, nodo_idx]

        return resultado

    # ── Estadísticas rápidas ─────────────────────────────────

    def stats_chunk(self, chunk_idx: int) -> dict:
        """
        Calcula min/max/mean de un chunk sin cargarlo todo.
        Útil para inspección rápida de un bloque.
        """
        skip = chunk_idx * self.chunk_size
        df = pd.read_csv(
            self.ruta,
            header=None,
            skiprows=skip,
            nrows=self.chunk_size,
            dtype=np.float32,
        )
        arr = df.to_numpy()
        return {
            "chunk_idx"  : chunk_idx,
            "estado_ini" : skip,
            "estado_fin" : skip + len(arr) - 1,
            "shape"      : arr.shape,
            "min"        : float(arr.min()),
            "max"        : float(arr.max()),
            "mean"       : float(arr.mean()),
            "n_zeros"    : int((arr == 0).sum()),
            "n_unos"     : int((arr == 1).sum()),
        }

    # ── Interno ──────────────────────────────────────────────

    def _contar_filas(self) -> int:
        """Cuenta líneas del CSV con lectura de buffer eficiente."""
        count = 0
        with open(self.ruta, 'rb') as f:
            buf = f.read(1 << 20)   # leer en bloques de 1 MB
            while buf:
                count += buf.count(b'\n')
                buf = f.read(1 << 20)
        return count

    def __repr__(self) -> str:
        return (f"LazyTPM(n_nodos={self.n_nodos}, "
                f"n_estados={self.n_estados:,}, "
                f"chunk_size={self.chunk_size})")


# ─────────────────────────────────────────────────────────────
#  Función de compatibilidad — reemplaza np.genfromtxt()
# ─────────────────────────────────────────────────────────────

def cargar_tpm(ruta: Path, forzar_lazy: bool = False) -> np.ndarray | LazyTPM:
    """
    Drop-in replacement para np.genfromtxt(ruta, delimiter=',').

    Para archivos pequeños (N ≤ 18) carga en memoria como antes.
    Para archivos grandes (N > 18) retorna un LazyTPM en su lugar.

    Parameters
    ----------
    ruta         : ruta al CSV
    forzar_lazy  : si True, siempre retorna LazyTPM sin importar el tamaño

    Returns
    -------
    np.ndarray  → sistemas pequeños (compatible con el código existente)
    LazyTPM     → sistemas grandes (iterar con .chunks())
    """
    ruta = Path(ruta)
    primera = pd.read_csv(ruta, header=None, nrows=1)
    n_nodos = primera.shape[1]

    UMBRAL_LAZY = 18  # N > 18 → ~6.7 millones de filas, ~400 MB

    if forzar_lazy or n_nodos > UMBRAL_LAZY:
        print(f"  → N={n_nodos} detectado: usando carga lazy (chunks).")
        return LazyTPM(ruta)
    else:
        print(f"  → N={n_nodos} detectado: carga directa en memoria.")
        return np.genfromtxt(ruta, delimiter=",")