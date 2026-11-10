from typing import Optional
from itertools import product
import numpy as np
import time

from src.models.base.application import aplicacion

from src.models.base.sia import SIA
from src.models.core.solution import Solution

from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import gestor_perfilado

from src.funcs.format import fmt_k_bloques
from src.funcs.force import particiones_en_k
from src.constants.base import COLS_IDX, NET_LABEL
from src.constants.models import (
    BRUTEFORCE_KMIP_LABEL,
    BRUTEFORCE_KMIP_STRAREGY_TAG,
)


class BruteForceKMIP(SIA):
    """
    Fuerza bruta EXHAUSTIVA para el problema k-MIP — ground truth de QNodes/KGeoMIP.

    Esta clase enumera TODAS las k-particiones VÁLIDAS del subsistema sobre el MISMO espacio
    asimétrico que explora QNodes y devuelve la de Φ mínimo. Sirve como "verdad de
    terreno" para validar la correctitud de las heurísticas en sistemas pequeños.

    Espacio de búsqueda (idéntico al de QNodes, ver `q_nodes.py`)
    ───────────────────────────────────────────────────────────
    Cada bloque es un par ASIMÉTRICO `(frozenset futuros_pos, frozenset presentes_pos)`
    de posiciones locales:
      • Los N nodos FUTUROS (`indices_ncubos`) se reparten en EXACTAMENTE k bloques
        no vacíos (partición propia: cada futuro en un solo bloque, ningún bloque
        sin futuro).
      • Los n_dims nodos PRESENTES (`dims_ncubos`) se asignan de forma INDEPENDIENTE
        a los k bloques. Si `permitir_presente_vacio=True` un bloque puede quedar con
        mecanismo ∅; si es `False` se exige que cada bloque tenga ≥1 presente.

    Para cada k → `Stirling2(N, k)` particiones de futuros × `k^n_dims` asignaciones
    de presentes. Es exponencial: antes de enumerar se estima el tamaño y, si supera
    `umbral_configuraciones`, se aborta (k explícito) o se omite ese nivel (k libre).

    Evaluación de Φ (idéntica bit a bit a `QNodes._emd_bloques`)
    ───────────────────────────────────────────────────────────
    Se reconstruye el vector de N marginales: por cada bloque se llama
    `sia_subsistema.bipartir(futuros_glob, presentes_glob).distribucion_marginal()`
    y se toman las entradas de los futuros del bloque; Φ = suma L1 contra la
    distribución marginal original (= `emd_efecto` de `iit.py`, lo que devuelve
    `seleccionar_emd()` por defecto). Esto garantiza Φ_bruta == Φ_QNodes (~1e-15)
    para una misma configuración de bloques.

    Attrs expuestos para validación:
        optimos_por_k : {k: (phi, particion_formateada)} — óptimo de cada k evaluado.
        bloques_por_k : {k: bloques}                     — la k-partición cruda óptima.
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.logeador = SafeLogger(BRUTEFORCE_KMIP_STRAREGY_TAG)
        self._N: int = 0
        self._n_dims: int = 0
        self._idx: np.ndarray = np.array([], dtype=np.int8)
        self._dims: np.ndarray = np.array([], dtype=np.int8)
        self._permitir_presente_vacio: bool = False
        self._cache_bloque: dict[tuple, np.ndarray] = {}
        self.optimos_por_k: dict[int, tuple[float, str]] = {}
        self.bloques_por_k: dict[int, list] = {}

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: Optional[int] = None,
        permitir_presente_vacio: bool = False,
        umbral_configuraciones: int = 200_000,
        max_futuros: int = 9,
    ) -> Solution:
        """
        Enumera exhaustivamente las k-particiones válidas y retorna la de Φ mínimo.

        Args:
            estado_inicial, condicion, alcance, mecanismo: cadenas de bits (igual
                que el resto de estrategias SIA del proyecto).
            k: si es entero, enumera solo k-particiones de ese tamaño; si es None,
                recorre k = 2..N y devuelve el mínimo global (expone el óptimo por k
                en `self.optimos_por_k`).
            permitir_presente_vacio: si True, un bloque puede quedar con mecanismo ∅
                (espacio asimétrico completo, garantiza Φ_bruta ≤ Φ_heurística); si
                False, cada bloque debe conservar ≥1 presente.
            umbral_configuraciones: tope de configuraciones a enumerar por k. Si se
                supera, aborta (k explícito) u omite el nivel (k libre).
            max_futuros: si N supera este valor el problema es intratable y se aborta.

        Returns:
            Solution con la k-partición de Φ mínimo (label de fuerza bruta k-MIP).

        Raises:
            ValueError: si k está fuera de rango, si N excede `max_futuros`, o si el
                espacio del k solicitado supera `umbral_configuraciones`.
        """
        self._permitir_presente_vacio = permitir_presente_vacio
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        self._idx = self.sia_subsistema.indices_ncubos
        self._dims = self.sia_subsistema.dims_ncubos
        self._N = len(self._idx)
        self._n_dims = len(self._dims)

        self._cache_bloque.clear()
        self.optimos_por_k.clear()
        self.bloques_por_k.clear()

        if self._N < 2:
            dist_trivial = self.sia_dists_marginales.copy()
            return Solution(
                estrategia=BRUTEFORCE_KMIP_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist_trivial,
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion="Sistema trivial (N<2)\n",
                quiere_hablar=False,
            )

        if k is not None and (k < 2 or k > self._N):
            raise ValueError(f"k={k} fuera del rango permitido [2, {self._N}]")

        if self._N > max_futuros:
            raise ValueError(
                f"N={self._N} futuros supera max_futuros={max_futuros}; el espacio "
                f"de k-particiones es intratable para fuerza bruta. Reduzca el "
                f"subsistema o aumente max_futuros bajo su propio riesgo."
            )

        ks_a_evaluar = [k] if k is not None else list(range(2, self._N + 1))

        for kk in ks_a_evaluar:
            n_config = self._estimar_configuraciones(kk)
            if n_config > umbral_configuraciones:
                msg = (
                    f"k={kk}: espacio de {n_config:,} configuraciones supera el "
                    f"umbral {umbral_configuraciones:,}"
                )
                if k is not None:
                    raise ValueError(
                        msg + " — aumente umbral_configuraciones o reduzca el sistema."
                    )
                self.logeador.warn(msg + " — nivel omitido.")
                continue

            self.logeador.info(
                f"k={kk}: enumerando {n_config:,} configuraciones "
                f"(Stirling2({self._N},{kk}) × {kk}^{self._n_dims})..."
            )
            phi_kk, bloques_kk = self._buscar_optimo_k(kk)
            if bloques_kk is None:
                self.logeador.warn(
                    f"k={kk}: sin k-particiones válidas "
                    f"(¿presente ∅ deshabilitado con n_dims<{kk}?)."
                )
                continue

            self.optimos_por_k[kk] = (
                phi_kk,
                fmt_k_bloques(bloques_kk, self._idx, self._dims),
            )
            self.bloques_por_k[kk] = bloques_kk
            self.logeador.info(f"k={kk}: Φ óptimo = {phi_kk:.6f}")

        if not self.bloques_por_k:
            raise ValueError(
                "No se evaluó ninguna k válida (todos los niveles superaron el "
                "umbral o no admitían partición). Ajuste umbral/flag o el subsistema."
            )

        mejor_k = min(
            self.bloques_por_k, key=lambda kk: self.optimos_por_k[kk][0]
        )
        mejor_phi = self.optimos_por_k[mejor_k][0]
        mejor_bloques = self.bloques_por_k[mejor_k]

        dist_reconstruida = np.empty(self._N, dtype=np.float32)
        for fut_pos, pre_pos in mejor_bloques:
            if not fut_pos:
                continue
            dist_bloque = self._dist_bloque(fut_pos, pre_pos)
            for p in fut_pos:
                dist_reconstruida[p] = float(dist_bloque[p])

        return Solution(
            estrategia=BRUTEFORCE_KMIP_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_reconstruida,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_k_bloques(mejor_bloques, self._idx, self._dims),
            quiere_hablar=False,
        )

    # ── Enumeración exhaustiva ─────────────────────────────────────────────

    def _buscar_optimo_k(self, k: int) -> "tuple[float, Optional[list]]":
        """
        Recorre TODAS las k-particiones asimétricas válidas y retorna la de Φ mínimo.

        Para cada partición de los N futuros en k bloques no vacíos, recorre todas
        las asignaciones de los n_dims presentes a los k bloques (k^n_dims). Con
        `permitir_presente_vacio=False` descarta las asignaciones que dejen algún
        bloque sin presente.
        """
        futuros = list(range(self._N))
        mejor_phi = float("inf")
        mejor_bloques: Optional[list] = None

        for part_fut in particiones_en_k(futuros, k):
            bloques_fut = [frozenset(b) for b in part_fut]

            # Sin presentes: una sola configuración (todos los bloques con ∅).
            if self._n_dims == 0:
                bloques = [(bloques_fut[i], frozenset()) for i in range(k)]
                phi = self._emd_bloques(bloques)
                if phi < mejor_phi:
                    mejor_phi, mejor_bloques = phi, bloques
                continue

            for asignacion in product(range(k), repeat=self._n_dims):
                pre_sets: list[list[int]] = [[] for _ in range(k)]
                for pos_pre, bloque_idx in enumerate(asignacion):
                    pre_sets[bloque_idx].append(pos_pre)

                if not self._permitir_presente_vacio and any(
                    len(s) == 0 for s in pre_sets
                ):
                    continue

                bloques = [
                    (bloques_fut[i], frozenset(pre_sets[i])) for i in range(k)
                ]
                phi = self._emd_bloques(bloques)
                if phi < mejor_phi:
                    mejor_phi, mejor_bloques = phi, bloques

        return mejor_phi, mejor_bloques

    def _estimar_configuraciones(self, k: int) -> int:
        """Estima el nº de configuraciones a enumerar para un k: S2(N,k)·k^n_dims."""
        return self._stirling2(self._N, k) * (k ** self._n_dims)

    @staticmethod
    def _stirling2(n: int, k: int) -> int:
        """Número de Stirling de segunda especie S(n, k) (DP iterativo)."""
        if k < 0 or k > n:
            return 0
        if k == 0:
            return 1 if n == 0 else 0
        fila = [0] * (k + 1)
        fila[0] = 1
        for _ in range(1, n + 1):
            nueva = [0] * (k + 1)
            for j in range(1, k + 1):
                nueva[j] = j * fila[j] + fila[j - 1]
            fila = nueva
        return fila[k]

    # ── Evaluación EMD (réplica EXACTA de QNodes para Φ idéntico) ──────────

    def _emd_bloques(self, bloques: "list[tuple[frozenset, frozenset]]") -> float:
        """
        Φ de una k-partición asimétrica — idéntico a `QNodes._emd_bloques`.

        Reconstruye el vector de N marginales (futuro de cada bloque condicionado
        por su propio presente) y devuelve la suma L1 contra la distribución
        original. Esta L1 es la Wasserstein-1 EXACTA con métrica de Hamming sobre
        productos de marginales (ver docstring de `q_nodes.py`).
        """
        dist_rec = np.empty(self._N, dtype=np.float64)
        for fut_pos, pre_pos in bloques:
            if not fut_pos:
                continue
            dist_bloque = self._dist_bloque(fut_pos, pre_pos)
            for p in fut_pos:
                dist_rec[p] = float(dist_bloque[p])
        return float(np.sum(np.abs(dist_rec - self.sia_dists_marginales)))

    def _dist_bloque(self, fut_pos: frozenset, pre_pos: frozenset) -> np.ndarray:
        """
        Distribución marginal del bloque (futuros, presentes), memoizada.

        Réplica exacta de `QNodes._dist_bloque`: el futuro del bloque (índices
        globales `indices_ncubos[p]`) se condiciona sobre su mecanismo (índices
        globales `dims_ncubos[p]`), que puede diferir del propio futuro (corte
        asimétrico) o estar vacío (mecanismo ∅).
        """
        clave = (fut_pos, pre_pos)
        cache = self._cache_bloque.get(clave)
        if cache is None:
            futuros = np.fromiter(
                (self._idx[p] for p in sorted(fut_pos)),
                dtype=np.int8,
                count=len(fut_pos),
            )
            presentes_pos = [p for p in sorted(pre_pos) if p < self._n_dims]
            presentes = (
                np.fromiter(
                    (self._dims[p] for p in presentes_pos),
                    dtype=np.int8,
                    count=len(presentes_pos),
                )
                if presentes_pos
                else np.array([], dtype=np.int8)
            )
            cache = (
                self.sia_subsistema.bipartir(futuros, presentes)
                .distribucion_marginal()
            )
            self._cache_bloque[clave] = cache
        return cache
