"""
QNodes — k-MIP heurístico mediante cortes ASIMÉTRICOS y greedy top-down.

El problema k-MIP (Partición de Mínima Información) requiere dividir un sistema
de N nodos en k subconjuntos minimizando la pérdida de información integrada
Phi (Φ). La búsqueda exhaustiva sobre B(N) particiones es intratable para
N ≥ 15, por lo que QNodes usa una estrategia greedy con refinamiento local.

Representación ASIMÉTRICA de bloques (clave para coherencia entre k)
───────────────────────────────────────────────────────────────────
Cada bloque es un par (futuros, presentes) de posiciones locales que se
particionan de forma INDEPENDIENTE:

  • futuros  (t+1): el alcance/efecto que el bloque produce.
  • presentes (t):  el mecanismo/causa que el bloque conserva como condicionante.

A diferencia de un corte SIMÉTRICO (donde el presente de un bloque es siempre
futuros ∩ dims, es decir cada grupo solo condiciona sobre sus propios nodos),
el corte asimétrico permite que un nodo aislado en su futuro siga actuando como
condicionante causal de otro bloque. Esto replica exactamente la lógica de
GeoMIP y evita el "sobre-corte" que infla Φ.

Motivación (medido en N10A, estado=1000…0):
  - Corte simétrico:  k=2 → 0.4746, k=3 → 2.5059  (salto incoherente)
  - Corte asimétrico: k=2 → 0.4746, k=3 → 0.9590  (monótono, coherente)

El caso k=2 asimétrico — aislar Xi con mecanismo ∅ y dejar al residual con TODOS
los presentes — ya reproducía el Φ de GeoMIP; aquí se generaliza a todo k.

Algoritmo
─────────
1. Pool de cortes O(N): por cada nodo i se generan
     ({i}, {pre_i})          aislamiento simétrico,
     (resto, resto_pre)      su complemento,
     ({i}, ∅)                aislamiento con mecanismo vacío (corte GeoMIP).
2. Greedy top-down: se parte del bloque único (TODOS los futuros, TODOS los
   presentes) y se aplican k-1 mejores divisiones usando el pool. Un solo
   descenso registra Φ para CADA k (jerarquía nido → coherencia garantizada).
3. Refinamiento local best-improvement sobre bloques:
     - movimiento futuro:  reubica un nodo futuro entre bloques,
     - movimiento presente (asimétrico): reubica el mecanismo de un nodo sin
       mover su futuro — imposible en representaciones simétricas.
4. ILS — Búsqueda Local Iterada: perturba y re-refina el k ganador.

Métrica EMD (suma L1 marginal — EXACTA, no aproximada):
   La reconstrucción de toda k-partición es un producto de marginales por nodo
   (independencia condicional garantizada por construcción), y la distribución
   original se modela igual. Para DOS distribuciones producto con métrica base
   de Hamming, la Wasserstein-1 (EMD real) coincide EXACTAMENTE con la suma de
   diferencias L1 marginales, porque la distancia de Hamming es separable por
   coordenada y el acoplamiento óptimo factoriza. Verificado numéricamente:
   |emd_causal − L1| < 1e-14 para N=2..12. Por tanto se usa L1 directo, que es
   O(N) en lugar de O(4^N) del pyemd, dando el MISMO Φ (p.ej. k=2 N10A = 0.4746,
   idéntico a GeoMIP) sin restricción de tamaño en N.

Memoización:
  • _cache_bloque[(futuros, presentes)]: distribución marginal del bloque.
  Los cachés se limpian entre ejecuciones independientes.
"""
import random as _random_module
import time
from typing import Generator, Optional

import numpy as np

from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import gestor_perfilado, profile
from src.funcs.format import fmt_k_bloques
from src.funcs.iit import HAMMING_EMD_MAX_N
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_LABEL,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import COLS_IDX, NET_LABEL, TYPE_TAG
from src.models.base.application import aplicacion


PERMITIR_PRESENTE_VACIO_POR_DEFECTO: bool = False

# Iteraciones base de Búsqueda Local Iterada (escala según N en _refinar_con_ils).
_N_ILS: int = 4

# Iteraciones de refinamiento por nivel en k=None (el ILS final refina el ganador).
_MAX_ITER_NIVEL: int = 5


# Un bloque asimétrico: (frozenset futuros_pos, frozenset presentes_pos).
Bloque = "tuple[frozenset, frozenset]"


def _bits_activos(mascara: int) -> Generator[int, None, None]:
    """Genera los índices de bits activos de `mascara` en orden ascendente."""
    m = mascara
    while m:
        bit = m & (-m)
        yield bit.bit_length() - 1
        m ^= bit


class QNodes(SIA):
    """
    Estrategia QNodes para k-MIP mediante cortes asimétricos y greedy top-down.

    Cada bloque mantiene futuros y presentes independientes, lo que permite
    cortes asimétricos coherentes para todo k (ver docstring del módulo). El
    greedy top-down construye una jerarquía nido (un Φ por cada k en un solo
    descenso) que luego se refina con búsqueda local best-improvement (1-move
    futuro + 1-move presente) e ILS.

    Attrs:
        _cache_bloque : (futuros, presentes) → distribución marginal del bloque.
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self._N: int = 0
        self._n_dims: int = 0
        self.logger = SafeLogger(QNODES_STRAREGY_TAG)
        self._permitir_presente_vacio: bool = False
        self._cache_bloque: dict[tuple, np.ndarray] = {}
        self._idx: np.ndarray = np.array([], dtype=np.int8)
        self._dims: np.ndarray = np.array([], dtype=np.int8)

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: Optional[int] = None,
        permitir_presente_vacio: bool = PERMITIR_PRESENTE_VACIO_POR_DEFECTO,
    ) -> Solution:
        self._permitir_presente_vacio = permitir_presente_vacio
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        self._idx = self.sia_subsistema.indices_ncubos
        self._dims = self.sia_subsistema.dims_ncubos
        self._N = len(self._idx)
        self._n_dims = len(self._dims)

        # Limpiar caché entre ejecuciones independientes
        self._cache_bloque.clear()

        if self._N < 2:
            dist_trivial = self.sia_dists_marginales.copy()
            return Solution(
                estrategia=QNODES_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist_trivial,
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion="Sistema trivial (N<2)\n",
            )

        if k is not None and (k < 2 or k > self._N):
            raise ValueError(f"k={k} fuera del rango permitido [2, {self._N}]")

        pool = self._construir_pool_cortes()

        if k is not None:
            # k especificado: greedy hasta k, refinamiento local, ILS final.
            mejor_phi, mejor_bloques = self._greedy_bloques(pool, k)
            mejor_phi, mejor_bloques = self._refinar_bloques(mejor_bloques, mejor_phi)
            mejor_phi, mejor_bloques = self._refinar_con_ils(mejor_bloques, mejor_phi)
        else:
            # k libre: descenso greedy completo (un Φ por cada k), refinamiento
            # ligero por nivel, y se elige el k ≥ 3 con menor Φ para el ILS final.
            historico = self._greedy_descenso(pool)

            historico_refinado: dict[int, tuple[float, list]] = {}
            for k_nivel, (phi_nivel, bloques_nivel) in historico.items():
                if k_nivel < 2:
                    continue
                phi_r, bloques_r = self._refinar_bloques(
                    bloques_nivel, phi_nivel, max_iter=_MAX_ITER_NIVEL
                )
                historico_refinado[k_nivel] = (phi_r, bloques_r)

            candidatos_k3 = {kk: ph for kk, (ph, _) in historico_refinado.items() if kk >= 3}
            if not candidatos_k3:
                candidatos_k3 = {kk: ph for kk, (ph, _) in historico_refinado.items()}
            mejor_k = min(candidatos_k3, key=candidatos_k3.get)
            mejor_phi, mejor_bloques = historico_refinado[mejor_k]
            mejor_phi, mejor_bloques = self._refinar_con_ils(mejor_bloques, mejor_phi)

        # Reconstrucción de la distribución de la partición óptima.
        dist_reconstruida = np.empty(self._N, dtype=np.float32)
        for fut_pos, pre_pos in mejor_bloques:
            if not fut_pos:
                continue
            dist_bloque = self._dist_bloque(fut_pos, pre_pos)
            for p in fut_pos:
                dist_reconstruida[p] = float(dist_bloque[p])

        fmt_mip = fmt_k_bloques(mejor_bloques, self._idx, self._dims)

        return Solution(
            estrategia=QNODES_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_reconstruida,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    # ── Pool de cortes asimétricos ─────────────────────────────────────────

    def _construir_pool_cortes(self) -> "list[tuple[frozenset, frozenset]]":
        """
        Construye el pool de cortes candidatos a partir de la estructura por nodo.

        Tres familias por cada nodo i (posición local):
          1. ({i}, {pre_i})            aislamiento simétrico (futuro+su mecanismo).
          2. (resto, resto_pre)        su complemento.
          3. ({i}, ∅)                  aislamiento con mecanismo vacío (corte GeoMIP):
             al aplicarse a un bloque B produce inside=({i}, ∅) y outside con el
             mecanismo de i intacto — i sigue condicionando causalmente al resto.

        Para K=2 una de estas familias reproduce el corte MIP exacto en la
        práctica totalidad de los casos; para K≥3 el greedy las encadena y el
        refinamiento las pule.
        """
        all_fut = frozenset(range(self._N))
        all_pre = frozenset(range(self._n_dims))
        pool: list = []
        vistos: set = set()

        def _add(eff: frozenset, pre: frozenset) -> None:
            if not eff:
                return
            clave = (eff, pre)
            if clave not in vistos:
                vistos.add(clave)
                pool.append(clave)

        for i in range(self._N):
            eff = frozenset((i,))
            pre = frozenset((i,)) if i < self._n_dims else frozenset()
            _add(eff, pre)
            _add(all_fut - eff, all_pre - pre)
            _add(eff, frozenset())  # corte de mecanismo vacío

        return pool

    # ── Greedy top-down sobre bloques asimétricos ──────────────────────────

    def _mejor_split_bloques(
        self,
        bloques: "list[tuple[frozenset, frozenset]]",
        pool: "list[tuple[frozenset, frozenset]]",
    ) -> "Optional[tuple[float, list]]":
        """
        Halla la mejor división de un único bloque sobre todas las (bloque, corte).

        Para cada bloque b y cada corte c calcula:
            inside  = (b.fut ∩ c.fut, b.pre ∩ c.pre)
            outside = (b.fut − c.fut, b.pre − c.pre)
        Se exige que AMBOS lados conserven al menos un futuro (partición limpia
        de los N nodos futuros); el presente puede quedar asimétrico o vacío.
        """
        mejor: list | None = None
        mejor_phi = float("inf")

        for pos, (eff_b, pre_b) in enumerate(bloques):
            if len(eff_b) <= 1 and len(pre_b) == 0:
                continue
            for cut_eff, cut_pre in pool:
                in_eff = eff_b & cut_eff
                out_eff = eff_b - cut_eff
                if not in_eff or not out_eff:
                    continue  # cada bloque debe conservar ≥1 futuro
                inside = (in_eff, pre_b & cut_pre)
                outside = (out_eff, pre_b - cut_pre)
                cfg = bloques[:pos] + [inside, outside] + bloques[pos + 1:]
                phi = self._emd_bloques(cfg)
                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor = cfg

        if mejor is None:
            return None
        return mejor_phi, mejor

    def _greedy_bloques(
        self,
        pool: "list[tuple[frozenset, frozenset]]",
        k: int,
    ) -> "tuple[float, list]":
        """Greedy top-down deteniéndose al alcanzar k bloques."""
        bloques: list = [(frozenset(range(self._N)), frozenset(range(self._n_dims)))]
        phi = self._emd_bloques(bloques)
        while len(bloques) < k:
            resultado = self._mejor_split_bloques(bloques, pool)
            if resultado is None:
                break
            phi, bloques = resultado
        return phi, bloques

    def _greedy_descenso(
        self,
        pool: "list[tuple[frozenset, frozenset]]",
    ) -> "dict[int, tuple[float, list]]":
        """
        Descenso greedy completo de k=1 a k=N registrando Φ en cada nivel.

        Un único descenso produce una jerarquía NIDO: cada k surge de dividir
        un bloque del nivel anterior, lo que garantiza coherencia (Φ monótono
        no decreciente) entre k consecutivos.
        """
        bloques: list = [(frozenset(range(self._N)), frozenset(range(self._n_dims)))]
        historico: dict[int, tuple[float, list]] = {1: (self._emd_bloques(bloques), list(bloques))}
        while len(bloques) < self._N:
            resultado = self._mejor_split_bloques(bloques, pool)
            if resultado is None:
                break
            phi, bloques = resultado
            historico[len(bloques)] = (phi, list(bloques))
        return historico

    # ── Refinamiento local best-improvement (1-move futuro + presente) ─────

    def _refinar_bloques(
        self,
        bloques: "list[tuple[frozenset, frozenset]]",
        phi: float,
        max_iter: int = 20,
    ) -> "tuple[float, list]":
        """
        Refinamiento best-improvement sobre bloques asimétricos.

        En cada ronda evalúa TODOS los vecinos y aplica el globalmente mejor:
          - movimiento futuro:  traslada un nodo futuro del bloque i al j.
          - movimiento presente: traslada el mecanismo de un nodo del bloque i
            al j SIN mover su futuro (exclusivo del esquema asimétrico).
        Repite hasta convergencia o max_iter rondas.
        """
        bloques = [(frozenset(e), frozenset(p)) for e, p in bloques]

        for _ in range(max_iter):
            mejor: list | None = None
            mejor_phi = phi
            k = len(bloques)

            # Movimientos futuros
            for i, (eff_i, pre_i) in enumerate(bloques):
                if len(eff_i) <= 1:
                    continue  # no vaciar el futuro de un bloque
                for nodo in eff_i:
                    for j in range(k):
                        if i == j:
                            continue
                        eff_j, pre_j = bloques[j]
                        cfg = list(bloques)
                        cfg[i] = (eff_i - {nodo}, pre_i)
                        cfg[j] = (eff_j | {nodo}, pre_j)
                        phi_cand = self._emd_bloques(cfg)
                        if phi_cand < mejor_phi - 1e-10:
                            mejor_phi = phi_cand
                            mejor = cfg

            # Movimientos presentes (asimétrico)
            for i, (eff_i, pre_i) in enumerate(bloques):
                for nodo in pre_i:
                    for j in range(k):
                        if i == j:
                            continue
                        eff_j, pre_j = bloques[j]
                        cfg = list(bloques)
                        cfg[i] = (eff_i, pre_i - {nodo})
                        cfg[j] = (eff_j, pre_j | {nodo})
                        phi_cand = self._emd_bloques(cfg)
                        if phi_cand < mejor_phi - 1e-10:
                            mejor_phi = phi_cand
                            mejor = cfg

            if mejor is None:
                break
            bloques = mejor
            phi = mejor_phi

        return phi, bloques

    # ── Perturbación + ILS ─────────────────────────────────────────────────

    def _perturbar_bloques(
        self,
        bloques: "list[tuple[frozenset, frozenset]]",
        n_movimientos: int = 2,
        semilla: int = 42,
    ) -> "list[tuple[frozenset, frozenset]]":
        """
        Perturba una configuración de bloques alternando movimientos futuros y
        presentes aleatorios, garantizando que ningún bloque quede sin futuro.
        """
        rng = _random_module.Random(semilla)
        result = [(frozenset(e), frozenset(p)) for e, p in bloques]
        k = len(result)
        if k < 2:
            return result

        for _ in range(n_movimientos):
            tipo = rng.randint(0, 1)
            if tipo == 0:  # movimiento futuro
                candidatos = [i for i, (e, _) in enumerate(result) if len(e) > 1]
                if not candidatos:
                    continue
                i = rng.choice(candidatos)
                eff_i, pre_i = result[i]
                nodo = rng.choice(sorted(eff_i))
                j = rng.choice([x for x in range(k) if x != i])
                eff_j, pre_j = result[j]
                result[i] = (eff_i - {nodo}, pre_i)
                result[j] = (eff_j | {nodo}, pre_j)
            else:  # movimiento presente (asimétrico)
                candidatos = [i for i, (_, p) in enumerate(result) if len(p) > 0]
                if not candidatos:
                    continue
                i = rng.choice(candidatos)
                eff_i, pre_i = result[i]
                nodo = rng.choice(sorted(pre_i))
                j = rng.choice([x for x in range(k) if x != i])
                eff_j, pre_j = result[j]
                result[i] = (eff_i, pre_i - {nodo})
                result[j] = (eff_j, pre_j | {nodo})

        return result

    def _refinar_con_ils(
        self,
        bloques: "list[tuple[frozenset, frozenset]]",
        phi: float,
    ) -> "tuple[float, list]":
        """
        ILS: refinamiento best-improvement + ciclos N-adaptativos de perturbación.

        n_ils y max_iter decrecen con N porque cada evaluación EMD es O(2^N) para
        N ≤ HAMMING_EMD_MAX_N: pocos ciclos de calidad superan muchos mediocres.
          max_iter = max(5, 20 - max(0, N - HAMMING_EMD_MAX_N))
          n_ils    = max(1, _N_ILS - max(0, (N - 16) // 2))
        """
        max_it = max(5, 20 - max(0, self._N - HAMMING_EMD_MAX_N))
        n_ils = max(1, _N_ILS - max(0, (self._N - 16) // 2))

        mejor_phi, mejor_bloques = self._refinar_bloques(bloques, phi, max_iter=max_it)

        n_mov = max(1, self._N // 4)
        for iter_ils in range(n_ils):
            perturbado = self._perturbar_bloques(
                mejor_bloques, n_movimientos=n_mov, semilla=42 + iter_ils * 17
            )
            phi_p = self._emd_bloques(perturbado)
            phi_r, bloques_r = self._refinar_bloques(perturbado, phi_p, max_iter=max_it)
            if phi_r < mejor_phi - 1e-9:
                mejor_phi = phi_r
                mejor_bloques = bloques_r

        return mejor_phi, mejor_bloques

    # ── Evaluación EMD de una partición de bloques ─────────────────────────

    def _emd_bloques(self, bloques: "list[tuple[frozenset, frozenset]]") -> float:
        """
        EMD total de una k-partición asimétrica de bloques.

        Reconstruye la distribución marginal a partir de las distribuciones de
        cada bloque (futuro condicionado por su propio presente) y la compara
        con la original mediante suma L1 marginal. Esta L1 es la Wasserstein-1
        EXACTA con métrica de Hamming, porque ambas distribuciones son productos
        de marginales (ver docstring del módulo): O(N) en vez de O(4^N).
        """
        dist_rec = np.empty(self._N, dtype=np.float64)
        for fut_pos, pre_pos in bloques:
            if not fut_pos:
                continue
            dist_bloque = self._dist_bloque(fut_pos, pre_pos)
            for p in fut_pos:
                dist_rec[p] = float(dist_bloque[p])

        return float(np.sum(np.abs(dist_rec - self.sia_dists_marginales)))

    # ── Memoización de distribuciones de bloque ────────────────────────────

    def _dist_bloque(self, fut_pos: frozenset, pre_pos: frozenset) -> np.ndarray:
        """
        Distribución marginal del bloque (futuros, presentes), calculada una vez.

        El futuro del bloque (índices globales indices_ncubos[p]) se condiciona
        sobre el mecanismo del bloque (índices globales dims_ncubos[p]), que puede
        diferir del propio futuro (corte asimétrico) o estar vacío (mecanismo ∅).
        """
        clave = (fut_pos, pre_pos)
        cache = self._cache_bloque.get(clave)
        if cache is None:
            futuros = np.fromiter(
                (self._idx[p] for p in sorted(fut_pos)), dtype=np.int8, count=len(fut_pos)
            )
            presentes_pos = [p for p in sorted(pre_pos) if p < self._n_dims]
            presentes = (
                np.fromiter((self._dims[p] for p in presentes_pos), dtype=np.int8,
                            count=len(presentes_pos))
                if presentes_pos
                else np.array([], dtype=np.int8)
            )
            cache = (
                self.sia_subsistema
                .bipartir(futuros, presentes)
                .distribucion_marginal()
            )
            self._cache_bloque[clave] = cache
        return cache


# Alias de compatibilidad con exec.py (que importa DynamicPartition)
DynamicPartition = QNodes
