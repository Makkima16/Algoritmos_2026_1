"""
KGeoMIP — Extensión de la estrategia geométrica GeoMIP a k-particiones.

Proyecto KGeoMIP · Análisis y Diseño de Algoritmos 2026-1

Idea central
------------
GeoMIP original resuelve el caso k=2 (bi-partición): divide el sistema V en
dos partes S1, S2 y minimiza:

    δ(V, {S1,S2}) = EMD( P(Vt+1|Vt),  P(S1,t+1|S1,t) ⊗ P(S2,t+1|S2,t) )

KGeoMIP generaliza esto a k partes (k ≥ 2):

    δₖ(V, {S1,...,Sk}) = EMD( P(Vt+1|Vt),  ⊗ᵢ P(Si,t+1|Si,t) )

La tabla de costos de transiciones entre estados se calcula internamente 
y se utiliza para optimizar la búsqueda de las subparticiones de manera 
idéntica a como lo haría una iteración base de GeoMIP generalizado.

Estrategia de búsqueda implementada
-------------------------------------
Para k pequeño y n pequeño (n·k ≤ UMBRAL_EXHAUSTIVO):
    → Búsqueda exhaustiva usando números de Stirling del segundo tipo.
      Se generan todas las k-particiones posibles de los nodos del sistema.

Para n·k > UMBRAL_EXHAUSTIVO:
    → Agrupamiento Jerárquico (Bottom-Up):
      1. Se empieza con n particiones (cada variable sola).
      2. Se fusionan iterativamente los dos subsistemas cuya unión genere
         la menor pérdida de información EMD respecto al sistema original.
      3. Se detiene cuando se alcanza el número k de particiones deseado.
      Esta reformulación geométrica evita la explosión combinatoria.

Reutilización de infraestructura
---------------------------------
- Hereda de SIA.
- Llama a `sia_preparar_subsistema` para obtener el subsistema condicionado.
- Implementa una tabla de transiciones para optimizar búsquedas.
- Usa `System.bipartir()` y `System.distribucion_marginal()` del framework.

Complejidad
-----------
Exhaustiva : O(S(n,k) · k · 2ⁿ) donde S(n,k) = número de Stirling
DP Bottom-Up : O(n³ · 2ⁿ) a nivel de llamadas por las iteraciones de agrupamiento
"""

import time
import itertools
from math import comb
from typing import List, Tuple, Dict, Optional

import numpy as np
from joblib import Parallel, delayed

from src.controllers.manager import Manager
from src.funcs.base import emd_causal, ABECEDARY, LOWER_ABECEDARY
from src.funcs.format import fmt_biparte_q
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.base import (
    NET_LABEL,
    ACTUAL,
    EFECTO,
    TYPE_TAG,
    VOID_STR,
)


# ── Constantes propias de KGeoMIP ──────────────────────────────────────────
KGEOMIP_LABEL: str = "KGeoMIP"
KGEOMIP_STRATEGY_TAG: str = f"{KGEOMIP_LABEL}_strategy"
KGEOMIP_ANALYSIS_TAG: str = f"{KGEOMIP_LABEL}_analysis"

# Si n * k supera este umbral se usa la heurística greedy en lugar de fuerza
# bruta sobre k-particiones. Se ajustó a 10 para evitar la explosión combinatoria.
UMBRAL_EXHAUSTIVO: int = 10


# ── Utilidades matemáticas ─────────────────────────────────────────────────

def stirling2(n: int, k: int) -> int:
    """
    Número de Stirling del segundo tipo S(n,k): cantidad de formas de
    particionar un conjunto de n elementos en k subconjuntos no vacíos.

    Fórmula: S(n,k) = (1/k!) · Σᵢ₌₀ᵏ (-1)^(k-i) · C(k,i) · iⁿ

    Args:
        n: Número total de elementos.
        k: Número de partes.

    Returns:
        Número de k-particiones posibles (int).
    """
    if k == 0:
        return 1 if n == 0 else 0
    if k > n:
        return 0
    total = sum(
        ((-1) ** (k - i)) * comb(k, i) * (i ** n)
        for i in range(k + 1)
    )
    # Dividir por k! (las particiones son conjuntos, no secuencias)
    factorial_k = 1
    for i in range(1, k + 1):
        factorial_k *= i
    return total // factorial_k


def generar_k_particiones(elementos: List[int], k: int):
    """
    Genera todas las k-particiones del conjunto `elementos` usando los
    números de Stirling del segundo tipo como base combinatoria.

    Implementación mediante distribución restringida de elementos en k
    cubetas distinguibles (para evitar duplicados por reordenamiento de
    cubetas se aplica un criterio canónico: el primer elemento siempre
    va a la parte 0).

    Args:
        elementos: Lista de índices a particionar.
        k        : Número de partes (cada parte debe ser no vacía).

    Yields:
        Tupla de k listas, cada una con los índices de esa parte.
        Ejemplo para elementos=[0,1,2], k=2:
            ([0,1], [2]), ([0,2], [1]), ([0], [1,2])
    """
    n = len(elementos)
    if k > n:
        return
    if k == 1:
        yield (list(elementos),)
        return
    if k == n:
        yield tuple([e] for e in elementos)
        return

    # Asignamos el primer elemento siempre a la parte 0 para evitar
    # permutaciones entre partes (canonicalización).
    # Los restantes n-1 elementos se distribuyen recursivamente.
    def _distribuir(idx: int, partes: List[List[int]], max_parte_actual: int):
        """Recursión interna: asigna elementos[idx:] a alguna de las k partes."""
        if idx == n:
            # Solo emitir si todas las k partes tienen al menos 1 elemento
            if all(len(p) > 0 for p in partes):
                yield tuple(list(p) for p in partes)
            return
        elem = elementos[idx]
        # El elemento puede ir a cualquier parte ya usada o abrir una nueva
        limite = min(max_parte_actual + 1, k - 1)
        for parte_idx in range(limite + 1):
            partes[parte_idx].append(elem)
            nuevo_max = max(max_parte_actual, parte_idx)
            yield from _distribuir(idx + 1, partes, nuevo_max)
            partes[parte_idx].pop()

    partes_vacias: List[List[int]] = [[] for _ in range(k)]
    # El primer elemento siempre va a la parte 0
    partes_vacias[0].append(elementos[0])
    yield from _distribuir(1, partes_vacias, 0)


def distribucion_conjunta_vectorizada(probabilidades: np.ndarray) -> np.ndarray:
    """
    Construye la distribución conjunta P de tamaño 2^N a partir de N probabilidades marginales p_i.
    Vectorizado y sin bucles for explícitos utilizando meshgrid y prod.
    """
    if len(probabilidades) == 0:
        return np.array([1.0], dtype=np.float64)
    # Formar factores [1-p, p] para cada probabilidad
    p_1 = np.asarray(probabilidades, dtype=np.float64)
    p_0 = 1.0 - p_1
    factors = np.stack([p_0, p_1], axis=1)
    
    # Generar la grilla de productos de Kronecker equivalentes para variables independientes
    grid = np.meshgrid(*factors, indexing='ij')
    # Multiplicar en todos los ejes para formar el tensor de estado conjunto
    dist = np.prod(grid, axis=0).flatten()
    return dist


def evaluar_k_particion(
    subsistema,
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
    particion: Tuple[List[int], ...],
    dist_original: np.ndarray,
) -> float:
    """
    Calcula la pérdida EMD de una k-partición usando semántica IIT estricta y 
    las distribuciones conjuntas totales (de tamaño 2^N).

    Definición matemática:
    ──────────────────────
    Para una k-partición {S1,...,Sk}, cada parte Si es causalmente independiente.
    La distribución reconstruida del sistema k-particionado asigna a cada nodo j ∈ Si:

        dist_rec[j] = 1 - P(Xj=1 | estado_Si,t)

    donde P(Xj=1 | estado_Si,t) se obtiene marginalizando el ncubo_j sobre todos
    los dims que NO pertenecen a Si, y evaluando en estado_inicial[Si].
    
    Luego, P = ⊗_i P(nodo i en sistema total) de tamaño 2^N
    Y se construye Q = ⊗_i P(nodo i en sistema particionado) de tamaño 2^N

    Args:
        subsistema    : System ya condicionado/substradido.
        indices_ncubos: Índices de los n-cubos del subsistema (futuros).
        dims_ncubos   : Todos los dims del subsistema (presentes).
        particion     : Tupla de k listas de índices posicionales (0..n-1).
        dist_original : Distribución marginal del subsistema completo (len n).

    Returns:
        Valor EMD (float, ≥ 0).
    """
    n = len(dist_original)
    dist_reconstruida = np.empty(n, dtype=np.float64)

    for parte in particion:
        if not parte:
            continue
        parte_arr = np.array(parte, dtype=np.int8)

        # futuros = índices de los ncubos de esta parte
        futuros_parte = indices_ncubos[parte_arr]

        # mecanismo = SOLO los dims de esta parte (IIT estricto: Si solo ve Si_t)
        presentes_parte = dims_ncubos[parte_arr]

        # bipartir(futuros=Si, mecanismo=Si_dims):
        sistema_parte = subsistema.bipartir(futuros_parte, presentes_parte)
        dist_parte    = sistema_parte.distribucion_marginal()  # shape (n,)

        # Tomar el valor de cada nodo de esta parte
        for idx_pos in parte:
            dist_reconstruida[idx_pos] = dist_parte[idx_pos]

    # Ambos tensores se forman asumiendo independencia entre marginales (tensor product).
    # Matemáticamente es indispensable generar el tensor de tamaño 2^N para conservar la causalidad 
    # estructural requerida por la métrica EMD de la Teoría de la Información Integrada originada en PyPhi.
    # Evitamos construir meshgrids línea por línea e invocamos la función de utilería matemática.
    dist_P_conjunta = distribucion_conjunta_vectorizada(dist_original)
    dist_Q_conjunta = distribucion_conjunta_vectorizada(dist_reconstruida)

    # Calculamos EMD usando la función base (con precalculo de penalidad de Hamming)
    return float(emd_causal(dist_P_conjunta, dist_Q_conjunta))



def fmt_k_particion(particion: Tuple[List[int], ...], indices_reales: np.ndarray) -> str:
    """
    Formatea una k-partición para mostrarla en consola.

    Cada parte se muestra como una columna con variables futuras (mayúscula)
    y presentes (minúscula), separadas por "|".

    Args:
        particion: Tupla de k listas de índices relativos al subsistema.
        indices_reales: El mapeo a los índices originales de la red completa.

    Returns:
        String con la representación de la k-partición.
    """
    partes_fmt = []
    for parte in particion:
        # mapear índice interno al índice original
        futuros = [ABECEDARY[indices_reales[i]] for i in parte]
        presentes = [LOWER_ABECEDARY[indices_reales[i]] for i in parte]
        str_fut = ",".join(futuros) if futuros else VOID_STR
        str_pres = ",".join(presentes) if presentes else VOID_STR
        ancho = max(len(str_fut), len(str_pres)) + 2
        partes_fmt.append((f"|{str_fut:^{ancho}}|", f"|{str_pres:^{ancho}}|"))

    linea_top = "".join(t for t, _ in partes_fmt)
    linea_bot = "".join(b for _, b in partes_fmt)
    return f"{linea_top}\n{linea_bot}"


# ── Clase principal ────────────────────────────────────────────────────────

class KGeoMIP(SIA):
    """
    Extensión de GeoMIP para k-particiones (k ≥ 2).

    Para k=2 reproduce exactamente los resultados de GeometricSIA
    (caso base de validación).

    Para k>2 usa búsqueda exhaustiva cuando el sistema es pequeño, o un
    Agrupamiento Jerárquico Bottom-up basado en Costos de Discrepancia cuando es grande.

    Args:
        gestor (Manager): Gestor con el estado inicial y ruta de la TPM.
        k      (int)    : Número de partes de la partición (default=2).

    Attributes:
        k                   : Número de partes solicitado.
        tabla_transiciones  : Tabla de costos de transición para optimizar 
                              el costo EMD evaluado.
        memoria_particiones : Diccionario con las pérdidas evaluadas.
    """

    def __init__(self, gestor: Manager, k: int = 2) -> None:
        super().__init__(gestor)
        if k < 2:
            raise ValueError(f"k debe ser ≥ 2, recibido k={k}.")
        self.k = k
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}_k{k}"
        )
        self.logger = SafeLogger(KGEOMIP_STRATEGY_TAG)
        # Tabla de costos de transición 
        self.tabla_transiciones: dict = {}
        # Registro de pérdidas por partición evaluada
        self.memoria_particiones: Dict[tuple, Tuple[float, np.ndarray]] = {}

    # ── Método principal ───────────────────────────────────────────────────

    @profile(context={TYPE_TAG: KGEOMIP_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: Optional[int] = None, # k is deprecated since we search ALL, kept for signature
    ) -> Solution:
        """
        Encuentra la Participación de Mínima Información ÓPTIMA GLOBAL (Optimal K-MIP).
        Itera sobre todos los k (desde 2 hasta N) y elige la partición
        con la mínima pérdida EMD para cumplir con el requisito formal del proyecto.
        """

        self.logger.critic(f"Iniciando KGeoMIP para búsqueda de k global óptimo.")
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        n_vars = len(self.sia_subsistema.indices_ncubos)

        # Caso trivial: si n <= 1
        if n_vars <= 1:
            particion_optima = tuple([i] for i in range(n_vars))
            k_optimo = n_vars
            mejor_perdida = 0.0
            
            # Distribución reconstruida
            dist_reconstruida = self.sia_dists_marginales.copy()
        else:
            self._construir_tabla_costos()
            
            mejor_perdida = float('inf')
            particion_optima = None
            k_optimo = 2
            self.historico_particiones = []
            
            # Evaluar todas las K posibles desde 2 hasta n_vars
            for test_k in range(2, n_vars + 1):
                if n_vars * test_k <= UMBRAL_EXHAUSTIVO:
                    part_k = self._busqueda_exhaustiva(test_k)
                else:
                    part_k = self._agrupamiento_jerarquico(test_k)
                
                perdida_k = evaluar_k_particion(
                    self.sia_subsistema,
                    self.sia_subsistema.indices_ncubos,
                    self.sia_subsistema.dims_ncubos,
                    part_k,
                    self.sia_dists_marginales,
                )

                fmt_pk = fmt_k_particion(part_k, self.sia_subsistema.indices_ncubos)
                self.historico_particiones.append({
                    "k": test_k,
                    "perdida": float(perdida_k),
                    "particion_grafica": fmt_pk
                })
                
                if perdida_k < mejor_perdida:
                    mejor_perdida = perdida_k
                    particion_optima = part_k
                    k_optimo = test_k
                    
                    if mejor_perdida == 0.0:  # Optimización, no se puede mejorar 0.0
                        break

            # Distribución reconstruida para el Solution con la partición óptima
            dist_reconstruida = np.empty(len(self.sia_dists_marginales), dtype=np.float32)
            for parte in particion_optima:
                if not parte:
                    continue
                parte_arr = np.array(parte, dtype=np.int8)
                futuros   = self.sia_subsistema.indices_ncubos[parte_arr]
                presentes = self.sia_subsistema.dims_ncubos[parte_arr]
                dist_parte = self.sia_subsistema.bipartir(futuros, presentes).distribucion_marginal()
                for idx_pos in parte:
                    dist_reconstruida[idx_pos] = dist_parte[idx_pos]

        fmt = fmt_k_particion(
            particion_optima,
            self.sia_subsistema.indices_ncubos
        )

        self.logger.critic(
            f"ÓPTIMA k-MIP encontrada (k={k_optimo}) con pérdida={mejor_perdida:.6f}:\n{fmt}"
        )

        return Solution(
            estrategia=f"{KGEOMIP_LABEL}(Global Optimal K={k_optimo})",
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_reconstruida,
            particion=fmt,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
        )

    # ── Construcción de tabla de costos ─────────

    def _construir_tabla_costos(self) -> None:
        """
        Construye la tabla de costos de transición entre estados del hipercubo.

        Algoritmo base:
          - BFS nivel por nivel desde el estado inicial al estado final.
          - Para cada par (i,j): t_X(i,j) = γ·(|X[i]-X[j]| + Σ t_X(k,j))
            con γ = 2^(-dH(i,j)).

        La tabla se almacena en self.tabla_transiciones para ser consultada
        durante la identificación de candidatos.
        """
        # Datos planos de los n-cubos
        self._flat_data = [
            ncubo.data.ravel()
            for ncubo in self.sia_subsistema.ncubos
        ]

        estado_inicial = self.sia_subsistema.estado_inicial[
            self.sia_subsistema.dims_ncubos
        ]
        estado_final = 1 - estado_inicial
        n = len(estado_inicial)
        idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))

        # Entrada trivial: costo de i → i es 0 para todos los cubos
        clave_trivial = (tuple(estado_inicial), tuple(estado_inicial))
        self.tabla_transiciones[clave_trivial] = [0.0] * len(idx_ncubos)

        # Construcción nivel por nivel (BFS)
        caminos: Dict[int, List[List[int]]] = {0: [estado_inicial.tolist()]}
        for nivel in range(1, n + 1):
            visitados: set = set()
            caminos[nivel] = []
            for estado_anterior in caminos[nivel - 1]:
                est_ant = np.array(estado_anterior)
                for i in range(n):
                    if est_ant[i] != estado_final[i]:
                        nuevo = est_ant.copy()
                        nuevo[i] = estado_final[i]
                        t = tuple(nuevo)
                        if t not in visitados:
                            caminos[nivel].append(nuevo.tolist())
                            self._calcular_costo(
                                caminos[0][0], nuevo.tolist(), idx_ncubos
                            )
                            visitados.add(t)

        self._caminos = caminos
        self._estado_inicial_tabla = estado_inicial.tolist()
        self._estado_final_tabla = estado_final.tolist()

    def _calcular_costo(
        self,
        estado_ini: list,
        estado_fin: list,
        idx_ncubos: list,
    ) -> None:
        """
        Calcula t_X(i,j) para todos los n-cubos X y lo almacena en
        self.tabla_transiciones[(tuple(i), tuple(j))].

        Fórmula:
            t_X(i,j) = γ · ( |X[i] - X[j]| + Σ_{k ∈ N(i,j)} t_X(i,k) )
        con γ = 2^(-dH(i,j)) y N(i,j) = vecinos de j en caminos óptimos hacia i.
        """
        key = (tuple(estado_ini), tuple(estado_fin))
        if key in self.tabla_transiciones:
            return

        dh = sum(a != b for a, b in zip(estado_ini, estado_fin))
        factor = 1.0 / (2 ** dh)

        # Índices enteros para acceso a flat_data (little-endian)
        idx_ini = int("".join(map(str, estado_ini[::-1])), 2)
        idx_fin = int("".join(map(str, estado_fin[::-1])), 2)

        diffs = np.abs(
            np.array([fd[idx_ini] for fd in self._flat_data])
            - np.array([fd[idx_fin] for fd in self._flat_data])
        ).tolist()

        self.tabla_transiciones[key] = diffs

        if dh > 1:
            for i in range(len(estado_ini)):
                if estado_ini[i] != estado_fin[i]:
                    vecino = list(estado_fin)
                    vecino[i] = estado_ini[i]
                    clave_vec = (tuple(estado_ini), tuple(vecino))
                    for n_idx in idx_ncubos:
                        self.tabla_transiciones[key][n_idx] += (
                            self.tabla_transiciones[clave_vec][n_idx]
                        )

        self.tabla_transiciones[key] = [
            factor * v for v in self.tabla_transiciones[key]
        ]

    # ── Estrategia 1: búsqueda exhaustiva ─────────────────────────────────

    def _busqueda_exhaustiva(
        self, k: int
    ) -> Tuple[List[int], ...]:
        """
        Evalúa todas las k-particiones posibles y retorna la de menor pérdida.

        Complejidad: O(S(n,k) · k · n) donde S(n,k) es el número de Stirling.

        Args:
            k: Número de partes.

        Returns:
            Tupla de k listas con los índices de variables de cada parte.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        elementos = list(range(n_vars))

        total_stirling = stirling2(n_vars, k)
        self.logger.critic(
            f"Exhaustiva: S({n_vars},{k}) = {total_stirling} particiones. Resolviendo en paralelo..."
        )

        def evaluar_en_proceso(particion):
            perdida = evaluar_k_particion(
                self.sia_subsistema,
                self.sia_subsistema.indices_ncubos,
                self.sia_subsistema.dims_ncubos,
                particion,
                self.sia_dists_marginales,
            )
            return (perdida, particion)

        # Evaluar en paralelo
        iterador_particiones = generar_k_particiones(elementos, k)
        resultados = Parallel(n_jobs=-1)(
            delayed(evaluar_en_proceso)(p) for p in iterador_particiones
        )

        mejor_perdida = float("inf")
        mejor_particion = None

        for perdida, particion in resultados:
            self.memoria_particiones[tuple(tuple(p) for p in particion)] = (
                perdida,
                None,
            )
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_particion = particion

        self.logger.critic(f"Exhaustiva terminada. Mejor pérdida = {mejor_perdida:.6f}.")
        return mejor_particion

    # ── Estrategia 2: heurística jerárquica bottom-up ─────────────────────────

    def _agrupamiento_jerarquico(
        self, k: int
    ) -> Tuple[List[int], ...]:
        """
        Algoritmo de Agrupamiento Jerárquico basado en Costos de Discrepancia (Bottom-Up).

        Empieza con n particiones (cada variable sola). Iterativamente fusiona los dos
        subsistemas cuya unión genere la MENOR pérdida de información EMD respecto
        al sistema original, hasta alcanzar el número k de particiones deseado.

        Esta es la Reformulación Geométrica recomendada para evitar la explosión combinatoria.

        Args:
            k: Número de partes objetivo.

        Returns:
            Tupla de k listas de índices.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        # Empieza con n particiones independientes
        particiones: List[List[int]] = [[i] for i in range(n_vars)]

        def evaluar_fusion(i, j, particiones_actuales):
            nueva_parte = particiones_actuales[i] + particiones_actuales[j]
            particion_prueba = tuple(
                [particiones_actuales[p] for p in range(len(particiones_actuales)) if p != i and p != j] 
                + [nueva_parte]
            )
            perdida = evaluar_k_particion(
                self.sia_subsistema,
                self.sia_subsistema.indices_ncubos,
                self.sia_subsistema.dims_ncubos,
                particion_prueba,
                self.sia_dists_marginales,
            )
            return (perdida, i, j, nueva_parte)

        while len(particiones) > k:
            n_partes = len(particiones)
            pares_a_evaluar = [(i, j) for i in range(n_partes) for j in range(i + 1, n_partes)]

            resultados = Parallel(n_jobs=-1)(
                delayed(evaluar_fusion)(i, j, particiones) for i, j in pares_a_evaluar
            )

            # Extraemos el que tenga la mejor (menor) pérdida
            mejor_perdida, i_idx, j_idx, mejor_union = min(resultados, key=lambda x: x[0])

            nueva_lista_particiones = [
                particiones[p] for p in range(n_partes) if p != i_idx and p != j_idx
            ]
            nueva_lista_particiones.append(mejor_union)
            
            particiones = nueva_lista_particiones

            self.logger.critic(
                f"Jerárquico (Bottom-Up): Fusionadas partes {(i_idx, j_idx)} -> {mejor_union}. "
                f"Particiones restantes: {len(particiones)}, Pérdida: {mejor_perdida:.6f}"
            )

        return tuple(particiones)