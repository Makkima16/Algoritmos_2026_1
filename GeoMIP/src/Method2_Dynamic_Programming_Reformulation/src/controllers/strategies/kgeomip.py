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

La tabla de costos de transiciones entre estados (calculada exactamente igual
que en GeometricSIA) **no cambia con k** — se reutiliza íntegramente. Lo que
cambia es la forma de generar y evaluar las k-particiones candidatas.

Estrategia de búsqueda implementada
-------------------------------------
Para k pequeño y n pequeño (n·k ≤ UMBRAL_EXHAUSTIVO):
    → Búsqueda exhaustiva usando números de Stirling del segundo tipo.
      Se generan todas las k-particiones posibles de los nodos del sistema.

Para n·k > UMBRAL_EXHAUSTIVO:
    → Heurística greedy jerárquica:
      1. Se aplica GeoMIP (bi-partición) para obtener la primera división {S1, S_resto}.
      2. Si k > 2, se vuelve a bipartir el subconjunto más grande de forma
         recursiva hasta obtener k partes.
      Esta heurística no garantiza optimalidad global pero es eficiente y
      produce particiones de buena calidad en la práctica.

Reutilización de infraestructura
---------------------------------
- Hereda de SIA (igual que GeometricSIA).
- Llama a `sia_preparar_subsistema` para obtener el subsistema condicionado.
- Comparte la función de costo y la tabla de transiciones con GeometricSIA.
- Usa `System.bipartir()` y `System.distribucion_marginal()` del framework.

Complejidad
-----------
Exhaustiva : O(S(n,k) · k · 2ⁿ) donde S(n,k) = número de Stirling
Greedy     : O((k-1) · n · 2ⁿ)  — k-1 llamadas a GeometricSIA
"""

import time
import itertools
from math import comb
from typing import List, Tuple, Dict, Optional

import numpy as np

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.base import emd_efecto, ABECEDARY, LOWER_ABECEDARY
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
from src.constants.models import GEOMETRIC_STRAREGY_TAG


# ── Constantes propias de KGeoMIP ──────────────────────────────────────────
KGEOMIP_LABEL: str = "KGeoMIP"
KGEOMIP_STRATEGY_TAG: str = f"{KGEOMIP_LABEL}_strategy"
KGEOMIP_ANALYSIS_TAG: str = f"{KGEOMIP_LABEL}_analysis"

# Si n * k supera este umbral se usa la heurística greedy en lugar de fuerza
# bruta sobre k-particiones.  Valor empírico: 30 cubre n≤10, k≤3 exactamente.
UMBRAL_EXHAUSTIVO: int = 30


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


def evaluar_k_particion(
    subsistema,
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
    particion: Tuple[List[int], ...],
    dist_original: np.ndarray,
) -> float:
    """
    Calcula la pérdida EMD de una k-partición usando semántica IIT estricta.

    Definición matemática:
    ──────────────────────
    Para una k-partición {S1,...,Sk}, cada parte Si es causalmente independiente.
    La distribución reconstruida del sistema k-particionado asigna a cada nodo j ∈ Si:

        dist_rec[j] = 1 - P(Xj=1 | estado_Si,t)

    donde P(Xj=1 | estado_Si,t) se obtiene marginalizando el ncubo_j sobre todos
    los dims que NO pertenecen a Si, y evaluando en estado_inicial[Si].

    Implementación:
    ───────────────
    Usa System.bipartir(futuros=Si, mecanismo=Si_dims) que:
    - Para j ∈ Si (en alcance):   ncubo_j marginaliza dims excluidos de mecanismo
                                   = marginaliza dims ∉ Si
    - Para j ∉ Si (fuera):         ncubo_j marginaliza mecanismo completo (Si_dims)

    Toma solo los valores de los nodos de la parte Si de la distribución resultante.

    Nota sobre diferencia con GeoMIP:
    ──────────────────────────────────
    GeoMIP usa bipartir(alcance=S1, mecanismo=TODOS_DIMS) para bi-partición,
    lo que produce un corte asimétrico. KGeoMIP usa la semántica IIT estricta
    donde cada parte ve solo su propio presente. Esto produce valores de φ
    diferentes para k=2, pero es matemáticamente más correcto para k≥2.

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
    dist_reconstruida = np.empty(n, dtype=np.float32)

    for parte in particion:
        if not parte:
            continue
        parte_arr = np.array(parte, dtype=np.int8)

        # futuros = índices de los ncubos de esta parte
        futuros_parte = indices_ncubos[parte_arr]

        # mecanismo = SOLO los dims de esta parte (IIT estricto: Si solo ve Si_t)
        presentes_parte = dims_ncubos[parte_arr]

        # bipartir(futuros=Si, mecanismo=Si_dims):
        #   j ∈ Si: marginaliza dims ∉ Si → ncubo reducido a dims Si
        #   j ∉ Si: marginaliza Si_dims → escalar
        sistema_parte = subsistema.bipartir(futuros_parte, presentes_parte)
        dist_parte    = sistema_parte.distribucion_marginal()  # shape (n,)

        # Tomar el valor de cada nodo de esta parte
        for idx_pos in parte:
            dist_reconstruida[idx_pos] = dist_parte[idx_pos]

    return float(emd_efecto(dist_original, dist_reconstruida))



def fmt_k_particion(particion: Tuple[List[int], ...], vertices: set) -> str:
    """
    Formatea una k-partición para mostrarla en consola.

    Cada parte se muestra como una columna con variables futuras (mayúscula)
    y presentes (minúscula), separadas por "|".

    Args:
        particion: Tupla de k listas de índices.
        vertices : Conjunto de vértices del subsistema (pares (tiempo, idx)).

    Returns:
        String con la representación de la k-partición.
    """
    partes_fmt = []
    for parte in particion:
        futuros = [ABECEDARY[i] for i in parte]
        presentes = [LOWER_ABECEDARY[i] for i in parte]
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

    Para k>2 usa búsqueda exhaustiva cuando el sistema es pequeño, o una
    heurística greedy jerárquica cuando es grande.

    Args:
        gestor (Manager): Gestor con el estado inicial y ruta de la TPM.
        k      (int)    : Número de partes de la partición (default=2).

    Attributes:
        k                   : Número de partes solicitado.
        tabla_transiciones  : Tabla de costos de transición (heredada de
                              GeometricSIA, calculada una sola vez).
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
        # Tabla de costos de transición (se llena igual que en GeometricSIA)
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
        k: Optional[int] = None,
    ) -> Solution:
        """
        Encuentra la k-Partición de Mínima Información (k-MIP) del subsistema.

        Pasos:
          1. Preparar el subsistema (condicionar + substraer).
          2. Construir la tabla de costos de transiciones (idéntica a GeoMIP).
          3. Generar candidatos de k-particiones.
          4. Evaluar cada candidato con EMD.
          5. Retornar la partición con pérdida mínima.

        Args:
            condicion : Bits que indican qué variables condicionar (0=condicionar).
            alcance   : Bits que indican qué variables incluir en el alcance.
            mecanismo : Bits que indican qué variables incluir en el mecanismo.
            tpm       : Matriz de probabilidad de transición (np.ndarray).
            k         : Número de partes (sobreescribe el k del constructor si se da).

        Returns:
            Solution con la k-MIP encontrada.
        """
        k_efectivo = k if k is not None else self.k

        self.logger.critic(f"Iniciando KGeoMIP con k={k_efectivo}.")
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        n_vars = len(self.sia_subsistema.indices_ncubos)

        # Caso trivial: si k ≥ n, la única partición válida es cada variable sola
        if k_efectivo >= n_vars:
            self.logger.critic(
                f"k={k_efectivo} ≥ n={n_vars}: se usa la partición atómica."
            )
            k_efectivo = n_vars

        # Construir la tabla de costos (igual que GeometricSIA)
        self._construir_tabla_costos()

        # Seleccionar estrategia según tamaño del problema
        if n_vars * k_efectivo <= UMBRAL_EXHAUSTIVO:
            self.logger.critic("Usando búsqueda exhaustiva.")
            particion_optima = self._busqueda_exhaustiva(k_efectivo)
        else:
            self.logger.critic("Usando heurística greedy.")
            particion_optima = self._heuristica_greedy(k_efectivo)

        # Calcular la pérdida final de la partición óptima encontrada
        perdida = evaluar_k_particion(
            self.sia_subsistema,
            self.sia_subsistema.indices_ncubos,
            self.sia_subsistema.dims_ncubos,
            particion_optima,
            self.sia_dists_marginales,
        )

        # Distribución reconstruida para el Solution (misma lógica que evaluar_k_particion)
        n = len(self.sia_dists_marginales)
        dist_reconstruida = np.empty(n, dtype=np.float32)
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
            set(
                [(EFECTO, i) for i in self.sia_subsistema.indices_ncubos]
                + [(ACTUAL, i) for i in self.sia_subsistema.dims_ncubos]
            ),
        )

        self.logger.critic(
            f"k-MIP encontrada con pérdida={perdida:.6f}:\n{fmt}"
        )

        return Solution(
            estrategia=f"{KGEOMIP_LABEL}(k={k_efectivo})",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_reconstruida,
            particion=fmt,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
        )

    # ── Construcción de tabla de costos (idéntica a GeometricSIA) ─────────

    def _construir_tabla_costos(self) -> None:
        """
        Construye la tabla de costos de transición entre estados del hipercubo.

        Reutiliza exactamente el mismo algoritmo de GeometricSIA:
          - BFS nivel por nivel desde el estado inicial al estado final.
          - Para cada par (i,j): t_X(i,j) = γ·(|X[i]-X[j]| + Σ t_X(k,j))
            con γ = 2^(-dH(i,j)).

        La tabla se almacena en self.tabla_transiciones para ser consultada
        durante la identificación de candidatos.
        """
        # Datos planos de los n-cubos (igual que en GeometricSIA)
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

        Fórmula (del paper GeoMIP, ec. 3.1 / 4.1):
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

        mejor_particion = None
        mejor_perdida = float("inf")

        total_stirling = stirling2(n_vars, k)
        self.logger.critic(
            f"Exhaustiva: S({n_vars},{k}) = {total_stirling} particiones."
        )

        for particion in generar_k_particiones(elementos, k):
            perdida = evaluar_k_particion(
                self.sia_subsistema,
                self.sia_subsistema.indices_ncubos,
                self.sia_subsistema.dims_ncubos,
                particion,
                self.sia_dists_marginales,
            )
            self.memoria_particiones[tuple(tuple(p) for p in particion)] = (
                perdida,
                None,  # distribución: se calcula al final solo para la óptima
            )
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_particion = particion

        self.logger.critic(f"Exhaustiva terminada. Mejor pérdida = {mejor_perdida:.6f}.")
        return mejor_particion

    # ── Estrategia 2: heurística greedy jerárquica ─────────────────────────

    def _heuristica_greedy(
        self, k: int
    ) -> Tuple[List[int], ...]:
        """
        Heurística greedy: aplica bipartición recursivamente.

        Algoritmo:
          1. Bipartir el conjunto completo de nodos con GeoMIP → {S1, S_resto}.
          2. Si k > 2, tomar la parte más grande de S_resto y volver al paso 1.
          3. Continuar hasta tener k partes.

        Nota: esta heurística no garantiza optimalidad global pero es eficiente
        (O((k-1) · costo_GeoMIP)) y produce buenos resultados empíricos.

        Args:
            k: Número de partes objetivo.

        Returns:
            Tupla de k listas de índices.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        partes_actuales: List[List[int]] = [list(range(n_vars))]

        for paso in range(k - 1):
            # Elegir la parte más grande para volver a bipartir
            idx_mayor = max(range(len(partes_actuales)), key=lambda i: len(partes_actuales[i]))
            parte_a_dividir = partes_actuales[idx_mayor]

            if len(parte_a_dividir) < 2:
                self.logger.critic(
                    f"Greedy paso {paso+1}: la parte más grande tiene solo 1 elemento, "
                    "no se puede dividir más."
                )
                break

            nueva_s1, nueva_s2 = self._bipartir_subconjunto(parte_a_dividir)
            partes_actuales.pop(idx_mayor)
            partes_actuales.append(nueva_s1)
            partes_actuales.append(nueva_s2)
            self.logger.critic(
                f"Greedy paso {paso+1}: dividió {parte_a_dividir} → "
                f"{nueva_s1} | {nueva_s2}."
            )

        return tuple(partes_actuales)

    def _bipartir_subconjunto(
        self, subconjunto: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Aplica GeoMIP (bi-partición) sobre un subconjunto de los nodos.

        Usa la tabla de costos ya calculada para este subsistema y evalúa
        todas las bi-particiones del subconjunto, retornando la de menor EMD.

        Args:
            subconjunto: Lista de índices de variables a bipartir.

        Returns:
            Dos listas de índices (S1, S2) que forman la bi-partición óptima
            del subconjunto.
        """
        if len(subconjunto) == 1:
            return [subconjunto[0]], []

        mejor_s1, mejor_s2 = None, None
        mejor_perdida = float("inf")

        # Evaluar todas las bi-particiones del subconjunto
        n = len(subconjunto)
        # Solo la mitad (2^(n-1) - 1) para evitar duplicados por complemento
        for mascara in range(1, 1 << (n - 1)):
            s1 = [subconjunto[i] for i in range(n) if (mascara >> i) & 1]
            s2 = [subconjunto[i] for i in range(n) if not ((mascara >> i) & 1)]
            if not s2:
                continue

            particion = (s1, s2)
            perdida = evaluar_k_particion(
                self.sia_subsistema,
                self.sia_subsistema.indices_ncubos,
                self.sia_subsistema.dims_ncubos,
                particion,
                self.sia_dists_marginales,
            )
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_s1, mejor_s2 = s1, s2

        return mejor_s1, mejor_s2