from colorama import Fore
from numpy.typing import NDArray
from typing import Callable, Optional
from itertools import product
import pandas as pd
import numpy as np
import time

from src.models.base.application import aplicacion

from src.models.base.sia import SIA
from src.models.core.system import System
from src.models.core.solution import Solution

from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profile, gestor_perfilado

from src.funcs.iit import seleccionar_emd, literales
from src.funcs.format import fmt_biparticion_fuerza_bruta, fmt_k_bloques
from src.funcs.force import (
    biparticiones,
    generar_candidatos,
    generar_particiones,
    generar_subsistemas,
    particiones_en_k,
)
from src.constants.base import (
    COLS_IDX,
    EXCEL_EXTENSION,
    FLOAT_ZERO,
    NET_LABEL,
    TYPE_TAG,
    EFFECT,
    ACTUAL,
)
from src.constants.models import (
    BRUTEFORCE_FULL_ANALYSIS_TAG,
    BRUTEFORCE_STRAREGY_TAG,
    BRUTEFORCE_LABEL,
    BRUTEFORCE_KMIP_LABEL,
    BRUTEFORCE_KMIP_STRAREGY_TAG,
    DUMMY_ARR,
    DUMMY_EMD,
    ERROR_PARTITION,
)


class BruteForce(SIA):
    """
    Generador de soluciones mediante fuerza bruta sobre una red específica.

    Para hacer uso del debug en diferentes zonas del proceso:

    >>>    self.logger.info("General status update")
    >>>    self.logger.debug("Detailed debugging info")
    >>>    self.logger.debuging("debuging message")
    >>>    self.logger.error("Error occurred")

    Así mismo este se almacenará en el archivo con el nombre que hayamos asociado en el `setup_logger(...)`.
    Este archivo de profilling de extensión HTML lo arrastras hasta tu navegador y se visualizará la depuración del aplicativo a lo largo del tiempo en dos vistas, temporal y cumulativa sobre el coste temporal en subrutinas.
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.distancia_metrica: Callable = seleccionar_emd()
        self.logeador = SafeLogger(BRUTEFORCE_STRAREGY_TAG)

    # @profile(
    #     context={TYPE_TAG: BRUTEFORCE_ANALYSIS_TAG}
    # )  # Descomentame y revisa el directorio `./review/profiling/`! #
    def aplicar_estrategia(
        self, estado_inicial: str, condiciones: str, alcance: str, mecanismo: str
    ):
        """
        Análisis por fuerza brutal sobre una red específica para un sistema candidato llevado a un subsistema determinado por el alcance y mecanismo indicado por el usuario.

        Args:
        ----
            conditions (str): Condiciones de fondo, dónde se va a condicionar el sistema original como candidato, sean las dimensiones en 0 las que se condicionen.
            alcance (str): Elementos futuros que serán marginalizados si el bit está en cero (0) para la posición de la variable asociada.
            mecanismo (str): Elementos presentes que serán marginalizados si su bit asociado en cero (0) para la posición de la variable.

        Returns:
        -------
            None: El análisis como se aprecia puede ser medido mediante el decorador de profiling, así como si se desea para algún otro método.
        """
        self.sia_preparar_subsistema(estado_inicial, condiciones, alcance, mecanismo)

        solucion_base = Solution(
            BRUTEFORCE_LABEL,
            DUMMY_EMD,
            self.sia_dists_marginales,
            DUMMY_ARR,
            ERROR_PARTITION,
            quiere_hablar=True,
        )

        small_phi = np.inf
        mejor_dist_marg: np.ndarray = DUMMY_ARR

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        biparticion_prim: tuple[tuple[int, ...], tuple[int, ...]]
        biparticion_dual: tuple[tuple[int, ...], tuple[int, ...]]
        m, n = futuros.size, presentes.size

        for subalcance, submecanismo in biparticiones(
            futuros, presentes, (1 << m) * (1 << n)
        ):
            subsistema = self.sia_subsistema
            arr_alcance = np.array(subalcance, dtype=np.int8)
            arr_mecanismo = np.array(submecanismo, dtype=np.int8)

            particion = subsistema.bipartir(arr_alcance, arr_mecanismo)

            part_marg_dist = particion.distribucion_marginal()
            emd_value = self.distancia_metrica(
                part_marg_dist, self.sia_dists_marginales
            )
            if emd_value < small_phi:
                small_phi = emd_value
                mejor_dist_marg = part_marg_dist
                biparticion_prim = submecanismo, subalcance
                biparticion_dual = (
                    set(presentes.data) - set(submecanismo),
                    set(futuros.data) - set(subalcance),
                )
                # La Fuerza Bruta (absoluta) no haría esto #
                if emd_value == FLOAT_ZERO:
                    solucion_base.perdida = emd_value
                    solucion_base.distribucion_particion = part_marg_dist
                    solucion_base.particion = fmt_biparticion_fuerza_bruta(
                        [biparticion_prim[ACTUAL], biparticion_prim[EFFECT]],
                        [biparticion_dual[ACTUAL], biparticion_dual[EFFECT]],
                    )
                    solucion_base.tiempo_ejecucion = (
                        time.time() - self.sia_tiempo_inicio
                    )
                    return solucion_base

        biparticion_formateada = fmt_biparticion_fuerza_bruta(
            [biparticion_prim[ACTUAL], biparticion_prim[EFFECT]],
            [biparticion_dual[ACTUAL], biparticion_dual[EFFECT]],
        )

        solucion_base.perdida = small_phi
        solucion_base.distribucion_particion = mejor_dist_marg
        solucion_base.particion = biparticion_formateada
        solucion_base.tiempo_ejecucion = time.time() - self.sia_tiempo_inicio
        return solucion_base

    @profile(context={TYPE_TAG: BRUTEFORCE_FULL_ANALYSIS_TAG})
    def analizar_completamente_una_red(self) -> None:
        """
        Se prepara el directorio de salida donde almacenaremos el análisis completo de una red específica.
        Este análisis consiste de para una red de N elementos en dos tiempos `t_0` y `t_1` para un único estado inicial, se crean todos los `{2^N}-1` factibles sistemas candidatos, posteriormente a cada uno sus `2^{m+n}` posibles biparticiones, excluyendo escenarios con alcances vacíos y finalmente cada bipartición de las `2^{m+n-1}-1` factibles.
        """
        self.tpm.output_dir.mkdir(parents=True, exist_ok=True)

        tpm = self.sia_cargar_tpm()
        initial_state = self.sia_subsistema.estado_inicial
        system = System(tpm, initial_state)
        self.__analizar_candidatos(system)
        print(f"""
{Fore.RED}Generación finalizada!{Fore.BLUE}\nRevisa tu directorio `review/resolver/`.
{Fore.WHITE}Tamaño de la red: {initial_state.size} nodos.
Estado incial: {initial_state}.
""")

    def __analizar_candidatos(self, sistema: System) -> None:
        """
        Genera todos los sistemas candidatos factibles para dar análisis, de forma que se almacenen luego como un documento excel para mejor visualización.

        Args:
        ----
            sistema (System): Sisteam completo que será condicionado según la combinación de dimensiones para condicionar/eliminar, formando el sistema candidato.
        """
        cantidad = len(self.tpm.estado_inicial)
        dim_candidatas = generar_candidatos(cantidad)

        for dimensiones in dim_candidatas:
            self.__procesar_candidato(sistema, np.array(dimensiones, dtype=np.int8))

    def __procesar_candidato(
        self, completo: System, condiciones: NDArray[np.int8]
    ) -> None:
        """Aplicamos condiciones de fondo sobre el sistema completo y continuamos la cadena para su análisis por subsistemas.

        Args:
        ----
            completo (System): Sistema completo a condicionar.
            condiciones (NDArray[np.int8]): Condiciones de fondo aplicadas sobre el sistema completo.
        """
        candidato = completo.condicionar(condiciones)
        nombre = literales(np.setdiff1d(candidato.dims_ncubos, condiciones))
        self.__procesar_subsistema(candidato, nombre)

    def __procesar_subsistema(
        self, mecanismo_removido: System, nombre_candidato: str
    ) -> None:
        """
        Genera todos los subsistemas para un sistema candidato.

        Args:
        ----
            mecanismo_removido (System): Mecanismo obtenido de algún condicionamiento realizado con anterioridad.
            nombre_candidato (str): El noombre del sistema candidato de forma amigable, este determinará el nombre del fichero donde se guardará la solución de su análisis, esto en el directorio `review/`.
        """
        results_file = self.tpm.output_dir / f"{nombre_candidato}.{EXCEL_EXTENSION}"

        with pd.ExcelWriter(results_file) as writer:
            for alcance_removido, sub_present in generar_subsistemas(
                mecanismo_removido.dims_ncubos
            ):
                if not self.__deberia_omitir_subsistema(
                    alcance_removido, mecanismo_removido
                ):
                    self.__analizar_subsistema(
                        mecanismo_removido,
                        np.array(alcance_removido, dtype=np.int8),
                        np.array(sub_present, dtype=np.int8),
                        writer,
                    )

    def __deberia_omitir_subsistema(
        self, alcance_removido: tuple[int, ...], candidate: System
    ) -> bool:
        """
        Revisa si el alcance o futuro que se va a condicionar genera un subsistema sin futuro y por ende, no útil en el análisis sistémico, no hay un non-trivial effect cual dar revisión.

        Args:
        ----
            alcance_removido (tuple[int, ...]): tupla con índices asociados a las dimensiones que serán removidas.
            candidate (System): Sistema cual se removeran los alcances.

        Returns:
        -------
            bool: Determina si tienen el mismo tamaño, de serlo su diferencia será 0 y por ende no habrá futuro.
        """
        return len(alcance_removido) == candidate.indices_ncubos.size

    def __analizar_subsistema(
        self,
        candidato: System,
        alcance_removido: NDArray[np.int8],
        mecanismo_removido: NDArray[np.int8],
        writer: pd.ExcelWriter,
    ) -> None:
        """Analiza un sistema candidato y genera un condicionamiento para analizar sus subsistemas restantes.

        Args:
        ----
            candidato (System): Subsistema candidato a ser substraído de sus elementos con el fin de obtener un subsistema.
            alcance_removido (NDArray[np.int8]): El alcance o elementos futuros que serán marginalizados.
            mecanismo_removido (NDArray[np.int8]): El mecanismo o elementos presentes que serán marginalizados.
            writer (pd.ExcelWriter): escritor en la hoja de cálculo para un documento excel ya asociado.

        Se almacena el resultado del análisis de este subsistema en una hoja de excel con la representación literal del mismo.
        """
        subsistema = candidato.substraer(alcance_removido, mecanismo_removido)
        dist_marginal = subsistema.distribucion_marginal()

        nombre_subsistema = self.__get_nombre_subsistema(
            candidato, alcance_removido, mecanismo_removido
        )
        resultado = self.__analizar_particiones(dist_marginal, subsistema)
        resultado.to_excel(writer, sheet_name=nombre_subsistema)

    def __analizar_particiones(
        self, distribucion: NDArray[np.float32], subsistema: System
    ) -> pd.DataFrame:
        """Para cada subsistema se realiza su análisis por cada partición. Como tenemos entendido la primera partición es tirivial de forma que es ignorada (esto es representado luego con i=1 para la selección de etiquetas).
        Primeramente se obtienen las dimensiones totales del subsistema, tanto para mecanismos/filas (n) como alcances/columnas (m), sabemos que la cantidad de particiones con `k=2` (biparticiones) `P_k(S_{n, m}) = 2^(m+n-1)-1 = [(2^m-1)*(2^{n})]-1`, con esto podemos generar una matriz de `2^m` filas por `2^(m-1)` columnas y sustraemos la partición trivial.
        Precomputamos las llaves y así mismo las posibles particiones, donde indexamos el resultado de la emd claramente en el iterando módulo m o n para asociar correctamente la clave e incrementamos ambos, pero sólo j cuando i haga una rotación.
        Como se aprecia en el fichero `resolver/<red específica>/<estado inicial>/` la partición que interseca las claves (0,0) siempre debe estar vacía puesto es la partición trivial (donde de hecho no es una partición pues toda variable pertenece al mismo lado).

        Args:
        ----
            distribucion (NDArray[np.float32]): Distribución marginal que se comparará con la distribución marginal de la partición
            subsistema (System): Subsistema que será particionado y su partición analizada con este mismo mediante la EMD Efecto

        Returns:
        -------
            pd.DataFrame: Matriz que asociará en las filas los elementos presente o mecanismos de la partición y en las columnas los elementos futuros o alcances de la partición, esto de forma que los elementos que pertenezcan al mismo bit (0|1), pertenecen a la misma partición.
        """
        m, n = subsistema.indices_ncubos.size, subsistema.dims_ncubos.size

        llave_presente = [f"{number:0{n}b}" for number in range(1 << n)]
        llave_futuro = [f"{number:0{m}b}" for number in range(1 << m - 1)]

        resultados = pd.DataFrame(
            columns=llave_futuro,
            index=llave_presente,
            dtype=np.float32,
        )

        i, j = 1, 0
        for alcance, mecanismo in generar_particiones(m, n):
            sub_alcance = np.array([i for i, bit in enumerate(alcance) if bit])
            sub_mecanismo = np.array([i for i, bit in enumerate(mecanismo) if bit])

            particion = subsistema.bipartir(
                np.array(sub_alcance, dtype=np.int8),
                np.array(sub_mecanismo, dtype=np.int8),
            )

            dist_parte_marginal = particion.distribucion_marginal()
            emd_value = self.distancia_metrica(dist_parte_marginal, distribucion)

            etiqueta_mecanismo = "".join(map(str, mecanismo.astype(int)))
            etiqueta_alcance = "".join(map(str, alcance.astype(int)))

            # Asignar el valor al DataFrame
            resultados.loc[etiqueta_mecanismo, etiqueta_alcance] = emd_value

        return resultados

    def __get_nombre_subsistema(
        self,
        candidato: System,
        sub_alcance: NDArray[np.int8],
        sub_mecanismo: NDArray[np.int8],
    ) -> str:
        """
        Muestra de forma amigable el subsistema analizado, utilizando literales asociados con la dimensión respectiva.

        Args:
            candidato (System): Sistema candidato del que se obtendrán las dimensiones a ser representadas de tanto el mecanismo presente, como el alcance futuro.
            sub_alcance (NDArray[np.int8]): Alcance que será eliminado en el proceso.
            sub_mecanismo (NDArray[np.int8]): Mecanismo que será eliminado en el proceso.

        Returns:
            str: Literal con la representación del subsistema
        """
        futuro_removido = np.setdiff1d(candidato.dims_ncubos, sub_alcance)
        presente_removido = np.setdiff1d(candidato.dims_ncubos, sub_mecanismo)
        return f"{literales(futuro_removido)}|{literales(presente_removido)}"


class BruteForceKMIP(SIA):
    """
    Fuerza bruta EXHAUSTIVA para el problema k-MIP — ground truth de QNodes/KGeoMIP.

    A diferencia de `BruteForce` (que solo enumera biparticiones k=2), esta clase
    enumera TODAS las k-particiones VÁLIDAS del subsistema sobre el MISMO espacio
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
