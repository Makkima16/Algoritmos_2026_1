import time
import itertools
from functools import lru_cache
import numpy as np

from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.iit import seleccionar_emd
from src.constants.models import DUMMY_ARR, ERROR_PARTITION
from src.constants.base import ACTUAL, EFFECT

class DynamicPartition(SIA):
    """
    Estrategia Eficiente / Programación Dinámica para encontrar la sub-partición 
    óptima (la de menor pérdida) evaluando todos los k posibles (k=2, 3, ..., N).
    
    Evitamos la explosión de tiempo de la fuerza bruta al "memoizar" (cachear) las
    distribuciones marginales de cada posible parte (subsistema) para no volver a 
    calcularlas en cada partición en la que aparezcan.
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()
        # Caché dinámica para las distribuciones marginales de cada subconjunto
        self._dp_cache_dists = {}

    def _obtener_distribucion_parte(self, parte: tuple) -> np.ndarray:
        """
        Consulta en la caché la distribución marginal reconstruida para un subconjunto 
        de nodos. Si no existe, se calcula y se almacena (Programación Dinámica Top-Down / Memoization).
        """
        if parte in self._dp_cache_dists:
            return self._dp_cache_dists[parte]

        parte_arr = np.array(parte, dtype=np.int8)
        
        # Filtramos los ncubos (futuros) y dimensiones (mecanismo) para la parte específica
        futuros_parte = self.sia_subsistema.indices_ncubos[parte_arr]
        presentes_parte = self.sia_subsistema.dims_ncubos[parte_arr]
        
        # Bipartición enfocada solo en esta parte (independencia causal)
        sistema_parte = self.sia_subsistema.bipartir(futuros_parte, presentes_parte)
        dist_parte = sistema_parte.distribucion_marginal()
        
        # Guardamos en nuestra caché DP
        self._dp_cache_dists[parte] = dist_parte
        return dist_parte

    def _generar_particiones_bell(self, elementos: list):
        """Genera iterativamente todas las particiones posibles de un conjunto (Números de Bell)"""
        if not elementos:
            yield []
            return
        
        elem = elementos[0]
        for sub_particion in self._generar_particiones_bell(elementos[1:]):
            # 1) Agregarlo a una parte existente
            for i in range(len(sub_particion)):
                copia = [list(p) for p in sub_particion]
                copia[i].append(elem)
                yield copia
            # 2) Crearlo como una nueva parte él solo
            copia = [list(p) for p in sub_particion]
            copia.append([elem])
            yield copia

    def _distribucion_conjunta_vectorizada(self, probabilidades: np.ndarray) -> np.ndarray:
        if len(probabilidades) == 0:
            return np.array([1.0], dtype=np.float64)
        p_1 = np.asarray(probabilidades, dtype=np.float64)
        p_0 = 1.0 - p_1
        factors = np.stack([p_0, p_1], axis=1)
        grid = np.meshgrid(*factors, indexing='ij')
        return np.prod(grid, axis=0).flatten()

    def evaluar_todas_las_k_particiones(self, estado_inicial: str, sistema_candidato: str) -> Solution:
        """
        Evaluación dinámica de la partición general óptima.
        k varía desde 2 hasta la longitud del sistema_candidato.
        """
        # Condiciones, alcance y mecanismo son el mismo sistema_candidato (cadena de bits)
        self.sia_preparar_subsistema(estado_inicial, sistema_candidato, sistema_candidato, sistema_candidato)
        
        n_elementos = len(self.sia_subsistema.indices_ncubos)
        indices_relativos = list(range(n_elementos))

        # P (el comportamiento del conjunto completo) - Se calcula una sola vez
        dist_P = self._distribucion_conjunta_vectorizada(self.sia_dists_marginales)

        mejor_emd = float('inf')
        mejor_particion = None
        mejor_dist_particion = None

        # Evaluamos cada posible partición generada (todas las k-particiones válidas)
        for particion in self._generar_particiones_bell(indices_relativos):
            k = len(particion)
            if k < 2: 
                continue # Omitimos k=1 (no particionar)

            # Reconstruimos la distribución Q mediante DP Memoized parts
            dist_reconstruida = np.empty(n_elementos, dtype=np.float64)

            for parte in particion:
                parte_tuple = tuple(sorted(parte)) # Identificador para caché
                dist_parte = self._obtener_distribucion_parte(parte_tuple)
                for idx_pos in parte:
                    dist_reconstruida[idx_pos] = dist_parte[idx_pos]
            
            dist_Q = self._distribucion_conjunta_vectorizada(dist_reconstruida)
            
            # EMD calculation
            emd_value = float(self.distancia_metrica(dist_P, dist_Q))

            if emd_value < mejor_emd:
                mejor_emd = emd_value
                mejor_particion = particion
                mejor_dist_particion = dist_reconstruida

                # Optimización temprana 
                if mejor_emd == 0.0:
                    break 
        
        particion_str = " | ".join(["{ " + ",".join(str(self.sia_subsistema.indices_ncubos[i]) for i in p) + " }" for p in mejor_particion]) if mejor_particion else "Ninguna"

        solucion = Solution(
            etiqueta="K-Partition Dynamic",
            perdida=mejor_emd,
            distribucion_particion=mejor_dist_particion if mejor_dist_particion is not None else DUMMY_ARR,
            particion=particion_str,
            tiempo_ejecucion=time.time() - self.sia_tiempo_inicio,
            quiere_hablar=True
        )

        return solucion
