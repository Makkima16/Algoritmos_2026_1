from src.funcs.iit import ABECEDARY, LOWER_ABECEDARY
from src.constants.base import BASE_TWO, COLON_DELIM, VOID_STR
import numpy as np

'''
Métodos para formatear particiones resultantes de estrategias específicas.
El objetivo es hacer estándar y presentable la salida de una k-partición. Si un
nuevo algoritmo necesita otro formato, es preferible añadir una función nueva.
'''

def fmt_k_bloques(
    bloques: "list[tuple[frozenset, frozenset]]",
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
) -> str:
    """
    Formatea una k-partición ASIMÉTRICA expresada como lista de bloques.

    Cada bloque es (futuros_pos, presentes_pos), conjuntos de posiciones locales
    independientes: el futuro (t+1) y el presente/mecanismo (t) de un bloque NO
    tienen por qué coincidir (corte asimétrico estilo GeoMIP). Un nodo puede
    aportar su mecanismo a un bloque mientras su futuro vive en otro.

    Fila superior (MAYÚSCULAS) = nodos futuros/alcance (t+1) del bloque.
    Fila inferior (minúsculas) = nodos presentes/mecanismo (t) del bloque; ∅ si vacío.

    Args:
        bloques        : Lista de (frozenset futuros_pos, frozenset presentes_pos).
        indices_ncubos : Mapeo posición → índice global del n-cubo (futuro).
        dims_ncubos    : Mapeo posición → índice global del mecanismo (presente).
    """
    partes_fmt = []
    n_dims = len(dims_ncubos)

    for fut_pos, pre_pos in bloques:
        idxs_fut = [int(indices_ncubos[p]) for p in sorted(fut_pos)]
        idxs_pre = [int(dims_ncubos[p]) for p in sorted(pre_pos) if p < n_dims]

        str_fut = COLON_DELIM.join(ABECEDARY[idx] for idx in idxs_fut) or VOID_STR
        str_pres = COLON_DELIM.join(LOWER_ABECEDARY[idx] for idx in idxs_pre) or VOID_STR

        ancho = max(len(str_fut), len(str_pres)) + BASE_TWO
        partes_fmt.append((f"|{str_fut:^{ancho}}|", f"|{str_pres:^{ancho}}|"))

    linea_top = "".join(t for t, _ in partes_fmt)
    linea_bot = "".join(b for _, b in partes_fmt)
    return f"{linea_top}\n{linea_bot}\n"
