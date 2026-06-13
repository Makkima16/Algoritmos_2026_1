from src.funcs.iit import ABECEDARY, LOWER_ABECEDARY
from src.constants.base import BASE_TWO, COLON_DELIM, VOID_STR
import numpy as np

'''
Métodos para formatear particiones resultantes de estrategias específicas.
Este fichero tiene el objetivo de hacer estándar y presentable la salida de resultados al hallarse una bipartición. Es importante aclarar cómo aunque cada función puede ser reutilizada para un nuevo algoritmo si se adaptan sus argumentos, es preferible crear una nueva función si se aprecia mayor dificultad en dicha adaptación.
'''

def fmt_biparticion_fuerza_bruta(
    parte_uno: list[tuple[int, ...], tuple[int, ...]],
    parte_dos: list[tuple[int, ...], tuple[int, ...]],
) -> str:
    '''
    Formatea una bipartición de una estrategia de fuerza bruta.

    Args:
        parte_uno: Mecanismo y purview de la primera parte.
    '''
    mech_p, pur_p = parte_uno
    mech_d, purv_d = parte_dos

    # Convertir índices a letras o símbolo vacío si no hay elementos
    purv_prim = COLON_DELIM.join(ABECEDARY[j] for j in pur_p) if pur_p else VOID_STR
    mech_prim = (
        COLON_DELIM.join(LOWER_ABECEDARY[i] for i in mech_p) if mech_p else VOID_STR
    )

    purv_dual = COLON_DELIM.join(ABECEDARY[i] for i in purv_d) if purv_d else VOID_STR
    mech_dual = (
        COLON_DELIM.join(LOWER_ABECEDARY[j] for j in mech_d) if mech_d else VOID_STR
    )

    width_prim = max(len(purv_prim), len(mech_prim)) + BASE_TWO
    width_dual = max(len(purv_dual), len(mech_dual)) + BASE_TWO

    return (
        f"⎛{purv_prim:^{width_prim}}⎞⎛{purv_dual:^{width_dual}}⎞\n"
        f"⎝{mech_prim:^{width_prim}}⎠⎝{mech_dual:^{width_dual}}⎠\n"
    )


def fmt_biparticion_q(
    prim: list[tuple[int, int]],
    dual: list[tuple[int, int]],
    to_sort: bool = True,
) -> str:
    top_prim, bottom_prim = fmt_parte_q(prim, to_sort)
    top_dual, bottom_dual = fmt_parte_q(dual, to_sort)

    return f"{top_prim}{top_dual}\n{bottom_prim}{bottom_dual}\n"


def _bits_activos_fmt(mascara: int):
    """Genera los índices de bits activos de la máscara en orden ascendente."""
    m = mascara
    while m:
        bit = m & (-m)
        yield bit.bit_length() - 1
        m ^= bit


def fmt_k_particion_dp(
    partes_mascaras: list[int],
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
    mascaras_vacio: "set[int] | None" = None,
) -> str:
    """
    Formatea una k-partición expresada como lista de máscaras de bits locales.

    Fila superior (MAYÚSCULAS) = nodos futuros/alcance (t+1).
    Fila inferior (minúsculas) = nodos presentes/mecanismo (t), o ∅ si ninguno
    o si la máscara aparece en mascaras_vacio (mecanismo vacío explícito).

    Args:
        partes_mascaras: Lista de enteros-máscara (índices locales 0..N-1).
        indices_ncubos : Mapeo local → índice global del n-cubo (futuro).
        dims_ncubos    : Índices globales activos en el mecanismo (presente).
        mascaras_vacio : Conjunto de máscaras que usan mecanismo vacío (∅).
    """
    dims_set = set(int(d) for d in dims_ncubos)
    vacio_set = mascaras_vacio or set()
    partes_fmt = []

    for mascara in partes_mascaras:
        indices_locales = sorted(_bits_activos_fmt(mascara))
        idxs_reales = [int(indices_ncubos[i]) for i in indices_locales]

        str_fut = COLON_DELIM.join(ABECEDARY[idx] for idx in idxs_reales)
        if mascara in vacio_set:
            str_pres = VOID_STR
        else:
            str_pres = COLON_DELIM.join(
                LOWER_ABECEDARY[idx] for idx in idxs_reales if idx in dims_set
            )
        str_fut = str_fut if str_fut else VOID_STR
        str_pres = str_pres if str_pres else VOID_STR

        ancho = max(len(str_fut), len(str_pres)) + BASE_TWO
        partes_fmt.append((f"⎛{str_fut:^{ancho}}⎞", f"⎝{str_pres:^{ancho}}⎠"))

    linea_top = "".join(t for t, _ in partes_fmt)
    linea_bot = "".join(b for _, b in partes_fmt)
    return f"{linea_top}\n{linea_bot}\n"


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
        partes_fmt.append((f"⎛{str_fut:^{ancho}}⎞", f"⎝{str_pres:^{ancho}}⎠"))

    linea_top = "".join(t for t, _ in partes_fmt)
    linea_bot = "".join(b for _, b in partes_fmt)
    return f"{linea_top}\n{linea_bot}\n"


def fmt_parte_q(
    parte: list[tuple[int, int]], a_ordenar: bool = True
) -> tuple[str, str]:
    if a_ordenar:
        # Ordenar por índice #
        parte.sort(key=lambda x: x[1])

    purv, mech = [], []
    for time, idx in parte:
        purv.append(ABECEDARY[idx]) if time else mech.append(LOWER_ABECEDARY[idx])

    str_purv = COLON_DELIM.join(purv) if purv else VOID_STR
    str_mech = COLON_DELIM.join(mech) if mech else VOID_STR
    width = max(len(str_purv), len(str_mech)) + 2

    return f"⎛{str_purv:^{width}}⎞", f"⎝{str_mech:^{width}}⎠"
