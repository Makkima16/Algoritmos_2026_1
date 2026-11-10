from typing import Generator


def particiones_en_k(items: list, k: int) -> Generator[list[list], None, None]:
    """
    Genera todas las particiones de `items` en EXACTAMENTE `k` bloques no vacíos.

    Son los números de Stirling de segunda especie S(len(items), k) particiones
    (conjuntos no ordenados de bloques no vacíos). Se implementa con el esquema
    clásico recursivo: el primer elemento siembra el primer bloque y cada
    elemento siguiente o entra en un bloque existente o abre uno nuevo, exigiendo
    que al final queden exactamente `k` bloques.

    Args:
        items: Lista de elementos a particionar (p.ej. posiciones de futuros).
        k:     Número exacto de bloques no vacíos.

    Yields:
        list[list]: Una partición como lista de `k` bloques (cada bloque, lista
        de elementos). El orden de los bloques es estable pero no significativo.
    """
    n = len(items)
    if k < 1 or k > n:
        return

    def _rec(idx: int, bloques: list[list]) -> Generator[list[list], None, None]:
        if idx == n:
            if len(bloques) == k:
                yield [bloque[:] for bloque in bloques]
            return

        restantes = n - idx
        # Poda: no abrir tantos bloques que sea imposible llegar a k,
        # ni dejar tan pocos elementos que no se puedan completar k bloques.
        elemento = items[idx]

        # 1. Colocar el elemento en un bloque existente (si aún se puede llegar a k).
        if len(bloques) + (restantes - 1) >= k:
            for i in range(len(bloques)):
                bloques[i].append(elemento)
                yield from _rec(idx + 1, bloques)
                bloques[i].pop()

        # 2. Abrir un bloque nuevo con el elemento (si no excede k).
        if len(bloques) < k:
            bloques.append([elemento])
            yield from _rec(idx + 1, bloques)
            bloques.pop()

    yield from _rec(0, [])
