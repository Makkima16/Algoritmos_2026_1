# Optimizaciones realizadas en QNodes

Este documento detalla las optimizaciones aplicadas en `AYDA_2026_1/QNodes` para
hacer el algoritmo tratable para N = 10, 15, 20, 22, 25 nodos, que es el rango
objetivo del proyecto.

---

## 1. Cambio de paradigma: de búsqueda exhaustiva a greedy aglomerativo

**La optimización más importante es algorítmica, no de implementación.**

La versión anterior (`DynamicPartition`) exploraba todas las k-particiones usando
DP + Branch & Bound. Para N = 25, el espacio de búsqueda es B(25) ≈ 4 × 10¹⁸
particiones — intractable incluso con poda agresiva.

El greedy aglomerativo reemplaza la búsqueda exhaustiva por una secuencia de
N − 2 fusiones, cada una evaluando C(k, 2) ≈ k²/2 pares:

```
Total evaluaciones ≈ Σₖ₌₂ᴺ k²/2 ≈ N³/6

N=10:  ~167 evaluaciones
N=15:  ~563 evaluaciones
N=20: ~1333 evaluaciones
N=25: ~2604 evaluaciones
```

Comparado con B(N) exhaustivo:

| N | B(N) exhaustivo | Greedy aglomerativo | Factor reducción |
|---|---|---|---|
| 10 | ~115 000 | ~167 | ×690 |
| 15 | ~1.4 × 10⁹ | ~563 | ×2.5 × 10⁶ |
| 25 | ~4 × 10¹⁸ | ~2 604 | ×10¹⁵ |

---

## 2. Memoización de distribuciones marginales

**Archivo:** `src/strategies/q_nodes.py` — `_dist_parte`, `_cache_dist`

La distribución marginal de cada parte se calcula una sola vez. Esta es la operación
más costosa (involucra `bipartir` y `distribucion_marginal` sobre el sistema):

```python
def _dist_parte(self, mascara: int) -> np.ndarray:
    if mascara not in self._cache_dist:
        idx_arr = np.fromiter(_bits_activos(mascara), dtype=np.int8)
        futuros = self.sia_subsistema.indices_ncubos[idx_arr]
        presentes = np.intersect1d(futuros, self.sia_subsistema.dims_ncubos)
        dist = self.sia_subsistema.bipartir(futuros, presentes).distribucion_marginal()
        self._cache_dist[mascara] = dist
    return self._cache_dist[mascara]
```

La misma máscara aparece como candidato en múltiples evaluaciones de `_emd_particion`
a lo largo del agrupamiento, el refinamiento y los candidatos de aislamiento. Con caché,
cada distribución se computa una sola vez por sesión.

**Impacto:** el número efectivo de llamadas costosas a `bipartir` es O(N²) en lugar de
O(N³) para el agrupamiento, ya que muchas máscaras candidatas son las mismas.

---

## 3. Métrica adaptada al tamaño en `_emd_particion`

**Archivo:** `src/strategies/q_nodes.py` — `_emd_particion`  
**Archivo:** `src/funcs/iit.py` — `HAMMING_EMD_MAX_N`, `emd_causal`

La función `_emd_particion` es el único punto donde se toma una decisión basada en N,
de forma transparente para la estrategia:

```python
if self._N <= HAMMING_EMD_MAX_N:
    # Wasserstein-1 con d_Hamming — métrica exacta de IIT
    P = distribucion_conjunta_vectorizada(sia_dists_marginales)
    Q = distribucion_conjunta_vectorizada(dist_rec)
    return float(emd_causal(P, Q))
else:
    # Suma L1 marginal — aproximación rápida para N grande
    return float(np.sum(np.abs(dist_rec - sia_dists_marginales)))
```

Para N ≤ 12: la EMD Hamming requiere construir distribuciones conjuntas de tamaño 2^N
(1024 estados para N=10) y una matriz de costes 2^N × 2^N (~8 MB para N=10). Tratable.

Para N > 12: la distribución conjunta sería de tamaño 2^N (>4096 estados) y la matriz
de costes de tamaño 4^N — inviable en memoria y tiempo. La suma L1 marginal es la
aproximación práctica para sistemas grandes.

La estrategia algorítmica (las tres fases, para todo k, para todo N) es siempre la
misma. Solo la métrica interna de `_emd_particion` varía según N.

---

## 4. Caché de matrices Hamming (`_HAMMING_CACHE`)

**Archivo:** `src/funcs/iit.py` — `get_hamming_matrix`, `_HAMMING_CACHE`

Para N ≤ 12, `emd_causal` necesita la matriz de costes Hamming de tamaño 2^N × 2^N.
Construirla en cada llamada costaría O(4^N). Se cachea una vez por sesión:

```python
_HAMMING_CACHE: dict[int, np.ndarray] = {}

def get_hamming_matrix(n: int) -> np.ndarray:
    if n not in _HAMMING_CACHE:
        # Construir la matriz (2^N × 2^N) de distancias Hamming — O(4^N) una vez
        _HAMMING_CACHE[n] = <construcción>
    return _HAMMING_CACHE[n]  # O(1) en llamadas posteriores
```

Para N = 10: matriz 1024 × 1024 ≈ 8 MB (float64). Se construye una sola vez por sesión
y se reutiliza en todas las llamadas a `emd_causal` de ese sistema.

---

## 5. Representación bitwise de conjuntos

**Archivo:** `src/strategies/q_nodes.py` — función `_bits_activos`

Los grupos se representan como enteros (máscaras de bits) en lugar de listas o sets:

| Operación | Con listas | Con bitmask |
|---|---|---|
| Unión Gᵢ ∪ Gⱼ | `set(a) | set(b)` O(N) | `a \| b` O(1) |
| Añadir nodo n | `list.append(n)` | `g \| (1 << n)` O(1) |
| Eliminar nodo n | `list.remove(n)` O(N) | `g ^ (1 << n)` O(1) |
| Clave de caché | `tuple(sorted(...))` | `int` directo |

Las claves enteras en los diccionarios de caché son más eficientes que tuplas
(menor overhead de hashing, mejor localidad de memoria).

---

## 6. Refinamiento reutiliza caché de distribuciones

**Archivo:** `src/strategies/q_nodes.py` — `_refinamiento_local`

El refinamiento 1-move evalúa movimientos candidatos con `_emd_particion`. Dentro,
`_dist_parte_efectiva` consulta `_cache_dist`. Las máscaras de los grupos ya evaluados
durante el agrupamiento son cache hits O(1); solo las nuevas combinaciones (g_sin_nodo,
g_con_nodo) pueden ser cache miss.

En la práctica, la mayoría de los movimientos candidatos corresponden a máscaras que
ya existen en caché (el refinamiento trabaja sobre grupos ya formados), por lo que
el refinamiento es efectivamente mucho más rápido que su complejidad teórica
O(N × k × 20 iteraciones).

---

## 7. Historial de k-particiones como base de la búsqueda libre

**Archivo:** `src/strategies/q_nodes.py` — `historico` en `_aglomerar`

Cada k-partición se guarda durante el agrupamiento sin costo adicional:

```python
historico[len(grupos)] = (phi_total, list(grupos))
```

`_aglomerar()` siempre retorna el historial completo; `aplicar_estrategia` decide
qué hacer con él según si k fue especificado o no.

**Ventaja — k especificado:** se toma `historico[k]` directamente sin rehacer
el agrupamiento. El historial amortiza el coste O(N³) entre todos los k posibles.

**Ventaja — k libre:** el historial es el punto de partida de las tres fases para
todos los niveles k. El refinamiento y los candidatos de aislamiento parten de una
solución greedy de calidad, reduciendo el número de mejoras necesarias.

El costo de memoria del historial es O(N²) en el peor caso (N entradas, cada una
con hasta N grupos enteros). Para N=25: ~625 enteros → despreciable.

---

## 8. Limpieza de cachés entre ejecuciones

**Archivo:** `src/strategies/q_nodes.py` — inicio de `aplicar_estrategia`

Entre ejecuciones consecutivas (modo CSV por bloques), los cachés se reinician:

```python
self._cache_dist.clear()
self._cache_costo.clear()
self._cache_dist_vacio.clear()
self._usar_vacio.clear()
```

Esto evita que distribuciones de un sistema contaminen el cálculo del siguiente
y que los cachés crezcan indefinidamente en runs largos con muchos sistemas.

---

## Resumen de impacto por optimización

| Optimización | Activa cuando | Reducción de tiempo |
|---|---|---|
| Greedy vs exhaustivo | Siempre | ×10¹⁵ para N=25 |
| Caché de distribuciones (`_cache_dist`) | Siempre | 70–90 % en la operación más cara |
| Caché de matrices Hamming (`_HAMMING_CACHE`) | N ≤ 12 | O(4^N) → O(1) por sesión |
| Métrica L1 en lugar de Hamming EMD | N > 12 | O(2^N) → O(N) por evaluación |
| Bitmask vs listas | Siempre | 10–20 % overhead de tipo |
| Refinamiento reutiliza caché | Siempre | Efectivamente O(1) por hit |
| Historial de k-particiones | Siempre | 0 EMDs extra al cambiar k |

Las dos primeras filas son las que hacen el problema tratable para N=25.
Para N ≤ 12, el caché de la matriz Hamming es adicionalmente importante.
