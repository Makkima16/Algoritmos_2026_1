# Cambios Realizados en GeoMIP vs. el Código Original

## Contexto

El código original (`projecto-analisis-20261`) era una implementación imperativa
genérica que resolvía únicamente **biparticiones** (k = 2). La versión actual
(`AYDA_2026_1 / GeoMIP`) reimplementa el problema con soporte para **k-particiones**,
una arquitectura OOP, y — en su estado vigente — un **motor greedy top-down sobre
bloques asimétricos** con métrica **L1 marginal = EMD de Hamming exacta**.

> **Estado actual:** la ruta principal de `aplicar_estrategia` es el motor greedy
> top-down asimétrico (idéntico en filosofía a QNodes). El pipeline de
> SpectralClustering quedó como **fallback**.

---

## 1. Métrica de Distancia — L1 marginal = EMD de Hamming EXACTA

### Original
```python
def emd_efecto(u, v):
    return np.sum(np.abs(u - v))   # diferencia marginal nodo a nodo, sin fundamento
```

Sumaba diferencias marginales sin justificación sobre el espacio de estados; el Φ
resultante no tenía interpretación causal y dependía del orden de los nodos.

### Actual
La misma forma L1 — pero ahora **fundamentada y exacta**. Como la distribución original
y la reconstruida son **productos de marginales** (independencia condicional por
construcción), la EMD de Wasserstein-1 con distancia base de Hamming **se descompone
exactamente** en la suma L1 marginal:

```python
# evaluar_bloques / evaluar_k_particion — O(N), exacta, para todo N
dist_rec = subsistema.particionar(particiones).distribucion_marginal()
return float(np.sum(np.abs(dist_original - dist_rec)))
```

```
EMD_Hamming(P, Q) = Σᵢ | P(nodo_i = 1) − Q(nodo_i = 1) |   (teorema de descomposición marginal)
```

- **Más rápido:** O(N) en vez de O(4^N) del solver `pyemd`.
- **Más preciso:** da el MISMO valor que la EMD real, sin límite de tamaño N.
- `emd_causal`/`get_hamming_matrix` (la ruta `pyemd`) permanecen sólo como verificación
  histórica; **no** están en el camino caliente.

---

## 2. Cortes asimétricos + Greedy Top-Down (motor principal)

### Original / versión intermedia
Selección pseudo-aleatoria (hill-climbing ciego), o cortes **simétricos** donde el
presente de cada bloque era `futuros ∩ dims` — sobre-cortaba e inflaba Φ para k ≥ 3.

### Actual
Cada bloque es `Block = (frozenset futuros, frozenset presentes)`, con futuro y presente
particionados de forma **independiente**. El motor:

```
_construir_cut_pool(...)   → O(N) cortes (3 familias por nodo), construido UNA vez
_greedy_k_particion(...)   → desde 1 bloque, k-1 mejores splits del pool
_refinar_bloques_1move(...) → 1-move futuro + 1-move presente (asimétrico)
ILS (N_ILS=4)              → perturbar + re-refinar, conservar el mejor
```

Resultado: Φ coherente y mínimo real, idéntico a QNodes para todo k.

---

## 3. Refinamiento Local — Ausente vs. 1-Move Asimétrico + ILS

### Original
Sin refinamiento: la primera partición era la final.

### Actual
- **1-move futuro:** mover un nodo futuro entre bloques.
- **1-move presente (asimétrico):** mover el mecanismo de un nodo sin tocar su futuro.
- **ILS:** perturbación + re-refinamiento N_ILS = 4 veces, conservando el mejor Φ.

---

## 4. Soporte para k-Particiones (k > 2)

### Original
Sólo k = 2.

### Actual
`System.particionar()` generaliza `bipartir()` y procesa todos los n-cubos en una sola
pasada:

```python
dist_rec = subsistema.particionar(
    [(futuros_i, presentes_i) for (futuros_i, presentes_i) in bloques]
).distribucion_marginal()
```

El bucle externo evalúa k secuencialmente (cada k con todos los núcleos vía joblib) y
reporta la k de Φ mínimo global.

---

## 5. Manejo del Mecanismo Vacío (∅)

### Original
Over-cutting con penalizaciones artificiales cuando una parte quedaba sin presente.

### Actual
El pool incluye el corte `({i}, ∅)`: el futuro del nodo i se evalúa sin mecanismo
presente, dejando que el resto conserve su causalidad. En el pipeline fallback se marca
con el centinela `-1`. Sin penalización falsa.

---

## 6. Arquitectura — Imperativa vs. OOP con Clases de Dominio

| Clase / Módulo | Responsabilidad |
|---|---|
| `System` | Condicionamiento, substracción, `bipartir`/`particionar`, marginales |
| `NCube` | Hipercubo de probabilidad por nodo, marginalización cacheada |
| `KGeoMIP` | Motor greedy top-down asimétrico (+ fallback heurístico) |
| `Manager` | Carga de TPM, enrutamiento de estrategias |
| `Solution` | Representación y visualización del resultado |
| `LazyTPM` | Lectura lazy de TPM por chunks para N ≥ 18 |

`NCube` memoiza marginalizaciones por `frozenset` de ejes (conmutatividad).

---

## 7. Modo de Entrada — Terminal Interactivo + Modo Bloque CSV

**Modo Manual:** TPM por diálogo de archivo, candidato, estado inicial, alcance,
mecanismo y k por terminal.

**Modo Bloque:** CSV con múltiples pruebas; candidato y estado se ingresan una vez para
todo el lote; las pruebas se ordenan de menor a mayor complejidad; resultados a `.xlsx`
formateado / JSON.

---

## 8. Gestión de Memoria para N Grande — LazyTPM

Para N ≥ 18 se activa `LazyTPM`: la matriz no se carga completa; un generador lee
fragmentos (`chunks`) y acumula sólo las marginales necesarias, permitiendo N = 25+ sin
colapso de memoria. Además `_construir_tabla_costos` avisa y estima memoria si n_dims > 20.

---

## Resumen de Cambios

| Criterio | Original | Actual (GeoMIP AYDA 2026-1) |
|---|---|---|
| Métrica EMD | L1 sin fundamento (`emd_efecto`) | **L1 = EMD Hamming EXACTA** (descomposición marginal) |
| Costo por evaluación | — | O(N), sin límite de tamaño |
| Particiones soportadas | Solo k = 2 | k = 2 hasta min(6, N) |
| Representación | Listas simétricas | `Block = (frozenset futuros, frozenset presentes)` asimétrico |
| Motor principal | Hill-climbing ciego | **Greedy top-down** + 1-move + ILS |
| Generación de candidatos (fallback) | — | Spectral + Agglomerative + Aislamiento |
| Refinamiento | Ninguno | 1-move futuro/presente + ILS |
| Mecanismo vacío (∅) | Over-cutting | Corte `({i}, ∅)` riguroso |
| Arquitectura | Funciones sueltas | OOP: System, NCube, KGeoMIP, Manager |
| Paralelismo | Ninguno | joblib, cpu_count-1 núcleos por k |
| Entrada | Excel fijo | Terminal interactivo + CSV en bloque |
| Memoria | Carga total | LazyTPM por chunks para N ≥ 18 |
