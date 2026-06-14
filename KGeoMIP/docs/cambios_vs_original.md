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
_refinar_bloques_1move(...) → 1-move futuro + 1-move presente (asimétrico) [fase final]
```

Resultado: Φ coherente y mínimo real, idéntico a QNodes para todo k. Desde 2026-06-12 el
motor es además **determinista** (se retiró la ILS — ver sección 9 y
[`decision_sin_ils.md`](decision_sin_ils.md)).

---

## 3. Refinamiento Local — Ausente vs. 1-Move Asimétrico

### Original
Sin refinamiento: la primera partición era la final.

### Actual
- **1-move futuro:** mover un nodo futuro entre bloques (sin vaciar el futuro del origen).
- **1-move presente (asimétrico):** mover el mecanismo de un nodo sin tocar su futuro.

> **Nota (2026-06-12):** antes existía una fase de **ILS** (perturbación +
> re-refinamiento ×4) tras el 1-move; fue retirada por mejora marginal y costo alto.
> El 1-move es ahora la fase final. Ver sección 9.

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

El comportamiento está controlado por `permitir_presente_vacio` y ahora se **respeta en
todo el camino greedy** (split, refinamiento): con `False`, ningún bloque puede quedar
con presente ∅; con `True`, sí. En ningún caso un bloque puede quedar con el **futuro
vacío**, lo que elimina partes degeneradas `(∅, ∅)`. Ver sección 9.

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
| Motor principal | Hill-climbing ciego | **Greedy top-down** + 1-move (determinista) |
| Generación de candidatos (fallback) | — | Spectral + Agglomerative + Aislamiento |
| Refinamiento | Ninguno | 1-move futuro/presente (ILS retirada — sección 9) |
| Mecanismo vacío (∅) | Over-cutting | Corte `({i}, ∅)` riguroso; flag respetado; sin `(∅,∅)` |
| Arquitectura | Funciones sueltas | OOP: System, NCube, KGeoMIP, Manager |
| Paralelismo | Ninguno | joblib, cpu_count-1 núcleos por k |
| Entrada | Excel fijo | Terminal interactivo + CSV en bloque |
| Memoria | Carga total | LazyTPM por chunks para N ≥ 18 |

---

## 9. GeoMIP vs QNodes — por qué GeoMIP es mejor en Φ para k≥3 a N grande

Ambos frameworks comparten la misma métrica (L1 = EMD Hamming exacta), la misma
representación asimétrica de bloques y el mismo motor greedy top-down + 1-move. La
diferencia está en la **tercera familia del pool de cortes**:

- **QNodes:** pool de O(N) cortes generado por 3 familias de aislamiento asimétrico
  (simétrico, complemento, mecanismo vacío). Exploración directa sobre el espacio de
  cortes; no usa geometría de la TPM.
- **GeoMIP:** añade cortes derivados de la **afinidad espectral** de las columnas de la
  tabla de costos `tabla_T` (similitud coseno entre nodos según su comportamiento
  probabilístico). Esto expone particiones geométricamente motivadas que el greedy
  asimétrico puro no genera.

A N ≤ 20, ambos encuentran el mismo Φ (la geometría espectral no aporta ventaja sobre
el greedy base a esos tamaños). A N ≥ 22, GeoMIP encuentra Φ menores para k≥3:

| k | GeoMIP N=22 φ | QNodes N=22 φ | Δ | GeoMIP t | QNodes t |
|---|--------------|--------------|---|---------|---------|
| 2 | 0.499575 | 0.499575 | 0 (empate) | 31.9 s | **6.5 s** |
| 3 | **0.999150** | 0.999189 | −0.000039 | 11.9 s | **6.1 s** |
| 4 | **1.498764** | 1.498915 | −0.000151 | 10.7 s | **5.5 s** |
| 5 | **1.998490** | 1.998667 | −0.000177 | 12.5 s | **5.8 s** |

**Lo que GeoMIP sacrifica:** velocidad. QNodes es 2×–5× más rápido en N=22. El tiempo
extra de GeoMIP es principalmente **arranque** (tabla de costos + matriz de afinidad),
no búsqueda; ver `GeoMIP_Optimizaciones.md` §8 y el README del proyecto.

**Para k=2:** QNodes usa el algoritmo de Queyranne (1998) que garantiza el **óptimo
global exacto** de la bipartición. GeoMIP es heurístico y puede perderlo. El empate en
N=22 es contingente; no hay garantía de que GeoMIP lo alcance siempre.

---

## 10. Changelog 2026-06-12 (correcciones y simplificación)

Tres cambios al motor greedy top-down de `KGeoMIP`:

1. **Bug `(∅, ∅)` corregido — invariante de futuro no vacío.**
   `_mejor_split_bloques` permitía crear un bloque con el **futuro vacío** `(∅, presente)`;
   el movimiento presente del refinamiento podía luego vaciar también su presente,
   produciendo una parte degenerada `(∅, ∅)` (alcance **y** mecanismo vacíos) que bajaba Φ
   artificialmente. Ahora **ningún split puede dejar un bloque sin futuro**, lo que vuelve
   imposible el `(∅, ∅)`.

2. **Flag `permitir_presente_vacio` respetado en el camino greedy.**
   El flag estaba conectado a la firma de `aplicar_estrategia` pero **no se propagaba** al
   greedy (split, 1-move, perturbación), así que el mecanismo ∅ aparecía siempre. Ahora se
   enhebra por `_mejor_split_bloques`, `_greedy_k_particion` y `_refinar_bloques_1move`:
   con `False`, ningún bloque queda con presente ∅; con `True`, sí.

3. **Búsqueda Local Iterada (ILS) retirada.**
   Se eliminaron la fase ILS, la función `_perturbar_bloques` y la constante `N_ILS`. La
   ILS aportaba mejoras marginales (rara vez superaba a QNodes) a un costo de ~5× el
   refinamiento, e introducía no-determinismo. El motor `greedy + 1-move` es ahora **más
   rápido y determinista**. Documento dedicado: [`decision_sin_ils.md`](decision_sin_ils.md).
