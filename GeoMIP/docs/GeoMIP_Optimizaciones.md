# Optimizaciones en GeoMIP (AYDA 2026-1)

Este documento detalla las optimizaciones que hacen el cálculo de la k-MIP en
GeoMIP **más rápido y más preciso**, para sistemas de alta dimensionalidad
(N ≥ 20).

Las dos optimizaciones de mayor impacto son conceptuales:

1. **La distancia L1 marginal es la EMD de Hamming EXACTA** (no una aproximación) →
   cada evaluación de Φ pasa de O(4^N) a O(N) **sin perder precisión** y sin límite
   de tamaño.
2. **Los cortes asimétricos + greedy top-down** (futuro y presente particionados
   independientemente) evitan el "sobre-corte" → Φ **coherente y mínimo real**,
   idéntico al de QNodes.

El resto (tabla de costos vectorizada, `particionar()` de una pasada, paralelismo
joblib, LazyTPM) acelera sobre estas dos bases.

---

## 1. ⭐ L1 marginal = EMD de Hamming EXACTA (más rápida Y más precisa)

**Archivo:** `src/controllers/strategies/kgeomip.py` — `evaluar_bloques`, `evaluar_k_particion`

### El teorema que lo hace exacto

La distribución original y la reconstruida de cualquier k-partición son **productos
de marginales por nodo** (independencia condicional garantizada por construcción).
Para dos distribuciones producto sobre el hipercubo booleano con métrica base de
Hamming, la Wasserstein-1 (EMD real) **se descompone exactamente** en la suma de
EMDs marginales unidimensionales:

```
EMD_Hamming(P, Q) = Σᵢ | P(nodo_i = 1) − Q(nodo_i = 1) |     (teorema de descomposición marginal)
```

```python
# evaluar_bloques(): O(N), exacta, válida para todo N
dist_rec = subsistema.particionar(particiones).distribucion_marginal()
return float(np.sum(np.abs(dist_original - dist_rec)))
```

### Por qué es más rápida

Construir la EMD "real" con `pyemd` exigía la distribución conjunta de 2^N estados y
una matriz de costes Hamming de 2^N × 2^N → **O(4^N)**. Para N = 20 esa matriz
requeriría cientos de gigabytes. La fórmula marginal es **O(N)**: un vector de N
restas absolutas.

### Por qué es más precisa

- **Da el mismo Φ que la EMD real**, no una aproximación.
- **Elimina el límite N ≤ 12.** Antes, N grande caía a una aproximación; ahora todos
  los N usan la fórmula exacta. Por eso `evaluar_k_particion` documenta explícitamente
  *"Valid for all N with no size restriction"*.
- Coincide **exactamente** con QNodes para todo k, porque ambos comparten métrica
  exacta y cortes asimétricos.

---

## 2. ⭐ Cortes asimétricos + greedy top-down (más preciso)

**Archivo:** `src/controllers/strategies/kgeomip.py` — `Block`, `_construir_cut_pool`,
`_greedy_k_particion`, `_mejor_split_bloques`

### El problema previo

Con cortes **simétricos**, el presente de cada bloque era siempre `futuros ∩ dims`:
cada grupo sólo condicionaba sobre sus propios nodos. Esto **sobre-cortaba** las
conexiones causales e inflaba Φ, sobre todo para k ≥ 3.

### La solución

Cada bloque es ahora un `Block = (frozenset futuros_globales, frozenset
presentes_globales)`, con futuro y presente particionados de forma **independiente**.
El motor principal de `aplicar_estrategia` es greedy top-down sobre estos bloques:

```
1. _construir_cut_pool(...)  → pool de O(N) cortes, construido UNA vez, compartido por todo k
2. _greedy_k_particion(...)  → desde 1 bloque (todo el subsistema), k−1 mejores splits
3. _refinar_bloques_1move(...) → 1-move futuro + 1-move presente (asimétrico)
4. ILS (N_ILS=4)             → perturbar + re-refinar, conservar el mejor
```

Cada split evalúa `inside = (b.fut ∩ c.fut, b.pre ∩ c.pre)` y
`outside = (b.fut − c.fut, b.pre − c.pre)`, eligiendo el de menor Φ. El corte de
**mecanismo vacío** `({i}, ∅)` deja que el nodo i siga condicionando causalmente al
resto, sin penalización falsa.

### Impacto

Φ resulta coherente y mínimo real; el pipeline antiguo basado en SpectralClustering
queda como **fallback** (`_evaluar_k_completo`, `_agrupamiento_jerarquico`) y ya no
es la ruta principal.

---

## 3. Tabla de costos vectorizada por niveles de Hamming (más rápido)

**Archivo:** `src/controllers/strategies/kgeomip.py` — `_construir_tabla_costos`

La tabla de costos `tabla_T` de forma (2^n_dims, n) se construye con un **BFS
vectorizado por niveles de Hamming**, no estado por estado:

```python
for d in range(1, n_dims + 1):
    estados_nivel = all_states[dist == d]          # todos los estados a distancia d
    gamma = np.float32(1.0 / (1 << d))             # decrecimiento exponencial 2^(-d)
    for start in range(0, len(estados_nivel), COST_TABLE_CHUNK_ROWS):
        chunk = estados_nivel[start : start + COST_TABLE_CHUNK_ROWS]
        diff = np.abs(flat[:, chunk].T - origin_values)
        # acumula contribuciones de vecinos a distancia d-1 (vectorizado)
        self.tabla_T[chunk] = (diff + accumulated) * gamma
```

Optimizaciones concretas:
- **Popcount vectorizado** (`_popcount_vec` con `np.unpackbits`) calcula las
  distancias de Hamming de los 2^n_dims estados de una vez.
- **Procesamiento por chunks** (`COST_TABLE_CHUNK_ROWS`) acota el pico de memoria al
  llenar la tabla.
- Complejidad O(n_dims · 2^n_dims) en tiempo, O(2^n_dims · n) en espacio, en vez de
  un cálculo recursivo por par de estados.

La misma tabla alimenta la matriz de afinidad geométrica y el pool de cortes, así que
su costo se amortiza.

---

## 4. Evaluación de una sola pasada con `particionar()` (más rápido)

**Archivo:** `src/controllers/strategies/kgeomip.py` — `evaluar_bloques`

`evaluar_bloques` usa `System.particionar()` para procesar **todos** los n-cubos en
una sola pasada, en vez de un `bipartir()` por bloque:

```python
particiones = [(np.array(sorted(fut)), np.array(sorted(pre)) or vacío)
               for fut, pre in bloques if fut]
dist_rec = subsistema.particionar(particiones).distribucion_marginal()
```

Cada cubo se marginaliza conservando sólo las dimensiones del mecanismo de **su
propio** bloque, manteniendo la independencia futuro/presente entre bloques sin forzar
intersección simétrica. Una pasada en lugar de k.

---

## 5. Paralelismo dedicado: todos los núcleos por k (más rápido)

**Archivo:** `src/controllers/strategies/kgeomip.py` — `N_JOBS_INTERNOS`, joblib

El bucle de k es **secuencial** (k=2, luego k=3, …), pero cada k usa
`cpu_count − 1` núcleos para evaluar candidatos y vecinos en paralelo con joblib
(`prefer="threads"`):

```python
perdidas = Parallel(n_jobs=min(len(vecinos), N_JOBS_INTERNOS), prefer="threads")(
    delayed(evaluar_bloques)(subsistema, cfg, dist_original) for cfg in vecinos
)
```

Enfocar todos los núcleos en un solo test a la vez evita dividir recursos entre k's,
no satura el bus de memoria y maximiza el rendimiento por tarea.

---

## 6. Refinamiento presente asimétrico + ILS (más preciso)

**Archivo:** `src/controllers/strategies/kgeomip.py` — `_refinar_bloques_1move`,
`_perturbar_bloques`

El refinamiento 1-move explora **dos** tipos de movimiento, el segundo exclusivo de la
representación asimétrica:

- **Movimiento futuro:** traslada un nodo futuro del bloque i al j.
- **Movimiento presente (asimétrico):** traslada un nodo del lado presente sin tocar
  su par futuro — imposible en cortes simétricos.

Tras converger, la **ILS** (N_ILS = 4) perturba alternando movimientos futuros y
presentes, y re-refina, conservando siempre el mejor Φ. Ambos amplían el espacio
explorado y bajan Φ por debajo del mínimo local del 1-move clásico.

---

## 7. Gestión de memoria para N grande: LazyTPM + aviso de tabla (más robusto)

**Archivos:** `src/lazy_tpm.py`; `_construir_tabla_costos` (aviso para n_dims > 20)

Para redes con N ≥ 18 nodos, cargar la TPM completa colapsa la RAM. **LazyTPM** lee la
matriz por **chunks** secuenciales y acumula sólo las marginales necesarias, sin
materializar nunca la matriz entera. Además, `_construir_tabla_costos` emite un aviso
y estima la memoria cuando n_dims > 20, para que el usuario anticipe el costo de
`tabla_T` antes de quedarse sin memoria.

---

## 8. Modo por bloque inteligente (de fácil a difícil)

**Archivo:** `exec_kgeomip.py`

El modo batch ordena la cola de pruebas **de menor a mayor complejidad** (más ceros en
la máscara = menos nodos activos = menor cardinalidad). Esto da feedback rápido de los
casos triviales mientras el motor escala hacia los costosos. El candidato y el estado
inicial se ingresan y cachean **una sola vez** para todo el lote, y se calientan las
cachés y los pools de joblib antes de la primera prueba.

---

## Resumen de impacto por optimización

| Optimización | Efecto | Velocidad | Precisión |
|---|---|---|---|
| **L1 = EMD Hamming exacta** | O(4^N) → O(N), sin límite N≤12 | ⭐⭐⭐ | ⭐⭐⭐ (mismo Φ, exacto) |
| **Cortes asimétricos + greedy top-down** | Corrige el sobre-corte | ⭐ | ⭐⭐⭐ (Φ coherente y mínimo) |
| Tabla de costos vectorizada (BFS + popcount + chunks) | Llenado O(n_dims·2^n_dims) | ⭐⭐ | — |
| `particionar()` de una pasada | k bipartir → 1 pasada | ⭐⭐ | — |
| Paralelismo joblib por k | Todos los núcleos por test | ⭐⭐ | — |
| Refinamiento presente + ILS | Vecindario asimétrico + escape de mínimos | ⭐ | ⭐⭐ |
| LazyTPM (N ≥ 18) | Evita Out-of-Memory | — | — (habilita N grande) |

Las dos primeras filas son las que hacen a GeoMIP simultáneamente más rápido **y** más
preciso, y las que lo alinean exactamente con QNodes para todo k.
