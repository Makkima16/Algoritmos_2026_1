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
3. _refinar_bloques_1move(...) → 1-move futuro + 1-move presente (asimétrico) [fase final]
```

> **Cambio (2026-06-12):** se retiró la **ILS** (paso 4: perturbar + re-refinar ×4) por
> ganancia marginal frente a su costo. El motor `greedy + 1-move` es ahora más rápido y
> **determinista**. Ver [`decision_sin_ils.md`](decision_sin_ils.md).

Cada split evalúa `inside = (b.fut ∩ c.fut, b.pre ∩ c.pre)` y
`outside = (b.fut − c.fut, b.pre − c.pre)`, eligiendo el de menor Φ. El corte de
**mecanismo vacío** `({i}, ∅)` deja que el nodo i siga condicionando causalmente al
resto, sin penalización falsa. **Invariante (2026-06-12):** ningún bloque puede quedar
con el **futuro vacío** (lo que eliminaba partes degeneradas `(∅, ∅)`); y si
`permitir_presente_vacio = False`, tampoco con el **presente vacío** (el flag se respeta
en split y refinamiento).

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

La tabla alimenta el pool de cortes geométricos (`_construir_cut_pool`, el mejor
representante por cáscara de Hamming). Su costo se amortiza entre todos los k del lote,
que reutilizan la misma tabla.

> **Cambio (2026-06-14):** la tabla **ya NO** se usa para construir una matriz de
> afinidad geométrica. Antes, `_construir_tabla_costos` terminaba llamando a
> `_construir_matriz_afinidad`, que recorría las `2^n_dims × n` celdas **dos veces en
> float64** (`astype(float64)` + división por normas) para producir una matriz N×N que
> **ningún camino de producción consumía** (solo la usaba `_evaluar_k_completo`, el
> camino espectral antiguo sin uso). Era CPU desperdiciado en **cada** corrida y para
> **todo N**, y el pico de RAM dominante en N grande (ver §11). Se eliminó la llamada.

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

## 6. Refinamiento presente asimétrico (más preciso)

**Archivo:** `src/controllers/strategies/kgeomip.py` — `_refinar_bloques_1move`

El refinamiento 1-move explora **dos** tipos de movimiento, el segundo exclusivo de la
representación asimétrica:

- **Movimiento futuro:** traslada un nodo futuro del bloque i al j (sin vaciar el futuro
  del bloque origen).
- **Movimiento presente (asimétrico):** traslada un nodo del lado presente sin tocar
  su par futuro — imposible en cortes simétricos. Si `permitir_presente_vacio = False`,
  no se permite vaciar el presente de un bloque.

Este es el **paso final** del motor: el movimiento presente asimétrico amplía el espacio
explorado y baja Φ por debajo del mínimo local del 1-move clásico.

> **Cambio (2026-06-12):** la **ILS** (`_perturbar_bloques` + re-refinar ×4) que seguía a
> esta fase fue **retirada** por mejora marginal a costo alto, y dejó el motor
> determinista. Detalle en [`decision_sin_ils.md`](decision_sin_ils.md).

---

## 7. El salto de 2 (VNS 2-move): por qué aporta poco

**Archivo:** `src/controllers/strategies/kgeomip.py` — `_refinar_bloques_2move`, bucle
VNS en `aplicar_estrategia` (`N_VNS_MAX`)

Tras converger el 1-move (mínimo local), se evaluó una **VNS** que aplica **pares de
movimientos 1-move simultáneos** (`_refinar_bloques_2move`) para escapar de ese mínimo,
repitiéndola mientras siguiera mejorando (`N_VNS_MAX` ciclos). La conclusión empírica es
que **casi nunca mejora el óptimo y cuesta mucho**, por lo que quedó desactivada
(`N_VNS_MAX = 0`).

### El costo

El 2-move genera todos los pares de movimientos válidos: con M ≈ O(N·k) movimientos
1-move hay **O(M²)** configuraciones, y **cada una** se evalúa marginalizando sobre 2^N.
Un solo pase ya es el término más caro del refinamiento; el bucle VNS lo repetía hasta 3
veces cada vez que el 1-move volvía a quedar atrapado.

### Por qué no se gana (conceptual)

1. **El objetivo es separable por nodo.** Como Φ es la suma L1 de marginales por nodo
   (sección 1), el movimiento de **un solo** nodo ya captura casi todo el gradiente de la
   función. Los pares de movimientos correlacionados rara vez desbloquean una
   configuración estrictamente mejor que la que el 1-move ya alcanza.
2. **Las particiones óptimas aquí son de aislamiento.** El cut-pool geométrico + los
   candidatos de aislamiento ya generan directamente las estructuras ganadoras (aislar
   nodos con ∅), y el **1-move asimétrico** (mover el presente sin el futuro) basta para
   llegar al óptimo local, que coincide con el global. El 2-move entonces solo
   **confirma** ese óptimo (sin mejora) a costo cuadrático.

### Evidencia (N22, estado todo-1, sistema completo, alcance ABDEGHJKMNPQSTV)

| k | VNS | Φ KGeoMIP | t búsqueda | Φ KQNodes | Resultado |
|---|---|---|---|---|---|
| 4 | 1 pase | 1.499087 | 59.2 s | 1.499087 | empate |
| 4 | **0 (off)** | **1.499087** | **14.7 s** | 1.499087 | empate, **≈4× más rápido** |
| 5 | 1 pase | 1.998851 | 176.9 s | 1.998958 | GeoMIP +1e-4 |
| 5 | **0 (off)** | **1.998851** | **48.5 s** | 1.998958 | GeoMIP +1e-4, **≈3.6× más rápido** |

Quitar el 2-move **no cambió Φ en ninguno de los dos k** (mismo empate en k=4, misma
ventaja por 1e-4 en k=5) y recortó la búsqueda 3–4×. La ventaja de KGeoMIP sobre KQNodes
en k=5 proviene del **1-move presente asimétrico** (sección 6), no del 2-move — por eso
eliminarlo no la pierde.

> **Decisión (2026-06-14):** `N_VNS_MAX = 0`. La VNS 2-move queda como código disponible
> pero inactivo; es reactivable subiendo `N_VNS_MAX` si una topología concreta lo
> justificara. Misma lógica que el retiro del ILS: mejora marginal a costo desproporcionado.

### Nota (2026-06-14): el ILS ligero corrió la misma suerte

El mismo día se volvió a probar un **ILS ligero** (`_perturbacion_bloques` +
re-refinamiento 1-move, 2 reinicios con semilla fija) sobre la salida del 1-move,
para confirmar si reactivarlo desde el retiro del 2026-06-12 cambiaba algo con la nueva
métrica marginal. **No mejoró Φ en ninguna prueba** (mismo argumento de separabilidad por
nodo de la sección anterior) y multiplicaba el refinamiento, así que también quedó
**desactivado**: `N_ILS_LIGHT = 0`. Tanto `_refinar_bloques_2move` como
`_perturbacion_bloques` permanecen en el código como referencia reactivable, pero el motor
en producción es **greedy top-down → 1-move → fin**, determinista. Ver
[`decision_sin_ils.md`](decision_sin_ils.md).

---

## 8. Gestión de memoria para N grande: LazyTPM + aviso de tabla (más robusto)

**Archivos:** `src/lazy_tpm.py`; `_construir_tabla_costos` (aviso para n_dims > 20)

Para redes con N ≥ 18 nodos, cargar la TPM completa colapsa la RAM. **LazyTPM** lee la
matriz por **chunks** secuenciales y acumula sólo las marginales necesarias, sin
materializar nunca la matriz entera. Además, `_construir_tabla_costos` emite un aviso
y estima la memoria cuando n_dims > 20, para que el usuario anticipe el costo de
`tabla_T` antes de quedarse sin memoria.

---

## 9. ⚠️ El principal cuello de botella: el arranque, no la búsqueda

**Archivos:** `exec_kgeomip.py`, `src/controllers/manager.py`, `kgeomip.py`

La búsqueda greedy + 1-move es rápida una vez preparada la infraestructura. Lo que
domina el tiempo de GeoMIP —sobre todo en N ≥ 20— es el **arranque del motor**:

| Paso del arranque | Coste | N=22 |
|---|---|---|
| LazyTPM: lectura chunks del CSV | O(2^N × N / chunk) | ~0.5 s |
| Condicionamiento del subsistema (NCubos) | O(N × 2^N) | ~2 s |
| `_construir_tabla_costos` (BFS Hamming) | O(N × 2^N) | ~5 s |
| ~~Matriz de afinidad geométrica~~ | ~~O(N² × 2^N) / joblib~~ | ~~~20 s~~ **eliminada (2026-06-14)** |
| `_construir_cut_pool` (pool geométrico) | O(N²) | <1 s |

> **Cambio (2026-06-14):** la matriz de afinidad era el paso **más caro** del arranque
> (~20 s en N=22) y se construía sin que nada la consumiera. Al eliminarla (§11), el
> arranque cae a ~7.5 s en N=22 y deja de ser, con diferencia, el cuello de botella.

El primer k llamado paga todo esto. Los k siguientes reutilizan el subsistema
(cacheado en Manager) y la tabla, por lo que son sensiblemente más rápidos.

En N=22: k=2 tardó **31.9 s** (arranque incluido) vs k=3: **11.9 s**, k=4: **10.7 s**.
En N=20: k=2 tardó **6.2 s** (arranque incluido) vs k=3: **2.7 s**.

### Modo manual

Cada problema en modo manual paga el arranque completo (subsistema + tabla de costos).
Cambiar candidato o estado invalida el caché. El tiempo reportado es arranque + búsqueda.

### Modo por bloque

El modo batch ordena las pruebas de menor a mayor complejidad, cachea el **candidato**
condicionado para todo el lote (un solo `condicionar`, lo más caro y común a todas las
pruebas), y ejecuta una corrida de **calentamiento descartable** (warmup) antes de la
primera prueba. La fila **"Arranque del motor (warmup)"** contabiliza solo ese arranque
**único** (Numba JIT + pool de hilos + condicionado del candidato).

> **Cambio (2026-06-14):** antes, la **preparación por prueba** (el `substraer` que
> construye el subsistema de cada prueba — depende de su alcance/mecanismo, que cambian
> por fila y por eso NO se cachean) se sumaba al "Arranque motor", y la fila de la prueba
> mostraba **solo su búsqueda**. Esto hacía que cada prueba pareciera más barata que su
> tiempo real de pared y que el "Arranque motor" creciera con cada prueba. Ahora esa
> preparación por prueba se **incluye en el tiempo de la prueba** (preparación + búsqueda),
> y el "Arranque motor" queda como costo único. El **tiempo total del lote no cambia**;
> solo se reparte de forma honesta. *(KQNodes tenía el mismo reparto y se corrigió igual;
> además no cachea nada, así que cada prueba reconstruye su subsistema completo — su fila
> "Arranque motor" queda en 0 porque no hay fase de calentamiento separada.)*

El arranque se reporta aparte como "Arranque del motor (warmup)" en el XLSX / SSE del
dashboard. En la comparación del dashboard, empate de Φ → **"Ambos"**.

**Implicación:** en lotes grandes el arranque se amortiza y GeoMIP es competitivo
en tiempo total. Para pruebas sueltas o k=2, QNodes (sin arranque costoso, exacto para
k=2 vía Queyranne) es preferible.

---

## 10. Comparación con QNodes — calidad vs velocidad

**Referencia:** mediciones 2026-06-13 (N22A, estado='1000…0', sistema completo)

| k | GeoMIP φ | GeoMIP t | QNodes φ | QNodes t | Mejor φ | Más rápido |
|---|---------|---------|---------|---------|--------|-----------|
| 2 | 0.499575 | 31.9 s | 0.499575 | **6.5 s** | empate | **QNodes ×4.9** |
| 3 | **0.999150** | 11.9 s | 0.999189 | **6.1 s** | **GeoMIP** | QNodes ×2.0 |
| 4 | **1.498764** | 10.7 s | 1.498915 | **5.5 s** | **GeoMIP** | QNodes ×2.0 |
| 5 | **1.998490** | 12.5 s | 1.998667 | **5.8 s** | **GeoMIP** | QNodes ×2.2 |

**Por qué GeoMIP es mejor en Φ para k≥3:** el pool de cortes geométricos (familia
derivada de `tabla_T` — el mejor representante por cáscara de Hamming en
`_construir_cut_pool`) expone particiones que el greedy asimétrico puro no genera. A
N≤20 ambos coinciden; a N≥22 la exploración geométrica empieza a dominar. *Nota: esta
ventaja proviene de la tabla de costos, NO de la matriz de afinidad eliminada en 2026-06-14
(que alimentaba el camino espectral sin uso); por eso quitarla no cuesta calidad.*

**Por qué QNodes es siempre más rápido:** no construye tabla de costos (KGeoMIP sí, aunque
desde 2026-06-14 ya no la matriz de afinidad); `marginal_valor` (desde 2026-06-13) reduce cada evaluación de bloque a
O(2^(N/2)) en vez de O(2^N). Para k=2, el algoritmo de Queyranne garantiza además el
óptimo global exacto en O(N²) evaluaciones, algo que GeoMIP (heurístico) no puede.

**Resumen práctico:**
- **k=2, cualquier N:** QNodes — exacto y ~4–5× más rápido.
- **k≥3, N≤20:** ambos equivalentes en Φ; QNodes más rápido.
- **k≥3, N≥22:** GeoMIP — mejor Φ; QNodes — mejor velocidad.

---

## 11. ⭐ Reducción de RAM pico (2026-06-14): afinidad muerta + copia única de `prob`

**Archivo:** `src/controllers/strategies/kgeomip.py` — `_construir_tabla_costos`

Dos cambios que **bajan el pico de RAM** sin tocar Φ, habilitando N grande (N25 completo)
en máquinas modestas. Para N=25 completo, `2^25 ≈ 33.5 M` filas × 25 cols × 4 B (float32) =
**~3.35 GB por cada arreglo `(2^n_dims, n)`**; el problema era cuántos coexistían.

### 11.1 Eliminar la matriz de afinidad (gana ~13 GB y CPU en TODO N)

`_construir_tabla_costos` llamaba `_construir_matriz_afinidad()` al final, que hacía
`C = tabla_T.astype(np.float64)` (+6.7 GB) y `C_norm = C / norms` (+6.7 GB) para producir
una matriz N×N que **nadie lee**: `_evaluar_k_completo` / `_particion_grafo_hipercubo`
(camino espectral) no se invocan desde `aplicar_estrategia`. Era doble pasada en float64 en
**cada** corrida (CPU desperdiciado para todo N) y el pico de RAM dominante en N grande.
Se eliminó la llamada. No cambia Φ (verificado: rutas Numba y numpy dan el mismo Φ).

### 11.2 Construir `prob_T` directo, sin duplicar (gana ~3.35 GB)

La ruta Numba creaba `prob` (n×2^d) y luego `prob_T = ascontiguousarray(prob.T)` (2^d×n):
**dos copias de los mismos datos** a la vez. Ahora `prob_T` se llena columna a columna
directamente desde los n-cubos y `prob` no se construye (solo existe en la ruta numpy sin
Numba, que sí opera por bloques de columnas). Una copia en vez de dos; mismos valores, mismo Φ.

### Presupuesto de RAM pico (N=25 completo)

| Etapa | Antes | Después |
|---|---|---|
| Cubos del subsistema (piso inevitable) | ~3.35 GB | ~3.35 GB |
| `prob` + `prob_T` (doble copia) | ~6.7 GB | **~3.35 GB** (solo `prob_T`) |
| `tabla_T` | ~3.35 GB | ~3.35 GB |
| Matriz de afinidad (`astype(f64)` + división) | **~13.4 GB** | **0** (eliminada) |
| **Pico aproximado** | **~20 GB** | **~7 GB** |

Con esto N25 entra en RAM. El piso real (~3.35 GB de cubos del subsistema) sigue ahí porque
las marginales necesitan los n-cubos completos; un futuro "modo bajo memoria" que omita
`tabla_T` para N ≥ umbral (usando solo aislamiento + complemento + greedy + 1-move, que no
requieren la tabla) bajaría el pico a ese piso.

### 11.3 Cachés acotados: fin de la fuga de RAM entre pruebas (2026-06-14)

**Archivo:** `src/models/base/sia.py` — `_CANDIDATO_CACHE`, `_SUBSISTEMA_CACHE`, `_acotar_cache`

`sia_preparar_subsistema` cachea el candidato condicionado y el subsistema. La clave del
**subsistema incluye alcance y mecanismo**, que cambian en cada prueba de un lote → su clave
es única y **nunca se reutiliza**. Como el caché no tenía límite, acumulaba **un subsistema
(con sus NCubos, enormes en N grande) por prueba**, y en el worker persistente del dashboard
a lo largo de TODOS los lotes. La RAM crecía prueba a prueba → el SO **paginaba a disco** →
las pruebas siguientes (incluso una idéntica) se volvían **progresivamente más lentas**.

Ahora ambos cachés se **acotan por inserción** (FIFO) a unas pocas entradas
(`_CANDIDATO_CACHE_MAX`, `_SUBSISTEMA_CACHE_MAX = 2`): el candidato es común a todo el lote
(basta 1, evita reconstruir el `System` completo O(N·2^N) por prueba) y el subsistema solo
se reutilizaría si se repite EXACTAMENTE la misma prueba (caso raro). El reuso legítimo del
subsistema entre los k de UNA misma llamada no se ve afectado (usa `self.sia_subsistema`, no
el caché). Esto elimina la degradación progresiva del modo bloque / dashboard sin tocar Φ.

> KQNodes no tiene estos cachés (reconstruye su subsistema por prueba), así que nunca sufrió
> esta fuga; el cambio es exclusivo de KGeoMIP.

---

## Resumen de impacto por optimización

| Optimización | Efecto | Velocidad | Precisión |
|---|---|---|---|
| **L1 = EMD Hamming exacta** | O(4^N) → O(N), sin límite N≤12 | ⭐⭐⭐ | ⭐⭐⭐ (mismo Φ, exacto) |
| **Cortes asimétricos + greedy top-down** | Corrige el sobre-corte | ⭐ | ⭐⭐⭐ (Φ coherente y mínimo) |
| Tabla de costos vectorizada (BFS + popcount + chunks) | Llenado O(n_dims·2^n_dims) | ⭐⭐ | — |
| `particionar()` de una pasada | k bipartir → 1 pasada | ⭐⭐ | — |
| Paralelismo joblib por k | Todos los núcleos por test | ⭐⭐ | — |
| Refinamiento presente 1-move (asimétrico) | Vecindario asimétrico; baja Φ | ⭐ | ⭐⭐ |
| Retiro de ILS (2026-06-12) | Menos ~5× del refinamiento; determinista | ⭐⭐ | — (Φ equivalente) |
| Desactivar VNS 2-move (2026-06-14) | Quita el término O(M²); 3–4× la búsqueda | ⭐⭐ | — (Φ idéntico) |
| **Eliminar matriz de afinidad (2026-06-14)** | Quita ~20 s de arranque y ~13 GB de RAM, en TODO N | ⭐⭐⭐ | — (Φ idéntico) |
| **`prob_T` sin doble copia (2026-06-14)** | −~3.35 GB de RAM pico en N25 | ⭐ | — (Φ idéntico) |
| LazyTPM (N ≥ 18) | Evita Out-of-Memory | — | — (habilita N grande) |

Las dos primeras filas son las que hacen a GeoMIP simultáneamente más rápido **y** más
preciso, y las que lo alinean exactamente con QNodes para todo k.
