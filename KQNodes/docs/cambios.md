# Cambios con respecto al proyecto original

**Referencia:** `projecto-analisis-20261/QNodes` → `AYDA_2026_1/QNodes`

> **Estado actual (2026-06-13):** QNodes es un **motor asimétrico unificado** con
> dos capas de optimización que lo hacen viable para N ≥ 20:
> - **Capa de evaluación:** `NCube.marginal_valor` reemplazó a `bipartir→marginalizar→distribucion_marginal`
>   → cada evaluación de bloque pasa de O(2^N) a O(2^(N-|mecanismo|)), speedup ×145 en N=20.
> - **Capa de búsqueda k=2:** el algoritmo de Queyranne (1998) con 2N átomos asimétricos
>   garantiza el óptimo global exacto de la bipartición en O(N²) evaluaciones.
>
> Sin embargo, para k ≥ 3 en N ≥ 22, la heurística greedy+ILS de QNodes queda por
> detrás de GeoMIP en calidad de Φ (ver §9). Este documento describe la evolución completa.

---

## 1. Cambio central: de bipartición greedy a k-partición asimétrica unificada

| Aspecto | `projecto-analisis-20261` | `AYDA_2026_1` (actual) |
|---|---|---|
| Particiones soportadas | Solo k = 2 (bipartición) | k ∈ [2, N], mismo motor para todo k |
| Representación | `list[tuple(tiempo, índice)]` | `Bloque = (frozenset futuros, frozenset presentes)` |
| Cortes | Simétricos | **Asimétricos** (futuro/presente independientes) |
| Algoritmo principal | Greedy incremental k=2 | Greedy **top-down** + 1-move + ILS |
| Métrica EMD | L1 marginal (asumida aproximación) | L1 marginal = **Wasserstein-1 Hamming EXACTA** |
| Límite de tamaño | N ≤ ~12 (por la EMD) | Sin límite — L1 es O(N) para todo N |
| Mecanismo vacío (∅) | No soportado | Soportado (corte `({i}, ∅)`), **controlado por `permitir_presente_vacio`** |

---

## 2. Representación asimétrica de bloques (el cambio que lo unifica todo)

Cada bloque es un par `(frozenset futuros, frozenset presentes)` donde el futuro (t+1)
y el presente/mecanismo (t) se particionan de forma **independiente**. Esto generaliza
a todo k el corte asimétrico que antes sólo se usaba para k=2, y elimina la distinción
entre el caso k=2 y k≥3.

**Causa raíz del salto k=2→k=3 (resuelto):** la versión intermedia usaba cortes
asimétricos sólo para k=2; para k≥3 usaba cortes **simétricos** que sobre-cortaban e
inflaban Φ. Medido en N10A: simétrico k=3 = 2.5059 vs asimétrico k=3 = 0.9590. El motor
actual usa asimétrico para **todo** k → Φ monótono y coherente.

---

## 3. Algoritmo: greedy top-down (no aglomerativo bottom-up)

El motor parte de **un solo bloque** (todo el subsistema) y aplica las mejores
divisiones:

```
_construir_pool_cortes()  → O(N) cortes (3 familias por nodo), construido UNA vez
_greedy_descenso(pool)    → un descenso k=1..N, registra Φ por cada k (jerarquía nido)
_refinar_bloques(...)     → 1-move futuro + 1-move presente (asimétrico)
_refinar_con_ils(...)     → perturbación + re-refinamiento (N-adaptativo)
```

Un único descenso produce una jerarquía **anidada** → coherencia (Φ monótono) entre k
consecutivos. (Detalles en `docs/estrategia-k-particion.md`.)

---

## 4. Corrección y aceleración del cálculo de EMD

El cambio de mayor impacto. La métrica L1 marginal **es** la Wasserstein-1 con Hamming
EXACTA (no una aproximación), porque tanto la distribución original como la
reconstruida de cualquier k-partición son **productos de marginales por nodo**:

```
EMD_Hamming(P, Q) = Σᵢ |P(nodo_i = ON) − Q(nodo_i = ON)|     (verificado |·| < 1e-14 para N=2..12)
```

Consecuencias:
- **Más rápido:** O(N) por evaluación en vez de O(4^N) del solver `pyemd`.
- **Más preciso:** da el MISMO Φ que la EMD real (k=2 N10A = 0.4746, idéntico a GeoMIP).
- **Sin límite de tamaño:** se eliminó la dependencia de `pyemd` y el techo N ≤ 12.

> Esto **corrige** la nota antigua que afirmaba que L1 = pyemd "sólo si P es producto,
> lo cual no ocurre en IIT": sí ocurre aquí por construcción.

---

## 5. Refinamiento 1-move asimétrico + ILS (nuevo)

La versión original no tenía mejora post-greedy. La actual añade:
- **Movimiento futuro:** reubica un nodo futuro entre bloques.
- **Movimiento presente (asimétrico):** reubica el mecanismo de un nodo sin mover su
  futuro — imposible en representaciones simétricas.
- **ILS:** perturbación + re-refinamiento N-adaptativo para escapar de mínimos locales.

---

## 6. Entrada de datos: hardcodeada → interactiva

**Original:** estado, TPM y sistema definidos en `src/main.py`; cambiar de sistema
exigía editar el código.

**Actual:** `exec.py` con menú de dos modos:
- **Modo manual:** ingreso por terminal de estado, candidato, alcance, mecanismo y k.
- **Modo por bloque (CSV):** múltiples sistemas; resultados volcados incrementalmente.

El candidato y el estado inicial se ingresan una sola vez para todo el lote.

---

## 7. Infraestructura

| Capacidad | Original | Actual |
|---|---|---|
| Profiling | Manual | `pyinstrument` con reporte HTML |
| Logger | `print` | `slogger.py` con niveles y colores |
| Salida | Manual | Incremental por sistema |
| Validaciones | Mínimas | Exhaustivas (alcance ⊆ candidato, estado coherente) |

---

## 8. Optimización `marginal_valor` + Queyranne exacto para k=2 (2026-06-13)

Dos cambios que hacen QNodes viable para N ≥ 20 (antes N=20 k=2 tardaba 392 s):

### 8.1 `NCube.marginal_valor` — evaluación O(2^(N-|mecanismo|))

**Archivo:** `src/models/core/ncube.py` — campo `valor_memo`, método `marginal_valor`

**Antes:** `_dist_bloque` llamaba `bipartir(futuros, presentes).distribucion_marginal()` →
`marginalizar(V-pre)` → `np.mean(data, axis=ejes)` sobre el array COMPLETO de 2^N
elementos → **O(2^N) por NCube**, independientemente del tamaño del mecanismo.

**Ahora:** `marginal_valor(ejes, estado_inicial)` fija con indexación numpy las dims
fuera del mecanismo al `estado_inicial`, luego promedia SOLO el sub-array de tamaño
2^|ejes| → **O(2^(N-|mecanismo|))**. Matemáticamente equivalente por linealidad:
`E_{V-pre}[data[pre=s0,•]] == marginalizar(V-pre)[s0]`.

Speedup promedio: **2^(N/2)**. Para N=20, |mecanismo|≈10 → speedup ×1024 por llamada;
total algoritmo **×145** medido (392 s → 2.7 s para k=2).

Los resultados son ligeramente **mejores** (Φ más bajo) que antes porque `marginal_valor`
devuelve `float64` vs `float32` de `distribucion_marginal` → mayor precisión numérica →
el greedy toma caminos distintos y encuentra mínimos más bajos.

### 8.2 Queyranne 1998 — k=2 exacto global con 2N átomos asimétricos

**Archivo:** `src/strategies/q_nodes.py` — `_queyranne`, `_atomos_asimetricos`

Para k=2 se usa el algoritmo de Queyranne (1998), que minimiza funciones submodulares
simétricas exactamente en O(N²) evaluaciones vía ordenamiento de máxima adyacencia,
pares colgantes y contracción. Los **2N átomos asimétricos** (N átomos `({i},∅)` para
futuros + N átomos `(∅,{j})` para presentes) cubren el espacio **completo** de
biparticiones asimétricas → garantía de mínimo global.

Se eliminó la constante `_QUEYRANNE_N_MAX = 15` que antes restringía los átomos
asimétricos a N ≤ 15 y usaba átomos simétricos ({i},{i}) para N > 15. Con
`marginal_valor`, los 2N átomos son viables para todo N: las primeras fases de
Queyranne trabajan sobre sub-arrays pequeños y el caché `valor_memo` amortiza el costo.

---

## 9. Convergencia con GeoMIP y sus límites

Para N=10 y N=20, QNodes y GeoMIP dan el **mismo Φ** para todos los k. Para k=2, la
coincidencia es garantizada por Queyranne (exacto). Para k≥3, es contingente: a N=22 la
convergencia se rompe — GeoMIP encuentra mejores soluciones:

| k | QNodes N=22 φ | GeoMIP N=22 φ | Ganador φ | Ganador velocidad |
|---|--------------|--------------|----------|-----------------|
| 2 | 0.499575 | 0.499575 | empate | **QNodes** (6.5s vs 31.9s) |
| 3 | 0.999189 | **0.999150** | GeoMIP | **QNodes** (6.1s vs 11.9s) |
| 4 | 1.498915 | **1.498764** | GeoMIP | **QNodes** (5.5s vs 10.7s) |
| 5 | 1.998667 | **1.998490** | GeoMIP | **QNodes** (5.8s vs 12.5s) |

QNodes es uniformemente **más rápido**, pero GeoMIP es **más preciso para k≥3 en N≥22**.
Si vuelve a aparecer un "salto" entre k consecutivos en QNodes, sospechar de cortes
simétricos colándose. La diferencia de Φ en N=22 no es un error: es la ventaja del
clustering espectral de GeoMIP para explorar cortes geométricamente distintos.

---

## 10. Por qué la estrategia pura de QNodes es inviable para N ≥ 20

El QNodes original (Queyranne 1998) fue diseñado para **bipartición** (k=2), donde es
exacto y óptimo. Para k ≥ 3 no existe ningún algoritmo polinomial que garantice el
óptimo global (el problema es NP-hard), por lo que **cualquier** implementación k≥3 es
necesariamente heurística.

El greedy top-down + ILS de QNodes funciona bien hasta N≈20, pero a N≥22 el espacio de
k-particiones crece tanto que la ILS con sus pocos ciclos adaptativos no alcanza a
explorar suficiente vecindario. GeoMIP, al añadir una matriz de afinidad geométrica y
explorar cortes basados en similitud espectral de las columnas de la TPM, accede a zonas
del espacio de búsqueda que el greedy asimétrico puro no visita.

**En resumen:** para N ≥ 20 el enfoque de QNodes debe entenderse como
- **exacto y preferido para k=2** (Queyranne garantiza el mínimo global),
- **heurístico y competitivo para k=3-5 en N=20** (mismo Φ que GeoMIP),
- **heurístico y subóptimo para k≥3 en N≥22** (GeoMIP lo supera en calidad).

Para sistemas grandes con k≥3, QNodes ofrece velocidad; GeoMIP ofrece mejor Φ a costa
de más tiempo de búsqueda y —sobre todo— un arranque más lento (ver `KGeoMIP/docs/`).

---

## 11. Precisión float64, solver exacto N≤6, `vacío=False` honrado y validación con fuerza bruta (2026-06-14)

### 11.1 Precisión unificada en float64 (QNodes = KGeoMIP = fuerza bruta)

`marginal_valor` calculaba la marginal del bloque en float64, pero `distribucion_marginal`
(usada por la fuerza bruta y KGeoMIP) la redondeaba a **float32**. Misma partición,
distinto redondeo → Φ diferían en **~1e-8** (y como cada motor es código aparte, el orden
de reducción de `np.mean` divergía 1 ULP). **Arreglo:** `System.distribucion_marginal` y
`QNodes._dist_bloque` ahora trabajan en **float64**. El vector de marginales es de tamaño
N (no la tabla de costos), así que float64 **no cuesta memoria**. Los tres motores ahora
coinciden a ~1e-15. (Antes se probó bajar QNodes a float32, pero subir todo a float64 es
lo correcto: KGeoMIP no podía igualar a la bruta en float32 por ser implementaciones
distintas.)

### 11.2 Solver EXACTO para N ≤ 6

Para N ≤ 6, `aplicar_estrategia` **enumera** el mismo espacio asimétrico que
`BruteForceKMIP` (`_resolver_exacto`, reusa `_emd_bloques`) y devuelve el **óptimo global
exacto**, honrando `permitir_presente_vacio`; si el espacio supera `_CAP_EXACTO` cae a la
heurística. Garantiza QNodes == KGeoMIP == fuerza bruta en CSVs pequeños. Para N > 6 sigue
el pipeline Queyranne + greedy + 1-move/2-move + ILS.

### 11.3 `permitir_presente_vacio=False` ahora se honra en la heurística (N > 6)

Antes, con `vacío=False` la heurística seguía generando bloques con mecanismo ∅ (Queyranne
usaba átomos asimétricos que incluyen ∅, y el greedy/refinamiento no tenían guardas). Se
implementó `_atomos_simetricos` (N átomos `({i},{i})`, usado cuando `vacío=False`), una
guarda en `_queyranne._f` (inf si un bloque con futuro queda sin presente) y guardas de
"presente no vacío" en `_mejor_split_bloques`, `_refinar_bloques`, `_refinar_bloques_2move`
y `_perturbar_bloques`. Verificado N7: con `vacío=False` ningún bloque queda sin mecanismo.

### 11.4 Default `permitir_presente_vacio` → `True`

Cambiado de `False` a `True` para igualar a KGeoMIP y a `run_suite_2026` (la k-MIP "real"
del proyecto permite ∅). Antes los dos motores exploraban espacios distintos por defecto.

### 11.5 Validación con fuerza bruta: todas verdaderas

`AYDA_2026_1/Brute_Force/comparar_fuerza_bruta.py` (solo N ≤ 6, lee `Brute_Force/samples_force/`,
**no** toca `samples_binary`) cruza BruteForce vs KQNodes vs KGeoMIP. Tras 11.1–11.2,
**todas las pruebas salen VERDADERO** (`Φ_igual = TRUE`, cota `QN_≥_BF = TRUE`) en todo
N≤6 y todo k, deterministas y no-deterministas. El `err_rel` residual ~5e-8 de XLSX viejos
era el float32 pre-arreglo; regenerado en float64 cae a ~0. Las **TPM no-deterministas no
fallan**: la métrica L1 y `np.mean`/`marginal_valor` son float-safe.

---

---

## Historial de evolución (referencia)

**2026-06-08 — Alineación con GeoMIP.** Se igualó la métrica con GeoMIP y se añadieron
candidatos de aislamiento. En esa etapa la métrica seleccionaba entre EMD-Hamming
(`pyemd`, N ≤ 12) y L1 (N > 12), y el motor era aglomerativo bottom-up.

**2026-06-10 (tarde) — Reescritura al motor asimétrico unificado.**
Se descubrió que la L1 marginal es la EMD-Hamming exacta (no aproximación), lo que
permitió eliminar `pyemd` y el límite N ≤ 12. Se reemplazó el motor aglomerativo por el
greedy top-down sobre bloques asimétricos `(futuros, presentes)`, con refinamiento
1-move futuro/presente e ILS. Quedó un único motor para todo k, sin el salto k=2→k=3.

**2026-06-12 — `permitir_presente_vacio` ahora tiene efecto.** El flag se asignaba a
`self._permitir_presente_vacio` pero **nunca se leía**: el corte de mecanismo vacío
`({i}, ∅)` (familia 3 del pool) se añadía siempre, así que el mecanismo ∅ aparecía aun
cuando se pedía `False`. Ahora la familia 3 se genera **solo si el flag es `True`**.

**2026-06-13 — `marginal_valor` + Queyranne exacto para k=2.**
- `NCube.marginal_valor(ejes, estado_inicial)` reemplaza la ruta `bipartir→marginalizar`
  en `_dist_bloque`: coste O(2^(N-|mecanismo|)) vs O(2^N) anterior. Speedup ×145 en N=20
  (392 s → 2.7 s). Resultados float64 vs float32 → Φ ligeramente más bajo.
- Queyranne 1998 con 2N átomos asimétricos garantiza óptimo global para k=2 en todo N.
  Se eliminó el umbral `_QUEYRANNE_N_MAX = 15` que antes forzaba átomos simétricos en N > 15.
- Descubierto que para k ≥ 3 en N ≥ 22, GeoMIP encuentra Φ menor: la ILS adaptativa de
  QNodes no cubre suficiente vecindario a esos tamaños (ver §9 y §10).

**2026-06-14 — Precisión float64, exacto N≤6, `vacío=False` honrado, validación BF (estado vigente).**
Ver §11. Resumen: vector de marginales a float64 (QNodes=KGeoMIP=fuerza bruta a ~1e-15);
solver exacto para N≤6; `permitir_presente_vacio=False` ahora respetado en la heurística
(átomos simétricos + guardas); default del flag → `True`; todas las pruebas vs fuerza
bruta (N≤6, det y no-det) salen VERDADERO. TPM no-deterministas verificadas (no fallan).

---

## Archivos sin cambios estructurales

- `src/models/core/ncube.py`, `src/models/core/system.py`, `src/models/base/sia.py`
- `src/strategies/force.py`, `src/strategies/phi.py`
- `src/funcs/format.py` (el formateador `fmt_k_bloques` ya era compatible)
