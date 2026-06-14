# KGeoMIP — Framework de k-Partición de Mínima Información

**Framework:** `AYDA_2026_1/KGeoMIP`
**Asignatura:** Análisis y Diseño de Algoritmos — 2026-1

---

## ¿Qué hace?

KGeoMIP encuentra la **k-partición de mínima información (k-MIP)** de un sistema de N
nodos descrito por una Transition Probability Matrix (TPM). Minimiza Φ (información
integrada perdida) sobre todas las particiones del sistema en k grupos, usando la
Wasserstein-1 con distancia de Hamming como métrica — equivalente a la suma L1 marginal
para distribuciones producto, que es el caso de cualquier k-partición de un sistema IIT.

---

## Por qué GeoMIP encuentra mejor Φ que QNodes para k ≥ 3 en N grande

GeoMIP y QNodes comparten el mismo motor base (greedy top-down asimétrico + 1-move
futuro/presente + métrica L1 exacta). La diferencia está en la **diversidad del pool de
cortes**: GeoMIP añade una tercera familia de candidatos derivada de la matriz de
afinidad geométrica de las columnas de la TPM (similitud espectral entre nodos), lo que
expone particiones que el greedy puro no genera.

A N pequeño (≤ 20) ambos frameworks coinciden exactamente en Φ para todos los k.
A N ≥ 22, la exploración geométrica empieza a dominar:

| k | QNodes N=22 φ | GeoMIP N=22 φ | Ganador φ | Ganador velocidad |
|---|--------------|--------------|----------|-----------------|
| 2 | 0.499575 | 0.499575 | empate | **QNodes** (6.5 s vs 31.9 s) |
| 3 | 0.999189 | **0.999150** | **GeoMIP** | QNodes (6.1 s vs 11.9 s) |
| 4 | 1.498915 | **1.498764** | **GeoMIP** | QNodes (5.5 s vs 10.7 s) |
| 5 | 1.998667 | **1.998490** | **GeoMIP** | QNodes (5.8 s vs 12.5 s) |

Para k=2, QNodes (algoritmo de Queyranne) tiene **garantía de óptimo global exacto**.
GeoMIP k=2 es heurístico y puede perder el óptimo. Para k≥3, ninguno garantiza el
óptimo (NP-hard), pero GeoMIP explora un vecindario más rico geométricamente.

**Lo que GeoMIP sacrifica:** tiempo. QNodes es 2×–5× más rápido en N=22 para todos
los k. La diferencia crece con N porque la construcción de la tabla de costos y la
matriz de afinidad de GeoMIP son O(N × 2^N), mientras que QNodes con `marginal_valor`
es O(N² × 2^(N/2)).

---

## El principal cuello de botella de GeoMIP: el arranque, no la búsqueda

El tiempo de GeoMIP está **dominado por el arranque del motor**, no por la búsqueda en
sí. Una vez construida la infraestructura (tabla de costos, subsistema condicionado,
pool de cortes), la búsqueda greedy + 1-move es rápida.

### ¿Qué paga el arranque?

1. **Carga de la TPM / LazyTPM** — para N ≥ 18, lectura por chunks del CSV.
2. **Condicionamiento del subsistema** — construir los N NCubos condicionados al estado
   inicial; coste O(N × 2^N) en tiempo y O(2^N) en memoria por NCubo.
3. **`_construir_tabla_costos`** — BFS vectorizado por niveles de Hamming:
   O(N × 2^N) en tiempo, O(2^N × N) en memoria. Para N=22: ~88 M operaciones.
4. **Matriz de afinidad geométrica** — producto interno de columnas de la tabla:
   O(N² × 2^N). Para N=22: ~1.7 G operaciones (paralelizado con joblib).
5. **`_construir_cut_pool`** — pool de cortes geométricos derivados de la afinidad.

En la práctica, para N=22, el tiempo del primer k (k=2: 31.9 s) incluye casi todo el
arranque. Los k posteriores (k=3: 11.9 s, k=4: 10.7 s, k=5: 12.5 s) son más rápidos
porque reutilizan el subsistema y la tabla ya construidos (Manager los cachea).

### Modo manual

En el modo manual (`exec_kgeomip.py` → Modo 1), cada problema paga el arranque
completo. Cambiar el candidato o el estado inicial invalida el caché del subsistema y
vuelve a pagar la construcción. El tiempo reportado incluye arranque + búsqueda.

### Modo por bloque

En el modo bloque (`exec_kgeomip.py` → Modo 2 / `dashboard/ api/block`), el candidato
y el estado inicial son fijos para todo el lote, así que el subsistema se cachea. Antes
de la primera prueba se ejecuta una corrida de **calentamiento descartable** con los
parámetros de la primera prueba válida, que paga el arranque sin contaminar el tiempo
registrado. Los tiempos del lote reflejan **solo la búsqueda**, no el arranque.

El arranque se reporta aparte como "Arranque del motor (warmup)" en el XLSX / SSE del
dashboard. En corridas con muchas pruebas (lotes de 8+), el arranque es un costo único
que se amortiza: la diferencia de velocidad contra QNodes pierde peso.

### Implicación práctica

| Contexto | Impacto del arranque | Recomendación |
|---|---|---|
| 1 prueba, modo manual | **Alto** — domina el tiempo total | Preferir QNodes para velocidad |
| Lote pequeño (2–5 pruebas) | **Significativo** | Depende de la prioridad: Φ mejor → GeoMIP |
| Lote grande (10+ pruebas) | **Amortizado** — bajo por prueba | GeoMIP para mejor Φ en k≥3 |
| k=2, cualquier N | **Irrelevante para Φ** | QNodes (Queyranne exacto + más rápido) |
| k≥3, N≥22 | **Cuenta** | GeoMIP si se prioriza Φ; QNodes si se prioriza velocidad |

---

## Cambios recientes (2026-06-14): precisión, arranque y validación exacta

### 1. Precisión unificada en float64 — KGeoMIP = QNodes = fuerza bruta

Antes, `System.distribucion_marginal` devolvía el vector de marginales en **float32** y
`evaluar_bloques` acumulaba la suma L1 también en float32. Como KGeoMIP, QNodes y la
fuerza bruta son implementaciones separadas, cada una redondeaba la media de `np.mean`
con distinto orden de reducción → los Φ diferían en **~1e-8** (incluso aparecía KGeoMIP
con Φ *por debajo* de la fuerza bruta, lo que parecía un imposible: misma partición,
distinto redondeo).

**Arreglo:** el vector de marginales pasa a **float64** (`System.distribucion_marginal`)
y `evaluar_bloques` / `evaluar_corte_asimetrico` acumulan con `dtype=np.float64`. Es
**gratis en memoria** porque ese vector es de tamaño N (no la `tabla_T` de 2^N×N, que
sigue en float32 para no doblar la memoria en N grande). Resultado: los tres motores
coinciden a **~1e-15** (idéntico en pantalla a cualquier número de decimales).

> Nota: la `tabla_T` y la matriz de afinidad **siguen en float32** a propósito. Subirlas
> a float64 doblaría la memoria (N=25: 3.35 → 6.7 GB, + la copia float64 de la afinidad
> → riesgo de OOM). La precisión que importa para el Φ reportado vive en el vector de
> marginales, no en la tabla.

### 2. Arranque optimizado: la TPM se lee en UNA sola pasada

`System.__init__` llamaba `LazyTPM.marginal_nodo(i)` **una vez por nodo**, y cada llamada
reparsea el CSV completo → para N=22 eran **22 relecturas** de 4.2 M filas (~929 s solo
en preparar el subsistema). Nuevo `LazyTPM.columnas()` lee **todas** las columnas en una
sola pasada → la construcción del subsistema N=22 baja de **~929 s a ~38 s** (medido).
Verificado idéntico a la ruta eager (`genfromtxt`).

`_construir_tabla_costos` usa un kernel **Numba** `@njit(cache=True)` para llenar la
tabla por capas de Hamming; `warmup_motor()` precalienta ese JIT y el pool de hilos
antes del lote (modo bloque) para que el costo de arranque no contamine la primera prueba.

### 3. Solver EXACTO para N ≤ 6 — coincidencia garantizada con fuerza bruta

Para N ≤ 6 (donde la enumeración es tratable), KGeoMIP **resuelve por fuerza bruta**
exhaustiva el mismo espacio asimétrico (`_resolver_exacto_geomip`, reusa `evaluar_bloques`)
y devuelve el **óptimo global exacto**, honrando `permitir_presente_vacio`. Así, en CSVs
pequeños, KGeoMIP == QNodes == fuerza bruta, deterministas o estocásticos. Para N > 6 se
usa la heurística **greedy top-down + 1-move** (el 2-move/VNS y el ILS ligero existen pero
están desactivados por no aportar Φ; ver `docs/decision_sin_ils.md`).

### 4. Validación con fuerza bruta: todas verdaderas

El comparador `AYDA_2026_1/Brute_Force/comparar_fuerza_bruta.py` (solo N ≤ 6, lee
`Brute_Force/samples_force/`, **no** toca `samples_binary`) cruza BruteForce vs KQNodes
vs KGeoMIP. Tras los arreglos de precisión y el solver exacto, **todas las pruebas
salen VERDADERO**: `Φ_igual = TRUE` y la cota inferior `GEO_≥_BF = TRUE` en todos los
N≤6 y todos los k, deterministas y no-deterministas. El `err_rel` residual de corridas
viejas (~5e-8) era float32; con float64 cae a ~0.

> El manejo de **TPM no-deterministas** quedó verificado: la métrica L1 y la
> marginalización (`np.mean`) son float-safe; con float64 coinciden con la fuerza bruta.

---

## Inicio rápido

```bash
# Activar entorno virtual (compartido con KQNodes)
source ../.venv/Scripts/activate      # Linux/Mac
..\.venv\Scripts\activate             # Windows PowerShell

# Modo interactivo
python exec_kgeomip.py

# Suite comparativa vs QNodes
python ../data/run_suite_2026.py --solo-n 10,20,22 --solo-k 3,4,5
```

## Documentación adicional

- [`docs/estrategia_k_particion.md`](docs/estrategia_k_particion.md) — pipeline detallado
- [`docs/GeoMIP_Optimizaciones.md`](docs/GeoMIP_Optimizaciones.md) — optimizaciones técnicas
- [`docs/cambios_vs_original.md`](docs/cambios_vs_original.md) — evolución desde el prototipo
- [`docs/decision_sin_ils.md`](docs/decision_sin_ils.md) — por qué se retiró la ILS
