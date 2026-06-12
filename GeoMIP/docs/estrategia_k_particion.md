# Estrategia para Encontrar la k-Partición con Mínima Pérdida de Información

**Clase:** `KGeoMIP` — `src/controllers/strategies/kgeomip.py`

## Contexto: ¿Qué es una k-Partición y por qué Minimizar Φ?

En la **Teoría de Información Integrada (IIT)**, un sistema de N nodos puede partirse en
k bloques. Al "cortar" las conexiones causales entre grupos, el sistema particionado
genera una distribución distinta a la del sistema integrado. La **pérdida de
información** Φ mide cuánta información se destruye con ese corte, usando la **Earth
Mover's Distance (EMD)**:

```
Φ(partición) = EMD( P_original(t+1) , P_particionada(t+1) )
```

La **k-MIP** es la partición que produce el **menor Φ**.

---

## Motor principal: Greedy Top-Down sobre bloques asimétricos

> **Importante:** la ruta principal de `aplicar_estrategia` es un motor **greedy
> top-down asimétrico**, idéntico en filosofía al de QNodes. El pipeline basado en
> SpectralClustering + AgglomerativeClustering (descrito más abajo en "Pipeline
> heurístico legado") quedó como **fallback** y ya no es el camino por defecto.

### Representación asimétrica de bloques

Cada bloque es un `Block = (frozenset futuros_globales, frozenset presentes_globales)`,
con futuro (t+1) y presente (t) particionados de forma **independiente**. Un nodo puede
aportar su mecanismo a un bloque mientras su futuro vive en otro, evitando el
"sobre-corte" que inflaba Φ para k ≥ 3.

### Pipeline

```
ENTRADA: subsistema condicionado, valores de k (k dado, o [2..min(6,N)] si k=None)
│
│  1. _construir_tabla_costos()     → tabla_T (2^n_dims, n) por BFS vectorizado
│  2. _construir_cut_pool(...)      → pool de O(N) cortes, construido UNA vez,
│                                      compartido por TODOS los k
│
│  Para cada k:
│     3. _greedy_k_particion(...)   → desde 1 bloque (todo el subsistema),
│                                      aplicar k-1 mejores splits del pool
│     4. _refinar_bloques_1move(...) → 1-move FUTURO + 1-move PRESENTE (asimétrico)
│                                      [fase final — sin ILS, ver más abajo]
│
└── Elegir la k con Φ mínimo global → k-MIP
```

> **Cambio (2026-06-12):** se **retiró la Búsqueda Local Iterada (ILS)** que antes
> ocupaba el paso 5 (perturbar + re-refinar ×4). Aportaba mejoras marginales a un
> costo de ~5× el refinamiento, y dejó al motor **determinista**. Detalle y evidencia
> en [`decision_sin_ils.md`](decision_sin_ils.md).

### El pool de cortes (`_construir_cut_pool`)

Tres familias de candidatos, todas como `(frozenset futuros, frozenset presentes)`:

1. **N aislamientos simétricos** `({i}, {present_i})` + complementos.
2. **N aislamientos de mecanismo vacío** `({i}, ∅)`: aísla el futuro de i SIN mecanismo
   presente; al aplicarse a un bloque B deja el mecanismo de i intacto en el resto.
3. **Mejor corte geométrico por nivel de Hamming** d = 1..n_dims//2+1 (desde `tabla_T`)
   + complementos.

### El mejor split (`_mejor_split_bloques`)

Para cada bloque b y corte c:

```
inside  = (b.F ∩ c.F,  b.P ∩ c.P)
outside = (b.F − c.F,  b.P − c.P)
```

Se evalúa Φ (`evaluar_bloques`) de cada configuración candidata en paralelo (joblib) y
se elige el split de menor pérdida.

**Invariantes de validez de un split (2026-06-12):**

- **Ningún bloque puede quedar con el futuro (alcance) vacío** — `inside.F` y `outside.F`
  deben ser no vacíos. Un bloque sin futuro no representa nada y abría la puerta a partes
  degeneradas `(∅, ∅)` (futuro **y** presente vacíos), que ahora son imposibles.
- Si `permitir_presente_vacio = False`, además **ningún bloque puede quedar con el
  presente (mecanismo) vacío** — el flag se respeta en todo el camino greedy (split,
  refinamiento 1-move y, antes, la perturbación). Con `True`, el presente ∅ sí se
  permite, pero el futuro nunca queda vacío.

---

## Evaluación de Φ: L1 marginal = EMD de Hamming EXACTA

`evaluar_bloques` reconstruye la marginal con `System.particionar()` en **una sola
pasada** y compara con la original mediante suma L1:

```python
dist_rec = subsistema.particionar(particiones).distribucion_marginal()
return float(np.sum(np.abs(dist_original - dist_rec)))       # O(N)
```

**Teorema de descomposición marginal:** como original y reconstruida son productos de
marginales (independencia condicional por construcción), la EMD con Hamming sobre la
conjunta 2^N **es igual** a la suma L1 marginal, **exactamente**, para todo N:

```
EMD_Hamming(P, Q) = Σᵢ | P(nodo_i = 1) − Q(nodo_i = 1) |
```

Esto convierte un problema O(4^N) (solver de transporte sobre 2^N estados) en O(N), sin
pérdida de precisión y **sin límite de tamaño**. (Detalle en `GeoMIP_Optimizaciones.md`.)

---

## Refinamiento 1-move asimétrico (fase final)

`_refinar_bloques_1move` explora dos vecindarios hasta convergencia:

- **Movimiento futuro:** traslada un nodo futuro del bloque i al j (no vacía el futuro
  del bloque origen).
- **Movimiento presente (asimétrico):** traslada un nodo del lado presente sin tocar su
  par futuro — imposible en cortes simétricos. Si `permitir_presente_vacio = False`, no
  se permite vaciar el presente de un bloque.

Este es el **último paso** del motor. La antigua **ILS** (perturbar + re-refinar ×4) se
retiró por ganancia marginal y costo alto; ver [`decision_sin_ils.md`](decision_sin_ils.md).
El resultado es **determinista**: misma entrada → misma partición.

---

## El producto de marginales y por qué NO se construye la conjunta

Para reconstruir la conjunta de bloques independientes haría falta el producto de
Kronecker `Q = Q₁ ⊗ Q₂ ⊗ … ⊗ Qₖ` (2^N elementos). **Pero el teorema de descomposición
evita construirla**: para el cálculo de Φ sólo se necesitan los N vectores marginales,
no la conjunta. Por eso no se usa PyTorch/TensorFlow (pensados para tensores continuos
con gradientes/GPU) ni siquiera `np.kron` en el camino caliente: la suma L1 sobre N
marginales basta y es O(N).

---

## Pipeline heurístico legado (fallback)

Antes del motor greedy top-down, GeoMIP generaba candidatos con heurísticas
geométricas. Este pipeline permanece en el código como **fallback** (`_evaluar_k_completo`,
`_particion_grafo_hipercubo`, `_agrupamiento_jerarquico`) pero **no** es la ruta por
defecto:

- **SpectralClustering** sobre la matriz de afinidad `A[i,j] = (1 + cos(col_i,col_j))/2`
  de la tabla de costos, con varias semillas.
- **AgglomerativeClustering** (average/complete/single linkage).
- **Aislamiento heurístico** C(N, k-1) y variantes con mecanismo vacío (centinela -1).
- Evaluación paralela + refinamiento 1-move.

Si scikit-learn no está disponible, hay un fallback jerárquico bottom-up determinista.

---

## Selección Global entre Valores de k

Para k=None se evalúan k = 2..min(6, N) secuencialmente (cada k usa todos los núcleos
internamente con joblib; paralelizar entre k's dividiría recursos sin ganar nada). Se
elige el de Φ mínimo global:

```
k_MIP = argmin_k { Φ_min(k) }
```

Como el descenso greedy es anidado, Φ resulta coherente entre k consecutivos, sin saltos.

---

## Ejemplo Conceptual (N = 3, k = 2)

Sistema ABC, estado `101`. Pool de cortes incluye `({A},{a})`, `({A},∅)`, complementos,
y el mejor corte geométrico. El greedy parte de `({A,B,C},{a,b,c})` y aplica el split de
menor Φ, p.ej. `({C},{c}) | ({A,B},{a,b})`. El 1-move (futuro y presente) intenta
mejorar; si ningún movimiento baja Φ, esa es la k-MIP para k=2.
