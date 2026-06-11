# Paradigmas Algorítmicos en GeoMIP — Qué se Usa, Qué No y Por Qué

## El Problema en Cuestión

Antes de analizar los algoritmos, es crucial entender la estructura del problema.

Dado un sistema de N nodos, se busca la **k-partición de mínima pérdida** de información (k-MIP):
dividir los N nodos en k grupos independientes de modo que la **Earth Mover's Distance** entre
la distribución original y la reconstruida desde las partes sea mínima (Φ mínimo).

El espacio de búsqueda está acotado por los **Números de Stirling de segunda especie** S(N, k):

| N  | k=2      | k=3          | k=4              | k=5               |
|----|----------|--------------|------------------|-------------------|
| 5  | 15       | 25           | 10               | 1                 |
| 10 | 511      | 9.330        | 42.525           | 42.525            |
| 15 | 16.383   | 2.375.101    | ~1,0 × 10⁹       | ~4,7 × 10¹¹       |
| 20 | 524.287  | ~580 × 10⁹   | ~4,5 × 10¹⁶      | ~4,7 × 10²¹       |

El crecimiento es **super-exponencial**. Esta es la restricción fundamental que determina por qué
se usa (o no) cada paradigma.

---

## 0. Motor principal actual: Greedy Top-Down sobre bloques asimétricos

> **Estado vigente.** La ruta principal de `aplicar_estrategia` es un **greedy top-down
> (divisivo)** sobre bloques asimétricos `Block = (frozenset futuros, frozenset presentes)`. El
> pipeline de SpectralClustering/AgglomerativeClustering descrito en la sección 1 quedó como
> **fallback** y ya no es el camino por defecto.

```
_construir_cut_pool(...)   → pool de O(N) cortes, construido UNA vez, compartido por todo k
_greedy_k_particion(...)   → desde 1 bloque (todo el subsistema), k-1 mejores splits
_refinar_bloques_1move(...) → 1-move futuro + 1-move presente (asimétrico)
ILS (N_ILS=4)              → perturbar + re-refinar, conservar el mejor Φ
```

El punto clave que vuelve **tratable** el top-down divisivo (que la sección 6 descartaba por
"evaluar O(2^N) cortes en el primer nivel") es que **no** se evalúan todos los cortes posibles:
sólo un **pool de O(N) cortes** construido una vez desde la tabla de costos. Así, cada split
evalúa O(N·|pool|) configuraciones, no O(2^N). El descenso es anidado → Φ coherente entre k.

Combina por tanto **Greedy** (sección 2) como motor de búsqueda y **Top-Down divisivo** como
estrategia de recorrido (sección 6), apoyados en la métrica L1 marginal exacta O(N).

---

## 1. Heurística / Aproximación — fallback (no es ya el paradigma principal)

### ¿Qué es?
Un algoritmo heurístico no garantiza encontrar el óptimo global, pero encuentra soluciones muy
buenas en tiempo polinomial, aprovechando la estructura del problema.

### ¿Cómo y dónde se usa en GeoMIP?

**a) Spectral Clustering — Heurística Geométrica**

En lugar de enumerar todas las particiones, se construye una **matriz de afinidad A[i,j]** donde
la entrada mide cuán "probabilísticamente similares" son dos nodos (similitud coseno de sus perfiles
de costo EMD). Luego `SpectralClustering(affinity="precomputed")` agrupa los nodos en el espacio
espectral del grafo de afinidad.

```python
# kgeomip.py — _particion_grafo_hipercubo()
SpectralClustering(n_clusters=k, affinity="precomputed", random_state=semilla).fit_predict(A)
```

La intuición: nodos que "se comportan igual" en el espacio de probabilidades bajo todas las
condiciones posibles tienden a pertenecer a la misma parte en la MIP.

**b) Agglomerative Clustering (Jerárquico Bottom-Up) — Heurística Estructural**

Complementa al Spectral con tres criterios de enlace: `average`, `complete`, `single`. Cada uno
produce agrupaciones geométricamente distintas, cubriendo regiones diferentes del espacio de
particiones que el Spectral puede no explorar.

**c) Aislamiento Heurístico**

Genera C(N, k-1) candidatos aislando exactamente k-1 nodos individualmente. En más del 80% de
los casos reales, la MIP exacta para k=2 se encuentra entre estos N candidatos, porque el corte
óptimo suele aislar el nodo con menor integración causal.

**d) Iterated Local Search (ILS) — Metaheurística**

Es una **metaheurística**: combina búsqueda local con perturbación para escapar de mínimos locales:

```
Repetir N_ILS = 4 veces:
    Perturbar la mejor solución conocida (mover nodos al azar entre bloques)
    Refinar con 1-move hasta convergencia local
    Si mejora → actualizar mejor solución global
```

### ¿Por qué heurística y no algo exacto?

Porque el espacio de búsqueda crece super-exponencialmente y cada evaluación de Φ requiere
calcular EMD sobre distribuciones de probabilidad. Para N = 15, k = 3: 2.375.101 particiones.
Evaluar cada una tomaría días incluso con la EMD rápida O(N). La heurística reduce esto a evaluar
decenas de candidatos con garantía práctica de encontrar la MIP exacta en la mayoría de sistemas.

---

## 2. Greedy (Voraz) — SÍ SE USA (motor de búsqueda principal + refinamiento)

### ¿Qué es?
Un algoritmo voraz toma la decisión localmente óptima en cada paso sin reconsiderar decisiones
anteriores, con la esperanza de que la secuencia de decisiones locales lleve a un óptimo global.

### ¿Cómo y dónde se usa en GeoMIP?

**a) Greedy top-down (motor de búsqueda principal — `_greedy_k_particion`)**

Desde un único bloque que cubre todo el subsistema, en cada paso se elige el par (bloque, corte)
cuya división minimiza Φ, hasta alcanzar k bloques. Es la decisión voraz central: nunca se
deshace una división. Un descenso de k=1 a k=N registra Φ por cada k (jerarquía anidada). Sólo se
prueban los O(N) cortes del pool, no O(2^N). (Ver sección 0.)

**b) Refinamiento 1-move (Hill Climbing Greedy)**

Es el componente voraz más explícito del sistema. Dado el mejor candidato inicial, evalúa todos
los "vecinos" posibles (mover un nodo a otro bloque) y acepta el mejor movimiento:

```python
# kgeomip.py — _refinar_particion_local()
while mejoro and not agotado():
    mejoro = False
    # Evaluar TODOS los vecinos 1-move
    resultados = Parallel(...)(delayed(evaluar)(vecino) for vecino in vecinos)
    perdida_mejor_vecino = min(resultados)

    if perdida_mejor_vecino < mejor_perdida - ε:
        # Aceptar el mejor vecino (decisión voraz) y repetir
        mejor_perdida = perdida_mejor_vecino
        mejor_particion = vecinos[argmin(resultados)]
        mejoro = True
```

Esto es un **Hill Climbing** estricto: nunca acepta un movimiento que empeore Φ, y siempre
elige el mejor movimiento disponible. Es voraz porque no "mira hacia adelante": no evalúa si
aceptar un movimiento peor ahora permitiría llegar a algo mejor después.

**b) Selección del mejor candidato en Fase 2**

Tras evaluar todos los candidatos generados en Fase 1 (en paralelo), se selecciona vorazmente
el de menor Φ como punto de partida para el refinamiento:

```python
idx_mejor = np.argmin(perdidas_candidatos)
mejor_candidato = candidatos_geo[idx_mejor]
```

**c) Fusión bottom-up en el fallback jerárquico**

Cuando scikit-learn no está disponible, el fallback construye la k-partición fusionando
iterativamente los dos grupos cuya unión produce el menor Φ:

```python
# Evaluar todos los pares posibles de fusión
# Elegir el par con menor pérdida → fusionar (decisión voraz)
# Repetir hasta tener k grupos
```

Cada fusión es irreversible: es una decisión voraz pura.

### Limitación del Greedy en este contexto

El 1-move greedy **garantiza un mínimo local**, no global. Por eso se complementa con ILS
(perturbación + re-refinamiento): para escapar de los mínimos locales superficiales.

---

## 3. Backtracking — NO SE USA

### ¿Qué es?
El backtracking explora el árbol de soluciones de forma sistemática: construye una solución
parcial paso a paso y, cuando detecta que no puede extenderse a una solución válida u óptima,
"retrocede" (`backtrack`) para probar otra rama.

### ¿Por qué no se usa en GeoMIP?

**Razón 1 — Explosión combinatoria sin poda efectiva**

Para que el backtracking sea útil, debe poder **descartar ramas tempranamente** mediante una
condición de poda. En el problema de k-MIP, una "asignación parcial" (nodo 1 → grupo A,
nodo 2 → grupo B, nodo 3 aún sin asignar) no tiene un valor de Φ significativo, porque Φ
sólo está definido para **particiones completas** (se necesita la distribución completa del
sistema particionado para calcular EMD).

Sin poda, el backtracking degenera en exploración exhaustiva del árbol de todas las particiones,
que tiene exactamente S(N, k) hojas — el mismo número que la fuerza bruta. Para N = 20 esto
es entre 10¹⁶ y 10²¹ nodos en el árbol.

**Razón 2 — No hay propiedad de monotonía**

Para que la poda sea efectiva, se necesita que una asignación parcial mala **implique** que todas
las extensiones también son malas. En la k-MIP esto no se cumple: asignar mal los primeros nodos
no implica nada sobre el valor de Φ de las extensiones completas, porque la EMD depende de la
distribución conjunta de todo el sistema particionado.

**Razón 3 — No hay "solución parcial"**

El backtracking funciona bien cuando se puede evaluar la calidad de una solución parcial
(ej. Sudoku: si hay conflicto en una celda, podar). En k-MIP no existe equivalente: una partición
parcial no tiene interpretación física en IIT hasta que es completa.

**Conclusión:** El backtracking no aporta ventaja sobre heurísticas en este problema. Su overhead
de gestión de la pila de recursión y el árbol sería puro costo sin beneficio de poda.

---

## 4. Branch & Bound (B&B) — NO SE USA

### ¿Qué es?
B&B es backtracking con una función de cota inferior (`lower bound`): en cada nodo del árbol de
búsqueda, si la cota inferior de cualquier extensión ya supera la mejor solución conocida, se
descarta esa rama completa sin explorarla.

### ¿Por qué no se usa en GeoMIP?

**Razón 1 — No existe una cota inferior útil para Φ parcial**

Para que B&B sea eficiente, la cota inferior debe ser:
1. **Calculable rápidamente** (más barata que evaluar la solución completa)
2. **Ajustada** (lo más cerca posible al valor real de Φ)
3. **Admisible** (nunca sobreestimar, i.e., siempre ≤ Φ real)

Para una asignación parcial en k-MIP no existe una cota inferior ajustada conocida. La EMD es
la medida mínima de transporte óptimo y no tiene una descomposición por subconjuntos de nodos
que pueda usarse como cota sin perder admisibilidad.

Una cota trivial sería Φ_parcial = 0 (siempre admisible, pero completamente inútil: nunca poda).
Cualquier cota no trivial requiere información sobre el sistema completo, lo que vuelve a ser
tan costoso como evaluar la partición completa.

**Razón 2 — La solución inicial (upper bound) ya es muy buena**

B&B requiere una buena solución inicial para definir el upper bound que permita podar. La
heurística de aislamiento + clustering ya produce en la práctica soluciones que están a ≤ 5%
del óptimo. Esto significa que B&B necesitaría que las cotas inferiores sean extremadamente
ajustadas para superar esto, lo cual no se puede garantizar.

**Razón 3 — Complejidad espacial del árbol**

B&B mantiene en memoria la frontera activa del árbol de búsqueda. En la k-MIP, el árbol tiene
ramificación O(N) en cada nivel y profundidad N, por lo que la frontera puede crecer hasta O(N!)
nodos activos simultáneamente en el peor caso — inmanejable para N ≥ 15.

**Conclusión:** B&B sería más lento que las heurísticas actuales porque gastaría tiempo enorme
en construir y gestionar el árbol sin poder podar efectivamente.

---

## 5. Programación Dinámica (DP) — SE USA PARCIALMENTE (memoización)

### ¿Qué es?
La programación dinámica divide el problema en subproblemas solapados, resuelve cada subproblema
una sola vez y almacena el resultado. Exige **subestructura óptima**: la solución óptima del
problema completo se construye a partir de soluciones óptimas de subproblemas.

### ¿Dónde SÍ se aplica DP en GeoMIP?

**Memoización de marginalizaciones en NCube (Top-Down DP)**

La operación de marginalización de un n-cubo (`NCube.marginalizar()`) es costosa y se llama
repetidamente con los mismos ejes durante el clustering y la evaluación de candidatos. Se
implementa **DP Top-Down** con memoización:

```python
# ncube.py — marginalizar()
cache_key = frozenset(marginable_axis.tolist())  # clave: conjunto de ejes a marginalizar
if cache_key in self._marginal_cache:
    return self._marginal_cache[cache_key]        # hit: O(1)

# ... computar marginalización (costosa) ...
self._marginal_cache[cache_key] = nuevo_ncube     # guardar para futuros accesos
return nuevo_ncube
```

Cada NCube mantiene su propio caché. La clave es un `frozenset` de índices de ejes porque la
marginalización es **conmutativa**: marginalizar sobre {A, B} da el mismo resultado que sobre
{B, A}. El frozenset captura esta propiedad de forma natural.

Esto es DP Top-Down puro: se computan marginalizaciones bajo demanda (cuando el algoritmo las
necesita) y se cachean. No se precomputan todas las posibles marginalizaciones (lo que sería
enfoque Bottom-Up y requeriría 2^N entradas por NCube).

### ¿Por qué NO se aplica DP al problema completo de k-MIP?

La DP completa requiere **subestructura óptima**: la mejor k-partición de N nodos debe poder
construirse a partir de la mejor (k-1)-partición de N-1 nodos (o similar).

**Esto no se cumple en k-MIP.** La razón es que Φ depende de la distribución **conjunta** del
sistema completo particionado. Quitar un nodo de la partición cambia la distribución marginal
de todos los otros nodos de su grupo, porque la marginalización promedia sobre ese nodo. No
hay una forma de descomponer el problema en subproblemas independientes que preserven el valor
de Φ.

Formalmente:
```
Φ(k-partición de N nodos) ≠ función(Φ(k-1)-partición de N-1 nodos)
```

Por eso GeoMIP no usa DP para la búsqueda de k-MIP sino heurísticas combinadas con búsqueda local.

---

## 6. Bottom-Up vs. Top-Down — Cuándo Se Usa Cada Uno y Por Qué

GeoMIP usa **ambos paradigmas** en partes distintas del sistema, y la elección en cada caso
responde a la naturaleza del subproblema.

---

### Top-Down con Memoización — Dónde y Por Qué

**Componente:** `NCube.marginalizar()` (programación dinámica con memoización)

**Estrategia:** El algoritmo pide marginalizaciones bajo demanda. Sólo se computan las que
realmente se necesitan, y se cachean para no repetirlas.

**¿Por qué Top-Down aquí?**

La alternativa Bottom-Up precomputaría todas las 2^|dims| marginalizaciones posibles de cada
n-cubo al construirlo. Para N = 10 nodos, cada NCube tendría 2^10 = 1024 entradas de caché.
Para N = 20: más de un millón de entradas por NCube, y hay N NCubos.

En contraste, durante la búsqueda real sólo se necesitan un subconjunto pequeño de
marginalizaciones (las que corresponden a los grupos de la partición candidata). El Top-Down
"paga" sólo por lo que se computa, mientras que el Bottom-Up pagaría por todo, se use o no.

```
Top-Down: computar sólo lo necesario → O(marginalizaciones_pedidas × costo_por_marginalización)
Bottom-Up: computar todo → O(2^N × costo_por_marginalización) por NCube
```

Para un sistema de N = 15 con 5 particiones candidatas evaluadas en paralelo, Top-Down
computa ~50 marginalizaciones; Bottom-Up habría precomputado 32.768 × 15 = ~500.000.

---

### Bottom-Up — Dónde y Por Qué

**Componentes:**
1. **Agglomerative Clustering** (`AgglomerativeClustering`)
2. **Fallback jerárquico** en `_evaluar_k_completo()` cuando sklearn no está disponible

**Estrategia:** Empezar con N singletons (cada nodo es su propio grupo) y fusionar iterativamente
el par cuya unión produce el menor Φ, hasta llegar a k grupos.

```python
# kgeomip.py — fallback en _evaluar_k_completo()
particiones = [[i] for i in range(n_vars)]   # Bottom: N singletons
while len(particiones) > k:
    # Evaluar todos los pares posibles de fusión
    # Fusionar el par con menor pérdida → Bottom-Up greedy
    ...
```

**¿Por qué Bottom-Up aquí?**

El clustering jerárquico busca estructura **local** entre nodos y la construye hacia arriba.
Cada fusión une los dos nodos o grupos más "similares" (según la afinidad o la pérdida EMD),
acumulando estructura de lo simple a lo complejo.

El **fallback** Bottom-Up (Agglomerative) construye estructura local entre pares. El Top-Down
divisivo *ingenuo* sería costoso si evaluara todos los cortes posibles, pero esto se resuelve con
el **pool de O(N) cortes** del motor principal (sección 0): en lugar de O(2^N) cortes por nivel,
sólo se prueban O(N). Por eso el motor vigente es **Top-Down divisivo con pool acotado**, y el
Bottom-Up queda relegado a fallback:
1. Un Top-Down sin pool requeriría evaluar O(2^N) cortes en el primer nivel — inviable.
2. Con el pool de O(N) cortes, cada split evalúa O(N·|pool|), tratable para todo N del proyecto.
3. El descenso anidado da una jerarquía coherente de Φ para todos los k en un solo recorrido.

Bottom-Up también tiene la ventaja de que produce una **jerarquía completa** de particiones
(dendrograma): con una sola ejecución se obtienen candidatos para k=2, k=3, ..., k=N cortando
el dendrograma a distintas alturas. Esto amortiza el costo del clustering entre todos los k.

---

### Comparación de Paradigmas de Recorrido en GeoMIP

| Componente                        | Paradigma      | Razón de la Elección                                           |
|-----------------------------------|----------------|----------------------------------------------------------------|
| **`_greedy_k_particion` (motor principal)** | **Top-Down + Greedy** | Divide el sistema en k-1 splits; pool de O(N) cortes lo hace tratable (no O(2^N)) |
| `NCube.marginalizar()` (caché)    | Top-Down (DP)  | Se piden sólo algunas marginalizaciones; precomputar todo sería 2^N entradas por NCube |
| `AgglomerativeClustering` (fallback) | Bottom-Up   | Estructura local entre pares → construir hacia arriba es natural y eficiente |
| Fallback jerárquico (`_evaluar_k_completo`) | Bottom-Up + Greedy | Fusionar el par de menor pérdida cuando no hay sklearn |
| Spectral Clustering (fallback)    | Global (ni B-U ni T-D) | Opera sobre el espectro de la matriz de afinidad completa; no es recursivo |
| Refinamiento 1-move (futuro+presente) | Greedy local | Explorar vecinos del punto actual; no requiere recorrido del árbol |
| ILS (Iterated Local Search)       | Metaheurística | Perturb + refine; escapa de mínimos locales del 1-move          |

---

## Resumen Final

| Paradigma                  | ¿Se Usa?              | Justificación                                                                       |
|----------------------------|-----------------------|-------------------------------------------------------------------------------------|
| **Backtracking**           | NO                    | Sin poda efectiva: Φ no está definido para particiones parciales. Degeneraría en fuerza bruta con overhead de recursión |
| **Branch & Bound**         | NO                    | No existe cota inferior ajustada para Φ parcial. Sin cota, B&B = backtracking = fuerza bruta |
| **Greedy Top-Down (divisivo)** | **SÍ (motor principal)** | `_greedy_k_particion`: divide en k-1 splits sobre un pool de O(N) cortes; jerarquía anidada coherente entre k |
| **Heurística/Aproximación**| SÍ (fallback)         | Spectral + Agglomerative + Aislamiento generan candidatos sin explorar el espacio completo |
| **Greedy (Voraz)**         | SÍ (búsqueda + refinamiento) | Greedy top-down (búsqueda) + 1-move Hill Climbing futuro/presente (refinamiento) |
| **Metaheurística (ILS)**   | SÍ                    | Perturbación + re-refinamiento N_ILS=4; escapa de mínimos locales |
| **DP Top-Down (Memo)**     | SÍ (parcial)          | `NCube.marginalizar()` paga sólo por lo que se computa |
| **DP Bottom-Up**           | NO (para k-MIP global)| k-MIP no tiene subestructura óptima: la mejor k-partición no se construye desde (k-1)-particiones óptimas |
| **Bottom-Up (clustering)** | SÍ (fallback)         | Agglomerative y fallback jerárquico cuando no hay sklearn |
