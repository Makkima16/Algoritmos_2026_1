# Paradigmas algorítmicos utilizados

Este documento analiza cada paradigma algorítmico clásico, determina cuáles se
utilizaron en `AYDA_2026_1/QNodes`, con qué justificación, y por qué los demás no
resultaron aplicables.

---

## Paradigmas utilizados

### 1. Algoritmo Voraz / Greedy (principal — top-down)

**Utilizado:** sí — en `QNodes._greedy_descenso` / `_greedy_bloques` y `_mejor_split_bloques`

**Descripción del uso:**

El greedy es el paradigma central de QNodes, pero **top-down (divisivo)**, no
aglomerativo. Se parte de un único bloque que cubre todo el subsistema (TODOS los
futuros, TODOS los presentes) y en cada paso se aplica la **mejor división** de un
bloque sobre todos los cortes del pool:

```
En cada paso: elegir (bloque b, corte c) cuya división minimiza _emd_bloques
Nunca deshacer divisiones tomadas.
```

Un único descenso de k=1 a k=N produce una **jerarquía anidada**: cada k surge de
dividir un bloque del nivel anterior, lo que garantiza Φ monótono no decreciente entre
k consecutivos (coherencia, sin saltos).

**Por qué greedy y no exhaustivo:** para N = 15, 20, 22, 25 la búsqueda exacta es
intratable (B(N) super-exponencial). El greedy top-down hace O(N³) evaluaciones, que
es polinomial y tratable para cualquier N del proyecto.

**Garantía:** óptimo local greedy en cada nivel (ninguna división distinta del último
paso mejora), no el óptimo global; por eso se complementa con refinamiento e ILS.

---

### 2. Heurístico (refinamiento local 1-move asimétrico)

**Utilizado:** sí — en `QNodes._refinar_bloques`

**Descripción del uso:**

Búsqueda local best-improvement hasta convergencia, con **dos** vecindarios:

```
Repetir hasta que ningún movimiento mejore Φ:
    Movimiento futuro:   trasladar un nodo futuro del bloque i al j
    Movimiento presente: trasladar el MECANISMO de un nodo del bloque i al j
                         SIN mover su futuro  (exclusivo del esquema asimétrico)
```

**Por qué se incluye:** el greedy top-down comete divisiones tempranas que no puede
deshacer; el 1-move redistribuye futuros **y** mecanismos entre bloques ya formados.
El movimiento presente abre un espacio de búsqueda imposible en representaciones
simétricas, capturando mínimos de menor Φ. Garantiza óptimo local respecto a ambos
vecindarios (no global).

---

### 3. Metaheurística (Búsqueda Local Iterada — ILS)

**Utilizado:** sí — en `QNodes._refinar_con_ils` / `_perturbar_bloques`

**Descripción del uso:**

```
Refinar (best-improvement) hasta converger
Repetir n_ils veces:
    Perturbar (mover nodos futuros/presentes al azar)
    Re-refinar
    Conservar el mejor Φ global
```

Los parámetros (`n_ils`, `max_iter`, intensidad de perturbación) son **N-adaptativos**:
pocos ciclos de calidad para N grande, muchos para N pequeño. La ILS escapa de los
mínimos locales superficiales del 1-move.

---

### 4. Programación Dinámica (memoización)

**Utilizado:** sí — caché de distribuciones de bloque

**Descripción del uso:**

`_dist_bloque(fut_pos, pre_pos)` es DP top-down: la distribución marginal de cada
bloque (la operación más cara, `bipartir().distribucion_marginal()`) se computa una
sola vez y se almacena en `_cache_bloque`:

```python
def _dist_bloque(self, fut_pos, pre_pos):
    clave = (fut_pos, pre_pos)
    cache = self._cache_bloque.get(clave)
    if cache is None:
        cache = self.sia_subsistema.bipartir(...).distribucion_marginal()
        self._cache_bloque[clave] = cache
    return cache
```

**Por qué top-down y no bottom-up:** con el greedy + refinamiento + ILS, sólo aparece
un subconjunto pequeño de los 2^N bloques posibles. La versión perezosa paga sólo por
lo que se computa. La clave es `(frozenset, frozenset)`: hashing barato y estable.

---

## Paradigmas no utilizados y justificación

### Branch & Bound (B&B)

**¿Se utiliza?** No.

**Por qué:** incluso con poda del 90 %, B(15)×0.1 ≈ 140 millones y B(25)×0.1 ≈ 4×10¹⁷
seguirían siendo intratables. La poda no salva la búsqueda exhaustiva a estos tamaños.
El greedy top-down hace O(N³) evaluaciones independientemente de la cota.

---

### Backtracking puro

**¿Se utiliza?** No. Sin memoización ni poda exploraría B(N) ramas con recomputaciones.
Estrictamente dominado por B&B, que ya es intratable para N ≥ 15.

---

### Búsqueda exhaustiva (fuerza bruta)

**¿Se utiliza?** No en QNodes, pero existe como estrategia separada (`BruteForce`) como
ground truth para N ≤ 10, para validar la calidad del greedy.

---

### Algoritmos genéticos / Simulated Annealing

**¿Se utilizan?** No. La ILS ya provee escape de mínimos locales con operadores simples
(movimientos futuro/presente). La función objetivo (EMD sobre la TPM) no tiene
gradiente aprovechable, y la claridad del greedy + ILS supera en valor pedagógico a un
metaheurístico de caja negra con parámetros difíciles de tunear.

---

### Programación lineal / ILP

**¿Se utiliza?** No. La EMD de distribuciones marginales de subconjuntos de la TPM no
tiene forma lineal natural en las variables binarias de la partición; la formulación
sería intratable o imprecisa, y el overhead de un solver superaría al greedy directo.

---

### Divide y vencerás

**¿Se utiliza?** No como esquema de combinación de sub-soluciones. Aunque el greedy es
top-down, las distribuciones marginales de cada bloque dependen de la TPM del sistema
completo, por lo que no hay descomposición en subproblemas independientes con garantía
de optimalidad global.

---

### Clustering espectral / métodos geométricos

**¿Se utilizan?** No en QNodes — son el dominio de GeoMIP. La separación es intencional:
QNodes trabaja directamente sobre cortes asimétricos y EMD; GeoMIP usa afinidades
geométricas. Permite comparar qué filosofía funciona mejor. (Nota: GeoMIP también
adoptó el mismo motor greedy top-down asimétrico como ruta principal; el clustering
espectral quedó como fallback.)

---

## Resumen

| Paradigma | Estado | Justificación |
|---|---|---|
| Greedy top-down (divisivo) | **Utilizado — principal** | O(N³); un descenso = jerarquía anidada de todos los k |
| Heurístico (1-move futuro + presente) | **Utilizado — refinamiento** | Mejora post-greedy; movimiento presente exclusivo del esquema asimétrico |
| Metaheurística (ILS) | **Utilizado — escape de mínimos** | Perturbación + re-refinamiento N-adaptativo |
| Programación Dinámica (memoización) | **Utilizado — soporte** | `_cache_bloque` evita recalcular distribuciones |
| Branch & Bound | **No** | Intratable para N ≥ 15 aun con poda |
| Backtracking puro | **No** | Dominado por B&B |
| Búsqueda exhaustiva | **Solo en BruteForce** | Ground truth para N ≤ 10 |
| Genéticos / SA | **No** | Sin ventaja sobre ILS; menos pedagógico |
| ILP | **No** | Objetivo no linealizable |
| Divide y vencerás | **No** | Subproblemas no independientes |
| Clustering espectral | **No (ver GeoMIP)** | Otro framework |
