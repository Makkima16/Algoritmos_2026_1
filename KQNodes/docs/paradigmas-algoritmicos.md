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

### 4. Algoritmo Exacto para Bipartición — Queyranne 1998 (k=2)

**Utilizado:** sí — en `QNodes._queyranne` / `_atomos_asimetricos` (desde 2026-06-13)

**Descripción del uso:**

Para k=2 se usa el **algoritmo de Queyranne** (1998) para minimización exacta de
funciones submodulares simétricas. La función de corte Φ(S) = Φ(bipartición S | V∖S)
es submodular, por lo que Queyranne la minimiza en **O(N²) evaluaciones** sin enumerar
los 2^(N-1) cortes posibles:

```
Fase de ordenamiento (tipo Prim / máxima adyacencia):
  Mantener un conjunto de "afines". En cada paso, añadir el elemento
  más afín al conjunto actual (máxima suma de aristas cruzadas). El
  penúltimo y último elemento forman un par colgante.

Contracción: el par colgante {s,t} da el mínimo corte {t}|{V∖{t}};
  se contraen s y t, y se repite sobre el sistema reducido.
  El mínimo de todos los cortes contraidoes es el global.
```

Los **2N átomos asimétricos** (N de futuro `({i},∅)` + N de presente `(∅,{j})`) son
los "vértices" sobre los que opera Queyranne, cubriendo el espacio COMPLETO de
biparticiones asimétricas.

**Garantía:** el óptimo **global** de Φ para k=2, sin aproximación. Ninguna heurística
(incluido GeoMIP k=2) puede garantizar esto.

**Por qué no se usa para k≥3:** la minimización submodular sobre biparticiones
(funciones de corte) no se extiende a k-particiones — el problema de k-corte para k≥3
es NP-hard. Por eso k≥3 sigue usando greedy + ILS.

---

### 5. Programación Dinámica (memoización)

**Utilizado:** sí — caché de distribuciones de bloque y de valores marginales

**Descripción del uso:**

Hay dos niveles de memoización desde 2026-06-13:

**Nivel 1 — `_cache_bloque` en QNodes:**
`_dist_bloque(fut_pos, pre_pos)` cachea el vector resultado por clave `(futuros, presentes)`.
El mismo bloque reaparece en el descenso, el refinamiento y la ILS; se computa una sola vez.

```python
def _dist_bloque(self, fut_pos, pre_pos):
    clave = (fut_pos, pre_pos)
    cache = self._cache_bloque.get(clave)
    if cache is None:
        pre_global = frozenset(int(self._dims[q]) for q in pre_pos ...)
        result = np.zeros(self._N, dtype=np.float64)
        for p in fut_pos:
            ncubo = self._ncubos_idx[int(self._idx[p])]
            ejes = np.array([d for d in ncubo.dims if int(d) not in pre_global], dtype=np.int8)
            result[p] = ncubo.marginal_valor(ejes, estado)   # ← nivel 2
        cache = result
        self._cache_bloque[clave] = cache
    return cache
```

**Nivel 2 — `valor_memo` en NCube:**
`NCube.marginal_valor(ejes, estado_inicial)` cachea el escalar P(nodo=1) por el
subconjunto de ejes a promediar (la clave no incluye `estado_inicial` porque es fijo
para toda la sesión). Coste de la primera vez: O(2^|ejes|). Coste de reusos: O(1).

**Por qué top-down y no bottom-up:** con el greedy + refinamiento + ILS, sólo aparece
un subconjunto pequeño de los bloques posibles. La versión perezosa paga sólo por lo
que se computa.

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
| Greedy top-down (divisivo) | **Utilizado — principal (k≥3)** | O(N³); un descenso = jerarquía anidada de todos los k |
| **Queyranne 1998 (exacto)** | **Utilizado — k=2** | Minimiza submodular simétrica exacto en O(N²); garantía global |
| Heurístico (1-move futuro + presente) | **Utilizado — refinamiento** | Mejora post-greedy; movimiento presente exclusivo del esquema asimétrico |
| Metaheurística (ILS) | **Utilizado — escape de mínimos** | Perturbación + re-refinamiento N-adaptativo |
| Programación Dinámica (memoización) | **Utilizado — soporte** | `_cache_bloque` + `valor_memo` evitan recalcular distribuciones y valores |
| Branch & Bound | **No** | Intratable para N ≥ 15 aun con poda |
| Backtracking puro | **No** | Dominado por B&B |
| Búsqueda exhaustiva | **Solo en BruteForce** | Ground truth para N ≤ 10 |
| Genéticos / SA | **No** | Sin ventaja sobre ILS; menos pedagógico |
| ILP | **No** | Objetivo no linealizable |
| Divide y vencerás | **No** | Subproblemas no independientes |
| Clustering espectral | **No (ver GeoMIP)** | Otro framework; supera a QNodes k≥3 en N≥22 |
