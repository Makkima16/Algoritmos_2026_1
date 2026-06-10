# Paradigmas algorítmicos utilizados

Este documento analiza cada paradigma algorítmico clásico, determina cuáles se
utilizaron en `AYDA_2026_1/QNodes`, con qué justificación, y por qué los demás
no resultaron aplicables o fueron descartados.

---

## Paradigmas utilizados

### 1. Algoritmo Voraz / Greedy (principal)

**Utilizado:** sí — en `QNodes._aglomerar` y `QNodes._refinamiento_local`

**Descripción del uso:**

El algoritmo greedy es el paradigma central de QNodes en ambas versiones del
proyecto. En `AYDA_2026_1`, se implementa como agrupamiento jerárquico aglomerativo:

```
En cada paso: elegir el par (Gᵢ, Gⱼ) cuya fusión minimiza _emd_particion
Nunca deshacer decisiones tomadas.
```

La función de costo usada es submodular (la reducción de Phi al añadir un elemento
decrece conforme el grupo crece), lo que hace que el greedy funcione bien en la
práctica para este tipo de problemas.

**Por qué greedy y no exhaustivo:** para N = 15, 20, 22, 25 la búsqueda exacta
es computacionalmente intractable (véase sección de Branch & Bound). El greedy
reduce la búsqueda a N³/6 evaluaciones de EMD, que es polinomial y tratable para
cualquier N del proyecto.

**Garantía:** el greedy garantiza un **óptimo local de primer orden** (no se puede
mejorar fusionando ningún par distinto en el último paso) pero no el óptimo global.
Esta es la misma garantía que ofrecía la versión original del proyecto.

**Fase 3 — candidatos de aislamiento (siempre activa):** para complementar el greedy
se evalúan exhaustivamente los candidatos donde k-1 nodos están completamente
aislados — el patrón de partición más frecuente en sistemas IIT. Para **k
especificado** se evalúan C(N, k-1) candidatos. Para **k libre**, se evalúan los
candidatos de aislamiento de *cada nivel k* del historial greedy (en total 2^N − 2
evaluaciones), y se elige el k globalmente óptimo entre todos los niveles refinados.
Esto hace que QNodes busque de forma completamente independiente, sin depender de
GeoMIP para saber qué k explorar, para todo N.

---

### 2. Heurístico (refinamiento local 1-move)

**Utilizado:** sí — en `QNodes._refinamiento_local`

**Descripción del uso:**

El refinamiento local es una heurística de mejora iterativa que complementa el
greedy aglomerativo. Después de que el agrupamiento produce una k-partición,
se aplica búsqueda local 1-move hasta convergencia:

```
Repetir hasta que ningún movimiento mejore Phi:
    Para cada nodo n ∈ Gᵢ (grupos con |Gᵢ| ≥ 2):
        Para cada grupo destino Gⱼ (j ≠ i):
            Si _emd_particion(candidato_con_movimiento) < phi_actual: aceptar
```

**Por qué es heurístico y no óptimo:** la búsqueda local 1-move garantiza un
óptimo local 1-opt (no existe movimiento individual que mejore), pero no garantiza
el óptimo 2-opt ni global. Es análogo al algoritmo de Lloyd (k-means) que tampoco
garantiza el óptimo global.

**Por qué se incluye:** el greedy aglomerativo comete errores tempranos que no
puede corregir (una vez fusionados, los grupos no se separan). El 1-move permite
redistribuir nodos entre grupos ya formados, compensando algunos de esos errores.
En la práctica mejora la calidad de la solución con muy poco costo adicional,
ya que la mayoría de las distribuciones ya están en caché.

---

### 3. Programación Dinámica (memoización)

**Utilizado:** sí — en el caché de distribuciones y costos

**Descripción del uso:**

La memoización de `_dist_parte(mascara)` y `_costo_parte(mascara)` es un caso
de programación dinámica top-down: se computan subproblemas (la distribución y el
costo de cada subconjunto de nodos) exactamente una vez y se almacenan para reutilización.

```python
def _dist_parte(self, mascara: int) -> np.ndarray:
    if mascara not in self._cache_dist:
        self._cache_dist[mascara] = <cálculo costoso: bipartir + distribucion_marginal>
    return self._cache_dist[mascara]
```

**Por qué top-down y no bottom-up:** un enfoque bottom-up precalcularía las
distribuciones de los 2^N subconjuntos posibles antes de comenzar. Con el greedy,
la mayoría de subconjuntos nunca se evalúan (solo los que aparecen como candidatos
de fusión, de refinamiento o de aislamiento). La versión perezosa calcula solo lo
necesario.

**Diferencia con la versión anterior:** en `DynamicPartition`, la DP era el motor
de búsqueda (recurrencia sobre particiones). En `QNodes`, la DP es solo soporte
de memoización para el greedy — los roles son distintos.

---

## Paradigmas no utilizados y justificación

### Branch & Bound (B&B)

**¿Se utiliza?** No — fue el paradigma central de la versión anterior
(`DynamicPartition`) y fue descartado en este reemplazo.

**Por qué se usaba antes:** en `DynamicPartition` el B&B podaba el árbol de
búsqueda exhaustiva cuando el costo acumulado superaba la mejor solución conocida.
Funcionaba bien para N ≤ 12 porque el número de Bell B(12) ≈ 4 millones es tratable.

**Por qué se descartó para N ≤ 25:** incluso con poda del 90 %, B(15) × 0.1 ≈
140 millones de evaluaciones y B(25) × 0.1 ≈ 4 × 10¹⁷ seguirían siendo intractables.
La poda no es suficiente para salvar la búsqueda exhaustiva a estos tamaños.
El greedy aglomerativo hace O(N³) evaluaciones independientemente de la cota,
lo que lo hace superior en el rango N = 15–25.

---

### Backtracking puro

**¿Se utiliza?** No.

**Por qué no:** el backtracking puro sin memoización ni poda exploraría B(N) ramas
con recomputaciones repetidas. Es estrictamente dominado por B&B, y B&B ya es
intractable para N ≥ 15. No aporta nada sobre el greedy en este contexto.

---

### Búsqueda exhaustiva (fuerza bruta)

**¿Se utiliza?** No en QNodes, pero existe como estrategia separada (`BruteForce`).

**Por qué existe en paralelo pero no en QNodes:** `BruteForce` sirve como ground
truth para sistemas pequeños (N ≤ 10). QNodes es la estrategia para sistemas grandes.
La co-existencia permite validar la calidad del greedy comparando ambas salidas en
sistemas donde la fuerza bruta es tratable.

---

### Algoritmos metaheurísticos (simulated annealing, algoritmos genéticos)

**¿Se utilizan?** No.

**Por qué no:**
1. El greedy + 1-move ya provee una solución de calidad razonable en tiempo polinomial.
2. Los metaheurísticos requieren definir operadores de vecindad y parámetros de
   enfriamiento/mutación que son dependientes del problema y difíciles de tunear.
3. La función objetivo (EMD sobre distribuciones de la TPM) no tiene gradiente
   aprovechable, lo que quita ventaja a métodos como gradiente estocástico.
4. En el contexto académico del proyecto, la claridad del greedy aglomerativo
   (análogo a clustering jerárquico, bien estudiado en la literatura) supera en
   valor pedagógico a un metaheurístico de caja negra.

---

### Programación lineal / ILP

**¿Se utiliza?** No.

**Por qué no:** formular la k-MIP como ILP requiere modelar la función objetivo
(EMD de distribuciones marginales de subconjuntos de la TPM) como restricciones
lineales enteras. Esta función no tiene una forma lineal natural en términos de
las variables de decisión binarias de la partición, lo que hace la formulación
intractable o imprecisa. Además, el overhead de solvers ILP para problemas pequeños
(N ≤ 25) es mayor que el del greedy directo.

---

### Divide y vencerás

**¿Se utiliza?** No.

**Por qué no:** divide y vencerás requiere que el problema se pueda descomponer en
subproblemas independientes que se combinan en tiempo polinomial. En la k-MIP, las
distribuciones marginales de cada parte dependen de la TPM del sistema completo
(no de subsistemas independientes), por lo que no existe una descomposición natural
que permita combinar sub-soluciones con garantía de optimalidad global.

---

### Clustering espectral / métodos geométricos

**¿Se utilizan?** No en QNodes — son el dominio de GeoMIP (el otro framework del proyecto).

**Por qué en GeoMIP y no en QNodes:** GeoMIP usa afinidades geométricas y matrices
de similitud para guiar la búsqueda, lo que es efectivo para sistemas con estructura
topológica clara. QNodes preserva su lógica original de función submodular sobre EMD
directamente, sin pasar por representaciones espectrales. La separación de enfoques
es intencional: permite comparar qué filosofía funciona mejor en distintos sistemas.

---

## Resumen

| Paradigma | Estado | Justificación |
|---|---|---|
| Greedy (aglomerativo + aislamiento) | **Utilizado — principal** | O(N³) + 2^N candidatos para k libre; aplica para todo N |
| Heurístico (1-move local) | **Utilizado — refinamiento** | Mejora calidad post-greedy sin costo significativo |
| Programación Dinámica (memoización) | **Utilizado — soporte** | Evita recalcular distribuciones, costos y matrices Hamming |
| Branch & Bound | **No** | Intractable para N≥15 aunque con poda del 90% |
| Backtracking puro | **No** | Dominado por B&B, y B&B ya es intractable |
| Búsqueda exhaustiva | **Solo en BruteForce** | Ground truth para N≤10 como validación |
| Metaheurísticas | **No** | Sin gradiente, sin ventaja sobre greedy; menos pedagógico |
| ILP / programación lineal | **No** | Objetivo no linealizable de forma natural |
| Divide y vencerás | **No** | Subproblemas no independientes entre sí |
| Clustering espectral | **No (ver GeoMIP)** | Filosófica y arquitecturalmente en el otro framework |
