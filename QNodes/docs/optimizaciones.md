# Optimizaciones realizadas en QNodes

Este documento detalla las optimizaciones aplicadas en `AYDA_2026_1/QNodes` que
hacen el algoritmo **más rápido y más preciso** para N = 10, 15, 20, 22, 25 nodos,
el rango objetivo del proyecto.

Las dos optimizaciones más importantes son conceptuales, no de implementación:

1. **La métrica L1 marginal es la EMD de Hamming EXACTA** (no una aproximación) →
   reduce cada evaluación de Φ de O(4^N) a O(N) **sin perder precisión** y elimina
   el límite N ≤ 12.
2. **Los cortes asimétricos** (futuro y presente particionados de forma
   independiente) corrigen el "sobre-corte" que inflaba Φ → **resultados más
   precisos y coherentes** (Φ monótono entre k consecutivos).

Todo lo demás (greedy top-down con pool compartido, movimiento presente, ILS,
memoización) acelera o afina sobre estas dos bases.

---

## 1. ⭐ Métrica L1 marginal = Wasserstein-1 Hamming EXACTA (más rápida Y más precisa)

**Archivo:** `src/strategies/q_nodes.py` — `_emd_bloques`

Esta es la optimización de mayor impacto, porque mejora **simultáneamente velocidad
y precisión**, algo poco común (normalmente una se sacrifica por la otra).

### El problema previo

La versión anterior calculaba Φ de dos formas según N:
- **N ≤ 12:** EMD real con `pyemd` sobre la distribución conjunta de 2^N estados y
  una matriz de costes Hamming de 2^N × 2^N → **O(4^N)**.
- **N > 12:** suma L1 marginal como **aproximación**, asumida menos precisa.

Esto imponía un techo duro: para N > 12 la EMD "real" era inviable en memoria
(la matriz de costes de N=20 pesaría ~10¹² entradas), así que se aceptaba que el
resultado para sistemas grandes era aproximado.

### La observación clave

Tanto la distribución original como la reconstruida de **cualquier** k-partición
son **productos de marginales por nodo** (independencia condicional garantizada por
construcción: cada bloque se reconstruye con `distribucion_marginal`). Para **dos
distribuciones producto** sobre el hipercubo booleano con métrica base de Hamming,
la Wasserstein-1 (EMD real) **coincide EXACTAMENTE** con la suma de diferencias L1
marginales, porque la distancia de Hamming es separable por coordenada y el
acoplamiento óptimo factoriza coordenada a coordenada:

```
EMD_Hamming(P, Q) = Σᵢ |P(nodo_i = ON) − Q(nodo_i = ON)|        (cuando P, Q son productos)
```

Verificado numéricamente: `|emd_causal(P,Q) − L1| < 1e-14` para N = 2..12.

### Consecuencia: más rápido

```python
def _emd_bloques(self, bloques) -> float:
    dist_rec = np.empty(self._N, dtype=np.float64)
    for fut_pos, pre_pos in bloques:
        if not fut_pos:
            continue
        dist_bloque = self._dist_bloque(fut_pos, pre_pos)   # memoizada
        for p in fut_pos:
            dist_rec[p] = float(dist_bloque[p])
    return float(np.sum(np.abs(dist_rec - self.sia_dists_marginales)))   # O(N)
```

Cada evaluación de Φ pasa de **O(4^N)** (construir conjunta + solver de transporte)
a **O(N)** (un vector de N restas absolutas). Para N=10 esto es la diferencia entre
construir y operar sobre matrices de 1024×1024 y operar sobre un vector de 10
elementos.

### Consecuencia: más preciso

- **Da el MISMO Φ que la EMD real**, no una aproximación: k=2 en N10A = 0.4746,
  idéntico al de GeoMIP y a la antigua ruta `pyemd`.
- **Elimina el límite N ≤ 12.** Antes, N > 12 usaba una aproximación; ahora todos
  los N usan la fórmula exacta. El resultado para N = 20, 22, 25 es tan exacto como
  para N = 5.
- Por eso se **eliminó la dependencia de `pyemd`**: era más lenta (O(4^N)), daba el
  mismo número, y restringía el tamaño del sistema.

> Esto **corrige** la nota antigua que afirmaba que "L1 = pyemd sólo si P es un
> producto, lo cual no ocurre en IIT". Sí ocurre aquí: ambas distribuciones son
> productos de marginales **por construcción**.

---

## 2. ⭐ Cortes asimétricos: corrigen el sobre-corte (más preciso)

**Archivo:** `src/strategies/q_nodes.py` — representación `Bloque`, `_construir_pool_cortes`

### El problema previo

Cada bloque se representaba de forma **simétrica**: el presente (mecanismo, t) de un
grupo era siempre `futuros ∩ dims`, es decir, cada grupo sólo condicionaba sobre sus
propios nodos. Esto **sobre-cortaba** el sistema: un nodo aislado en su futuro perdía
también su rol causal como condicionante de otros bloques, inflando Φ.

El síntoma era un **salto incoherente** entre k=2 y k=3 (medido en N10A,
estado=1000…0):

| k | Corte simétrico (viejo) | Corte asimétrico (nuevo) |
|---|---|---|
| 2 | 0.4746 | 0.4746 |
| 3 | **2.5059** (salto) | **0.9590** (monótono) |

### La solución

Cada bloque es ahora un par **`(frozenset futuros, frozenset presentes)`** donde
futuro (t+1) y presente (t) se particionan de forma **independiente**:

- **futuros:** el alcance/efecto que el bloque produce.
- **presentes:** el mecanismo/causa que el bloque conserva como condicionante.

Un nodo puede aportar su mecanismo a un bloque mientras su futuro vive en otro. Esto
replica la lógica de GeoMIP y evita penalizaciones causales falsas.

### Impacto en precisión

- Φ crece de forma **monótona y coherente** con k (patrón Φ(k) ≈ (k−1)×~0.5).
- QNodes y GeoMIP dan el **mismo Φ** para todo k (ambos asimétricos + L1 exacta).
- Desaparece la inflación artificial: el `({i}, ∅)` (mecanismo vacío) deja que el
  nodo i siga condicionando causalmente al resto.

---

## 3. Pool de cortes O(N) construido una sola vez (más rápido)

**Archivo:** `src/strategies/q_nodes.py` — `_construir_pool_cortes`

En lugar de generar cortes candidatos en cada paso, se construye **una sola vez** un
pool de **O(N)** cortes y se reutiliza en todo el descenso greedy y para todos los k:

```python
for i in range(self._N):
    eff = frozenset((i,))
    pre = frozenset((i,)) if i < self._n_dims else frozenset()
    _add(eff, pre)                    # 1. aislamiento simétrico ({i}, {pre_i})
    _add(all_fut - eff, all_pre - pre)  # 2. su complemento
    _add(eff, frozenset())            # 3. aislamiento con mecanismo vacío ({i}, ∅)
```

Tres familias por nodo → **3N cortes** en total, deduplicados. Construir el pool es
O(N); el costo se **amortiza** entre todos los splits y todos los niveles k del
descenso, en vez de regenerarse en cada paso.

---

## 4. Greedy top-down con jerarquía nido (más rápido Y más coherente)

**Archivo:** `src/strategies/q_nodes.py` — `_greedy_descenso`, `_greedy_bloques`

Un **único descenso** top-down (de 1 bloque a N bloques) registra Φ para **cada k**:

```python
historico = {1: (phi_inicial, bloques)}
while len(bloques) < self._N:
    phi, bloques = self._mejor_split_bloques(bloques, pool)
    historico[len(bloques)] = (phi, list(bloques))
```

- **Más rápido (k libre):** no se rehace la búsqueda por cada k; un solo descenso
  O(N²·|pool|) produce la solución de todos los k simultáneamente.
- **Más coherente:** cada k surge de dividir un bloque del nivel anterior →
  **jerarquía anidada** → Φ monótono no decreciente entre k consecutivos, sin saltos.
- **Más rápido (k fijo):** `_greedy_bloques` detiene el descenso exactamente en k.

---

## 5. Movimiento presente asimétrico en el refinamiento (más preciso)

**Archivo:** `src/strategies/q_nodes.py` — `_refinar_bloques`

El refinamiento best-improvement explora **dos** tipos de vecinos, el segundo
imposible en representaciones simétricas:

- **Movimiento futuro:** traslada un nodo futuro del bloque i al j.
- **Movimiento presente (asimétrico):** traslada el **mecanismo** de un nodo sin
  mover su futuro.

El movimiento presente abre una dimensión de búsqueda completamente nueva: permite
afinar *qué bloque condiciona sobre qué nodo* sin alterar la partición de los
futuros. Captura mínimos que el 1-move clásico (simétrico) no puede alcanzar →
**soluciones de menor Φ**. Aplica el globalmente mejor vecino por ronda hasta
convergencia.

---

## 6. ILS N-adaptativa (más preciso sin penalizar el tiempo)

**Archivo:** `src/strategies/q_nodes.py` — `_refinar_con_ils`, `_perturbar_bloques`

La Búsqueda Local Iterada perturba el óptimo local y re-refina para escapar de
mínimos locales. Sus parámetros **decrecen con N** para mantener el tiempo acotado
sin perder calidad donde más importa:

```python
max_it = max(5, 20 - max(0, self._N - HAMMING_EMD_MAX_N))
n_ils  = max(1, _N_ILS - max(0, (self._N - 16) // 2))
n_mov  = max(1, self._N // 4)   # intensidad de la perturbación
```

Para N pequeño se hacen muchos ciclos de calidad; para N grande, pocos pero
dirigidos. Pocos ciclos buenos superan a muchos mediocres cuando cada evaluación es
cara. `HAMMING_EMD_MAX_N` ya **no** selecciona métrica (la métrica es siempre L1
exacta); sólo calibra cuántas iteraciones de ILS valen la pena.

---

## 7. Memoización de distribuciones de bloque (más rápido)

**Archivo:** `src/strategies/q_nodes.py` — `_dist_bloque`, `_cache_bloque`

La operación más cara es `bipartir(...).distribucion_marginal()`. Se memoiza por
clave `(futuros, presentes)`:

```python
def _dist_bloque(self, fut_pos, pre_pos):
    clave = (fut_pos, pre_pos)
    cache = self._cache_bloque.get(clave)
    if cache is None:
        cache = self.sia_subsistema.bipartir(futuros, presentes).distribucion_marginal()
        self._cache_bloque[clave] = cache
    return cache
```

El mismo bloque aparece como candidato muchas veces a lo largo del descenso, el
refinamiento y la ILS. Con caché, cada distribución se computa **una sola vez por
sesión**. Las claves son `frozenset` (no listas), con hashing barato y estable.

El caché se limpia al inicio de cada `aplicar_estrategia` para que un sistema no
contamine al siguiente en runs por bloques.

---

## Resumen de impacto por optimización

| Optimización | Efecto | Velocidad | Precisión |
|---|---|---|---|
| **L1 = EMD Hamming exacta** | O(4^N) → O(N), sin límite N≤12 | ⭐⭐⭐ | ⭐⭐⭐ (mismo Φ, exacto) |
| **Cortes asimétricos** | Corrige el sobre-corte | — | ⭐⭐⭐ (sin saltos en k) |
| Pool de cortes O(N) único | Amortizado entre k y splits | ⭐⭐ | — |
| Greedy top-down nido | 1 descenso = todos los k | ⭐⭐ | ⭐ (Φ monótono) |
| Movimiento presente | Vecindario asimétrico nuevo | — | ⭐⭐ |
| ILS N-adaptativa | Escapa de mínimos locales | ⭐ | ⭐⭐ |
| Memoización `_cache_bloque` | Distribución 1 vez por sesión | ⭐⭐ | — |

**Tiempos medidos (estado='1000…0', candidato/alcance/mecanismo completos):**

- **N10A:** k=2 = 0.4746 / k=3 = 0.9590 / k=4 = 1.4453, todos en **~0.2–0.5 s**
  (antes k=2 tardaba ~34 s).
- **N15A:** k=2 = 0.4956 / k=3 = 0.9919 (~1 s).
- **N20A:** k=2 = 0.4992 / k=3 = 0.9985 (~15–20 s); k=None ~60 s.

Las dos primeras filas (L1 exacta + cortes asimétricos) son las que convierten a
QNodes en simultáneamente más rápido **y** más preciso que la versión anterior.
