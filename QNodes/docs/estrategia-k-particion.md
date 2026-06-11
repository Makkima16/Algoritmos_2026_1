# Estrategia implementada para hallar la k-partición

**Clase:** `QNodes` (alias `DynamicPartition`) — `src/strategies/q_nodes.py`

---

## 1. Definición del problema

Dado un sistema de N nodos con una distribución de probabilidad conjunta, se busca la
**k-partición de mínima información** (MIP): la división del sistema en k bloques tal
que la pérdida Phi (Φ) sea mínima.

Cuando k no está fijado, se busca sobre todos los k ∈ [2, N] y se reporta el k con
menor Φ global (priorizando k ≥ 3 como objetivo principal del proyecto).

---

## 2. Representación ASIMÉTRICA de bloques (clave de todo el algoritmo)

Cada bloque es un par **`(frozenset futuros, frozenset presentes)`** de posiciones
locales que se particionan de forma **independiente**:

- **futuros (t+1):** el alcance/efecto que el bloque produce.
- **presentes (t):** el mecanismo/causa que el bloque conserva como condicionante.

A diferencia de un corte **simétrico** (donde el presente de un bloque es siempre
`futuros ∩ dims`, es decir cada grupo sólo condiciona sobre sus propios nodos), el
corte **asimétrico** permite que un nodo aislado en su futuro siga actuando como
condicionante causal de otro bloque. Esto evita el "sobre-corte" que inflaba Φ y
genera coherencia entre k (ver `docs/optimizaciones.md`, §2).

```
Bloque = (frozenset futuros_pos, frozenset presentes_pos)

Ejemplo k=3 sobre {A,B,C,D,E}:
  ({A,B}, {a,b})   ({C,D}, {c})   ({E}, ∅)
   futuro  presente  futuro presente  futuro  mecanismo vacío
```

---

## 3. Por qué no se usa búsqueda exhaustiva

La búsqueda exhaustiva sobre todas las k-particiones requiere explorar B(N) casos
(número de Bell), que crece super-exponencialmente:

| N | B(N) | Tiempo estimado |
|---|---|---|
| 10 | ~115 000 | segundos |
| 15 | ~1 400 millones | horas |
| 20 | ~5 × 10¹³ | intratable |
| 25 | ~4 × 10¹⁸ | intratable |

Para N = 15, 20, 22, 25 — los tamaños objetivo — la búsqueda exhaustiva es inviable
incluso con poda agresiva.

---

## 4. Estructura general del algoritmo

El **mismo motor** se usa para todo k, sin distinguir el caso k=2:

```
aplicar_estrategia()
  ├── 1. Preparar subsistema (condicionar TPM al estado inicial, marginalizar)
  ├── 2. _construir_pool_cortes()  → pool de O(N) cortes, construido UNA vez
  │
  ├── [k especificado]
  │       ├── _greedy_bloques(pool, k)   — top-down hasta k bloques
  │       ├── _refinar_bloques(...)      — best-improvement 1-move (futuro + presente)
  │       └── _refinar_con_ils(...)      — perturbación + re-refinamiento (ILS)
  │
  └── [k libre — k=None]
          ├── _greedy_descenso(pool)     — un descenso = un Φ por CADA k (jerarquía nido)
          ├── _refinar_bloques(nivel)    — refinamiento ligero por cada nivel k
          ├── elegir k ≥ 3 con menor Φ
          └── _refinar_con_ils(ganador)  — ILS final sobre el k ganador
```

La memoización de `_cache_bloque` hace que las distribuciones de bloque vistas sean
O(1) en evaluaciones posteriores, amortizando el costo de las cuatro fases.

---

## 5. Pool de cortes (`_construir_pool_cortes`)

Se construye **una sola vez** un pool de O(N) cortes candidatos. Por cada nodo i se
generan tres familias:

```python
for i in range(self._N):
    eff = frozenset((i,))
    pre = frozenset((i,)) if i < self._n_dims else frozenset()
    _add(eff, pre)                       # 1. aislamiento simétrico ({i}, {pre_i})
    _add(all_fut - eff, all_pre - pre)   # 2. su complemento
    _add(eff, frozenset())               # 3. aislamiento con mecanismo vacío ({i}, ∅)
```

La familia 3 (mecanismo vacío, estilo GeoMIP) es la que, al aplicarse a un bloque B,
produce `inside=({i}, ∅)` dejando el mecanismo de i intacto en `outside` — i sigue
condicionando al resto. El pool se comparte entre todos los splits y todos los niveles
k del descenso.

---

## 6. Greedy top-down sobre bloques

### 6.1 Mejor split (`_mejor_split_bloques`)

Para cada bloque b y cada corte c del pool se evalúa dividir b en:

```
inside  = (b.fut ∩ c.fut,  b.pre ∩ c.pre)
outside = (b.fut − c.fut,  b.pre − c.pre)
```

Se exige que **ambos** lados conserven al menos un futuro (partición limpia de los N
nodos futuros); el presente puede quedar asimétrico o vacío. Se elige el split con
menor Φ (`_emd_bloques`).

### 6.2 k especificado (`_greedy_bloques`)

Parte de un único bloque (TODOS los futuros, TODOS los presentes) y aplica k−1 mejores
splits, deteniéndose exactamente en k bloques.

### 6.3 k libre (`_greedy_descenso`)

Un **único descenso** de k=1 a k=N registra Φ en cada nivel:

```python
historico = {1: (self._emd_bloques(bloques), list(bloques))}
while len(bloques) < self._N:
    phi, bloques = self._mejor_split_bloques(bloques, pool)
    historico[len(bloques)] = (phi, list(bloques))
```

Como cada k surge de dividir un bloque del nivel anterior, la jerarquía es **anidada**
→ Φ monótono no decreciente entre k consecutivos (coherencia garantizada, sin saltos).

---

## 7. Refinamiento local best-improvement (`_refinar_bloques`)

En cada ronda evalúa TODOS los vecinos y aplica el globalmente mejor. Hay dos tipos de
movimiento:

- **Movimiento futuro:** traslada un nodo futuro del bloque i al j (sin vaciar el
  futuro de i).
- **Movimiento presente (asimétrico):** traslada el **mecanismo** de un nodo del
  bloque i al j **sin** mover su futuro — exclusivo del esquema asimétrico.

```python
# movimiento presente
cfg[i] = (eff_i, pre_i - {nodo})
cfg[j] = (eff_j, pre_j | {nodo})
phi_cand = self._emd_bloques(cfg)
if phi_cand < mejor_phi - 1e-10:
    mejor_phi, mejor = phi_cand, cfg
```

Repite hasta convergencia o `max_iter` rondas (20 por defecto, 5 por nivel en k libre).
Garantiza un óptimo local respecto a ambos vecindarios.

---

## 8. Búsqueda Local Iterada (`_refinar_con_ils`)

Perturba el óptimo local (`_perturbar_bloques`, alternando movimientos futuros y
presentes aleatorios) y re-refina, conservando el mejor Φ. Sus parámetros son
**N-adaptativos** para acotar el tiempo sin perder calidad:

```python
max_it = max(5, 20 - max(0, self._N - HAMMING_EMD_MAX_N))
n_ils  = max(1, _N_ILS - max(0, (self._N - 16) // 2))
n_mov  = max(1, self._N // 4)
```

`HAMMING_EMD_MAX_N` (=12) ya **no** selecciona métrica; sólo calibra cuántas
iteraciones de ILS valen la pena para cada N.

---

## 9. Evaluación de Φ (`_emd_bloques`)

Reconstruye la distribución marginal a partir de la distribución de cada bloque
(futuro condicionado por su propio presente) y la compara con la original mediante
**suma L1 marginal**, que es la Wasserstein-1 con Hamming **EXACTA** (ambas
distribuciones son productos de marginales — ver `docs/optimizaciones.md`, §1):

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

Es **O(N)** por evaluación, da el mismo Φ que la EMD real, y vale para todo N sin
restricción de tamaño.

---

## 10. Reconstrucción de la solución

Al terminar, `mejor_bloques` se formatea con `fmt_k_bloques` (futuros en MAYÚSCULAS,
presentes en minúsculas por bloque, ∅ si el mecanismo está vacío):

```
|  A,B  ||  C,D  || E |
|  a,b  ||   c   || ∅ |
```

---

## 11. Complejidad

| Fase | Complejidad | Nota |
|---|---|---|
| Pool de cortes | O(N) | Una vez |
| Mejor split | O(k · |pool|) = O(k · N) evaluaciones | Por paso del descenso |
| Greedy top-down (k libre) | O(N² · N) = O(N³) evaluaciones | Un descenso, todos los k |
| Refinamiento 1-move | O(rondas · (Σ|fut| + Σ|pre|) · k) | Futuro + presente |
| ILS | n_ils × (refinamiento) | N-adaptativo |
| Una evaluación `_emd_bloques` | **O(N)** | Exacta, todo N |

**Tiempos medidos** (estado='1000…0'): N10A ~0.2–0.5 s por k; N15A ~1 s; N20A k≥3
~15–20 s, k=None ~60 s. Patrón Φ(k) ≈ (k−1)×~0.5.
