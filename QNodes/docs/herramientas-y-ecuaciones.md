# Herramientas y ecuaciones utilizadas

---

## 1. Librerías y herramientas externas

### NumPy

Usada en prácticamente todo el código numérico.

- **Propósito:** representar TPMs, distribuciones de probabilidad y n-cubos como
  arrays.
- **Uso clave:**
  - `np.ndarray` con `dtype=float32/float64` para distribuciones marginales.
  - `np.fromiter((self._idx[p] for p in sorted(fut_pos)), ...)` para mapear
    posiciones de bloque a índices globales.
  - `np.abs` y `np.sum` para el cálculo de EMD L1 (O(N)).

**Archivos:** `src/models/core/ncube.py`, `src/funcs/iit.py`, `src/strategies/q_nodes.py`

---

### PyPhi (`pyphi 1.2.0`)

- **Propósito:** ground truth teórico de IIT. La estrategia `Phi` delega en PyPhi.
- **Uso:** validación de `QNodes` y `BruteForce`; no se usa en la búsqueda principal.

**Archivo:** `src/strategies/phi.py`

---

### pyinstrument (`5.1.2`)

- **Propósito:** profiling estadístico con reportes HTML.
- **Uso:** identificar cuellos de botella en `_greedy_descenso`, `_refinar_bloques` y
  `_dist_bloque`.

**Archivo:** `src/middlewares/profile.py`

---

### pandas + openpyxl

- **Propósito:** exportación de resultados en modo por bloques (CSV → Excel/JSON),
  escritos incrementalmente.

**Archivo:** `src/controllers/manager.py`

---

### colorama + pyttsx3 + tkinter

- **colorama:** color en terminal. **pyttsx3:** síntesis de voz del resultado.
  **tkinter:** diálogos de selección de archivo.

**Archivos:** `src/middlewares/slogger.py`, `src/models/core/solution.py`, `exec.py`

---

### pyemd — ELIMINADO de la ruta principal

- **Estado:** ya **no** se usa para calcular Φ. Se descubrió que la suma L1 marginal es
  la Wasserstein-1 con Hamming **exacta** (ver §2.2), que es O(N) en vez de O(4^N) y da
  el mismo valor sin límite de tamaño. `emd_causal`/`get_hamming_matrix` permanecen en
  `src/funcs/iit.py` sólo como utilidad de verificación histórica, no en el camino
  caliente.

---

## 2. Ecuaciones fundamentales

### 2.1 Información integrada Φ (phi)

Para una k-partición en bloques asimétricos `B = {(F₁,P₁), …, (Fₖ,Pₖ)}`, Φ es la EMD
entre la distribución marginal del subsistema y la reconstruida desde los bloques:

```
Φ(B) = EMD_Hamming( p_subsistema , p_reconstruida )

p_reconstruida[i] = distribución marginal del nodo futuro i dentro de su bloque,
                    condicionada por el presente (mecanismo) de ESE bloque
```

La k-MIP es `B* = argmin_{|B|=k} Φ(B)`, y `Φ = min_{k≥2} Φ(B*_k)` (prioridad k ≥ 3).

---

### 2.2 La métrica EMD: L1 marginal = Wasserstein-1 Hamming EXACTA

**Teorema (descomposición marginal en el hipercubo de Hamming):** si P y Q son
**productos de marginales** por nodo, la Wasserstein-1 con distancia base de Hamming se
descompone exactamente en la suma de las EMDs marginales unidimensionales:

```
EMD_Hamming(P, Q) = Σᵢ | P(nodo_i = ON) − Q(nodo_i = ON) |
d_Hamming(s₁, s₂) = popcount(s₁ XOR s₂)
```

En QNodes **ambas** distribuciones (original y reconstruida) son productos de
marginales por construcción, así que esta igualdad es **exacta** (verificado
`|emd_causal − L1| < 1e-14` para N = 2..12). Por eso la métrica única, para todo N, es:

```python
# _emd_bloques — O(N)
return float(np.sum(np.abs(dist_rec - self.sia_dists_marginales)))
```

No hay bifurcación por tamaño: la misma fórmula exacta vale para N = 5 y para N = 25.

---

### 2.3 Criterio de división greedy (top-down)

En cada paso del descenso se elige el par (bloque b, corte c) que minimiza Φ de la
configuración resultante de dividir b en `inside`/`outside`:

```
inside  = (b.F ∩ c.F,  b.P ∩ c.P)
outside = (b.F − c.F,  b.P − c.P)

(b*, c*) = argmin  _emd_bloques( bloques con b reemplazado por {inside, outside} )
```

Se exige que ambos lados conserven al menos un futuro.

---

### 2.4 Movimientos del refinamiento 1-move

Dos vecindarios, evaluados con `_emd_bloques` sobre la partición candidata completa:

```
Movimiento futuro:   F_i ← F_i \ {n},  F_j ← F_j ∪ {n}     (n nodo futuro)
Movimiento presente: P_i ← P_i \ {m},  P_j ← P_j ∪ {m}     (m nodo del mecanismo; futuro intacto)
```

Se acepta el globalmente mejor por ronda si `Φ_cand < Φ_actual − 1e-10`.

---

### 2.5 Distribución marginal de un bloque

Dado un bloque (F, P), su distribución marginal condiciona el futuro F sobre el
mecanismo P (que puede diferir del propio futuro — corte asimétrico — o estar vacío):

```python
# _dist_bloque
futuros   = indices_ncubos[posiciones de F]
presentes = dims_ncubos[posiciones de P]        # ∅ si P vacío
dist = bipartir(futuros, presentes).distribucion_marginal()
```

Memoizada en `_cache_bloque` por clave `(frozenset F, frozenset P)`.

---

### 2.6 Extracción de bits activos

```python
def _bits_activos(mascara: int):
    m = mascara
    while m:
        bit = m & (-m)               # aísla el bit menos significativo activo (O(1))
        yield bit.bit_length() - 1
        m ^= bit
```

El truco `m & (-m)` funciona por complemento a dos: `-m` invierte y suma 1, aislando
exactamente el primer bit activo.

---

### 2.7 `distribucion_conjunta_vectorizada` (utilidad de verificación)

Construye la distribución conjunta 2^N a partir de N marginales asumiendo independencia.
Se usa **sólo** para verificar la equivalencia L1 = EMD-Hamming (§2.2), no en el camino
caliente:

```python
factors = np.stack([1 - p_on, p_on], axis=1)
grid = np.meshgrid(*factors, indexing="ij")
return np.prod(grid, axis=0).flatten()          # shape (2^N,)
```

**Archivo:** `src/funcs/iit.py`
