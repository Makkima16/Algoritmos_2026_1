# Herramientas y ecuaciones utilizadas

---

## 1. Librerías y herramientas externas

### NumPy

Usada en prácticamente todo el código numérico del proyecto.

- **Propósito:** representar TPMs, distribuciones de probabilidad y n-cubos
  como arrays N-dimensionales.
- **Uso clave:**
  - `np.ndarray` con `dtype=float32` para distribuciones marginales
  - `np.fromiter(_bits_activos(mascara), dtype=np.int8)` para extraer índices
    de grupos representados como máscaras
  - `np.abs` y `np.sum` para el cálculo de EMD L1
  - `np.intersect1d` para hallar los índices presentes de una parte

**Archivos principales:** `src/models/core/ncube.py`, `src/funcs/iit.py`,
`src/strategies/q_nodes.py`

---

### SciPy

- **Propósito:** cálculos científicos auxiliares.
- **Uso:** soporte para operaciones estadísticas sobre distribuciones que NumPy
  no cubre directamente.

---

### PyPhi (`pyphi 1.2.0`)

- **Propósito:** ground truth teórico de IIT 3.0. La estrategia `Phi` actúa como
  wrapper que delega el cálculo de Φ a PyPhi.
- **Uso:** validación de resultados de `QNodes` y `BruteForce`. No se usa en la
  búsqueda principal porque es más lento y menos configurable.

**Archivo:** `src/strategies/phi.py`

---

### pyinstrument (`5.1.2`)

- **Propósito:** profiling estadístico de llamadas a función. Genera reportes HTML.
- **Uso:** activado opcionalmente para identificar cuellos de botella en
  `_aglomerar`, `_costo_parte` y las operaciones de EMD.

**Archivo:** `src/middlewares/profile.py`

---

### pandas + openpyxl

- **Propósito:** exportación de resultados a Excel para análisis posterior.
- **Uso:** en modo por bloques (CSV), los resultados se escriben incrementalmente
  en un archivo `.xlsx` sin esperar a que termine el batch completo.

**Archivo:** `src/controllers/manager.py`

---

### colorama + pyttsx3

- **colorama:** resaltado de colores en la salida de terminal.
- **pyttsx3:** síntesis de voz para narrar el resultado final de cada sistema.

**Archivos:** `src/middlewares/slogger.py`, `src/models/core/solution.py`

---

### tkinter (stdlib)

- **Propósito:** diálogos de selección de archivo nativos del SO.
- **Uso:** explorador de archivos para seleccionar TPM o CSV en modo interactivo.

**Archivo:** `exec.py`

---

### pyemd (`2.0.0`)

- **Propósito:** solver de Earth Mover's Distance (Wasserstein-1) con matriz de
  costes arbitraria. Envuelve una implementación C eficiente del algoritmo de
  transporte óptimo.
- **Uso:** en `emd_causal` (`src/funcs/iit.py`) para calcular la EMD real con
  distancia de Hamming como métrica base, activo para N ≤ 12.

```python
from pyemd import emd as _pyemd
resultado = _pyemd(P.astype(np.float64), Q.astype(np.float64), mat_hamming)
```

**Archivo:** `src/funcs/iit.py` — funciones `emd_causal`, `get_hamming_matrix`

---

## 2. Ecuaciones fundamentales

### 2.1 Información integrada Φ (phi)

La medida central del proyecto. Para una k-partición P = {P₁, P₂, …, Pₖ},
la métrica exacta de IIT es la distancia de Wasserstein-1 con métrica base Hamming
entre la distribución conjunta del sistema y la distribución reconstruida asumiendo
independencia entre partes:

```
Φ(P) = EMD_Hamming(p_sistema_conjunta, p_particion_conjunta)

p_particion_conjunta = ⊗ᵢ p_Pᵢ     (producto tensorial — asume independencia)
```

Para N ≤ 12 (donde 2^N distribución conjunta es manejable en memoria) se calcula
la EMD real. Para N > 12 se usa la aproximación L1 marginal:

```
Φ_L1(P) = Σᵢ costo_L1(Pᵢ)

costo_L1(Pᵢ) = Σⱼ∈Pᵢ |p_Pᵢ(j) − p_S(j)|
```

La k-partición de mínima información (MIP) es:

```
P* = argmin_{P, |P|=k} Φ(P)
Φ  = min_{k≥2} Φ(P*_k)      (con prioridad a k ≥ 3)
```

---

### 2.2 Métricas de EMD: Hamming (N ≤ 12) y L1 (N > 12)

**Para N ≤ 12 — EMD real con distancia Hamming:**

```
EMD_Hamming(P, Q) = Wasserstein-1(P_conjunta, Q_conjunta, d_Hamming)
d_Hamming(s₁, s₂) = popcount(s₁ XOR s₂)    (bits diferentes entre estados)
```

Implementado en `emd_causal` usando `pyemd` con la matriz de costes Hamming
de tamaño 2^N × 2^N, cacheada en `_HAMMING_CACHE`:

```python
mat_costes = get_hamming_matrix(2**N)
return _pyemd(P.astype(float64), Q.astype(float64), mat_costes)
```

**Para N > 12 — suma L1 marginal (aproximación):**

```
costo_L1(Pᵢ) = Σⱼ∈Pᵢ |p_Pᵢ(j) − p_S(j)|
```

En el código (`src/strategies/q_nodes.py`):

```python
costo_normal = float(
    np.sum(np.abs(dist[idx_arr] - self.sia_dists_marginales[idx_arr]))
)
```

donde `idx_arr` son los índices de bits activos de la máscara.

---

### 2.3 Función de fusión greedy (criterio de selección de par)

**Para N > 12 (L1 aditivo):**

```
Δ(Gᵢ, Gⱼ) = costo(Gᵢ ∪ Gⱼ) − costo(Gᵢ) − costo(Gⱼ)
```

El par con **menor Δ** se fusiona. La aditividad del L1 permite calcular Δ
sin evaluar toda la partición.

**Para N ≤ 12 (EMD Hamming sobre la partición completa):**

```
phi_cand(candidato) = _emd_particion(grupos_sin_i_j + [Gᵢ ∪ Gⱼ])
```

Se elige la fusión que minimiza `phi_cand`. No hay delta: se evalúa la EMD
Wasserstein-1 de la partición completa resultante.

---

### 2.4 Delta de movimiento local 1-move

**Para N > 12 (L1 aditivo):**

```
Δ_move(n, Gᵢ → Gⱼ) = costo(Gᵢ \ {n}) + costo(Gⱼ ∪ {n}) − costo(Gᵢ) − costo(Gⱼ)
```

Se acepta si `Δ_move < 0`. La aditividad permite calcular el delta con solo
dos grupos, sin recalcular el sistema completo.

**Para N ≤ 12 (EMD Hamming sobre la partición candidata):**

```
phi_cand = _emd_particion(grupos con Gᵢ → Gᵢ\{n}, Gⱼ → Gⱼ∪{n})
```

Se acepta si `phi_cand < phi_total − ε`.

---

### 2.5 Distribución marginal de una parte

Dado el sistema S con N nodos y una parte Pᵢ ⊆ S, su distribución marginal se
obtiene marginalizando sobre los nodos fuera de Pᵢ:

```
p_Pᵢ(xᵢ) = Σ_{xⱼ : j ∉ Pᵢ} p(S = x)
```

En la representación de n-cubos esto corresponde a colapsar las dimensiones
de los nodos no incluidos en Pᵢ mediante promedio condicional:

```python
# ncube.py
def marginalizar(self, ejes: tuple[int, ...]) -> np.ndarray:
    return self.data.mean(axis=ejes)
```

---

### 2.6 Bipartición de un subsistema (para evaluar una parte)

Para evaluar el costo de la parte Pᵢ se construye la bipartición del subsistema:
- `futuros` = índices de nodos futuros (alcance/purview) en Pᵢ
- `presentes` = intersección de futuros con los nodos del mecanismo

```python
futuros  = indices_ncubos[bits_de_mascara]
presentes = intersect(futuros, dims_ncubos)
dist_Pi = bipartir(futuros, presentes).distribucion_marginal()
```

Esto implementa la condición de independencia de IIT: si la partición es la MIP,
las partes interactúan mínimamente, y la distribución bipartida se aleja poco
de la distribución real.

---

### 2.7 Aditividad del costo (aplica solo para N > 12)

Para N > 12, el costo de la k-partición se descompone como suma de costos por parte:

```
Φ_L1(P₁ | P₂ | … | Pₖ) = Φ_L1(P₁) + Φ_L1(P₂) + … + Φ_L1(Pₖ)
```

Esto permite:
1. Calcular deltas de fusión sin evaluar la partición completa.
2. Calcular deltas de movimiento 1-move con solo dos grupos.
3. Reutilizar el caché de costos en ambas fases.

**Para N ≤ 12, la aditividad ya no aplica:** la EMD Wasserstein-1 con Hamming
se evalúa sobre la distribución conjunta 2^N de la partición completa (no es
suma de EMDs por parte). Por eso el greedy y el refinamiento local deben evaluar
`_emd_particion` sobre la partición completa en cada paso.

---

### 2.8 Distribución conjunta vectorizada (`distribucion_conjunta_vectorizada`)

Dado un vector de N probabilidades P(nodo=ON), construye la distribución conjunta
2^N asumiendo independencia entre nodos (igual que GeoMIP para la reconstrucción):

```
p_conjunta(s) = ∏ᵢ [P(nodo_i = ON)]^{s_i} × [1 − P(nodo_i = ON)]^{1−sᵢ}
```

donde `s` es un estado de N bits y `sᵢ` es el i-ésimo bit de `s`.

Implementación eficiente usando broadcasting con `np.meshgrid`:

```python
def distribucion_conjunta_vectorizada(probabilidades: np.ndarray) -> np.ndarray:
    p_on  = np.asarray(probabilidades, dtype=np.float64)
    p_off = 1.0 - p_on
    factors = np.stack([p_off, p_on], axis=1)  # factors[i] = [P(OFF), P(ON)]
    grid = np.meshgrid(*factors, indexing="ij")
    return np.prod(grid, axis=0).flatten()      # shape (2^N,)
```

El resultado es un vector de 2^N probabilidades ordenado en little-endian (bit 0
varía más rápido). Usado por `_emd_particion` para construir P y Q antes de
llamar a `emd_causal`.

**Archivo:** `src/funcs/iit.py`

---

### 2.8 Extracción de bits activos (iteración sobre nodos de un grupo)

Para iterar sobre los nodos de un grupo representado como máscara:

```python
def _bits_activos(mascara: int):
    m = mascara
    while m:
        bit = m & (-m)          # aísla el bit menos significativo activo
        yield bit.bit_length() - 1   # índice del nodo (0-based)
        m ^= bit                # apaga ese bit
```

El truco `m & (-m)` funciona por la representación en complemento a dos:
`-m` invierte todos los bits y suma 1, lo que aísla exactamente el primer
bit activo de `m` en O(1).
