# Herramientas, Librerías y Ecuaciones Utilizadas en GeoMIP

## 1. Librerías Principales

### NumPy
**¿Para qué?** Toda la manipulación numérica del proyecto.
- Arrays N-dimensionales para representar los n-cubos de probabilidad
- Indexación avanzada para condicionamiento y marginalización
- `np.kron()`: producto de Kronecker para reconstrucción de distribuciones conjuntas
- `np.setdiff1d`, `np.intersect1d`: operaciones de conjuntos sobre índices de nodos
- `np.empty`, `np.float32`: gestión eficiente de memoria

**¿Por qué NumPy y no otra cosa?** Es el estándar de facto para álgebra lineal en Python, sin
overhead de frameworks más pesados. Las operaciones sobre vectores de tamaño 2^N (donde N ≤ 20)
son exactamente el caso de uso para el que NumPy está optimizado. No se necesita autodiferenciación
(PyTorch/TF) ni procesamiento de grafos (NetworkX en este contexto).

---

### PyEMD — relegado a verificación (ya NO en el camino caliente)
**¿Para qué?** Históricamente, calcular la EMD exacta vía solver de transporte para N ≤ 12.

```python
from pyemd import emd
resultado = emd(u, v, cost_matrix)   # O(4^N): solver de transporte sobre 2^N estados
```

**Estado actual:** se descubrió que la suma L1 marginal es la EMD de Hamming **exacta** para
distribuciones producto (ver §2.3), que es O(N) en vez de O(4^N), da el **mismo** valor y vale
para **todo N** sin límite. Por eso PyEMD ya **no** se usa para calcular Φ; `emd_causal` /
`get_hamming_matrix` permanecen sólo como utilidad de verificación de esa equivalencia. La ruta
principal (`evaluar_bloques`) usa la fórmula marginal directa.

---

### scikit-learn
**¿Para qué?** Generación de candidatos de partición mediante clustering.

```python
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
```

- `SpectralClustering(affinity="precomputed", n_clusters=k)`: toma la matriz de afinidad
  geométrica precalculada y agrupa los nodos
- `AgglomerativeClustering(n_clusters=k, linkage="average"|"complete"|"single")`: clustering
  jerárquico bottom-up con distintos criterios de enlace

**¿Por qué scikit-learn?** Es la librería de ML más completa de Python para clustering. Ofrece
los algoritmos que necesitamos con parámetros de afinidad personalizada y es una dependencia
opcional (si no está instalada, GeoMIP hace fallback a heurísticas de varianza). No se necesita
ningún otro framework de ML porque no se entrena ningún modelo: el clustering es puramente
heurístico para generar candidatos.

---

### joblib
**¿Para qué?** Paralelismo en la evaluación de candidatos de partición.

```python
from joblib import Parallel, delayed
resultados = Parallel(n_jobs=N_JOBS_INTERNOS)(
    delayed(evaluar_candidato)(c) for c in candidatos
)
```

`N_JOBS_INTERNOS = max(1, cpu_count() - 1)` — usa todos los núcleos menos uno (el principal).

**¿Por qué joblib?** Es la librería de paralelismo recomendada por scikit-learn y la más madura
para paralelismo CPU-bound en Python. Maneja automáticamente el backend (loky, multiprocessing,
threading) y la serialización de objetos entre procesos. La alternativa `concurrent.futures` es
más verbosa y tiene menos opciones de control de memoria. `multiprocessing` puro requeriría más
código boilerplate para el mismo resultado.

---

### openpyxl
**¿Para qué?** Exportación de resultados del modo bloque a archivos `.xlsx`.

```python
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
```

Genera hojas de cálculo con encabezados coloreados, wrap de texto en la columna de partición,
filas alternas sombreadas y `freeze_panes` en la primera fila.

**¿Por qué openpyxl?** Es el escritor de Excel nativo para Python sin dependencias de Microsoft
Office. `pandas.to_excel()` lo usa internamente de todas formas; usar openpyxl directamente da
control total sobre el formato de celdas (algo que `pandas` no expone completamente).

---

### tkinter
**¿Para qué?** Diálogos de selección de archivos en la interfaz terminal.

```python
from tkinter import filedialog
ruta = filedialog.askopenfilename(initialdir=SAMPLES_DIR, ...)
```

**¿Por qué tkinter?** Es la librería de GUI incluida en la distribución estándar de Python, sin
instalación adicional. Para un diálogo de selección de archivo es más que suficiente. Si tkinter
no está disponible (entornos sin display), el código hace fallback a ingreso manual por terminal.

---

### pyttsx3 (opcional)
**¿Para qué?** Anuncio por voz del resultado al terminar la búsqueda (`Solution`).

**¿Por qué?** Funcionalidad de accesibilidad y retroalimentación para ejecuciones largas sin
monitorear la pantalla. Es opcional: si no está instalado, el resultado sólo se imprime.

---

## 2. Ecuaciones Centrales

### 2.1 Earth Mover's Distance (EMD) — Problema de Transporte Óptimo

La EMD entre dos distribuciones de probabilidad P y Q sobre un espacio métrico (S, d) es:

```
EMD(P, Q) = min_{f ∈ Π(P,Q)} Σᵢ Σⱼ f(i,j) · d(i,j)
```

Donde:
- `f(i,j)` es el "flujo" de masa desde el estado i al estado j
- `d(i,j)` es el costo de mover masa de i a j (distancia Hamming en GeoMIP)
- `Π(P,Q)` es el conjunto de planes de transporte válidos:
  - `Σⱼ f(i,j) = P(i)` para todo i (se envía exactamente P(i) desde i)
  - `Σᵢ f(i,j) = Q(j)` para todo j (se recibe exactamente Q(j) en j)
  - `f(i,j) ≥ 0` (no se puede mover masa negativa)

Intuitivamente, la EMD es el mínimo "esfuerzo" necesario para transformar la distribución P en
la distribución Q, donde mover una unidad de masa una distancia d cuesta d.

**Esta forma general (solver de transporte) ya no se ejecuta para calcular Φ.** Como las
distribuciones comparadas son productos de marginales, la EMD se reduce a la suma L1 marginal
(§2.3), que es el caso particular exacto y O(N). El solver `pyemd` queda sólo como verificación.

---

### 2.2 Distancia de Hamming

La distancia de Hamming entre dos estados binarios i y j es el número de bits en que difieren:

```
H(i, j) = popcount(i XOR j)
```

Para estados binarios representados como enteros:
- `i XOR j` da un entero con un `1` en cada bit donde i y j difieren
- `popcount` (o `.bit_count()` en Python 3.10+) cuenta esos unos

```python
def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()
```

Esta es la métrica natural sobre el hipercubo booleano {0,1}^N. Refleja la idea de que dos
estados que difieren en más bits son "más distintos" causalmente.

La matriz de Hamming completa se pre-calcula una sola vez y se cachea:
```python
costs[i, j] = hamming_distance(i, j)   # para todo par (i,j) en [0, 2^N)
```

---

### 2.3 ⭐ Descomposición marginal de la EMD — la métrica EXACTA (para TODO N)

**Teorema fundamental:** Si P y Q son distribuciones conjuntas de N variables binarias
condicionalmente independientes (es decir, **productos de marginales**), la EMD de
Wasserstein-1 sobre el hipercubo booleano con distancia base de Hamming se descompone
**exactamente** en:

```
EMD_Hamming(P, Q) = Σᵢ₌₁ᴺ | P(nodo_i = 1) - Q(nodo_i = 1) |
```

**¿Por qué es válido y por qué es EXACTO (no una aproximación)?** En cada k-partición, tanto la
distribución original como la reconstruida son productos de marginales **por construcción**. Para
dos productos, el problema de transporte óptimo sobre el hipercubo se **desacopla** coordenada a
coordenada: la masa se mueve óptimamente por cada dimensión de forma independiente, y el costo
total es la suma de los costos por dimensión (la diferencia absoluta de probabilidades en
variables binarias). Verificado numéricamente: `|emd_causal − L1| < 1e-14` para N = 2..12.

Esto **no** es un atajo sólo para N grande: es la métrica única y exacta para **todo N**. Por eso
no hay bifurcación por tamaño y se eliminó el solver `pyemd` del camino caliente.

**Reducción de complejidad (sin perder precisión):**
- EMD vía solver de transporte: O(4^N) en tiempo, O(4^N) en memoria — inviable para N grande.
- Suma marginal exacta: **O(N)** en tiempo y memoria — válida para todo N.

```python
# evaluar_bloques — ruta principal
dist_rec = subsistema.particionar(particiones).distribucion_marginal()
return float(np.sum(np.abs(dist_original - dist_rec)))
```

---

### 2.4 Producto de Kronecker para Reconstruir la Distribución Conjunta

Dada una k-partición P = {S₁, S₂, ..., Sₖ}, la distribución conjunta del sistema particionado
(asumiendo independencia entre partes) es el producto de Kronecker de las distribuciones
marginales de cada parte:

```
Q_conjunta = Q_{S₁} ⊗ Q_{S₂} ⊗ ... ⊗ Q_{Sₖ}
```

La operación de Kronecker entre dos vectores u ∈ ℝᵐ y v ∈ ℝⁿ produce un vector w ∈ ℝᵐⁿ:

```
w[i·n + j] = u[i] · v[j]   para todo i ∈ [0,m), j ∈ [0,n)
```

Esto es exactamente la distribución de probabilidad conjunta de dos variables independientes:
P(X=i, Y=j) = P(X=i) · P(Y=j).

**Importante:** En GeoMIP, el producto de Kronecker se evita para el cálculo de Φ (gracias a
la descomposición del apartado 2.3). Se usa sólo para reconstrucción explícita en sistemas
pequeños o para exportación de resultados.

---

### 2.5 Phi (Φ) — Pérdida de Información Integrada

Para una k-partición P = {S₁, ..., Sₖ} aplicada al subsistema (con alcance y mecanismo dados):

```
Φ(P) = EMD( distribución_marginal(subsistema_original) ,
            distribución_marginal(subsistema_particionado) )
```

Las distribuciones marginales son vectores de tamaño N donde la entrada i-ésima es:

```
dist[i] = P(nodo_i = 1 | estado_inicial)
```

`distribucion_marginal` almacena directamente P(nodo = ON) (la misma convención que QNodes),
seleccionando el subestado correspondiente al estado inicial. Esta convención unificada hace que
los valores reportados por GeoMIP y QNodes sean directamente comparables.

La **k-MIP** es la partición que minimiza Φ:

```
k-MIP = argmin_{P ∈ k-particiones} Φ(P)
```

---

### 2.6 Matriz de Afinidad Geométrica (para Spectral Clustering)

Para construir la matriz de afinidad entre nodos se usa la similitud coseno entre columnas
de la "tabla de costos EMD" del subsistema:

```
A[i, j] = (1 + cos_sim(col_i, col_j)) / 2

cos_sim(u, v) = (u · v) / (||u|| · ||v||)
```

La normalización a [0, 1] convierte la similitud coseno (rango [-1, 1]) en una afinidad válida
para SpectralClustering con `affinity="precomputed"`.

---

### 2.7 Condicionamiento (Background Conditioning)

Fijar las variables de "fondo" (nodos no activos en el sistema candidato) a sus valores en el
estado inicial. Matemáticamente, corresponde a seleccionar una "cara" del hipercubo N-dimensional:

```
ncubo_condicionado = ncubo[ :, :, ..., estado_inicial[dim_a_fijar], :, ..., : ]
```

Para cada dimensión `d` a condicionar, se selecciona el "slice" correspondiente al bit del
estado inicial en esa posición. El resultado es un hipercubo de dimensión reducida.

---

### 2.8 Marginalización

Integrar (promediar) sobre una o más dimensiones de un n-cubo:

```
ncubo_marginalizado = mean(ncubo, axis=ejes_a_marginalizar)
```

El promedio sobre un eje equivale a eliminar la dependencia del n-cubo en esa variable del
presente, es decir, "ignorar" el estado de ese nodo en t como información causal. El resultado
es un n-cubo de dimensión reducida.

En código:
```python
nuevo_data = np.mean(self.data, axis=tuple(ejes_a_eliminar))
```

---

## 3. Indexación Little-Endian

GeoMIP usa convenio **Little-Endian** para la indexación de estados binarios: el nodo A (índice 0)
corresponde al **bit menos significativo**. El estado `001` representa A=1, B=0, C=0, que como
entero es 1 (no 4 como sería en Big-Endian).

La función `lil_endian(N)` precomputa la permutación de índices necesaria para reordenar las
filas de la TPM de la convención estándar Big-Endian a Little-Endian.

Esta convención afecta la construcción de los n-cubos (`reshape`) y la selección del estado
inicial en `distribucion_marginal`. Es **consistente en todo el código** y no afecta los
resultados finales de Φ, sólo la correspondencia entre índices enteros y estados binarios.

---

## 4. Resumen de Decisiones de Diseño

| Componente              | Decisión                             | Alternativa Descartada               | Razón                                                |
|-------------------------|--------------------------------------|--------------------------------------|------------------------------------------------------|
| EMD (métrica única)     | **Suma L1 marginal = EMD Hamming exacta, todo N** | PyEMD / scipy.wasserstein | L1 es O(N), exacta para productos, sin límite de tamaño |
| EMD para N grande       | Misma fórmula marginal O(N)          | Construir conjunta 2^N + solver O(4^N) | Imposible en memoria para N = 20+; e innecesario (es exacta) |
| Producto tensorial      | Evitado (sólo verificación/export)   | PyTorch / TensorFlow                 | Φ no necesita la conjunta; basta la marginal O(N)    |
| Motor de búsqueda       | **Greedy top-down asimétrico** + 1-move (determinista; ILS retirada 2026-06-12) | Spectral (fallback) / Fuerza bruta | Pool O(N) compartido; cortes asimétricos coherentes; sin partes `(∅,∅)` |
| Paralelismo             | joblib (cpu_count - 1)               | multiprocessing puro / asyncio       | joblib maneja serialización y backends automáticamente |
| Memoria para N ≥ 18     | LazyTPM por chunks                   | Cargar CSV completo en RAM           | N=20 → 2^20×20 valores ≈ 80MB mínimo, más copias     |
| Memoización             | NCube._marginal_cache (frozenset)    | Ninguna / recomputar siempre         | Mismas marginalizaciones se piden decenas de veces   |
| Arquitectura            | OOP (System, NCube, KGeoMIP)        | Funciones sueltas imperativas        | Encapsulamiento, memoización, herencia, testabilidad |
