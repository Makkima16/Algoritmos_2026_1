# Cambios Realizados en GeoMIP vs. el Código Original

## Contexto

El código original (`projecto-analisis-20261`) era una implementación imperativa genérica que resolvía
únicamente **biparticiones** (k = 2). La versión actual (`AYDA_2026_1 / GeoMIP`) reimplementa el
problema con rigor matemático, soporte para **k-particiones generalizadas** y una arquitectura
completamente diferente.

---

## 1. Métrica de Distancia — EMD Real vs. Suma L1 Simple

### Original
```python
# emd_efecto: diferencia absoluta nodo por nodo, independiente del espacio
def emd_efecto(u, v):
    return np.sum(np.abs(u - v))
```

La función `emd_efecto` sumaba directamente las diferencias de probabilidad marginal nodo a nodo.
Esta operación es correcta **sólo cuando se asume que los nodos son completamente independientes**,
lo que rara vez se cumple en un sistema real. El resultado era un Phi (Φ) artificialmente bajo.

**Problema:** Al ignorar la topología del espacio de estados (el hipercubo de Hamming), el valor de Φ
no tenía interpretación causal rigurosa y dependía del orden de los nodos, no de su estructura.

### Actual
```python
# emd_causal: EMD verdadera con distancia de Hamming como métrica base
def emd_causal(u, v):
    n = u.size
    cost_mat = get_hamming_matrix(n)   # matriz (2^N × 2^N) de distancias Hamming
    return emd(u, v, cost_mat)         # PyEMD resuelve el problema de transporte
```

La función `emd_causal` calcula la **Earth Mover's Distance** real. La matriz de costos usa la
**distancia de Hamming** entre estados binarios, que es la métrica correcta sobre el hipercubo
booleano. El resultado es un Φ matemáticamente realista (típicamente > 1.0 para sistemas con
integración causal real).

---

## 2. Generación de Candidatos — Hill-Climbing Ciego vs. Heurísticas Geométricas

### Original
El código original seleccionaba biparticiones de forma pseudo-aleatoria (hill-climbing estocástico)
sin información previa sobre qué cortes son más prometedores.

### Actual
Se generan candidatos a través de tres estrategias complementarias:

**a) Spectral Clustering con matriz de afinidad geométrica**
```
A[i, j] = similitud_coseno(columna_i, columna_j) de la tabla de costos EMD
         → normalizada a [0, 1]
→ SpectralClustering(affinity="precomputed", n_clusters=k)
```
La afinidad mide cuán probabilísticamente similares son dos nodos bajo todas las condiciones.
Nodos que "se comportan igual" en el espacio de probabilidades tienden a agruparse juntos.

**b) Agglomerative Clustering (bottom-up)**
Tres variantes de enlace: `average`, `complete`, `single`. Cada una produce candidatos con
distintas estructuras de agrupamiento, cubriendo diferentes geometrías del espacio de particiones.

**c) Aislamiento heurístico**
Para k=2: N candidatos (cada nodo aislado vs. el resto).
Para k=3: C(N, 2) candidatos (dos nodos individuales + residual). Etc.
En la práctica, uno de estos candidatos reproduce el corte MIP exacto en la mayoría de los sistemas.

---

## 3. Refinamiento Local — Ausente vs. 1-Move + ILS

### Original
No había fase de refinamiento. La primera partición encontrada era la solución final.

### Actual

**Refinamiento 1-move:**
```
Para cada nodo n en bloque Bi:
    Para cada bloque Bj (j ≠ i):
        Mover n de Bi a Bj → evaluar nuevo Φ
        Si Φ mejora → aceptar y repetir desde el principio
Terminar cuando ningún movimiento mejore Φ
```

**Iterated Local Search (ILS):**
Tras el refinamiento, se aplica una perturbación aleatoria (mover un nodo a un bloque distinto)
y se refina de nuevo. Se repite N_ILS = 4 veces, conservando siempre el mejor resultado global.
Este proceso escapa de mínimos locales superficiales.

---

## 4. Soporte para k-Particiones (k > 2)

### Original
Sólo soportaba biparticiones (k = 2). La generalización a más grupos no existía en la arquitectura.

### Actual
El método `System.particionar()` generaliza `bipartir()`:

```python
def particionar(self, particiones: list[tuple[alcance_i, mecanismo_i]]) -> System:
    for cube in self.ncubos:
        for alcance_i, mecanismo_i in particiones:
            if cube.indice in alcance_i:
                # Marginaliza sólo lo que NO pertenece al mecanismo de su grupo
                cube.marginalizar(setdiff1d(cube.dims, mecanismo_i))
```

El bucle externo evalúa k = 2, 3, 4, 5 secuencialmente. Para cada k se usan todos los núcleos
disponibles. Se reporta la k con el Φ mínimo global como la k-MIP.

---

## 5. Manejo del Mecanismo Vacío (∅)

### Original
Cuando una bipartición dejaba una parte sin variables en el presente, el sistema "destrozaba"
las distribuciones y acumulaba penalizaciones artificiales (over-cutting), produciendo cortes
inválidos con Φ inflado.

### Actual
La función `_generar_candidatos_presente_vacio()` genera explícitamente variantes donde los
nodos aislados usan **mecanismo vacío** (∅). Se marca con el centinela `-1` al inicio de la lista
de la parte:

```python
partes = [[-1, a] for a in aislados] + [residual]
#          ↑
#      centinela ∅
```

`evaluar_k_particion` interpreta este centinela y usa `presentes_parte = np.array([], dtype=np.int8)`,
lo que produce una distribución uniforme para ese subconjunto, sin penalización artificial.

---

## 6. Arquitectura — Imperativa vs. OOP con Clases de Dominio

### Original
Código imperativo genérico con funciones sueltas. El estado del sistema se pasaba como argumentos
entre funciones sin encapsulamiento ni memoización.

### Actual
Arquitectura orientada a objetos con separación clara de responsabilidades:

| Clase / Módulo       | Responsabilidad                                              |
|----------------------|--------------------------------------------------------------|
| `System`             | Condicionamiento, substracción, particionamiento, marginales |
| `NCube`              | Hipercubo de probabilidad por nodo, marginalización cacheada |
| `KGeoMIP`            | Algoritmo de búsqueda k-MIP completo                        |
| `Manager`            | Carga de TPM, enrutamiento de estrategias                    |
| `Solution`           | Representación y visualización del resultado                 |
| `SafeLogger`         | Logging thread-safe con archivos por fecha                   |
| `LazyTPM`            | Lectura lazy de TPM por chunks para N ≥ 18                  |

La clase `NCube` implementa **memoización de marginalizaciones**:
```python
cache_key = frozenset(marginable_axis.tolist())
if cache_key in self._marginal_cache:
    return self._marginal_cache[cache_key]
```
Esto evita recomputar la misma marginalización decenas de veces durante el clustering.

---

## 7. Modo de Entrada — Excel Iterado vs. Terminal Interactivo + Modo Bloque CSV

### Original
El código iteraba sobre un archivo Excel fijo como fuente de parámetros. No había interacción
con el usuario en tiempo de ejecución.

### Actual
Dos modos independientes:

**Modo Manual:** El usuario ingresa por terminal (o diálogo de archivo):
1. Selecciona la TPM via explorador de archivos (`tkinter.filedialog`)
2. Ingresa `sistema candidato` (máscara binaria)
3. Ingresa `estado inicial` (o genera uno aleatorio)
4. Ingresa `alcance` (t+1) y `mecanismo` (t)
5. Ingresa K (o deja en blanco para evaluar todas)

**Modo Bloque:** Carga un CSV con múltiples pruebas:
```
#Prueba,Alcance o Purview (t+1),Mecanismo(t)
1,ABCDE,ABCDE
2,ABCDE,ABCD
```
- Candidato y estado inicial se ingresan una sola vez para todo el lote
- Las pruebas se ordenan automáticamente de menor a mayor complejidad
- Se calienta la caché del sistema candidato y los pools de joblib antes de la primera prueba
- Los resultados se exportan a `.xlsx` con formato profesional (color, wrap, freeze panes)

---

## 8. Gestión de Memoria para N Grande — LazyTPM

### Original
La TPM se cargaba íntegramente en memoria. Para N = 20 esto representa 2²⁰ × 20 = ~20 millones
de valores en float32 ≈ 80 MB mínimo, pero el procesamiento posterior multiplicaba ese requerimiento.

### Actual
Para N ≥ 18 se activa `LazyTPM`: la matriz no se carga nunca completa. En su lugar, un generador
lee fragmentos (`chunks`) secuencialmente y calcula la marginal de cada columna acumulando:

```python
def marginal_nodo(self, idx: int) -> np.ndarray:
    acum = np.zeros(2 ** self.n_nodos, dtype=np.float32)
    for chunk_inicio, chunk_data in self.chunks():
        acum[chunk_inicio : chunk_inicio + len(chunk_data)] = chunk_data[:, idx]
    return acum
```

Esto permite procesar sistemas de hasta N = 25+ sin colapso de memoria.

---

## Resumen de Cambios

| Criterio                      | Original                          | Actual (GeoMIP AYDA 2026-1)                   |
|-------------------------------|-----------------------------------|-----------------------------------------------|
| Métrica EMD                   | Suma L1 simple (`emd_efecto`)     | EMD real con Hamming (`emd_causal`)           |
| Valores de Φ                  | Artificialmente bajos             | Matemáticamente realistas                     |
| Particiones soportadas        | Solo k = 2                        | k = 2 hasta min(6, N)                         |
| Generación de candidatos      | Hill-climbing estocástico ciego   | Spectral + Agglomerative + Aislamiento        |
| Refinamiento                  | Ninguno                           | 1-move + ILS                                  |
| Mecanismo vacío (∅)           | Over-cutting sin control          | Soporte riguroso con centinela -1             |
| Arquitectura                  | Funciones imperativas sueltas     | OOP: System, NCube, KGeoMIP, Manager         |
| Memoización                   | Ninguna                           | NCube._marginal_cache por frozenset de ejes   |
| Paralelismo                   | Ninguno / básico                  | joblib con cpu_count-1 núcleos               |
| Entrada de datos              | Excel fijo                        | Terminal interactivo + CSV en modo bloque     |
| Gestión de memoria            | Carga total en RAM                | LazyTPM por chunks para N ≥ 18               |
| Exportación de resultados     | Excel manual                      | JSON (manual) y .xlsx formateado (bloque)    |
