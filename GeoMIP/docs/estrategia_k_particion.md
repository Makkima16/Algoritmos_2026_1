# Estrategia para Encontrar la k-Partición con Mínima Pérdida de Información

## Contexto: ¿Qué es una k-Partición y por qué Minimizar Φ?

En la **Teoría de Información Integrada (IIT)**, un sistema de N nodos puede partirse en k
subconjuntos independientes. Al "cortar" las conexiones causales entre grupos, el sistema
particionado genera una distribución de probabilidad diferente a la del sistema integrado.

La **pérdida de información** (Φ, "phi") mide cuánta información se destruye con ese corte,
usando la **Earth Mover's Distance (EMD)** entre la distribución original y la reconstruida
desde las partes independientes:

```
Φ(partición) = EMD( P_original(t+1) , P_particionada(t+1) )
```

La **k-MIP** (k-Partition of Minimum Information) es la partición que produce el **menor Φ**:
si el sistema tiene poca información integrada, se puede cortar casi sin pérdida.

---

## Pipeline Completo del Algoritmo KGeoMIP

```
ENTRADA: subsistema (condicionado + sustraído), valores de k a evaluar
│
│  ┌──────────────────────────────────────────────────────────────┐
│  │  Para cada k en [2, 3, 4, 5, min(6, N)]                     │
│  │                                                              │
│  │  FASE 1: Generación de candidatos                            │
│  │  ├── Spectral Clustering (múltiples semillas y enlazados)    │
│  │  ├── Agglomerative Clustering (avg / complete / single)      │
│  │  ├── Aislamiento heurístico (C(N, k-1) candidatos)          │
│  │  └── Variantes con mecanismo vacío (∅)                       │
│  │                                                              │
│  │  FASE 2: Evaluación paralela (joblib, cpu_count-1 núcleos)   │
│  │  └── Para cada candidato → calcular Φ con EMD               │
│  │                                                              │
│  │  FASE 3: Refinamiento local (1-move)                         │
│  │  └── Mover un nodo a otro bloque si mejora Φ, repetir       │
│  │                                                              │
│  │  FASE 4: ILS (Iterated Local Search, 4 iteraciones)          │
│  │  ├── Perturbar (mover nodo aleatorio)                        │
│  │  ├── Re-refinar                                              │
│  │  └── Conservar mejor Φ                                       │
│  │                                                              │
│  └─── Resultado: Φ_min para este k                              │
│
└── Retornar la k y la partición con Φ mínimo global
```

---

## Fase 1: Generación de Candidatos

### ¿Por qué no buscar exhaustivamente?

El número de k-particiones de N elementos está dado por los **Números de Stirling de segunda
especie** S(N, k). Para N = 10, k = 5: S(10, 5) = 42.525 particiones. Para N = 20, k = 5:
S(20, 5) ≈ 1.77 × 10¹³. La búsqueda exhaustiva es computacionalmente imposible.

La solución es generar un conjunto pequeño pero inteligente de candidatos mediante heurísticas
que explotan la **geometría del espacio de probabilidades**.

### a) Spectral Clustering — Afinidad Geométrica

Se construye una **matriz de afinidad** A donde cada entrada mide la similitud entre dos nodos:

```
A[i, j] = (1 + similitud_coseno(col_i, col_j)) / 2
```

`col_i` es la columna i de la "tabla de costos EMD" del subsistema: un vector que describe cómo
contribuye ese nodo a la distancia con cada posible estado. Nodos con perfiles de transición
similares tendrán alta afinidad y tenderán a agruparse en la misma parte.

Con esta matriz se ejecuta `SpectralClustering(affinity="precomputed", n_clusters=k)` con
varias semillas aleatorias (para evitar mínimos locales del propio clustering).

### b) Agglomerative Clustering — Jerárquico Bottom-Up

Complementa al Spectral con tres estrategias de enlace:
- **Average linkage**: la afinidad entre grupos es el promedio de todas las afinidades entre pares
- **Complete linkage**: la afinidad entre grupos es el mínimo entre los pares (máxima separación)
- **Single linkage**: la afinidad es el máximo entre los pares (enlaza cadenas de nodos)

Cada estrategia produce agrupaciones de forma diferente y el conjunto cubre más del espacio.

### c) Aislamiento Heurístico

Para k grupos se generan todas las formas de aislar exactamente k-1 nodos individualmente,
dejando el resto en un único grupo residual:

```
Para k = 2: [nodo_0 | resto], [nodo_1 | resto], ..., [nodo_N-1 | resto]  → N candidatos
Para k = 3: [nodo_i, nodo_j | resto] para todo par (i,j)               → C(N,2) candidatos
```

En la práctica, el corte MIP exacto para k = 2 casi siempre se encuentra entre estos N candidatos,
porque la partición óptima suele aislar el nodo con menor integración causal.

### d) Variantes con Mecanismo Vacío (∅)

En IIT, un corte puede dejar a un nodo sin ninguna dependencia causal en el pasado. Esto se
representa como `mecanismo = ∅`. GeoMIP genera candidatos donde los nodos aislados tienen
mecanismo vacío (marcados con el centinela `-1`):

```python
partes = [[-1, nodo_i], [nodo_j], [resto...]]
#          ↑ centinela ∅: sin dependencias en t
```

Esto produce una distribución uniforme para ese nodo, sin penalización artificial por over-cutting.

---

## Fase 2: Evaluación de Φ para cada Candidato

Para cada candidato se calcula Φ comparando la distribución marginal original con la reconstruida
desde las partes independientes.

### Para N ≤ 12: EMD exacta con Hamming

Se construye la distribución conjunta completa (2^N estados) y se calcula la EMD exacta:

```python
cost_mat = get_hamming_matrix(2**N)   # matriz 2^N × 2^N de distancias Hamming
Φ = pyemd.emd(P_original, P_particionada, cost_mat)
```

### Para N > 12: Suma de EMDs Marginales (O(N))

Construir la matriz de Hamming para 2^N estados (N > 12 → más de 4096 estados) requiere
cientos de gigabytes de RAM. Se usa la propiedad matemática fundamental del hipercubo booleano:

**Teorema:** Si dos distribuciones multivariadas están compuestas de **variables condicionalmente
independientes**, la EMD sobre el hipercubo de Hamming se descompone exactamente en la suma de
las EMDs marginales unidimensionales:

```
EMD(P, Q) = Σᵢ | P(nodo_i = 1) - Q(nodo_i = 1) |
```

Esto convierte un problema de O(2^N³) (solver de transporte sobre 2^N estados) en O(N)
(suma de N diferencias absolutas).

---

## El Rol del Producto de Kronecker — y por qué NO se usa la librería `tensor`

### ¿Qué hace el Producto de Kronecker aquí?

Cuando partimos el sistema en k grupos independientes {S₁, S₂, ..., Sₖ}, cada grupo tiene
su propia distribución marginal Qᵢ. Para reconstruir la distribución **conjunta** del sistema
particionado se usa el producto de Kronecker:

```
Q_conjunta = Q₁ ⊗ Q₂ ⊗ ... ⊗ Qₖ  =  np.kron(np.kron(Q₁, Q₂), ..., Qₖ)
```

Esto equivale a asumir **independencia total** entre los grupos: la probabilidad conjunta de
cualquier estado es el producto de las probabilidades marginales de cada parte.

```python
def producto_tensorial(self, distribuciones_marginales):
    Q = distribuciones_marginales[0]
    for dist in distribuciones_marginales[1:]:
        Q = np.kron(Q, dist)
    return Q
```

### ¿Por qué los valores de la TPM son 0 ó 1, y qué implica eso?

Las TPMs en GeoMIP son **binarias/deterministas**: cada entrada P(nodo_i = 1 | estado_j) ∈ {0, 1}.
Esto refleja sistemas donde el estado de cada nodo en t+1 está **completamente determinado** por
el estado del sistema en t (sin ruido ni estocasticidad).

Este requisito **no es arbitrario**: surge de que el espacio de estados es el **hipercubo booleano**
{0, 1}^N. La distancia entre dos estados se mide como **distancia de Hamming** (número de bits
que difieren). Para que la EMD con Hamming tenga sentido físico como métrica de pérdida causal,
las distribuciones deben ser **distribuciones de probabilidad sobre vértices del hipercubo**,
es decir, vectores de probabilidad indexados por estados binarios.

Con valores en {0, 1}, cada nodo en un estado dado está **encendido o apagado con certeza**,
y la distribución resultante (al marginalizar sobre todos los estados iniciales posibles) es
una distribución de Bernoulli por nodo, que es exactamente lo que se compara con EMD.

### ¿Por qué no se usa una librería tipo `torch` o `tensorflow` para el producto tensorial?

Hay dos razones fundamentales:

**Razón 1: No se construye la distribución conjunta para el cálculo real de Φ.**

El teorema de descomposición del EMD en hipercubo evita construir Q_conjunta. En la práctica,
`producto_tensorial` se usa sólo para verificación o para casos pequeños (N ≤ 12). Para el
cálculo de Φ en la inmensa mayoría de los sistemas, sólo se necesitan los vectores marginales
de tamaño N, no la conjunta de tamaño 2^N:

```python
# Lo que realmente se compara (O(N)):
Φ = Σᵢ | P(nodo_i=1) - Q(nodo_i=1) |

# No necesitamos construir Q_conjunta (2^N elementos) para esto.
```

**Razón 2: `np.kron()` de NumPy es suficiente y sin dependencias extra.**

Cuando sí se requiere construir la conjunta (para N pequeño o para exportación), `np.kron()` es
la operación nativa correcta en NumPy. Librerías como PyTorch o TensorFlow operan sobre tensores
de punto flotante de alta dimensión pensados para aprendizaje automático (gradientes, GPU, etc.).

El producto de Kronecker en GeoMIP opera sobre **distribuciones de probabilidad de tamaño 2^Nₖ**
donde Nₖ es el número de nodos en la parte k-ésima (típicamente 1 a 5). Para estos tamaños,
`np.kron()` es instantáneo y sin overhead. Agregar PyTorch añadiría cientos de MB de dependencias
para hacer exactamente la misma operación en vectores de 2 a 32 elementos.

Además, los tensores de PyTorch/TF representan conceptualmente **espacios de características
continuos**, no **distribuciones discretas sobre hipercubos booleanos**. Usar esa abstracción
introduciría confusión conceptual sin ningún beneficio.

---

## Fases 3 y 4: Refinamiento Local e ILS

### Refinamiento 1-Move

Después de evaluar todos los candidatos y quedarse con el mejor, se intenta mejorar moviendo
un nodo a la vez entre bloques:

```
Mejor partición actual: P* con Φ*
Para cada nodo n en bloque Bᵢ:
    Para cada bloque Bⱼ (j ≠ i):
        P' = P* con n movido de Bᵢ a Bⱼ
        Φ' = EMD(P_original, P'_particionada)
        Si Φ' < Φ*:
            P* = P', Φ* = Φ'
            Reiniciar búsqueda desde el principio
Parar cuando ningún movimiento mejore Φ*
```

El costo de cada evaluación es O(N) (suma de marginales), por lo que este proceso es muy rápido
incluso para N = 20.

### Iterated Local Search (ILS)

El refinamiento 1-move puede quedar atrapado en un mínimo local. Para escapar, se aplica una
**perturbación aleatoria** (mover un nodo elegido al azar a un bloque aleatorio) y se vuelve
a refinar. Se hacen N_ILS = 4 iteraciones, conservando siempre la mejor partición encontrada.

---

## Selección Global entre Valores de k

El bucle externo evalúa k = 2, 3, 4, ..., min(6, N) secuencialmente. Para cada k se obtiene
la mejor partición (con su Φ_min). Al finalizar todos los k, se selecciona el que tenga el
**Φ mínimo global**:

```
k_MIP = argmin_k { Φ_min(k) }
```

La evaluación de k es **secuencial** (no paralela entre k's) porque cada k ya usa todos los
núcleos disponibles internamente. Paralelizar entre k's dividiría los recursos y ralentizaría
cada uno sin ganar nada neto.

---

## Ejemplo Conceptual (N = 3, k = 2)

Sistema ABC con estado inicial `101`. Se busca bipartición (k = 2).

1. Candidatos generados: `{A|BC}`, `{B|AC}`, `{C|AB}`, `{A∅|BC}`, `{B∅|AC}`, `{C∅|AB}`
2. Para cada candidato se calcula Φ = EMD(P_original, P_reconstruida)
3. Supongamos que `{A|BC}` da Φ = 0.125, `{C|AB}` da Φ = 0.031
4. El mejor candidato es `{C|AB}` con Φ = 0.031
5. Refinamiento 1-move: intentar mover A, B o C entre grupos → ninguno mejora
6. ILS: perturbar y re-refinar → tampoco mejora
7. Resultado: k-MIP = `{C|AB}` con Φ = 0.031
