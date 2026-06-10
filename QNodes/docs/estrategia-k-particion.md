# Estrategia implementada para hallar la k-partición

**Clase:** `QNodes` — `src/strategies/q_nodes.py`

---

## 1. Definición del problema

Dado un sistema de N nodos con una distribución de probabilidad conjunta, se busca la
**k-partición de mínima información** (MIP): la división del sistema en k subconjuntos
disjuntos P₁, P₂, …, Pₖ tal que la pérdida Phi (Φ) sea mínima.

Cuando k no está fijado, se busca sobre todos los k ∈ [2, N] y se reporta el k con
menor Phi global (priorizando k ≥ 3 como objetivo principal del proyecto).

---

## 2. Por qué no se usa búsqueda exhaustiva

La búsqueda exhaustiva sobre todas las k-particiones requiere explorar B(N) casos
(número de Bell), que crece super-exponencialmente:

| N | B(N) | Tiempo estimado |
|---|---|---|
| 10 | ~115 000 | segundos |
| 15 | ~1 400 millones | horas |
| 20 | ~5 × 10¹³ | intractable |
| 25 | ~4 × 10¹⁸ | intractable |

Para N = 15, 20, 22, 25 — los tamaños objetivo del proyecto — la búsqueda
exhaustiva es inviable incluso con Branch & Bound agresivo.

---

## 3. Estructura general del algoritmo

QNodes aplica **las mismas tres fases para todo N**, sin excepciones:

```
aplicar_estrategia()
  ├── 1. Preparar subsistema (condicionar, marginalizar)
  ├── 2. _aglomerar() — agrupamiento greedy O(N³)
  │       ├── Inicializar N singletons
  │       ├── Pre-poblar _cache_dist para todos los singletons
  │       ├── Calcular phi_total inicial con _emd_particion
  │       └── Repetir N-2 veces:
  │               Evaluar _emd_particion para cada par candidato de fusión
  │               Fusionar el par con menor Phi resultante
  │               Registrar historico[k] = (phi, grupos)
  │           Retornar historial completo {k: (phi, grupos)} para k ∈ [2, N]
  │
  ├── [k especificado]
  │       ├── 3a. _refinamiento_local(historico[k]) — 1-move hasta convergencia
  │       └── 4a. _candidatos_aislamiento(k) — C(N, k-1) candidatos
  │               Evaluar con _emd_particion; si mejoran → adoptar + refinar
  │
  └── [k libre — búsqueda independiente sobre todos los niveles]
          Para cada k_nivel en historico (k_nivel ∈ [2, N]):
              _refinamiento_local(historico[k_nivel])
              _candidatos_aislamiento(k_nivel) → C(N, k_nivel-1) evaluaciones
              Si mejoran: adoptar + _refinamiento_local nuevamente
          Elegir k_nivel con menor Phi (≥ 3 preferido) entre todos los niveles
```

La memoización de `_cache_dist` hace que todas las evaluaciones de `_emd_particion`
sobre máscaras ya vistas sean O(1), amortizando el costo de las tres fases.

---

## 4. Representación del estado

Cada subconjunto de nodos se representa como una **máscara entera** de N bits.

```
Nodo:       E   D   C   B   A
Índice:     4   3   2   1   0
Bit:       16   8   4   2   1

{A, C, E} = 0b10101 = 21
{B, D}    = 0b01010 = 10
```

Operaciones sobre máscaras:

| Operación | Expresión |
|---|---|
| Unión Gᵢ ∪ Gⱼ | `Gᵢ \| Gⱼ` |
| Eliminar nodo n de G | `G ^ (1 << n)` |
| Añadir nodo n a G | `G \| (1 << n)` |
| Bit mínimo (primer nodo) | `m & (-m)` |

---

## 5. Fase 1: agrupamiento aglomerativo greedy

### 5.1 Inicialización

Se crean N grupos (singletons), uno por nodo. Se pre-populan los cachés de
distribuciones y se calcula el Phi inicial de la partición trivial de N partes:

```python
grupos = [1 << i for i in range(N)]

for g in grupos:
    self._dist_parte(g)          # pre-pobla _cache_dist

phi_total = self._emd_particion(grupos)   # Phi de la N-partición inicial
historico[N] = (phi_total, list(grupos))
```

### 5.2 Criterio de fusión

En cada paso se evalúan todos los C(k, 2) pares de grupos actuales.
Para cada par (Gᵢ, Gⱼ) se construye la partición candidata completa y se
evalúa con `_emd_particion`:

```python
candidato = grupos_sin_i_j + [Gᵢ | Gⱼ]
phi_cand  = self._emd_particion(candidato)
```

Se elige la fusión con **menor phi_cand** total.

### 5.3 Registro del historial

Después de cada fusión se guarda el estado en `historico[k_actual]`:

```python
historico[len(grupos)] = (phi_total, list(grupos))
```

Esto da acceso a cualquier nivel k desde k=N hasta k=2 sin rehacer el cálculo.
`_aglomerar()` retorna el historial completo — la selección de k y las fases
de refinamiento ocurren en `aplicar_estrategia`.

### 5.4 Selección de la k óptima

**Si k fue especificado:** `aplicar_estrategia` toma `historico[k]` y aplica
refinamiento + candidatos de aislamiento para ese k exacto.

**Si k es libre (k=None):** se aplican refinamiento + candidatos de aislamiento
para **cada nivel k** del historial, luego se elige el k ∈ {3, …, N} con menor
Phi entre todos los niveles refinados:

```python
historico_refinado = {}
for k_nivel, (phi_nivel, grupos_nivel) in historico.items():
    phi_r, grupos_r = self._refinamiento_local(grupos_nivel, phi_nivel)
    if k_nivel < self._N:
        for candidato in self._candidatos_aislamiento(k_nivel):
            phi_cand = self._emd_particion(candidato)
            if phi_cand < phi_r - 1e-10:
                phi_r, grupos_r = phi_cand, candidato
        phi_r, grupos_r = self._refinamiento_local(grupos_r, phi_r)
    historico_refinado[k_nivel] = (phi_r, grupos_r)

candidatos_k3 = {kk: ph for kk, (ph, _) in historico_refinado.items() if kk >= 3}
mejor_k = min(candidatos_k3, key=candidatos_k3.get)
```

---

## 6. Fase 2: refinamiento local 1-move

### 6.1 Motivación

El greedy aglomerativo puede producir fusiones subóptimas en pasos tempranos que
luego no se pueden deshacer. El refinamiento local corrige las más evidentes.

### 6.2 Movimiento 1-move

Para cada nodo n en cada grupo Gᵢ (con |Gᵢ| ≥ 2), se evalúa moverlo a cada
otro grupo Gⱼ usando `_emd_particion`:

```python
candidato = grupos con grupos[idx_origen]=Gᵢ\{n}, grupos[idx_dest]=Gⱼ∪{n}
phi_cand  = self._emd_particion(candidato)
```

Se acepta si `phi_cand < phi_total − ε`.

### 6.3 Convergencia

El proceso repite hasta que ningún movimiento mejora Phi, o hasta un máximo de
20 pasadas. Al converger se garantiza un **óptimo local 1-move**: no existe
ningún movimiento individual de un nodo que mejore la solución.

### 6.4 Restricción

Los singletons (grupos con un solo nodo) no pueden donar su nodo, ya que eso
crearía un grupo vacío. Solo grupos con ≥ 2 nodos participan como donantes.

---

## 7. Fase 3: candidatos de aislamiento

### 7.1 Motivación

El greedy aglomerativo puede pasar por alto particiones donde k-1 nodos están
completamente aislados, si en algún paso previo los fusionó con otros nodos por
error local. Esta fase evalúa exhaustivamente todos esos candidatos.

Se aplica **para todo N**: para k especificado, C(N, k-1) candidatos de ese k;
para k libre, candidatos de todos los niveles k del historial.

### 7.2 Generación de candidatos

Para k partes y N nodos totales, se generan C(N, k-1) candidatos:

```python
for aislados in combinations(range(N), k-1):
    residual = nodos no en aislados
    mascaras = [1 << a for a in aislados] + [máscara_residual]
    yield mascaras
```

Ejemplo para N=10, k=3: C(10,2) = 45 candidatos. Cada uno tiene 2 nodos
aislados individualmente más un grupo residual con los 8 restantes.

Para k libre con todos los niveles: Σₖ₌₂^{N-1} C(N, k-1) = 2^N − 2 candidatos.

### 7.3 Evaluación y selección

**k especificado:**

```python
for candidato_grupos in self._candidatos_aislamiento(k):
    phi_cand = self._emd_particion(candidato_grupos)
    if phi_cand < mejor_phi - 1e-10:
        mejor_phi   = phi_cand
        mejor_grupos = candidato_grupos
mejor_phi, mejor_grupos = self._refinamiento_local(mejor_grupos, mejor_phi)
```

**k libre — búsqueda sobre todos los niveles:**

```python
for k_nivel in historico:
    phi_r, grupos_r = _refinamiento_local(historico[k_nivel])
    if k_nivel < N:
        for candidato in _candidatos_aislamiento(k_nivel):
            if _emd_particion(candidato) < phi_r - 1e-10:
                phi_r, grupos_r = phi_cand, candidato
        phi_r, grupos_r = _refinamiento_local(grupos_r, phi_r)
    historico_refinado[k_nivel] = (phi_r, grupos_r)
mejor_k = min({k: phi for k, (phi, _) in historico_refinado.items() if k >= 3})
```

QNodes puede retornar un k diferente al que el greedy había elegido inicialmente.

---

## 8. Método `_emd_particion`: EMD de la partición completa

Recibe una lista de máscaras `grupos` y devuelve el Phi total de la partición.
La métrica usada es la más precisa que el tamaño del sistema permite:

```python
def _emd_particion(grupos: list[int]) -> float:
    # Construir distribución reconstruida (N componentes marginales)
    dist_rec = np.empty(N, dtype=np.float64)
    for mascara in grupos:
        dist_parte = _dist_parte_efectiva(mascara)   # memoizada
        for i in bits_activos(mascara):
            dist_rec[i] = float(dist_parte[i])

    # Métrica según tamaño — transparente para la estrategia
    if N <= HAMMING_EMD_MAX_N:
        # Wasserstein-1 con d_Hamming sobre el espacio conjunto 2^N
        P = distribucion_conjunta_vectorizada(sia_dists_marginales)
        Q = distribucion_conjunta_vectorizada(dist_rec)
        return emd_causal(P, Q)
    else:
        # Suma L1 marginal sobre N nodos (rápida, tratable para N grande)
        return np.sum(np.abs(dist_rec - sia_dists_marginales))
```

---

## 9. Reconstrucción de la solución

Al terminar, `mejor_grupos` contiene una lista de máscaras. Se reconstruye
la distribución de la partición óptima:

```python
dist_reconstruida = np.empty(N, dtype=np.float32)
for mascara in mejor_grupos:
    dist_parte = _dist_parte_efectiva(mascara)
    for i in _bits_activos(mascara):
        dist_reconstruida[i] = float(dist_parte[i])
```

El resultado se formatea visualmente con marcos:

```
⎛A:B:C⎞⎛  D  ⎞⎛ E ⎞
⎝a:b  ⎠⎝a:d:e⎠⎝ ∅ ⎠
```

---

## 10. Complejidad

| Fase | Complejidad | Nota |
|---|---|---|
| Agrupamiento greedy | O(N³) llamadas a `_emd_particion` | Siempre |
| Refinamiento 1-move (k fijo) | O(N × k × iteraciones) llamadas | Siempre |
| Candidatos de aislamiento (k fijo) | C(N, k-1) llamadas | Siempre |
| Candidatos de aislamiento (k libre) | Σₖ₌₂^{N-1} C(N, k-1) = 2^N − 2 llamadas | k=None |
| Refinamiento por nivel (k libre) | N−1 pasadas adicionales | k=None |
| Una llamada `_emd_particion` (N ≤ 12) | O(2^N) — construye distribución conjunta | N ≤ 12 |
| Una llamada `_emd_particion` (N > 12) | O(N) — suma L1 marginal | N > 12 |

**Para N = 10, k libre:**
- Candidatos de aislamiento: 2¹⁰ − 2 = 1 022 candidatos
- Refinamiento por nivel: 9 pasadas adicionales de 1-move
- **Tiempo estimado: < 1 segundo**

**Para N = 20, k libre:**
- Candidatos de aislamiento: 2²⁰ − 2 ≈ 1M candidatos × O(20) cada uno
- **Tiempo estimado: pocos segundos**

**Para N = 25, k libre:**
- Candidatos de aislamiento: 2²⁵ − 2 ≈ 33M candidatos × O(25) cada uno
- **Tiempo estimado: decenas de segundos a pocos minutos**

El agrupamiento exhaustivo (B&B) requería B(25) ≈ 4 × 10¹⁸ evaluaciones.
El greedy aglomerativo lo reduce a ~2 600, un factor de reducción de ~10¹⁵.
La búsqueda libre sobre todos los k añade O(2^N) evaluaciones adicionales.
