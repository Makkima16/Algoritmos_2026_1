# Cambios con respecto al proyecto original

**Referencia:** `projecto-analisis-20261/QNodes` → `AYDA_2026_1/QNodes`

---

## 1. Cambio central: de bipartición greedy a k-partición aglomerativa

El cambio más importante es el alcance del problema y el algoritmo que lo resuelve.

| Aspecto | `projecto-analisis-20261` | `AYDA_2026_1` |
|---|---|---|
| Particiones soportadas | Solo k = 2 (bipartición) | k ∈ [2, N] (cualquier k) |
| Algoritmo principal | Greedy iterativo (k=2 único) | Agrupamiento aglomerativo greedy |
| Complejidad | O(N² × EMD) | O(N³ × EMD) — tratable para N ≤ 25 |
| Garantía | Óptimo local (greedy k=2) | Óptimo local greedy para cada k |
| Mecanismo vacío (∅) | No soportado | Soportado con `permitir_presente_vacio` |
| Tamaños manejables | N ≤ ~12 (limitado por EMD, no por partición) | N ≤ 25 en tiempo razonable |

---

## 2. Cambio de algoritmo: greedy k=2 → agrupamiento aglomerativo greedy k∈[2,N]

### Versión original (`QNodes`)

Construía la bipartición de forma incremental usando una función submodular:
comienza con ω = {primer_vértice}, Δ = {resto}, y en cada paso añade a ω el
vértice de Δ que minimiza `EMD(ω ∪ δ) − EMD(δ)`. Producía solo una bipartición
(k = 2) y era intrínsecamente imposible extenderlo a k > 2 sin perder la
propiedad de optimalidad local.

### Versión nueva (`QNodes` en AYDA_2026_1)

Usa agrupamiento jerárquico aglomerativo (bottom-up) con la misma función de
costo basada en EMD:

```
Inicializar N singletons: G₀ = {nodo_0}, G₁ = {nodo_1}, ..., G_{N-1}
phi_total = Σᵢ costo(Gᵢ)

Repetir hasta tener 2 grupos:
    Encontrar par (Gᵢ, Gⱼ) que minimice:
        Δ = costo(Gᵢ ∪ Gⱼ) − costo(Gᵢ) − costo(Gⱼ)
    Fusionar Gᵢ y Gⱼ → G_nuevo
    phi_total += Δ
    Registrar {k_actual: (phi_total, grupos)}

Después del agrupamiento → refinar con búsqueda local 1-move
Retornar k con menor phi (k ≥ 3 como prioridad)
```

El resultado natural es una jerarquía completa de k-particiones de k=N a k=2,
de la que se extrae la de menor phi.

---

## 3. Filosofía greedy preservada, generalizada a k dimensiones

El principio submodular original —elegir la operación que minimiza el incremento
de EMD— se preserva en la función de selección de fusión:

```
Δ(Gᵢ, Gⱼ) = costo(Gᵢ ∪ Gⱼ) − costo(Gᵢ) − costo(Gⱼ)
```

Si Δ < 0, fusionar los grupos REDUCE el phi total. Si Δ > 0, lo aumenta.
El greedy elige siempre el par con menor Δ, exactamente el mismo principio que
la versión original aplicaba a la incorporación de vértices uno a uno.

---

## 4. Representación de conjuntos: tuplas (tiempo, índice) → máscaras enteras

| | `projecto-analisis-20261` | `AYDA_2026_1` |
|---|---|---|
| Tipo | `list[tuple[int, int]]` | `int` (bitmask) |
| Ejemplo nodos {A, C} | `[(0,0),(0,2),(1,0),(1,2)]` | `0b0101 = 5` |
| Unión | `list + list` | `a \| b` |
| Complemento | bucle explícito | `total ^ a` |
| Bit mínimo | `min(lista)` | `m & (-m)` |

La representación en máscaras enteras elimina la distinción entre vértices
presentes y futuros de la misma representación, simplificando el código.

---

## 5. Refinamiento local 1-move (nuevo)

La versión original no tenía ningún paso de mejora post-greedy. `AYDA_2026_1`
añade un refinamiento que itera hasta convergencia probando mover cada nodo
a otro grupo:

```
Para cada nodo n en grupo Gᵢ:
    Para cada grupo Gⱼ (j ≠ i):
        Δ = costo(Gᵢ \ {n}) + costo(Gⱼ ∪ {n}) − costo(Gᵢ) − costo(Gⱼ)
        Si Δ < 0: aceptar el movimiento y reiniciar
```

Esto captura mejoras locales que el greedy aglomerativo pierde al fusionar
en el orden incorrecto, mejorando la calidad de la solución sin aumentar
la complejidad asintótica significativamente.

---

## 6. Corrección del cálculo de EMD

En el proyecto original el EMD reportado era "artificial" — se calculaba
directamente con L1 puro sobre el vector completo.

En `AYDA_2026_1` el phi es **aditivo sobre las partes**:

```
phi(particion) = Σᵢ costo(Pᵢ) = Σᵢ Σⱼ∈Pᵢ |dist_parte[j] − dist_sistema[j]|
```

La aditividad es la propiedad que hace que el greedy y la búsqueda local
sean matemáticamente coherentes: el delta de una fusión se puede calcular
directamente como diferencia de costos, sin recalcular el sistema completo.

---

## 7. Entrada de datos: hardcodeada → interactiva

**Original:** estado inicial, TPM y sistema definidos en `src/main.py`.
Para cambiar de sistema era necesario editar el código fuente.

**Nueva versión:** `exec.py` con menú interactivo de dos modos:
- **Modo manual:** ingreso por terminal de nodos, estado y estrategia.
- **Modo por bloque (CSV):** selección de CSV con múltiples sistemas;
  resultados volcados incrementalmente a Excel.

---

## 8. Nuevas capacidades de infraestructura

| Capacidad | Original | Nuevo |
|---|---|---|
| Profiling | Básico / manual | `pyinstrument` con reporte HTML |
| Logger | `print` simple | `slogger.py` con niveles y colores |
| Excel | Post-ejecución manual | Actualización incremental por sistema |
| Validaciones | Mínimas | Exhaustivas (alcance ⊆ candidato, estado coherente) |
| Síntesis de voz | No | `pyttsx3` en `Solution` para narrar resultados |

---

## 9. Archivos sin cambios estructurales

- `src/models/core/ncube.py` — idéntico entre versiones
- `src/models/core/system.py` — idéntico entre versiones
- `src/models/base/sia.py` — idéntico entre versiones
- `src/strategies/force.py` — cambios menores de estilo
- `src/strategies/phi.py` — sin cambios
- `src/funcs/format.py` — sin cambios (el formato de k-partición ya era compatible)

---

## 10. Alineación con GeoMIP: EMD real + candidatos de aislamiento (2026-06-08)

### Problema detectado

Al comparar los resultados de QNodes y GeoMIP sobre el mismo sistema (N10A, k=3),
los valores de `perdida_phi` diferían significativamente (3.46 vs 2.51) y las
`distribucion_subsistema` eran complementarias entre sí.

Se identificaron dos bugs independientes:

---

### Bug 1 — Métrica EMD incorrecta (causa raíz de la diferencia en phi)

**Antes:** QNodes usaba siempre la suma L1 marginal como "EMD":

```
phi_parte(Pᵢ) = Σⱼ∈Pᵢ |p_Pᵢ(j) − p_S(j)|
phi_total    = Σᵢ phi_parte(Pᵢ)          ← suma aditiva por partes
```

Aunque este costo es aditivo (propiedad usada por el greedy), **no es la EMD
matemáticamente correcta de IIT**. La verdadera EMD de IIT es la distancia de
Wasserstein-1 con métrica base Hamming sobre el espacio conjunto de estados:

```
EMD_Hamming(P, Q) = Wasserstein-1(P_conjunta, Q_conjunta, d_Hamming)
```

La suma L1 marginal es una aproximación que sobreestima phi y conduce el greedy
a una partición subóptima distinta de la que GeoMIP encuentra.

**Después:** todas las evaluaciones de la estrategia usan `_emd_particion`, que
internamente aplica la métrica más precisa posible: Wasserstein-1 con d_Hamming
cuando la distribución conjunta 2^N es tratable (N ≤ HAMMING_EMD_MAX_N), y suma
L1 marginal como aproximación rápida para N grandes. Esta decisión es interna a
`_emd_particion` y es transparente para la estrategia: las tres fases del algoritmo
(agrupamiento, refinamiento, candidatos de aislamiento) llaman siempre a
`_emd_particion` sin bifurcarse según N.

**Archivos modificados:**

- `src/funcs/iit.py`:
  - Añadida constante `HAMMING_EMD_MAX_N = 12`.
  - Añadida función `get_hamming_matrix(n)` con caché en `_HAMMING_CACHE`.
  - Añadida función `distribucion_conjunta_vectorizada(probabilidades)`.
  - `emd_causal` actualizado para usar `get_hamming_matrix` (primer llamada O(4^N),
    siguientes O(1)).

- `src/strategies/q_nodes.py`:
  - Añadido método `_emd_particion(grupos)`: calcula el Phi total de la partición;
    aplica Hamming EMD o L1 según N de forma interna y transparente.
  - `_aglomerar`: siempre evalúa cada fusión candidata con `_emd_particion`.
  - `_refinamiento_local`: siempre evalúa cada movimiento con `_emd_particion`.
  - `aplicar_estrategia`: las tres fases (agrupamiento, refinamiento, candidatos
    de aislamiento) se aplican para todo N sin bifurcaciones por tamaño.

---

### Bug 2 — `distribucion_subsistema` complementaria (cosmético)

**Antes:** `GeoMIP/system.py` calculaba `1 − probabilidad` en `distribucion_marginal`,
almacenando P(nodo = OFF). QNodes almacenaba P(nodo = ON). Los valores mostrados
en los JSON de salida eran complementarios (donde uno tenía 1.0 el otro tenía 0.0).

**Después:** `GeoMIP/system.py` corregido para almacenar `probabilidad` directamente,
es decir P(nodo = ON), igual que QNodes. Ambos módulos ahora reportan la misma
convención. La corrección no afecta los valores de EMD (la distancia de Hamming
es simétrica bajo inversión de bits, y la L1 tampoco cambia: `|p − q| = |(1−p) − (1−q)|`).

**Archivo modificado:** `GeoMIP/src/.../models/core/system.py` línea 316.

---

### Fase 3: candidatos de aislamiento (nuevo, activo para todo N)

Incluso con la métrica correcta, el greedy aglomerativo puede pasar por alto
particiones estructuralmente simples donde k-1 nodos están completamente aislados.
GeoMIP evalúa explícitamente todos los C(N, k-1) candidatos de este tipo.

**Añadido:** método `_candidatos_aislamiento(k)` que genera las mismas C(N, k-1)
particiones que `_generar_candidatos_aislamiento` de GeoMIP. Después del refinamiento
local, si algún candidato de aislamiento tiene menor phi que la solución del greedy,
se adopta como nueva solución y se refina nuevamente.

Se aplica para todo N: tanto en modo k especificado (C(N, k-1) candidatos) como
en modo k libre (candidatos de todos los niveles k del historial, 2^N − 2 en total).
Esto garantiza que QNodes busca de forma completamente independiente sin depender
de GeoMIP para ningún tamaño de sistema.

---

### Resultado

Con ambas correcciones, QNodes y GeoMIP obtienen resultados comparables.
Para k especificado, coinciden exactamente:

| Sistema | GeoMIP phi | QNodes phi (antes) | QNodes phi (k=3 fijo) |
|---|---|---|---|
| N10A, k=3, vacio=False | 2.505859375 | 3.458984375 | **2.505859375** ✓ |

Cuando k=None (libre), QNodes puede retornar un k distinto al de GeoMIP si existe
un nivel k diferente con menor phi global — esto es intencional (ver sección 12).

---

---

## 12. Búsqueda independiente: QNodes al 100% sin depender de GeoMIP (2026-06-08)

### Motivación

La Fase 3 (candidatos de aislamiento) se añadió originalmente solo para el caso
`k especificado`, replicando el comportamiento de GeoMIP. Cuando `k=None` (búsqueda
libre), esa fase se saltaba y QNodes solo usaba el greedy + refinamiento.

Esto significaba que QNodes en modo libre era menos exhaustivo que GeoMIP, y que
su resultado dependía indirectamente de que el greedy coincidiera con lo que GeoMIP
hubiera encontrado para algún k.

### Cambio

`_aglomerar()` ahora **siempre retorna el historial completo** `{k: (phi, grupos)}`
para todos los k ∈ [2, N]. La selección de k y las fases de refinamiento quedan
en `aplicar_estrategia`.

**Para `k` especificado:** comportamiento idéntico al anterior (refinamiento + candidatos
de aislamiento para ese k exacto).

**Para `k=None` (cualquier N):** búsqueda exhaustiva independiente:
1. Construir la jerarquía greedy completa.
2. Para **cada nivel k** del historial (k ∈ [2, N−1]):
   - Refinar con 1-move.
   - Evaluar todos los candidatos de aislamiento C(N, k−1).
   - Refinar de nuevo si se encontró mejor candidato.
3. Elegir el k con menor phi (≥3 preferido) entre todos los niveles refinados.

### Consecuencia

Para k libre, QNodes puede retornar un k o una partición **distinta** a la de GeoMIP
si existe un nivel k diferente con menor phi global. Ambos son matemáticamente
válidos; QNodes maximiza su propia búsqueda sin estar anclado a la estrategia de
GeoMIP, para cualquier tamaño de sistema.

---

## 11. Archivos sin cambios estructurales (vigente)

- `src/models/core/ncube.py`
- `src/models/core/system.py`
- `src/models/base/sia.py`
- `src/strategies/force.py`
- `src/strategies/phi.py`
- `src/funcs/format.py`
