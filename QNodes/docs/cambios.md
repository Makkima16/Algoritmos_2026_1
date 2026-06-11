# Cambios con respecto al proyecto original

**Referencia:** `projecto-analisis-20261/QNodes` → `AYDA_2026_1/QNodes`

> **Estado actual (2026-06-10):** QNodes fue reescrito a un **motor asimétrico
> unificado** (greedy top-down + refinamiento 1-move futuro/presente + ILS, con
> métrica L1 marginal = EMD de Hamming exacta). Este documento describe el estado
> vigente; las secciones finales conservan el historial de la evolución.

---

## 1. Cambio central: de bipartición greedy a k-partición asimétrica unificada

| Aspecto | `projecto-analisis-20261` | `AYDA_2026_1` (actual) |
|---|---|---|
| Particiones soportadas | Solo k = 2 (bipartición) | k ∈ [2, N], mismo motor para todo k |
| Representación | `list[tuple(tiempo, índice)]` | `Bloque = (frozenset futuros, frozenset presentes)` |
| Cortes | Simétricos | **Asimétricos** (futuro/presente independientes) |
| Algoritmo principal | Greedy incremental k=2 | Greedy **top-down** + 1-move + ILS |
| Métrica EMD | L1 marginal (asumida aproximación) | L1 marginal = **Wasserstein-1 Hamming EXACTA** |
| Límite de tamaño | N ≤ ~12 (por la EMD) | Sin límite — L1 es O(N) para todo N |
| Mecanismo vacío (∅) | No soportado | Soportado (corte `({i}, ∅)`) |

---

## 2. Representación asimétrica de bloques (el cambio que lo unifica todo)

Cada bloque es un par `(frozenset futuros, frozenset presentes)` donde el futuro (t+1)
y el presente/mecanismo (t) se particionan de forma **independiente**. Esto generaliza
a todo k el corte asimétrico que antes sólo se usaba para k=2, y elimina la distinción
entre el caso k=2 y k≥3.

**Causa raíz del salto k=2→k=3 (resuelto):** la versión intermedia usaba cortes
asimétricos sólo para k=2; para k≥3 usaba cortes **simétricos** que sobre-cortaban e
inflaban Φ. Medido en N10A: simétrico k=3 = 2.5059 vs asimétrico k=3 = 0.9590. El motor
actual usa asimétrico para **todo** k → Φ monótono y coherente.

---

## 3. Algoritmo: greedy top-down (no aglomerativo bottom-up)

El motor parte de **un solo bloque** (todo el subsistema) y aplica las mejores
divisiones:

```
_construir_pool_cortes()  → O(N) cortes (3 familias por nodo), construido UNA vez
_greedy_descenso(pool)    → un descenso k=1..N, registra Φ por cada k (jerarquía nido)
_refinar_bloques(...)     → 1-move futuro + 1-move presente (asimétrico)
_refinar_con_ils(...)     → perturbación + re-refinamiento (N-adaptativo)
```

Un único descenso produce una jerarquía **anidada** → coherencia (Φ monótono) entre k
consecutivos. (Detalles en `docs/estrategia-k-particion.md`.)

---

## 4. Corrección y aceleración del cálculo de EMD

El cambio de mayor impacto. La métrica L1 marginal **es** la Wasserstein-1 con Hamming
EXACTA (no una aproximación), porque tanto la distribución original como la
reconstruida de cualquier k-partición son **productos de marginales por nodo**:

```
EMD_Hamming(P, Q) = Σᵢ |P(nodo_i = ON) − Q(nodo_i = ON)|     (verificado |·| < 1e-14 para N=2..12)
```

Consecuencias:
- **Más rápido:** O(N) por evaluación en vez de O(4^N) del solver `pyemd`.
- **Más preciso:** da el MISMO Φ que la EMD real (k=2 N10A = 0.4746, idéntico a GeoMIP).
- **Sin límite de tamaño:** se eliminó la dependencia de `pyemd` y el techo N ≤ 12.

> Esto **corrige** la nota antigua que afirmaba que L1 = pyemd "sólo si P es producto,
> lo cual no ocurre en IIT": sí ocurre aquí por construcción.

---

## 5. Refinamiento 1-move asimétrico + ILS (nuevo)

La versión original no tenía mejora post-greedy. La actual añade:
- **Movimiento futuro:** reubica un nodo futuro entre bloques.
- **Movimiento presente (asimétrico):** reubica el mecanismo de un nodo sin mover su
  futuro — imposible en representaciones simétricas.
- **ILS:** perturbación + re-refinamiento N-adaptativo para escapar de mínimos locales.

---

## 6. Entrada de datos: hardcodeada → interactiva

**Original:** estado, TPM y sistema definidos en `src/main.py`; cambiar de sistema
exigía editar el código.

**Actual:** `exec.py` con menú de dos modos:
- **Modo manual:** ingreso por terminal de estado, candidato, alcance, mecanismo y k.
- **Modo por bloque (CSV):** múltiples sistemas; resultados volcados incrementalmente.

El candidato y el estado inicial se ingresan una sola vez para todo el lote.

---

## 7. Infraestructura

| Capacidad | Original | Actual |
|---|---|---|
| Profiling | Manual | `pyinstrument` con reporte HTML |
| Logger | `print` | `slogger.py` con niveles y colores |
| Salida | Manual | Incremental por sistema |
| Validaciones | Mínimas | Exhaustivas (alcance ⊆ candidato, estado coherente) |

---

## 8. Equivalencia con GeoMIP

QNodes y GeoMIP dan el **mismo Φ** para todo k, porque ambos comparten ahora:
cortes asimétricos, greedy top-down con pool de cortes, refinamiento 1-move
futuro/presente, ILS, y métrica L1 = EMD Hamming exacta. Si vuelve a aparecer un
"salto" entre k consecutivos, sospechar de cortes simétricos colándose.

---

---

## Historial de evolución (referencia)

**2026-06-08 — Alineación con GeoMIP.** Se igualó la métrica con GeoMIP y se añadieron
candidatos de aislamiento. En esa etapa la métrica seleccionaba entre EMD-Hamming
(`pyemd`, N ≤ 12) y L1 (N > 12), y el motor era aglomerativo bottom-up.

**2026-06-10 (tarde) — Reescritura al motor asimétrico unificado (estado vigente).**
Se descubrió que la L1 marginal es la EMD-Hamming exacta (no aproximación), lo que
permitió eliminar `pyemd` y el límite N ≤ 12. Se reemplazó el motor aglomerativo por el
greedy top-down sobre bloques asimétricos `(futuros, presentes)`, con refinamiento
1-move futuro/presente e ILS. Quedó un único motor para todo k, sin el salto k=2→k=3.

---

## Archivos sin cambios estructurales

- `src/models/core/ncube.py`, `src/models/core/system.py`, `src/models/base/sia.py`
- `src/strategies/force.py`, `src/strategies/phi.py`
- `src/funcs/format.py` (el formateador `fmt_k_bloques` ya era compatible)
