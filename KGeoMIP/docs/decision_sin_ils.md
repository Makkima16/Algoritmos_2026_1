# Decisión de Diseño: ILS y VNS 2-move desactivados en GeoMIP

**Fecha:** 2026-06-12 (retiro inicial del ILS) · revisado 2026-06-14 (reevaluación de ILS + VNS 2-move)
**Archivo afectado:** `src/controllers/strategies/kgeomip.py`
**Estado:** Vigente — supersede la descripción de la "Fase 4 / ILS" en
`estrategia_k_particion.md`, `cambios_vs_original.md`, `GeoMIP_Optimizaciones.md`,
`herramientas_y_ecuaciones.md` y `paradigmas_algoritmicos.md`.

> **Actualización (2026-06-14).** Tras el retiro del 12-06 se volvieron a implementar,
> a modo de experimento, **dos** mejoras de búsqueda local: (a) una **VNS 2-move**
> (`_refinar_bloques_2move`) y (b) un **ILS ligero** (`_perturbacion_bloques` + re-refinar,
> 2 reinicios con semilla fija). Ambas se midieron contra `greedy + 1-move` y **ninguna
> mejoró Φ** en la suite, a un costo de 3–5×. Conclusión: se dejaron en el código como
> referencia pero **desactivadas** mediante `N_VNS_MAX = 0` y `N_ILS_LIGHT = 0`. El motor
> en producción sigue siendo `greedy top-down → 1-move → fin`, determinista. El detalle
> empírico del 2-move (con tabla N22) está en `GeoMIP_Optimizaciones.md`, sección 7.

---

## Resumen

Se **retiró la Búsqueda Local Iterada (ILS)** del motor greedy top-down de GeoMIP.
La ILS era la fase que, una vez hallado un óptimo local, lo **perturbaba
aleatoriamente y volvía a refinar** ("buscar hacia los lados") repetidas veces para
intentar escapar de mínimos locales. En la práctica aportaba mejoras **marginales y
poco frecuentes** a un **costo de tiempo alto** (multiplicaba el trabajo de
refinamiento por `1 + N_ILS = 5`), por lo que se eliminó en favor de **resultados
rápidos y precisos**.

---

## Pipeline ANTES vs. DESPUÉS

El motor principal de `KGeoMIP.aplicar_estrategia` es el **greedy top-down sobre
bloques asimétricos** `(futuro, presente)`. Su pipeline era:

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | **Greedy top-down**: parte del bloque completo y aplica k−1 mejores cortes del *cut pool*. | ✅ Se mantiene |
| 2 | **Construcción del cut pool** (afinidad geométrica + cortes derivados). | ✅ Se mantiene |
| 3 | **Refinamiento 1-move**: mueve un nodo (futuro o presente) entre bloques hasta convergencia local. | ✅ Se mantiene |
| 4 | **ILS**: `N_ILS = 4` iteraciones de *perturbación aleatoria + re-refinamiento 1-move*, conservando el mejor. | ❌ **Retirada** |

**Después:** `Greedy top-down → Refinamiento 1-move → fin`.

---

## Por qué se retiró

1. **Ganancia marginal.** Sobre el espacio de pruebas (CSVs `Pruebas_N*`), la ILS
   muy rara vez mejoraba la pérdida Φ ya alcanzada por `greedy + 1-move`, y cuando lo
   hacía, casi nunca lograba **superar a QNodes** (que realiza una exploración
   determinista más completa). El óptimo local del 1-move sobre bloques asimétricos
   ya es, en la abrumadora mayoría de los casos, la mejor partición que GeoMIP
   encontraría con o sin ILS.

2. **Costo de tiempo desproporcionado.** El 1-move es la fase más cara (evalúa
   O(N²·k) vecinos por ronda). La ILS lo repetía `N_ILS` veces adicionales sobre
   configuraciones perturbadas → **~5× el costo del refinamiento** para una mejora
   esperada cercana a cero.

3. **No-determinismo.** La perturbación usaba semillas pseudoaleatorias, introduciendo
   variabilidad entre corridas del mismo caso. Sin ILS, GeoMIP es **determinista**:
   misma entrada → misma partición, lo que facilita la comparación con QNodes y la
   reproducibilidad de la suite de pruebas.

---

## Evidencia (rendimiento tras el retiro)

Muestra de 8 pruebas de `data_scripts/Pruebas/Pruebas_N10.csv`, TPM `N10A`, k=3,
`permitir_presente_vacio=False`, en una máquina de desarrollo:

| Métrica | Valor |
|---------|-------|
| Tiempo total (8 casos) | **2.59 s** |
| Tiempo promedio por caso | **0.324 s** |
| Particiones degeneradas (∅,∅) | 0 |
| Bloques (∅, ∅) degenerados | 0 |

Todas las particiones resultantes son válidas (ningún bloque tiene futuro y presente
vacíos simultáneamente; tanto futuro ∅ como presente ∅ son permitidos por separado).
La calidad de Φ se mantiene equivalente a la del pipeline con ILS, a una fracción del tiempo.

---

## Qué cambió en el código

**Retiro inicial (2026-06-12):** se sacó la Fase 4 del flujo activo de `aplicar_estrategia`,
dejando `greedy + 1-move` como motor.

**Estado actual (2026-06-14):** ambas mejoras existen pero están **inactivas por
constante de bucle = 0**, no eliminadas:

- `_refinar_bloques_2move` (VNS) + bucle `N_VNS_MAX = 0` en `aplicar_estrategia`.
- `_perturbacion_bloques` (ILS ligero) + bucle `N_ILS_LIGHT = 0` en `aplicar_estrategia`.
- El refinamiento 1-move (`_refinar_bloques_1move`) y el invariante de no generar
  bloques `(∅, ∅)` permanecen como única fase de mejora activa.

> Reactivar cualquiera es subir su constante (`N_VNS_MAX` / `N_ILS_LIGHT`) a > 0; el ILS
> ligero ya usa semilla fija, así que conserva el determinismo si se reactiva.

> Nota: la función `_evaluar_k_completo` (ruta simétrica alternativa, **no invocada**
> por el flujo principal) conserva su propia lógica de ILS (`N_ILS = 4`), irrelevante en
> runtime por ser código inactivo.

---

## Reversión

El código de ambas mejoras ya está presente; reactivar exploración adicional (p. ej. para
N grande donde los óptimos locales sean más frecuentes) solo requiere subir la constante
del bucle correspondiente en `aplicar_estrategia`:

- **ILS ligero:** `N_ILS_LIGHT > 0` (usa `_perturbacion_bloques`, con semilla fija → sigue
  siendo determinista).
- **VNS 2-move:** `N_VNS_MAX > 0` (usa `_refinar_bloques_2move`; recuerda su costo O(M²)
  por pase, ver `GeoMIP_Optimizaciones.md` §7).
