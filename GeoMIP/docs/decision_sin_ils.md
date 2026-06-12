# Decisión de Diseño: Retiro de la Búsqueda Local Iterada (ILS) en GeoMIP

**Fecha:** 2026-06-12
**Archivo afectado:** `src/controllers/strategies/kgeomip.py`
**Estado:** Vigente — supersede la descripción de la "Fase 4 / ILS" en
`estrategia_k_particion.md`, `cambios_vs_original.md`, `GeoMIP_Optimizaciones.md`,
`herramientas_y_ecuaciones.md` y `paradigmas_algoritmicos.md`.

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
| Bloques con futuro vacío | 0 |

Todas las particiones resultantes son válidas (cada bloque conserva ≥1 nodo futuro;
el presente vacío solo aparece cuando `permitir_presente_vacio=True`). La calidad de
Φ se mantiene equivalente a la del pipeline con ILS, a una fracción del tiempo.

---

## Qué cambió en el código

- Eliminado el bucle ILS (Fase 4) dentro de `aplicar_estrategia`.
- Eliminada la función auxiliar `_perturbar_bloques` (solo la usaba la ILS).
- Eliminada la constante de módulo `N_ILS`.
- El refinamiento 1-move (`_refinar_bloques_1move`) y el invariante de no generar
  bloques con futuro vacío permanecen intactos.

> Nota: la función `_evaluar_k_completo` (ruta simétrica alternativa, **no invocada**
> por el flujo principal) conserva su propia lógica de ILS, irrelevante en runtime por
> ser código inactivo.

---

## Reversión

Si en el futuro se desea reintroducir exploración tipo ILS (p. ej. para N grande donde
los óptimos locales sean más frecuentes), basta con restaurar `_perturbar_bloques` y el
bucle de perturbación + re-refinamiento tras la Fase 3, parametrizando `N_ILS` y, de
preferencia, **fijando una semilla** para conservar el determinismo.
