# Registro de Cambios (Changelog) - AYDA 2026-1 vs Proyecto Anterior

Este documento expone las justificaciones algorítmicas y de arquitectura técnica detrás del nuevo desarrollo en AYDA_2026_1, haciendo comparaciones directas contra la anterior implementación del repositorio projecto-analisis-20261.

## 1. Ajuste en la Métrica Central (EMD) y Pérdida φ
* **Antes**: Se empleaba el método emd_efecto, el cual calculaba la suma llana de las distancias L1 de cada distribución de variable de manera individualizada y aislada. Esto causaba que la pérdida final $ aparente fuese artificialmente minúscula (0.2, 0.4), rompiendo indirectamente las asunciones probabilísticas conjuntas del sistema.
* **Ahora (AYDA_2026_1)**: Se usa emd_causal soportado por pyemd.emd. El conjunto se reconstruye como una matriz densa topológica de dimensiones ^N$ basada en Distancias de Hamming, resolviendo un puro problema de *Earth Mover's Distance*. De manera esperada, los $ reportados son mayores numéricamente puesto que escalan logarítmicamente contra las asunciones cruzadas del grafo (Ej: un particionamiento k=3 producirá φ superiores a 2.0+).  

## 2. Incorporación Rigurosa de Nodos Incondicionales (Presente ∅)
* **Antes**: Generar una partición con nodos que cayeran en estados incondicionados originaba una desconexión total artificial (el nodo desconectado perdía la memoria y su entropía quedaba suprimida defectuosamente en los resultados).
* **Ahora (AYDA_2026_1)**: KGeoMIP implementa nativamente _generar_candidatos_presente_vacio. Se admite la inserción legítima del centinela -1 o ∅. En lugar de romper el sistema, evalúa esta sub-partición con justicia métrica, permitiendo que existan aislamientos topológicos donde el futuro es causalmente independiente, sin sesgar o engañar a la métrica $.

## 3. Heurística e Inferencia del Clustering (KGeoMIP) vs Random Hill-Climbing
| Característica | Geometric SIA Anterior | KGeoMIP Nuevo (AYDA_2026_1) |
|--------------|-------------------------|--------------------------|
| **Inicio de Búsqueda** | Pseudo-al azar + Greedy plano. | Reconstrucción de Grafo (Spectral + Agglomerative Clustering). |
| **Poda de Explosión** | Búsqueda exhaustiva sin un perfilador robusto, lo que creaba cuellos combinatorios excesivos para grafos N ≥ 12. | Determinación calculada con Factor Stirling. Umbral fijo rigidez (3000 max combinaciones). Al cruzar, muta rápidamente a búsqueda heurística topológica. |
| **Refinamiento de Borde**| No se refinaban bordes después de los cruces combinatorios del Hypercubo. | Implementación de Búsqueda de vecindad **1-Move** (_refinar_particion_local), con un reacomodo de precisión. Controlado de manera programática un *time bucket limit* sin ahogar el cluster procesador. |

## 4. Programación Dinámica y Múltiples-K en Paralelo 
* **GeoMIP Paralelizado**: La distribución del K ahora se lanza vía concurrent.futures.ProcessPoolExecutor de forma estricta reservando un worker duro. Un modelo antes demoraba X minutos; ahora evalúa subconjuntos combinacionales utilizando todo el músculo multicore disponible hasta K-máximo de manera casi sincrónica (evitando el bloqueo GIL Python).
* **Escala Vertical QNodes**: DynamicPartition introdujo una memoización estructural de variables matricial (*lazy-dicts*) que rescata los cómputos iterativos EMD y los superpone cuando sub-cálculos son repetitivos entre cruces de distintas $. En projecto-analisis-20261 recalcular k-múltiples era un bucle sin retención ni estado de memoria persistente.
