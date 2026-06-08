# Optimizaciones en GeoMIP (AYDA 2026-1)

Este documento detalla las técnicas de optimización y agilización implementadas específicamente en el framework **GeoMIP** para garantizar el cálculo eficiente de la Partición de Mínima Información (MIP) en sistemas de alta dimensionalidad ($N \ge 20$).

## 1. Gestión Inteligente de Memoria (LazyTPM)
El manejo de Matrices de Probabilidad de Transición (TPM) para redes con 18 nodos o más provoca exponenciales requerimientos de memoria RAM (una TPM de $N=20$ puede pesar varios gigabytes). 
* **Solución:** Se implementó una lógica de lectura "perezosa" o por fragmentos (*Lazy Reading / Chunks*). En lugar de cargar la matriz completa a la memoria de golpe, el sistema extrae de forma secuencial solo los datos necesarios en tiempo de ejecución, previniendo caídas del sistema por "Out of Memory".

## 2. Paralelismo Dedicado (Un Test a la vez)
Aunque se cuenta con procesamiento multi-núcleo (utilizando `ProcessPoolExecutor`), la estrategia arquitectónica se ajustó para **enfocar todos los núcleos disponibles en un solo test a la vez** en lugar de correr múltiples tests en paralelo.
* **Beneficio:** Al paralelizar las sub-tareas de un mismo test (como la evaluación de diferentes particiones $k$ o cálculos dentro del mismo espacio candidato), nos aseguramos de no saturar el bus de memoria ni generar cuellos de botella de I/O. Todo el poder de la CPU trabaja en conjunto para destruir la complejidad del test actual antes de pasar al siguiente.

## 3. Modo por Bloque Inteligente (De Fácil a Difícil)
El modo bloque o procesamiento por lotes (*Batch Mode*) fue optimizado en su flujo de trabajo:
* **Ordenamiento por Complejidad:** El sistema lee el archivo de pruebas (ej. `.csv`) y organiza la cola de ejecución testeando **primero los casos más fáciles** (aquellos con mayor cantidad de ceros en su máscara, lo que implica menos nodos activos y menor cardinalidad). 
* **Beneficio:** Permite obtener un *feedback* rápido de las pruebas triviales o pequeñas mientras el motor progresivamente escala hacia los tests donde el costo computacional es mayor. Además, el estado y el candidato se ingresan y guardan en memoria una sola vez para todo el lote completo.

## 4. Reducción Combinatoria y Heurísticas
La búsqueda por fuerza bruta es matemáticamente imposible en el particionamiento de conjuntos grandes (Números de Bell/Stirling). GeoMIP lo solventa con:
* **Matrices de Afinidad Geométrica:** En vez de hacer cortes a ciegas, calcula distancias probabilísticas e infiere grupos mediante clustering jerárquico (*Bottom-Up*).
* **Refinamiento Local (1-move):** Toma una partición candidata y evalúa mover solo un nodo a la vez de un subconjunto a otro. Si mejora el valor de $\Phi$, lo acepta, encontrando mínimos locales de forma extremadamente veloz.
* **Espacio de Hipercubo optimizado:** Se hace una Búsqueda en Anchura (BFS) aprovechando matrices EMD y la distancia de Hamming para acelerar los mapeos topológicos.

## 5. Algoritmo Tolerante al "Estado Vacío" ($\emptyset$)
En previas iteraciones de IIT algorítmico, cuando un corte dejaba una parte sin componentes (vacía), el sistema "destrozaba" las distribuciones y sumaba castigos artificales (*over-cutting*).
* **Solución (`generar_candidatos_presente_vacio`):** El mecanismo contempla rigurosamente el corte $\emptyset$, ponderándolo contra su peso probabilístico de Hamming de forma limpia usando `PyEMD`. Esto restringe que la búsqueda iterativa divague calculando penalizaciones falsas, agilizando enormemente el cálculo iterativo del $\Phi$ real.