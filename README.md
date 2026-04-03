# KGeoMIP: Particionamiento Óptimo de Sistemas Complejos

Este proyecto implementa el algoritmo computacional **KGeoMIP** diseñado para encontrar particiones óptimas de sistemas de información compleja (redes de gran escala, como N >= 15). El objetivo matemático central es dividir la red en **k-particiones** mientras se minimiza rigurosamente la pérdida de información del sistema, una métrica que se evalúa mediante la distancia de transporte óptimo **Earth Mover's Distance (EMD)**.

---

## 🚀 El Algoritmo: ¿Por qué es tan rápido y eficiente?

El problema clásico de particionamiento de sistemas de grafos sufre de la infame **explosión combinatoria** (descrita asintóticamente por los Números de Bell y de Stirling). Una búsqueda exhaustiva tradicional tiene una complejidad exponencial $O(2^N)$ por cada bipartición, lo que convierte la evaluación de redes de más de 10 nodos en un proceso de días o meses de cómputo ininterrumpido.

Para superar esta barrera, la arquitectura de KGeoMIP renuncia a las iteraciones por "fuerza bruta" e implementa una **estrategia híbrida iterativa** altamente optimizada y matemáticamente guiada:

### 1. Enfoque "Greedy" (Codicioso) de División Iterativa
En lugar de intentar evaluar y mapear todas las posibles particiones k de golpe, el algoritmo fracciona el sistema de manera iterativa. En cada paso, toma el subconjunto más denso (con mayor cantidad de nodos) y lo divide en dos, repitiendo el proceso hasta consolidar exactamente k particiones. Sin embargo, dividir ingenuamente un subconjunto de N=15 en dos porciones arrojaba **más de 16,000 combinaciones** que debían someterse a la costosa métrica EMD.

### 2. Búsqueda Local Estocástica (Hill Climbing) - *El núcleo de la eficiencia*
Para esquivar el análisis exponencial en las redes complejas (superando el `UMBRAL_EXHAUSTIVO`), inyectamos un método de **Búsqueda Local de Escalada (Hill Climbing)** que funciona de forma magistral:

*   **Estado Inicial Aleatorio:** Las biparticiones inician dividiendo los nodos de forma agnóstica al azar (50/50), garantizando únicamente que ambas orillas tengan elementos.
*   **Evaluación de Vecindario $O(N)$:** A partir de la posición inicial, el algoritmo empieza a jugar con márgenes minúsculos: toma *un único nodo* de la "Partición A" y evalúa matemáticamente su impacto si lo pasamos a la "Partición B" (y viceversa).
*   **First Improvement (Aceptación Temprana):** En el instante exacto en el que trasladar un nodo resulta en una **disminución del costo de información (EMD)**, el algoritmo interrumpe la exploración del resto, asume este nuevo ordenamiento como el estándar dorado temporal, y re-comienza a iterar desde esta nueva posición mejorada.
*   **Convergencia Hiper-rápida:** Esta "escalada" se repite iterativamente persiguiendo caídas drásticas en el EMD hasta chocar con el "fondo" o mínimo local; es decir, llega a un punto donde mover cualquier otro nodo individual del subconjunto solamente encarece la pérdida o la mantiene estática. 

*Resultado:* El crecimiento del tiempo de ejecución cambió desde una escalabilidad **Exponencial** prohibitiva a unas cuantas docenas de comprobaciones **Lineales/Polinómicas** por cada rama o vecindario, resolviendo particiones masivas en escasos segundos.

### 3. Computación Paralela (Multiprocessing)
Para blindar el rendimiento, el gestor orquesta múltiples estados topológicos y lotes de procesamiento utilizando colas de concurrencia multinúcleo (`multiprocessing`). Cada caso de prueba viaja independientemente y tiene su propia trampa térmica (timeout) sin entorpecimiento entre hilos.

---

## ⚙️ Arquitectura de Datos

El sistema KGeoMIP se alimenta automáticamente de dos fuentes principales sin necesitar ajustes en rutas de código:
1. **Matrices de Transición TPM (`.csv`):** Localizadas en `GeoMIP/data/samples/` (Ej. `N15A.csv`). El script extrae e infiere automáticamente la topología más grande posible para empujar el límite del sistema. Estos datos representan conceptualmente la evolución natural del grafo complejo evaluada a través de PYPHI/EMD.
2. **Casos Control / Sub-Sistemas (`.xlsx`):** A través del script central, extrae lotes de particiones semilla (ej. cadenas lógicas en binario que definen subconjuntos estáticos iniciales) extraidas directamente como casos de prueba desde `GeoMIP/results/Pruebas_Metodo2.xlsx`.

---

## 💻 Requisitos y Entorno
*   **SO:** Compatible multiplataforma (Probado con alta eficiencia en Windows y Linux).
*   **Python:** 3.11 o superior.
*   **Dependencias principales:** `numpy`, `pandas`, `pyemd`, `openpyxl`.
*   *(Sugerido)*: Administrador de entornos `uv` (opcional).

*(Si cuentas con instalación base, instala localmente mediante `uv sync` o tu gestor tradicional dentro de `GeoMIP/src/Method2_Dynamic_Programming_Reformulation`)*

---

## ▶️ Ejecución y Uso

Para procesar un lote de pruebas masivas del algoritmo en la consola, ejecuta:

```bash
python GeoMIP/src/Method2_Dynamic_Programming_Reformulation/exec.py
```

El log transaccional (`CRITICAL`) te irá mostrando en vivo las iteraciones y el costo EMD de las particiones óptimas de Hill Climbing (Por ej. `| D,E || B,G,I,A,C,H... |` con `pérdida=0.1896`). 
Los diagnósticos temporales de ejecución, las métricas, las particiones devueltas y su costo definitivo se exportan sin intervención manual a una tabla de Excel estructurada en la siguiente ruta:

* **Salida de Resultados:** `GeoMIP/results/resultados_Geometric.xlsx`
