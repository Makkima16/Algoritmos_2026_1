## Instrucciones para el Desarrollo de QNodes

**Contexto del Proyecto:**
QNodes es un framework hermano de GeoMIP, diseñado para resolver el problema de la Partición de Mínima Información (MIP) en el marco de la Teoría de Información Integrada (IIT). Ambos frameworks realizan $k$-particiones buscando minimizar la pérdida de información integrada ($\Phi$), utilizando la métrica Earth Mover's Distance (EMD) y garantizando tolerar el "Estado Vacío" ($\emptyset$). Sin embargo, sus estrategias de resolución son fundamentalmente diferentes, y **deben mantenerse completamente independientes** (no pueden compartir pre-cálculos ni resultados entre sí).

**Estrategia Pure de QNodes:**
Mientras que GeoMIP utiliza heurísticas espaciales y geométricas (Clustering Bottom-Up y refinamiento 1-move) que pueden caer en mínimos locales, **QNodes tiene la responsabilidad de ser computacionalmente y matemáticamente exhaustivo para encontrar siempre el óptimo global**. Para ello, QNodes se basa íntegramente en la **Programación Dinámica con Memoización** (Top-Down/Bottom-Up).

Su núcleo funcional (`DynamicPartition`) debe:
1. Calcular la distribución marginal de sub-partes independientes del sistema causal.
2. Guardar (cachear) estas distribuciones e iterar construyendo las $k$-particiones combinando el "costo" de piezas más pequeñas.
3. Evaluar de forma ágil desde $k=2$ hasta $N$ evitando recalcular subsistemas idénticos múltiples veces.

**Objetivo de Optimización para esta iteración:**
Actualmente el enfoque exhaustivo causa un límite duro de rendimiento cuando $N \ge 13$ debido a la explosión combinatoria. **Tu tarea es optimizar drásticamente la velocidad y eficiencia de memoria de QNodes SIN alterar su esencia de Programación Dinámica y garantizando que encuentre el óptimo global**.

Por favor, implementa y evalúa las siguientes mejoras en la arquitectura del motor de QNodes (`DynamicPartition`):

### 1. Poda Guiada en Programación Dinámica (Branch & Bound Intrínseco)
Implementa una variable global de "mejor costo parcial conocido" ($Upper Bound$ local). Mientras compones iterativamente una partición sumando los costos de la caja 1, caja 2, etc., si el costo acumulado supera al mejor $\Phi$ que ya habías calculado para esa rama en tu árbol de Programación Dinámica, debe existir un _early return_ (Poda). Esto evitará explorar y sumar sub-grafos que no tienen oportunidad de mejorar el global, ahorrando masivamente tiempo en $N$ altos.

### 2. Cacheo Perezoso (Lazy Memoization / Top-Down)
Revisa cómo se está guardando la información. En vez de "inundar" iterativamente la RAM armando los $2^N-1$ estados posibles (Bottom-Up ineficiente), asegúrate de que el algoritmo solo llame a la función objetivo recursivamente. De esta forma, **únicamente calcularemos la distribución y la cachearemos en el momento exacto en que esa sub-partición sea instanciada** por una rama válida de corte del sistema.

### 3. Operaciones Bitwise Estrictas
Dado que las llaves de la Caché (y los cruces de comprobación) se tocan millones de veces, revisa el tipado de Python: 
* Asegúrate de que las entidades candidatas (Mecanismos/Alcances) y las particiones se mapeen obligatoriamente a representaciones de **Enteros (Máscaras de Bits)**. 
* Refactoriza las uniones e intersecciones de la lógica de evaluación causal para que usen operadores binarios en lugar de búsquedas en arrays, tuplas o strings.

### Requisitos Adicionales:
* No utilices librerías probabilísticas aleatorias ni heurísticas (como recocido simulado) para intentar evitar la explosión combinatoria; perderías la precisión que justifica la existencia de QNodes.
* El código debe mantenerse legible, documentado en español (especialmente la lógica matemática de la memoización) y respetando la topología planteada en el `CLAUDE.md`.
* Los resultados de QNodes, testeando los archivos (ej. `N10A.csv`), deben dar la misma partición y el mismo valor $\Phi$ que tira GeoMIP, garantizando equivalencia matemática estricta.