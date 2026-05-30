# KGeoMIP: Particionamiento Óptimo de Sistemas Complejos

Este proyecto implementa el algoritmo computacional **KGeoMIP** diseñado para encontrar particiones óptimas de sistemas de información compleja (redes de gran escala, como N >= 15). El objetivo matemático central es dividir la red en **k-particiones** mientras se minimiza rigurosamente la pérdida de información del sistema.

---

## 🧠 ¿Qué hicimos? Lógica a Nivel Simple

Imagina que tienes una gran red de personas conectadas (o neuronas, o servidores) y necesitas agruparlos en exactamente $K$ equipos diferentes. 
El problema es que, al separarlos, se pierde "comunicación" o información vital entre ellos. Queremos encontrar la división exacta que **pierda la menor cantidad de información posible**.

Si intentamos probar todas las combinaciones posibles una por una, nos tomaría años (o siglos) incluso en computadoras modernas. Así que, en lugar de probar todo, nuestro algoritmo hace lo siguiente:
1. **Divide de a poco:** En lugar de hacer las $K$ divisiones de golpe, toma el grupo más grande y lo corta en dos, repitiendo esto hasta llegar a $K$ grupos.
2. **Empieza al azar y mejora iterativamente:** Para dividir un grupo en dos, hace una partición inicial aleatoria. Luego, toma *un solo elemento* y lo pasa al otro lado. Si la comunicación mejora (se pierde menos información), nos quedamos con ese cambio y seguimos probando. 
3. **Se detiene en la mejor jugada:** Cuando mover cualquier elemento a otro lado empeora todo, el algoritmo dice "este es el mejor agrupamiento" y termina. 

¡Esto nos permite encontrar agrupamientos casi perfectos en cuestión de segundos, en lugar de días!

---

## 🔬 Lógica del Algoritmo a Nivel Detallado

El problema clásico de particionamiento de sistemas de grafos sufre de la infame **explosión combinatoria** (descrita asintóticamente por los Números de Bell y de Stirling). Una búsqueda exhaustiva tradicional tiene una complejidad exponencial $O(2^N)$ por cada bipartición.

Para superar esta barrera, la arquitectura de KGeoMIP implementa una **estrategia híbrida iterativa** matemáticamente guiada:

### 1. Evaluación de Pérdida de Información (EMD)
Para medir qué tan "buena" es una partición, usamos el **Earth Mover's Distance (EMD)**. Esta métrica compara la matriz de transición de estados del sistema original contra la del sistema particionado, calculando el "costo de transporte" de la distribución de probabilidad (qué tan lejos se movió la información tras la partición).

### 2. Enfoque "Greedy" (Codicioso) Jerárquico
Tratar de evaluar particiones k-múltiples simultáneamente es ineficiente. El gestor codicioso selecciona iterativamente el sub-sistema actual más grande y aplica una bipartición óptima sobre él, apilándolos hasta satisfacer la restricción exacta de $K$ particiones.

### 3. Búsqueda Local Estocástica y Refinamiento (Local Search) - *El núcleo de la eficiencia*
Para realizar la bipartición (y generalizaciones a k-particiones) tras superar el umbral exhaustivo (`UMBRAL_STIRLING`), inyectamos un híbrido de Búsqueda Heurística Global seguida de un Refinamiento Local:
*   **Clustering sobre Grafo de Afinidad:** En lugar de aleatoriedad pura, se construye una matriz geométrica con las distancias de transición de probabilidad. Luego se emplea *Spectral Clustering* para acercarse muy rápidamente a la partición cuasi-óptima.
*   **Aislamiento Heurístico y Nodos Vacíos (∅):** El algoritmo explora combinaciones formales donde futuros (t+1) pueden ser causados por estados presentes vacíos (la inclusión del ∅), dadas exigencias algebraicas.
*   **Evaluación de Vecindario 1-Move (Refinamiento Local):** Tomando como base la partición heurística, el algoritmo evalúa iterativamente el traslado (swap) de *un único nodo* hacia una partición colindante.
*   **Aceptación Temprana:** Apenas se detecta que un intercambio reduce la EMD conjunta (`emd_causal`), se acepta el nuevo estado óptimo local.
*   **Mínimo Local:** Si mover cualquier nodo incrementa la EMD, o se agota el presupuesto de tiempo asignado, el algoritmo converge y finaliza. Esto pule enormemente el valor final de φ comparado con heurísticas superficiales.

### 4. Computación Paralela (Multiprocessing)
El programa principal aprovecha todos los hilos del procesador aislando la evaluación de vecindarios y casos de prueba usando procesos concurrentes (`multiprocessing`), garantizando que no haya cuellos de botella por el Global Interpreter Lock (GIL) de Python.

---

## 🛠️ Herramientas y Librerías Utilizadas

El algoritmo fue estructurado para ser ligero, robusto y altamente especializado en manipulaciones numéricas. 

*   **`Python 3.11+`:** Escogido como lenguaje base por su flexibilidad, facilidad de prototipado algorítmico y amplio ecosistema.
*   **`pyemd`:** La librería reina del proyecto. Es una envoltura (wrapper) en C++ hiper-optimizada que calcula la métrica matemática de transporte óptimo (*Earth Mover's Distance*).
*   **`numpy`:** Maneja todas las estructuras de datos matriciales, vectores de distancias y topologías espaciales de los grafos. Permite realizar operaciones matemáticas vectorizadas a velocidades nativas de C.
*   **`pandas`:** Fundamental para ingerir velozmente matrices masivas de comportamiento desde los archivos CSV originales.
*   **`multiprocessing` (Nativa de Python):** Orquesta los lotes de procesamiento asíncrono para exprimir el rendimiento de la CPU moderna dividiendo casos de test.

### 💡 ¿Por qué NO usamos Tensores (TensorFlow o PyTorch)?
Al pensar en cálculos intensivos, es fácil imaginar el uso de tensores de Deep Learning en la GPU. Sin embargo, no los implementamos por las siguientes razones de arquitectura:
1. **Naturaleza discreta del problema:** Estamos haciendo **optimización combinatoria discreta** (evaluando qué nodo va en qué partición: 0 o 1). Los tensores y el descenso de gradiente de PyTorch o TensorFlow están diseñados matemáticamente para espacios continuos y derivables, no para agrupaciones estáticas de grafos bidimensionales.
2. **El cálculo EMD es exacto:** Para medir la pérdida de información iteramos sobre la distancia *Earth Mover's Distance*. La librería que usamos (`pyemd`) utiliza el algoritmo *Network Simplex* programado directamente en C++. Intentar calcular EMD en tensores de GPU mediante aproximaciones no suele garantizar el resultado exacto, o lo hace de manera excesivamente costosa para nuestro contexto.
3. **Sobrecarga (Overhead) innecesaria:** Instalar y cargar el contexto CUDA en la GPU a través de PyTorch añadiría un tiempo muerto enorme y un peso de Gigabytes a las dependencias. Las operaciones algebraicas aquí requeridas (`numpy`) vectorizadas sobre CPU junto con un *Hill Climbing* veloz resulta ser abismalmente más rápido para ejecutar nuestras iteraciones de forma paralela.

---

## ▶️ Ejecución y Uso Interactivo

El algoritmo ahora se lanza a través de un script interactivo que te permite seleccionar el dataset fácilmente. Para ejecutarlo en consola:

```bash
python GeoMIP/src/Method2_Dynamic_Programming_Reformulation/exec_kgeomip.py
```

**Flujo de ejecución:**
1. Se abrirá una pequeña ventana (explorador de archivos) permitiéndote seleccionar la Matriz de Probabilidad TPM (`.csv`) almacenada en `GeoMIP/data/samples/`.
2. La consola te pedirá ingresar el estado inicial de la red en binario (o puedes simplemente presionar `ENTER` para dejar que el programa genere uno aleatorio válido).
3. Podrás indicar el valor $K$ de particiones requeridas (por ej. `2`, `5`, etc.).
4. El programa ejecutará el KGeoMIP e **imprimirá el resultado (y la partición óptima)** directamente en la pantalla de la consola. 
   > *(Nota: En esta versión interactiva, los resultados se visualizan inmediatamente en la terminal al finalizar, sin escribir archivos en disco).*
