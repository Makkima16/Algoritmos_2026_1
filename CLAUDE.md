# CLAUDE.md — GeoMIP & QNodes / AYDA 2026-1

## Descripción del Proyecto

El repositorio contiene dos frameworks académicos para resolver el problema de la **Partición de Mínima Información (MIP)**, concepto central en la Teoría de Información Integrada (IIT). El proyecto es desarrollado para la asignatura *Análisis y Diseño de Algoritmos (AYDA) 2026-1*.

El objetivo principal de ambos frameworks es encontrar la partición óptima de un sistema lógico o neurobiológico que minimice la pérdida de información integrada (Phi, Φ), empleando la **Distancia del Transportador de Tierra (EMD)** para medir la pérdida.

Existen dos enfoques principales en el repositorio:
1. **GeoMIP:** Framework original interactivo (`GeoMIP/src/Method2_Dynamic_Programming_Reformulation/`) que aborda el problema k-MIP paralelizando los cálculos de múltiples tamaños de partición $k$ e implementando heurísticas combinatorias (ej. clustering jerárquico bottom-up) para reducir la complejidad cuando las dimensiones son muy altas.
2. **QNodes:** Nuevo framework (`QNodes/`) centrado en una estrategia de **Programación Dinámica con Memoización** (`DynamicPartition`). Su enfoque consiste en precalcular y cachear distribuciones marginales de cada subparte en lugar de recalcularlas por cada partición entera, para poder evaluar sub-particiones óptimas sobre todos los valores posibles de $k$ (desde 2 hasta $N$) evitando la explosión combinatoria.

---

## Estructura de Carpetas

```
AYDA_2026_1/
├── CLAUDE.md                          <- este archivo
├── .venv/                             <- entorno virtual Python (compartido)
├── GeoMIP/
│   ├── data/                          <- Muestras de TPMs y scripts de generación (creation.py)
│   ├── results/                       <- JSONs con resultados
│   ├── viewer/                        <- Frontend de visualización
│   ├── view_result.py                 <- Visualizador de JSON
│   └── src/Method2_Dynamic_Programming_Reformulation/
│       ├── exec_kgeomip.py            <- Entrypoint interactivo principal para GeoMIP
│       └── src/                       <- Modelos (System, NCube), Controladores (KGeoMIP), etc.
└── QNodes/
    ├── exec.py                        <- Entrypoint principal para QNodes
    ├── pyproject.toml                 <- Dependencias de QNodes
    ├── review/                        <- Scripts de testeo y profiling de QNodes
    └── src/
        ├── main.py                    <- Lógica del lanzador CLI/GUI
        └── strategies/
            └── dynamic.py             <- Algoritmo core de Programación Dinámica con memoización
```

---

## Conceptos de Dominio (IIT y MIP)

| Término | Descripción |
|---------|-------------|
| **TPM** | *Transition Probability Matrix*. Matriz estocástica (2^N filas x N columnas). Describe la dinámica de los nodos. |
| **Estado inicial**| Secuencia que representa el estado del sistema en un tiempo $t$. Ej: `"101"`. |
| **Condición / Candidato**| Máscara donde los bits en 0 son condiciones de fondo marginadas (background). |
| **Mecanismo ($t$) / Alcance ($t+1$)**| Selecciones de qué nodos intervienen como causa (pasado/presente) o efecto (futuro). |
| **EMD** | *Earth Mover's Distance*. Métrica de similitud usada para comparar la distribución del sistema original y el distribuido en sub-partes. |
| **Phi (Φ)** | Información Integrada. La pérdida EMD que da como resultado el mínimo sobre todas las particiones válidas es la MIP. |
| **k-particiones** | Dividir los $N$ nodos en $k$ subconjuntos no vacíos. Existen números de Stirling de segunda especie $S(n,k)$ sub-particiones posibles. |

---

## Flujos de Ejecución

### GeoMIP Pipelíne (`exec_kgeomip.py`)
```
exec_kgeomip.py
  └── KGeoMIP.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
        ├── 1. Extraer y construir subsistema candidato.
        ├── 2. BFS sobre hipercubo (o matriz EMD optimizada para N>20).
        └── 3. Distribuir heurísticas por k concurrentes (ProcessPoolExecutor).
```

### QNodes Pipelíne (`exec.py`)
```
exec.py
  └── DynamicPartition(aplicacion)
        ├── 1. Cargar estado inicial y TPM.
        ├── 2. Identificar alcance y mecanismo a evaluar.
        ├── 3. Encontrar particiones minimizando pérdida cacheando distribuciones marginales.
        └── 4. Retornar las particiones con Phi mínimo explorando ágilmente todo el rango de k.
```

---

## Comandos de Uso

Todo el código depende del mismo entorno virtual (`.venv/`) con Python >= 3.11.

**Ejecutar GeoMIP (elige modo al inicio):**
```bash
source .venv/Scripts/activate
python GeoMIP/src/Method2_Dynamic_Programming_Reformulation/exec_kgeomip.py
```

Al ejecutar se ofrece:
- **Modo 1 — Manual**: ingreso interactivo de candidato, estado, alcance, mecanismo y K.
- **Modo 2 — Por bloque**: selecciona un CSV de pruebas y guarda los resultados en el destino indicado.

Formato del CSV para modo bloque (`GeoMIP/data/pruebas_ejemplo.csv`):
```
#prueba,alcance,mecanismo,k
1,1111111111,1111111111,
2,1110000000,1111111111,2
```
El candidato y el estado inicial se piden una sola vez para todo el lote.
Los resultados se guardan como JSON en la ruta que el usuario indique.
La partición se almacena con su formato de dos líneas (futuros/presentes).
El tiempo se reporta en s / min s / h min s según su magnitud.

**Ejecutar QNodes con profiling activado (según su config global):**
```bash
source .venv/Scripts/activate
python QNodes/exec.py
```

---

## Diferencias Clave de k-MIP (vs Prototipo Antiguo)

AYDA reimplementa estructuralmente la búsqueda y evaluación de k-particiones óptimas integrando rigor topológico-probabilístico sobre las heurísticas originales.

| Criterio | `projecto-analisis-20261` (Antiguo) | `AYDA_2026_1` (Nuevo) |
|----------|---------------------------------------|-------------------------|
| **Cálculo EMD (Métrica)** | Distancia simple tipo L1 (`emd_efecto`) sumando las diferencias aisladas de variables de causa/efecto marginales. Generaba valores artificialmente bajos $φ$. | Verdadera Earth Mover's Distance (`emd_causal`) interlazada mediante PyEMD sobre el espacio probabilístico conjunto con base de Distancia de Hamming. |
| **Puntaje $φ$ Reportado** | Valores reducidos artificialmente (p.ej. $0.4$) al omitir interconexión causal cruzada de los sub-estados. | Magnitudes matemáticamente realistas sobre el espacio matricial ($φ > 1.0$) según los cortes iterativos reales. |
| **Búsqueda Óptima** | "Hill-Climbing" simple con selección estocástica ciega inicial. | Generación de Matrices de Afinidad Geométrica (*Spectral Clustering*) + Refinamiento Local iterativo (1-move). |
| **Manejo del "Estado Vacío"** | Los nodos quedaban causalmente destrozados en estados vacíos (Over-cutting) sin penalización alta. | Causalidad topológica conservada; el mecanismo ∅ se permite pero evalúa contra su peso probabilístico de Hamming rigurosamente (`generar_candidatos_presente_vacio`). |
| **Arquitectura Algorítmica** | Implementación imperativa genérica para bipartir iterativamente. | Modelado OOP (`System`, `NCube`), memorización de sub-distribuciones (QNodes `DynamicPartition`), paralelismo con `ProcessPoolExecutor` y control K-múltiple por hilos locales de CPU. |

---

## Convenciones de Desarrollo
- **Idioma:** Código, comentarios y nombres de variable están en **español**.
- **Variables Futuras/Presentes:** t+1 (efecto) van en MAYÚSCULAS; t (causa/mecanismo) en minúsculas.
- **Indexación:** *Little-Endian* por defecto (bit menos significativo = primero nodo).
- **Lectura Lazy:** GeoMIP usa un `LazyTPM` para leer por *chunks* archivos CSV sin colapsar memoria cuando $N > 18$.
- **Logs y Rendimiento:** Se utilizan `pynstrument` y middlewares para logs ordenados (`.logs/`). En `QNodes/exec.py` se puede configurar `aplicacion.activar_profiling()`.
