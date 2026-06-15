# CLAUDE.md — KGeoMIP & KQNodes / AYDA 2026-1

## Descripción del Proyecto

El repositorio contiene dos frameworks académicos para resolver el problema de la **Partición de Mínima Información (MIP)**, concepto central en la Teoría de Información Integrada (IIT). El proyecto es desarrollado para la asignatura *Análisis y Diseño de Algoritmos (AYDA) 2026-1*.

El objetivo principal de ambos frameworks es encontrar la partición óptima de un sistema lógico o neurobiológico que minimice la pérdida de información integrada (Phi, Φ), empleando la **Distancia del Transportador de Tierra (EMD)** para medir la pérdida.

Existen dos enfoques principales en el repositorio:
1. **KGeoMIP:** Framework original interactivo (carpeta `KGeoMIP/`) que aborda el problema k-MIP paralelizando los cálculos de múltiples tamaños de partición $k$ e implementando heurísticas combinatorias (ej. clustering jerárquico bottom-up) para reducir la complejidad cuando las dimensiones son muy altas.
2. **KQNodes:** Nuevo framework (`KQNodes/`) centrado en **agrupamiento jerárquico aglomerativo greedy con memoización** (`QNodes` / alias `DynamicPartition`). Aplica tres fases para todo N sin excepciones: (1) agrupamiento greedy O(N³) que construye un historial completo de k-particiones, (2) refinamiento local 1-move hasta convergencia, y (3) evaluación exhaustiva de candidatos de aislamiento C(N, k-1). Para k libre, las tres fases se ejecutan sobre cada nivel k del historial y se elige el k con menor Φ global. La métrica interna de `_emd_particion` usa Wasserstein-1 con d_Hamming para N ≤ 12 y suma L1 para N > 12 (detalle de implementación transparente para la estrategia).

---

## Estructura de Carpetas

```
AYDA_2026_1/
├── CLAUDE.md                          <- este archivo
├── .venv/                             <- entorno virtual Python (compartido)
├── data/                              <- Muestras de TPMs y scripts COMPARTIDOS (creation.py,
│   │                                     run_suite_2026.py, _worker_motor.py, samples_binary/, Pruebas/)
├── results_test/                      <- Copias fechadas de DatosPruebas2026_1.xlsx con resultados del suite
├── KGeoMIP/
│   ├── results/                       <- JSONs/XLSX con resultados de KGeoMIP
│   ├── exec_kgeomip.py                <- Entrypoint interactivo principal para KGeoMIP
│   └── src/                           <- Modelos (System, NCube), Controladores (KGeoMIP), etc.
└── KQNodes/
    ├── exec.py                        <- Entrypoint principal para KQNodes
    ├── results/                       <- JSONs/XLSX con resultados de KQNodes
    ├── pyproject.toml                 <- Dependencias de KQNodes
    ├── review/                        <- Scripts de testeo y profiling de KQNodes
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

### KGeoMIP Pipelíne (`exec_kgeomip.py`)
```
exec_kgeomip.py (modo bloque)
  ├── warmup_motor()                  ← Numba JIT (carga binario @njit o compila)
  │                                      + pool de hilos joblib; arrays mínimos 2 nodos.
  └── KGeoMIP.aplicar_estrategia(condicion, alcance, mecanismo, tpm, k)
        ├── 1. sia_preparar_subsistema  → condicionar TPM al estado inicial.
        ├── 2. _construir_tabla_costos  → tabla (2^n_dims × N) por recurrencia de
        │                                 capas de Hamming; kernel @njit si Numba
        │                                 disponible, numpy vectorizado si no.
        ├── 3. _construir_cut_pool      → O(N) cortes asimétricos + Hamming.
        └── Para cada k ∈ {2..min(5,N)} (secuencial, joblib threads interno):
              ├── _greedy_k_particion   → top-down: k-1 divisiones del pool.
              └── _refinar_bloques_1move → best-improvement futuro + presente [fase final].
              # VNS (_refinar_bloques_2move) e ILS ligero (_perturbacion_bloques) existen
              # como código pero están DESACTIVADOS (N_VNS_MAX = 0, N_ILS_LIGHT = 0):
              # ninguno mejora Φ y cuestan 3–5×. Ver KGeoMIP/docs/decision_sin_ils.md.
```

### KQNodes Pipelíne (`exec.py`)
```
exec.py
  └── KQNodes.aplicar_estrategia(estado, condicion, alcance, mecanismo, k)
        ├── 1. Preparar subsistema (condicionar TPM al estado inicial).
        ├── 2. _aglomerar() — jerarquía greedy O(N³); retorna historico{k: (phi, grupos)}.
        ├── 3. Para cada k evaluado (el dado o todos si k=None):
        │       _refinamiento_local() — 1-move hasta convergencia.
        │       _candidatos_aislamiento(k) — C(N, k-1) candidatos con nodos aislados.
        │       _refinamiento_local() nuevamente si mejora.
        └── 4. Retornar k-partición con Phi mínimo (k ≥ 3 preferido si k=None).
```

---

## Comandos de Uso

Todo el código depende del mismo entorno virtual (`.venv/`) con Python >= 3.11.

**Ejecutar KGeoMIP (elige modo al inicio):**
```bash
source .venv/Scripts/activate
python KGeoMIP/exec_kgeomip.py
```
Las TPMs y CSV de pruebas se leen de la carpeta `data/` de la raíz (compartida);
los resultados se guardan en `KGeoMIP/results/`. El modo bloque/manual registra
además el tiempo de "arranque del motor" (warmup) aparte del tiempo de las pruebas.
Desde 2026-06-14 ese "arranque" es solo el coste **único** (Numba JIT + pool +
condicionado del candidato); la preparación del subsistema **por prueba** (substraer
con su alcance/mecanismo) se incluye en el tiempo de la prueba, no en el warmup.
KQNodes hace lo mismo, pero al no tener caché ni warmup separado su fila de arranque
queda en 0 (cada prueba reconstruye su subsistema y ese coste va en su propio tiempo).

Al ejecutar se ofrece:
- **Modo 1 — Manual**: ingreso interactivo de candidato, estado, alcance, mecanismo y K.
- **Modo 2 — Por bloque**: selecciona un CSV de pruebas y guarda los resultados en el destino indicado.

Formato del CSV para modo bloque (`data/Pruebas/Pruebas_N10.csv`):
```
#prueba,alcance,mecanismo,k
1,1111111111,1111111111,
2,1110000000,1111111111,2
```
El candidato y el estado inicial se piden una sola vez para todo el lote.
Los resultados se guardan como JSON en la ruta que el usuario indique.
La partición se almacena con su formato de dos líneas (futuros/presentes).
El tiempo se reporta en s / min s / h min s según su magnitud.

**Ejecutar KQNodes con profiling activado (según su config global):**
```bash
source .venv/Scripts/activate
python KQNodes/exec.py
```

**Ejecutar la suite comparativa (KGeoMIP vs KQNodes, k=3,4,5):**
```bash
source .venv/Scripts/activate
python data/run_suite_2026.py            # copia DatosPruebas2026_1.xlsx a
                                         # results_test/DatosPruebas2026_1_<fecha>.xlsx
# Opciones: --solo-n 10,15  --solo-k 3,4  --solo-motor qnodes  --no-vacio
```
Por defecto usa `permitir_presente_vacio=True` (pérdida mínima real; con `--no-vacio`
las pérdidas de KGeoMIP se inflan por sobre-corte). Por cada N/motor/k escribe, debajo
de las pruebas, el Σ del tiempo de búsqueda y —justo debajo— el tiempo de arranque
del motor (warmup). El original NO se modifica.

**Dashboard GUI (`dashboard/`, React + FastAPI):**
```bash
.venv/Scripts/python.exe -m uvicorn main:app --app-dir dashboard/server --port 8000
cd dashboard && npm run dev      # frontend en :5173
```
Mantiene un worker persistente por motor (aislamiento de `src`) y ejecuta corridas
manuales o por bloque (SSE). Dos comportamientos a tener presentes en el modo análisis:
- **Arranque del motor (warmup):** `/api/block` ejecuta una corrida de calentamiento
  DESCARTABLE con los parámetros de la primera prueba válida ANTES del lote, y la
  contabiliza aparte como "arranque del motor" (fila *Arranque motor (warmup)* del XLSX,
  campo `tiempo_arranque` del SSE). El "Tiempo Total Lote" excluye el warmup.
  Desde 2026-06-14 el tiempo guardado de cada prueba es **preparación de su subsistema +
  búsqueda** (costo real por prueba), no solo la búsqueda: la preparación es trabajo por
  prueba (varía con alcance/mecanismo) y el "arranque" queda solo con el warmup único.
  `worker_runner` normaliza `tiempo_total_segundos` = prep+búsqueda para AMBOS motores
  (antes `tiempo_ejecucion` incluía la prep en KGeoMIP pero no en KQNodes); el SSE de
  bloque reporta el Σ como `tiempo_pruebas`/`tiempo_pruebas_fmt`.
- **Ganador "Ambos":** en la comparación por bloque (vista *Análisis*), si KQNodes y
  KGeoMIP arrojan la misma pérdida Φ, la columna *Mejor* muestra **"Ambos"** (antes el
  empate se adjudicaba a KQNodes por el `<=`). Detalle en `dashboard/README.md`.

---

## Diferencias Clave de k-MIP (vs Prototipo Antiguo)

AYDA reimplementa estructuralmente la búsqueda y evaluación de k-particiones óptimas integrando rigor topológico-probabilístico sobre las heurísticas originales.

| Criterio | `projecto-analisis-20261` (Antiguo) | `AYDA_2026_1` (Nuevo) |
|----------|---------------------------------------|-------------------------|
| **Cálculo EMD (Métrica)** | Distancia simple tipo L1 (`emd_efecto`) sumando las diferencias aisladas de variables de causa/efecto marginales. Generaba valores artificialmente bajos $φ$. | Verdadera Earth Mover's Distance (`emd_causal`) interlazada mediante PyEMD sobre el espacio probabilístico conjunto con base de Distancia de Hamming. |
| **Puntaje $φ$ Reportado** | Valores reducidos artificialmente (p.ej. $0.4$) al omitir interconexión causal cruzada de los sub-estados. | Magnitudes matemáticamente realistas sobre el espacio matricial ($φ > 1.0$) según los cortes iterativos reales. |
| **Búsqueda Óptima** | "Hill-Climbing" simple con selección estocástica ciega inicial. | Greedy top-down asimétrico sobre un *cut pool* geométrico derivado de la tabla de costos + Refinamiento Local iterativo (1-move). *(El camino de Spectral Clustering sobre matriz de afinidad quedó como código legado sin uso; su construcción se eliminó del arranque en 2026-06-14 — ver `KGeoMIP/docs/GeoMIP_Optimizaciones.md` §11.)* |
| **Manejo del "Estado Vacío"** | Los nodos quedaban causalmente destrozados en estados vacíos (Over-cutting) sin penalización alta. | Causalidad topológica conservada; el mecanismo ∅ se permite pero evalúa contra su peso probabilístico de Hamming rigurosamente (`generar_candidatos_presente_vacio`). |
| **Arquitectura Algorítmica** | Implementación imperativa genérica para bipartir iterativamente. | Modelado OOP (`System`, `NCube`), memorización de sub-distribuciones (KQNodes `DynamicPartition`), paralelismo con `ProcessPoolExecutor` y control K-múltiple por hilos locales de CPU. |

---

## Convenciones de Desarrollo
- **Idioma:** Código, comentarios y nombres de variable están en **español**.
- **Variables Futuras/Presentes:** t+1 (efecto) van en MAYÚSCULAS; t (causa/mecanismo) en minúsculas.
- **Indexación:** *Little-Endian* por defecto (bit menos significativo = primero nodo).
- **Lectura Lazy:** KGeoMIP usa un `LazyTPM` para leer por *chunks* archivos CSV sin colapsar memoria cuando $N > 18$.
- **Logs y Rendimiento:** Se utilizan `pynstrument` y middlewares para logs ordenados (`.logs/`). En `KQNodes/exec.py` se puede configurar `aplicacion.activar_profiling()`.
