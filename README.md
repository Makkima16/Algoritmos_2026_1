# AYDA 2026-1 — Partición de Mínima Información

Repositorio del proyecto académico para la asignatura **Análisis y Diseño de Algoritmos (AYDA) 2026-1**.

Contiene dos frameworks independientes para resolver el problema de la **k-Partición de Mínima Información (k-MIP)**, concepto central de la Teoría de Información Integrada (IIT): encontrar la división de un sistema dinámico en $k$ subconjuntos disjuntos que minimice la pérdida de información integrada **Phi (Φ)**.

---

## Índice

1. [Conceptos de dominio](#conceptos-de-dominio)
2. [Frameworks del proyecto](#frameworks-del-proyecto)
   - [KGeoMIP](#kgeomip)
   - [QNodes](#qnodes)
3. [Comparación entre frameworks](#comparación-entre-frameworks)
4. [Estructura de carpetas](#estructura-de-carpetas)
5. [Requisitos previos](#requisitos-previos)
6. [Instalación paso a paso](#instalación-paso-a-paso)
7. [Uso — KGeoMIP](#uso--kgeomip)
8. [Uso — QNodes](#uso--qnodes)
9. [Dashboard (GUI)](#dashboard-gui)
10. [Convenciones de desarrollo](#convenciones-de-desarrollo)

---

## Conceptos de dominio

| Término | Descripción |
|---------|-------------|
| **TPM** | *Transition Probability Matrix*. Matriz estocástica de $2^N$ filas × $N$ columnas que describe la dinámica de probabilidad de transición de un sistema de $N$ nodos binarios. |
| **Estado inicial** | Secuencia binaria que representa el estado del sistema en el tiempo $t$. Ejemplo: `"1000000000"`. |
| **Sistema candidato** | Máscara binaria que indica qué nodos pertenecen al subsistema de análisis; los bits en `0` son condiciones de fondo que se marginalizan. |
| **Mecanismo ($t$) / Alcance ($t+1$)** | El **mecanismo** selecciona los nodos del presente (causa); el **alcance** selecciona los nodos del futuro (efecto). |
| **Marginalización** | Operación que "elimina" un nodo del hipercubo de probabilidades sumando su distribución sobre sus estados posibles (integración). |
| **EMD** | *Earth Mover's Distance* (Wasserstein-1). Métrica de transporte óptimo entre dos distribuciones de probabilidad usando la distancia de Hamming como costo base. Es la métrica correcta de IIT para medir pérdida de información. |
| **Phi (Φ)** | Información Integrada. Valor numérico que cuantifica cuánta información se pierde al particionar el sistema. La **k-MIP** es la partición que minimiza Φ. |
| **k-partición** | División de los $N$ nodos en $k$ subconjuntos no vacíos y disjuntos. El número de posibilidades para un $k$ fijo son los números de Stirling de segunda especie $S(N, k)$; el total sobre todos los $k$ es el número de Bell $B(N)$. |

---

## Frameworks del proyecto

### KGeoMIP

**Ubicación:** `KGeoMIP/`

KGeoMIP aborda el problema k-MIP mediante una **estrategia heurística geométrica** combinada con **paralelismo por hilos**. Su arquitectura está pensada para sistemas de tamaño medio-grande ($N \geq 15$) donde la búsqueda exhaustiva es inviable.

#### Algoritmo

El flujo principal corre en `KGeoMIP.aplicar_estrategia()`:

```
1. Extraer subsistema candidato y condicionar la TPM al estado inicial.
2. Si N ≤ 6: resolver por enumeración exhaustiva (_resolver_exacto_geomip).
3. Si no:
   a. Construir la tabla de costos (2^n_dims × N), con kernel Numba opcional.
   b. Construir el cut pool: aislamientos simétricos, mecanismo vacío y el
      mejor representante por nivel de Hamming.
   c. Para cada k ∈ [2, min(5, N)] (secuencial):
      · _greedy_k_particion  → k-1 divisiones óptimas del cut pool.
      · _refinar_bloques_1move → best-improvement futuro + presente.
      · _candidatos_aislamiento_bloques → siembra acotada por CAP_AISLAMIENTO_BLOQUES.
   d. Elegir el k con menor Φ global.
```

> **Nota:** VNS (2-move) e ILS ligero existen en el código pero están **desactivados**
> (`N_VNS_MAX = 0`, `N_ILS_LIGHT = 0`): se midieron y no mejoran Φ, con un coste de
> 3–5×. Detalle en [`KGeoMIP/docs/decision_sin_ils.md`](KGeoMIP/docs/decision_sin_ils.md).

#### Características principales

- **EMD exacta vía L1 marginal:** `evaluar_bloques` compara las marginales por nodo y suma su diferencia L1. Para distribuciones producto con métrica de Hamming, esa suma es exactamente la Wasserstein-1, con coste O(N) en vez de O(4^N).
- **Paralelismo:** `joblib` con `prefer="threads"` (`cpu_count − 1` hilos) evalúa candidatos y vecinos; los valores de $k$ se recorren secuencialmente.
- **LazyTPM:** Para $N \geq 18$, la TPM no se carga completa en RAM. Un generador lee por *chunks* y calcula las marginales incrementalmente, permitiendo sistemas de hasta $N = 25+$.
- **Modo bloque CSV:** Carga lotes de pruebas, ordena por complejidad creciente, calienta el motor antes de la primera prueba y exporta resultados a `.xlsx`.
- **Mecanismo vacío (∅):** Soporte explícito para partes con mecanismo vacío mediante un centinela `-1`.

#### Arquitectura OOP

| Clase / Módulo | Responsabilidad |
|----------------|-----------------|
| `System` | Condicionamiento, particionamiento, marginales del subsistema |
| `NCube` | Hipercubo de probabilidad por nodo; memoización de marginalizaciones con `frozenset` |
| `KGeoMIP` | Algoritmo completo de búsqueda k-MIP |
| `Manager` | Carga de TPM, enrutamiento de estrategias |
| `Solution` | Representación y visualización del resultado |
| `LazyTPM` | Lectura perezosa de TPM por *chunks* para $N \geq 18$ |
| `SafeLogger` | Logging thread-safe con archivos por fecha |

---

### QNodes

**Ubicación:** `KQNodes/`

QNodes aborda el problema k-MIP mediante **cortes asimétricos** (futuro y presente se particionan de forma independiente), **Queyranne** para la bipartición exacta $k=2$, **greedy top-down**, refinamiento local e **ILS**, con búsqueda libre sobre todos los valores de $k$ si el usuario no fija uno.

#### Algoritmo

El flujo principal corre en `QNodes.aplicar_estrategia()`:

```
1. Preparar subsistema: condicionar la TPM, seleccionar alcance y mecanismo.
2. Si N ≤ 6: _resolver_exacto() por enumeración de k-particiones asimétricas.
3. Si no, _resolver_heuristico():
   a. _queyranne() sobre 2N átomos asimétricos (o N simétricos si no se permite ∅):
      obtiene la bipartición k=2 globalmente óptima y un pool de cortes colgantes.
   b. Combinar ese pool con _construir_pool_cortes() (aislamientos + mecanismo ∅).
   c. k fijo: _greedy_bloques() + _refinar_bloques() + _refinar_con_ils().
   d. k libre: _greedy_descenso() registra cada nivel k, se refina por nivel y se
      elige el k ≥ 3 con menor Φ; después se aplica ILS al ganador.
4. Reconstruir la distribución marginal de la partición y devolver Solution.
```

#### Características principales

- **Métrica exacta en O(N):** `_emd_bloques` reconstruye las marginales y suma su diferencia L1. Para distribuciones producto con métrica de Hamming, esa L1 es exactamente la Wasserstein-1, sin el coste O(4^N) de `pyemd`.
- **k=2 exacto vía Queyranne:** con átomos asimétricos, el algoritmo de Queyranne garantiza la bipartición óptima global en O(N²) evaluaciones.
- **Búsqueda libre sobre k:** si no se especifica $k$, se evalúan todos los niveles del descenso greedy y se elige el de menor Φ (prefiriendo $k \geq 3$).
- **Memoización de bloques:** `_cache_bloque[(futuros, presentes)]` guarda la distribución marginal de cada bloque; `NCube.marginal_valor` reduce cada evaluación a O(2^(N−|pre|)).
- **Representación asimétrica:** cada bloque es `(frozenset futuros, frozenset presentes)`, lo que permite movimientos de presente sin mover el futuro correspondiente.

#### Complejidad

| Fase | Complejidad | Condición |
|------|-------------|-----------|
| Queyranne (k=2 exacto) | O((2N)²) evaluaciones | Siempre |
| Greedy top-down | O(N) divisiones sobre el pool | Siempre |
| Refinamiento 1-move | O(N × k × iteraciones) | Siempre |
| ILS | O(N_ILS × N) evaluaciones | Siempre |
| Enumeración exacta | Stirling2(N,k) × k^n_dims | N ≤ 6 |
| Una evaluación `_emd_bloques` | O(N) — suma L1 marginal | Siempre |

Para $N = 10$ la ruta exacta cubre todo el espacio tratable; para $N = 20$–$25$ se usa la heurística (Queyranne + greedy + ILS) en segundos o decenas de segundos según $k$.

---

## Comparación entre frameworks

| Criterio | KGeoMIP | QNodes |
|----------|---------|--------|
| **Estrategia principal** | Greedy top-down + cut pool geométrico + 1-move | Queyranne k=2 + greedy top-down + 1-move + ILS |
| **k de entrada** | Requerido (o rango completo en bloque) | Opcional — búsqueda libre sobre todos los k |
| **Métrica EMD** | L1 marginal exacta (Wasserstein-1 con Hamming) | L1 marginal exacta (Wasserstein-1 con Hamming) |
| **Paralelismo** | joblib threads (candidatos y vecinos) | Single-thread; eficiencia por memoización |
| **Escalabilidad** | N ≤ 25 con LazyTPM | N ≤ 25 con `NCube.marginal_valor` |
| **Garantía de optimalidad** | Exacto N ≤ 6; óptimo local (greedy + 1-move) | Exacto N ≤ 6 y k=2 exacto vía Queyranne; óptimo local para k≥3 |
| **Modo de entrada** | Manual / bloque CSV | Manual / bloque CSV |
| **Salida** | JSON (manual) / Excel (bloque) | JSON (manual) / Excel (bloque) |
| **Mecanismo vacío (∅)** | Soportado | Soportado (`permitir_presente_vacio`) |

---

## Estructura de carpetas

```
AYDA_2026_1/
├── README.md, CLAUDE.md               ← documentación e instrucciones del proyecto
├── requirements.txt                   ← dependencias Python (todas las carpetas)
├── setup.sh / setup.ps1               ← creación del .venv e instalación
├── .venv/                             ← entorno virtual Python (no se sube al repo)
│
├── data/                              ← datos y scripts COMPARTIDOS
│   ├── creation.py                    ← generación de TPMs
│   ├── run_suite_2026.py              ← suite comparativa KGeoMIP vs KQNodes
│   ├── _worker_motor.py               ← worker aislado por motor (para la suite)
│   ├── compare_pruebas.py             ← benchmark y gráficas (matplotlib)
│   ├── graficos_suite.py              ← gráficas nativas de Excel
│   ├── generar_graficos_cli.py        ← CLI de gráficas
│   ├── samples_binary/                ← TPMs binarias (N=3..25; no versionadas)
│   ├── samples_no_binary/             ← TPMs no binarias (no versionadas)
│   └── Pruebas/                       ← CSVs de lotes de prueba
│
├── KGeoMIP/
│   ├── pyproject.toml                 ← dependencias de KGeoMIP
│   ├── exec_kgeomip.py                ← entrypoint interactivo principal
│   ├── docs/                          ← documentación técnica
│   ├── tests/                         ← pytest sobre la tabla de costos
│   ├── results/                       ← JSONs y Excel con resultados (no versionados)
│   └── src/
│       ├── controllers/
│       │   ├── manager.py             ← carga TPM, enruta estrategias
│       │   └── strategies/kgeomip.py  ← algoritmo KGeoMIP completo
│       ├── models/
│       │   ├── base/                  ← application.py, sia.py (caché del candidato)
│       │   ├── core/                  ← system.py, ncube.py, solution.py, lazy_tpm.py
│       │   └── enums/                 ← distance.py, notation.py
│       ├── funcs/base.py              ← ABECEDARY, emd_causal, reindexado
│       ├── constants/                 ← base.py, error.py, models.py
│       └── middlewares/               ← slogger.py, profile.py
│
├── KQNodes/
│   ├── pyproject.toml                 ← dependencias de KQNodes
│   ├── exec.py                        ← entrypoint interactivo principal
│   ├── docs/                          ← documentación técnica
│   └── src/
│       ├── strategies/q_nodes.py      ← algoritmo QNodes (Queyranne + greedy + ILS)
│       ├── strategies/force.py        ← BruteForceKMIP (ground truth)
│       ├── funcs/
│       │   ├── iit.py                 ← reindexado, Hamming y utilidades
│       │   ├── force.py               ← generación de k-particiones (Stirling)
│       │   └── format.py              ← formateo de particiones
│       ├── models/                    ← System, NCube, Solution, SIA
│       ├── constants/                 ← base.py, error.py, models.py
│       └── middlewares/               ← slogger.py, profile.py
│
├── Brute_Force/
│   ├── comparar_fuerza_bruta.py       ← BF vs KQNodes vs KGeoMIP (N ≤ 6)
│   └── samples_force/                 ← TPMs pequeñas para la comparación
│
├── dashboard/                         ← GUI React + FastAPI
│   ├── server/                        ← backend (main.py, workers.py, metrics.py…)
│   └── src/                           ← frontend (App.jsx, views/, components/)
│
└── docs/                              ← PDFs, manuales y material de la asignatura
```

---

## Requisitos previos

- **Python 3.11 o superior**
- **Git**
- Sistema operativo: Windows 10/11, Linux o macOS

Verificar la versión de Python instalada:

```bash
python --version
```

---

## Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd AYDA_2026_1
```

### 2. Crear el entorno virtual

El `.venv` **no se incluye en el repositorio** (está en `.gitignore`). Cada colaborador debe crearlo localmente:

```bash
python -m venv .venv
```

### 3. Activar el entorno virtual

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS / Git Bash:**
```bash
source .venv/Scripts/activate
# o, en Linux/macOS:
source .venv/bin/activate
```

El prompt de la terminal mostrará `(.venv)` al estar activo.

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

También puedes instalar cada framework en modo *editable* si vas a modificar su
código (`pip install -e KGeoMIP/` y `pip install -e KQNodes/`). Las dependencias
principales son numpy, pandas, pyemd, joblib, numba, openpyxl, matplotlib,
colorama, pyttsx3, pyinstrument y, para el dashboard, fastapi/uvicorn/pydantic.

### 5. Verificar la instalación

```bash
python -c "import numpy, pandas, pyemd, openpyxl; print('Todo OK')"
```

---

## Uso — KGeoMIP

```bash
# Con el .venv activo:
python KGeoMIP/exec_kgeomip.py
```

Al ejecutar, el programa ofrece dos modos:

### Modo 1 — Manual

Ingreso interactivo de una prueba individual:

1. Seleccionar tipo de TPM (binaria / no binaria) y archivo.
2. Ingresar sistema candidato en binario (`1111111111` = sistema completo).
3. Ingresar estado inicial en binario (o `ENTER` para uno aleatorio).
4. Ingresar alcance (t+1) y mecanismo (t) como etiquetas de nodos (`ABCDE`, `ABC`, etc.).
5. Ingresar $K$ (o `ENTER` para evaluar $k = 2, \ldots, \min(5, N)$).
6. Elegir si se permite mecanismo vacío (∅).

### Modo 2 — Por bloque (CSV)

Procesa un lote de pruebas desde un archivo CSV:

```
#Prueba,Alcance o Purview (t+1),Mecanismo(t)
1,ABCDEFGHIJ,ABCDEFGHIJ
2,ABCDE,ABCDE
3,ABCDE,ABCD
```

- El candidato y el estado inicial se ingresan una sola vez para todo el lote.
- Las pruebas se reordenan automáticamente de menor a mayor complejidad.
- Los resultados se exportan a un archivo `.xlsx` en `KGeoMIP/results/block/`.

Los CSVs de prueba se encuentran en `data/Pruebas/`.

---

## Uso — QNodes

```bash
# Con el .venv activo:
python KQNodes/exec.py
```

El flujo es análogo al de KGeoMIP:

1. Seleccionar tipo de TPM y archivo.
2. Ingresar sistema candidato, estado inicial, alcance y mecanismo.
3. Ingresar $K$ (o `ENTER` para búsqueda libre sobre todos los k ≥ 3).
4. Elegir modo de ejecución (manual o bloque CSV).

**Activar profiling** (genera reporte HTML con pyinstrument):

```python
# En KQNodes/exec.py, antes de ejecutar:
aplicacion.activar_profiling()
```

---

## Dashboard (GUI)

Interfaz gráfica (React + Vite + FastAPI) en `dashboard/` para operar y analizar ambos
frameworks desde el navegador. Mantiene un worker persistente por motor (aislamiento del
paquete `src`) y permite ejecutar por bloque (streaming SSE), explorar resultados guardados
y comparar QNodes vs KGeoMIP con métricas. Detalles y endpoints en
[`dashboard/README.md`](dashboard/README.md).

```bash
# Backend (desde la raíz, con el .venv activo):
.venv/Scripts/python.exe -m uvicorn main:app --app-dir dashboard/server --port 8000
# Frontend (desde dashboard/):
npm install && npm run dev      # http://localhost:5173
```

Dos detalles del **modo análisis** que conviene conocer:

- **Arranque del motor (warmup):** antes de cada lote se ejecuta una corrida de
  calentamiento descartable (con los parámetros de la primera prueba válida) para pagar
  los costos de una sola vez del motor. Así la **primera prueba ya no incluye el arranque**
  en su tiempo guardado; el arranque se reporta aparte (fila *Arranque motor (warmup)*).
- **Ganador "Ambos":** en la comparación por bloque, si ambos motores arrojan la misma
  pérdida Φ, la columna *Mejor* muestra **"Ambos"** en lugar de adjudicar el empate a uno.

---

## Convenciones de desarrollo

| Convención | Descripción |
|------------|-------------|
| **Idioma** | Código, comentarios y nombres de variable en **español** |
| **Nodos futuros/presentes** | Variables de t+1 (efecto/alcance) en MAYÚSCULAS; variables de t (causa/mecanismo) en minúsculas |
| **Indexación** | *Little-Endian* — el bit menos significativo corresponde al primer nodo (nodo A = bit 0) |
| **Representación de particiones** | Bloques asimétricos `(frozenset futuros, frozenset presentes)` de índices globales |
| **Entorno virtual** | `.venv/` en la raíz del repo, compartido por KGeoMIP y KQNodes, nunca comiteado |
| **Logs** | Archivos `.log` en `.logs/`, ignorados por git |
