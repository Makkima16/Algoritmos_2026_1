# AYDA 2026-1 — Partición de Mínima Información

Repositorio del proyecto académico para la asignatura **Análisis y Diseño de Algoritmos (AYDA) 2026-1**.

Contiene dos frameworks independientes para resolver el problema de la **k-Partición de Mínima Información (k-MIP)**, concepto central de la Teoría de Información Integrada (IIT): encontrar la división de un sistema dinámico en $k$ subconjuntos disjuntos que minimice la pérdida de información integrada **Phi (Φ)**.

---

## Índice

1. [Conceptos de dominio](#conceptos-de-dominio)
2. [Frameworks del proyecto](#frameworks-del-proyecto)
   - [GeoMIP](#geomip)
   - [QNodes](#qnodes)
3. [Comparación entre frameworks](#comparación-entre-frameworks)
4. [Estructura de carpetas](#estructura-de-carpetas)
5. [Requisitos previos](#requisitos-previos)
6. [Instalación paso a paso](#instalación-paso-a-paso)
7. [Uso — GeoMIP](#uso--geomip)
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

### GeoMIP

**Ubicación:** `GeoMIP/`

GeoMIP aborda el problema k-MIP mediante una **estrategia heurística geométrica** combinada con **paralelismo por CPU**. Su arquitectura está pensada para sistemas de tamaño medio-grande ($N \geq 15$) donde la búsqueda exhaustiva es inviable.

#### Algoritmo

El flujo principal corre en `KGeoMIP.aplicar_estrategia()`:

```
1. Extraer subsistema candidato y condicionar la TPM al estado inicial.
2. Para cada k ∈ [2, K_máximo] (en paralelo, un proceso por k):
   a. Generar candidatos iniciales:
      · Spectral Clustering sobre matriz de afinidad geométrica (similitud coseno).
      · Agglomerative Clustering (variantes: average, complete, single).
      · Candidatos de aislamiento: C(N, k-1) particiones con nodos individuales.
   b. Refinar cada candidato con búsqueda local 1-move hasta convergencia.
   c. Aplicar Iterated Local Search (ILS): perturbar + refinar × 4.
3. Retornar la k-partición con menor Φ global.
```

#### Características principales

- **EMD real con Hamming:** `emd_causal` calcula la verdadera Wasserstein-1 sobre el espacio conjunto de $2^N$ estados con distancia de Hamming como métrica base. Produce valores de Φ matemáticamente rigurosos (típicamente > 1.0).
- **Paralelismo:** `ProcessPoolExecutor` con `cpu_count − 1` núcleos; cada proceso evalúa un valor de $k$ independiente.
- **LazyTPM:** Para $N \geq 18$, la TPM no se carga completa en RAM. Un generador lee por *chunks* y calcula las marginales incrementalmente, permitiendo sistemas de hasta $N = 25+$.
- **Modo bloque CSV:** Carga lotes de pruebas, ordena por complejidad creciente, calienta caché antes de la primera prueba y exporta resultados a `.xlsx` con formato profesional.
- **Mecanismo vacío (∅):** Soporte explícito para partes con mecanismo vacío mediante un centinela `-1`, produciendo distribuciones uniformes sin penalización artificial.

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

**Ubicación:** `QNodes/`

QNodes aborda el problema k-MIP mediante **agrupamiento aglomerativo greedy** (bottom-up) con **memoización de distribuciones** y búsqueda libre sobre todos los valores posibles de $k$, sin necesidad de especificar $k$ de antemano.

#### Algoritmo

El flujo principal corre en `QNodes.aplicar_estrategia()`:

```
1. Preparar subsistema: condicionar la TPM, seleccionar alcance y mecanismo.
2. _aglomerar() — O(N³):
   · Inicializar N singletons (un grupo por nodo).
   · Pre-poblar caché de distribuciones y costos.
   · Repetir N-2 veces:
       N ≤ 12 → evaluar EMD Hamming de la partición completa para cada par.
       N > 12 → evaluar Δ = costo(Gᵢ∪Gⱼ) − costo(Gᵢ) − costo(Gⱼ)  (L1 aditivo).
       Fusionar el par con menor incremento de Φ.
       Registrar historial[k] = (phi, grupos) para cada nivel k.
   · Retornar historial completo {k: (phi, grupos)} para k ∈ [2, N].
3. Para k especificado:
   · Refinar historial[k] con 1-move hasta convergencia.
   · Evaluar C(N, k-1) candidatos de aislamiento [solo N ≤ 12].
4. Para k libre (k=None):
   · N ≤ 12: refinar + evaluar candidatos de aislamiento para CADA nivel k.
     Seleccionar el k ∈ [3, N] con menor Φ global entre todos los niveles.
   · N > 12: seleccionar el k ≥ 3 con menor Φ del historial greedy + refinar una vez.
5. Retornar la partición óptima con su Φ y k.
```

#### Características principales

- **Estrategia uniforme para todo N:** Las tres fases (agrupamiento greedy, refinamiento 1-move, candidatos de aislamiento) se aplican para cualquier tamaño de sistema, sin excepciones por N.
- **Búsqueda libre sobre k:** A diferencia de GeoMIP, QNodes puede encontrar la k óptima global sin que el usuario la especifique. En modo k libre, las tres fases se ejecutan sobre cada nivel k del historial y se elige el k con menor Φ global.
- **Memoización de distribuciones:** `_cache_dist` almacena la distribución marginal de cada subconjunto (representado como máscara entera). `_emd_particion` obtiene distribuciones en O(1) para máscaras ya vistas, amortizando el costo de las tres fases.
- **Representación por máscaras enteras:** Cada subconjunto de nodos es un entero de $N$ bits. Las operaciones de unión (`|`), diferencia (`^`) y extracción de bits (`& -m`) son operaciones nativas de hardware.
- **Métrica adaptada en `_emd_particion`:** Para $N \leq 12$ usa Wasserstein-1 con distancia de Hamming (métrica exacta de IIT, misma que GeoMIP). Para $N > 12$ usa la suma L1 marginal (aproximación rápida, necesaria porque la distribución conjunta $2^N$ y la matriz Hamming $4^N$ serían inmanejables en memoria). Esta decisión es interna y transparente para la estrategia.
- **Independencia de GeoMIP:** Para k libre, QNodes realiza su propia búsqueda exhaustiva sobre todos los niveles $k$ para todo N y puede retornar una partición distinta a la de GeoMIP si existe un $k$ diferente con menor $\Phi$ global — ambos resultados son matemáticamente válidos.

#### Complejidad

| Fase | Complejidad | Condición |
|------|-------------|-----------|
| Agrupamiento greedy | O(N³) llamadas a `_emd_particion` | Siempre |
| Refinamiento 1-move | O(N × k × iteraciones) llamadas | Siempre |
| Candidatos de aislamiento (k fijo) | C(N, k−1) evaluaciones | Siempre |
| Candidatos de aislamiento (k libre) | $2^N − 2$ evaluaciones en total | k=None |
| Una llamada `_emd_particion` (N ≤ 12) | O($2^N$) — distribución conjunta Hamming | N ≤ 12 |
| Una llamada `_emd_particion` (N > 12) | O(N) — suma L1 marginal | N > 12 |

Para $N = 10$ con $k$ libre: ~1 022 evaluaciones adicionales → **< 1 segundo**.  
Para $N = 20$ con $k$ libre: ~1 M evaluaciones × O(20) cada una → **pocos segundos**.  
Para $N = 25$ con $k$ libre: ~33 M evaluaciones × O(25) cada una → **decenas de segundos**.

---

## Comparación entre frameworks

| Criterio | GeoMIP | QNodes |
|----------|--------|--------|
| **Estrategia principal** | Heurística geométrica + ILS | Agrupamiento aglomerativo greedy |
| **k de entrada** | Requerido (o rango completo en bloque) | Opcional — búsqueda libre sobre todos los k |
| **Métrica EMD** | Hamming real siempre | Hamming (N ≤ 12) / L1 (N > 12) en `_emd_particion` |
| **Paralelismo** | ProcessPoolExecutor (un proceso por k) | Single-thread; eficiencia por memoización |
| **Escalabilidad** | N ≤ 25 con LazyTPM | N ≤ 25 (L1 para N > 12) |
| **Garantía de optimalidad** | Óptimo local + ILS | Óptimo local greedy + 1-move |
| **Modo de entrada** | Manual / bloque CSV | Manual / bloque CSV |
| **Salida** | JSON (manual) / Excel (bloque) | Excel (bloque) |
| **Mecanismo vacío (∅)** | Soportado | Soportado (`permitir_presente_vacio`) |

---

## Estructura de carpetas

```
AYDA_2026_1/
├── CLAUDE.md                          ← instrucciones del proyecto para el asistente
├── .venv/                             ← entorno virtual Python (compartido, no se sube al repo)
│
├── GeoMIP/
│   ├── pyproject.toml                 ← dependencias de GeoMIP
│   ├── exec_kgeomip.py                ← entrypoint interactivo principal
│   ├── data/
│   │   ├── samples_binary/            ← TPMs binarias de prueba (N=3..25)
│   │   ├── samples_no_binary/         ← TPMs no binarias
│   │   └── Pruebas/                   ← CSVs de lotes de prueba
│   ├── results/                       ← JSONs y Excel con resultados
│   ├── viewer/                        ← frontend de visualización de resultados
│   ├── view_result.py                 ← visualizador de JSON en terminal
│   └── src/
│       ├── controllers/
│       │   ├── manager.py             ← carga TPM, enruta estrategias
│       │   └── strategies/
│       │       └── kgeomip.py         ← algoritmo KGeoMIP completo
│       ├── models/
│       │   ├── base/
│       │   │   ├── application.py     ← configuración global
│       │   │   └── sia.py             ← caché del candidato condicionado
│       │   └── core/
│       │       ├── system.py          ← condicionamiento, particionamiento
│       │       ├── ncube.py           ← hipercubo por nodo, marginalización cacheada
│       │       └── solution.py        ← representación del resultado
│       ├── funcs/
│       │   ├── base.py                ← utilidades generales (ABECEDARY, etc.)
│       │   ├── emd_optimized.py       ← emd_causal con matriz Hamming
│       │   └── format.py              ← formateo de particiones
│       └── lazy_tpm.py                ← lectura lazy por chunks para N ≥ 18
│
└── QNodes/
    ├── pyproject.toml                 ← dependencias de QNodes
    ├── exec.py                        ← entrypoint interactivo principal
    ├── review/                        ← scripts de testing y profiling
    └── src/
        ├── main.py                    ← lógica del lanzador CLI
        ├── strategies/
        │   ├── q_nodes.py             ← algoritmo QNodes (aglomerativo + 1-move)
        │   ├── force.py               ← búsqueda exhaustiva (referencia)
        │   └── phi.py                 ← cálculo de Phi unitario
        ├── funcs/
        │   ├── iit.py                 ← emd_causal, Hamming matrix, distribución conjunta
        │   └── format.py              ← formateo de particiones
        └── models/
            ├── base/sia.py            ← caché del candidato condicionado
            └── core/
                ├── system.py          ← condicionamiento, marginales
                └── ncube.py           ← hipercubo por nodo
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

### 4. Instalar las dependencias de ambos frameworks

```bash
pip install -e GeoMIP/
pip install -e QNodes/
```

El flag `-e` instala en modo *editable*: los cambios en el código fuente se reflejan sin reinstalar. Pip resolverá automáticamente todas las dependencias transitivas (numpy, scipy, pyphi, pyemd, POT, joblib, pandas, openpyxl, etc.).

### 5. Verificar la instalación

```bash
python -c "import pyphi, numpy, scipy, pyemd; print('Todo OK')"
```

---

## Uso — GeoMIP

```bash
# Con el .venv activo:
python GeoMIP/exec_kgeomip.py
```

Al ejecutar, el programa ofrece dos modos:

### Modo 1 — Manual

Ingreso interactivo de una prueba individual:

1. Seleccionar tipo de TPM (binaria / no binaria) y archivo.
2. Ingresar sistema candidato en binario (`1111111111` = sistema completo).
3. Ingresar estado inicial en binario (o `ENTER` para uno aleatorio).
4. Ingresar alcance (t+1) y mecanismo (t) como etiquetas de nodos (`ABCDE`, `ABC`, etc.).
5. Ingresar $K$ (o `ENTER` para evaluar todas las k posibles).
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
- Los resultados se exportan a un archivo `.xlsx` en `GeoMIP/results/block/`.

Los CSVs de prueba se encuentran en `GeoMIP/data/Pruebas/`.

---

## Uso — QNodes

```bash
# Con el .venv activo:
python QNodes/exec.py
```

El flujo es análogo al de GeoMIP:

1. Seleccionar tipo de TPM y archivo.
2. Ingresar sistema candidato, estado inicial, alcance y mecanismo.
3. Ingresar $K$ (o `ENTER` para búsqueda libre sobre todos los k ≥ 3).
4. Elegir modo de ejecución (manual o bloque CSV).

**Activar profiling** (genera reporte HTML con pyinstrument):

```python
# En QNodes/exec.py, antes de ejecutar:
aplicacion.activar_profiling()
```

---

## Dashboard (GUI)

Interfaz gráfica (React + Vite + FastAPI) en `dashboard/` para operar y analizar ambos
frameworks desde el navegador. Mantiene un worker persistente por motor (aislamiento del
paquete `src`) y permite ejecutar por bloque (streaming SSE), explorar resultados guardados
y comparar QNodes vs GeoMIP con métricas. Detalles y endpoints en
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
| **Representación de subconjuntos** | Máscaras enteras de N bits: `{A, C} = 0b0101 = 5` |
| **Entorno virtual** | `.venv/` en la raíz del repo, compartido por GeoMIP y QNodes, nunca comiteado |
| **Logs** | Archivos `.log` en `.logs/`, ignorados por git |
