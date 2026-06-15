# Dashboard MIP — QNodes & GeoMIP

Interfaz gráfica (React + Vite) para operar y analizar los frameworks de k-MIP
del repositorio. Permite:

1. **Ejecutar por bloque** QNodes y GeoMIP desde la GUI (orquestando corridas manuales).
2. **Explorar resultados guardados** (`results/manual`, `results/manually`, `results/block`).
3. **Generar métricas** sobre cada resultado: pérdida relativa Φ, crecimiento de la
   distribución, cohesión por columna/nodo y comparativa QNodes vs GeoMIP.

## Arquitectura

```
React (Vite, :5173)  ──/api──▶  FastAPI (uvicorn, :8000)
                                   │
                                   ├─ worker QNodes  (proceso vivo, cwd=QNodes/)
                                   └─ worker GeoMIP  (proceso vivo, cwd=GeoMIP/)
```

QNodes y GeoMIP usan ambos un paquete `src` de nivel superior, por lo que **no pueden
convivir en un mismo intérprete**. Cada uno corre en su propio proceso worker
persistente, precargado al arrancar el backend y anclado a modo manual
(una corrida — un `aplicar_estrategia` — por petición). El "bloque" de la GUI
itera corridas manuales fila a fila sobre un CSV de pruebas.

## Requisitos

- Python ≥ 3.11 con el `.venv` del repo y `pip install -r server/requirements.txt`.
- Node ≥ 18 (`npm install` dentro de `dashboard/`).

## Arranque (dos terminales)

**Backend** (desde la raíz del repo):
```bash
.venv/Scripts/python.exe -m uvicorn main:app --app-dir dashboard/server --port 8000
```

**Frontend** (desde `dashboard/`):
```bash
npm install   # solo la primera vez
npm run dev
```

Abre http://localhost:5173. El indicador "Motores" en la barra lateral muestra
si los workers QNodes/GeoMIP están listos.

## Endpoints principales (backend)

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/health` | Estado de los dos workers |
| GET | `/api/datasets?algo=` | TPMs disponibles |
| GET | `/api/pruebas?algo=&n=` | CSVs de pruebas por N |
| POST | `/api/run` | Una corrida manual + métricas |
| POST | `/api/block` | Lote (streaming SSE), guarda XLSX. Calienta el motor antes del lote (ver abajo) |
| GET | `/api/results?algo=&tipo=` | Índice de resultados guardados |
| GET | `/api/results/detail?ruta=` | Detalle parseado + métricas |
| GET | `/api/comparativa` | Cruce QNodes vs GeoMIP (resultados manuales) |

## Arranque del motor (warmup) en el lote

La PRIMERA corrida real de un worker paga costos de una sola vez (primer toque de
numpy, construcción de tablas, carga/caché de la TPM, JIT). Si ese costo cayera
dentro de la primera prueba, su tiempo guardado quedaría **inflado y engañoso**.

Por eso `/api/block`, antes de iterar el lote, ejecuta una **corrida de calentamiento
descartable** con los parámetros de la primera prueba válida y la contabiliza aparte
como **"arranque del motor"** (fila *Arranque motor (warmup)* del XLSX y campo
`tiempo_arranque` del evento SSE `fin`). El "Tiempo Total Lote" (wall-clock) excluye el
arranque.

> **Cambio (2026-06-14).** El tiempo guardado de cada prueba es ahora **preparación de su
> subsistema + búsqueda**, no solo la búsqueda. La preparación (`substraer` con el
> alcance/mecanismo de la prueba; en KQNodes además reconstruye todo el `System` por no
> cachear) es trabajo **específico por prueba** y debe contar en su tiempo; el "arranque"
> queda solo con el warmup único. `worker_runner` emite `tiempo_total_segundos` ya
> normalizado a prep+búsqueda para **ambos** motores (antes `tiempo_ejecucion` lo incluía
> en KGeoMIP pero no en KQNodes), y el SSE `fin` reporta el Σ como `tiempo_pruebas`. Mismo
> criterio aplicado en los `exec.py` de terminal y en `run_suite_2026.py`.

## Análisis comparativo (KQNodes vs KGeoMIP)

En la vista *Análisis*, la comparación **por bloque** (fila a fila) decide el ganador
por menor pérdida Φ. Cuando ambos motores arrojan **la misma pérdida**, la columna
*Mejor* muestra **"Ambos"** (antes el empate se adjudicaba silenciosamente a KQNodes
por el `<=`).
