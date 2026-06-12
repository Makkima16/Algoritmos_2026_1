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
| POST | `/api/block` | Lote (streaming SSE), guarda XLSX |
| GET | `/api/results?algo=&tipo=` | Índice de resultados guardados |
| GET | `/api/results/detail?ruta=` | Detalle parseado + métricas |
| GET | `/api/comparativa` | Cruce QNodes vs GeoMIP (resultados manuales) |
