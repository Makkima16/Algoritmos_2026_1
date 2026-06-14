import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { fmt } from "../lib/metrics.js";
import MetricCards from "../components/MetricCards.jsx";
import DistribucionChart from "../components/DistribucionChart.jsx";
import CohesionTable from "../components/CohesionTable.jsx";

export default function Resultados() {
  const [comparar, setComparar] = useState(false);

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>Resultados guardados</h2>
        <label style={{ display: "inline", marginBottom: 0 }}>
          <input type="checkbox" checked={comparar} onChange={(e) => setComparar(e.target.checked)} />{" "}
          Comparar 2 resultados
        </label>
      </div>
      <p className="muted">
        {comparar
          ? "Dos paneles independientes: elige un resultado en cada uno para verlos en paralelo."
          : "Explora y analiza un resultado guardado. Activa “Comparar 2” para ver dos a la vez."}
      </p>

      <div className="row" style={{ alignItems: "flex-start" }}>
        <ResultadoPanel titulo={comparar ? "Panel A" : null} compacto={comparar} />
        {comparar && <ResultadoPanel titulo="Panel B" compacto />}
      </div>
    </div>
  );
}

// Panel autónomo: selector (algo/tipo/archivo) + detalle del resultado elegido.
function ResultadoPanel({ titulo, compacto }) {
  const [algo, setAlgo] = useState("qnodes");
  const [tipo, setTipo] = useState("manual");
  const [lista, setLista] = useState([]);
  const [sel, setSel] = useState(null);
  const [detalle, setDetalle] = useState(null);
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState(null);
  const [fEstado, setFEstado] = useState("");
  const [fCandidato, setFCandidato] = useState("");

  useEffect(() => {
    setSel(null);
    setDetalle(null);
    api.results(algo, tipo).then((r) => setLista(r.resultados)).catch((e) => setError(String(e)));
  }, [algo, tipo]);

  // Filtra el listado por estado inicial y candidato (coincidencia por substring).
  const listaFiltrada = lista.filter(
    (it) =>
      (!fEstado || (it.estado || "").includes(fEstado)) &&
      (!fCandidato || (it.candidato || "").includes(fCandidato))
  );

  async function abrir(item) {
    setSel(item.ruta);
    setDetalle(null);
    setError(null);
    setCargando(true);
    try {
      setDetalle(await api.resultDetail(item.ruta));
    } catch (e) {
      setError(String(e));
    } finally {
      setCargando(false);
    }
  }

  return (
    <div className="col" style={{ minWidth: compacto ? 360 : 420 }}>
      {titulo && <h3 style={{ marginTop: 0 }}>{titulo}</h3>}
      <div className="card">
        <div className="row">
          <div className="col">
            <label>Algoritmo</label>
            <select value={algo} onChange={(e) => setAlgo(e.target.value)}>
              <option value="qnodes">KQNodes</option>
              <option value="geomip">KGeoMIP</option>
            </select>
          </div>
          <div className="col">
            <label>Tipo</label>
            <select value={tipo} onChange={(e) => setTipo(e.target.value)}>
              <option value="manual">Manual (JSON)</option>
              <option value="block">Bloque (XLSX)</option>
            </select>
          </div>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <div className="col">
            <label>Filtrar por estado inicial</label>
            <input className="mono" value={fEstado} placeholder="p.ej. 1111…"
                   onChange={(e) => setFEstado(e.target.value.trim())} />
          </div>
          <div className="col">
            <label>Filtrar por candidato</label>
            <input className="mono" value={fCandidato} placeholder="p.ej. 1111…"
                   onChange={(e) => setFCandidato(e.target.value.trim())} />
          </div>
        </div>
        <div className="scroll" style={{ marginTop: 10, maxHeight: 220 }}>
          {listaFiltrada.length === 0 && (
            <p className="muted">
              {lista.length === 0 ? "Sin resultados." : "Ningún resultado coincide con el filtro."}
            </p>
          )}
          {listaFiltrada.map((it) => (
            <div
              key={it.ruta}
              className={`list-item ${sel === it.ruta ? "active" : ""}`}
              onClick={() => abrir(it)}
            >
              {it.archivo}
              {it.subcarpeta && <div className="tag">{it.subcarpeta}</div>}
              {(it.estado || it.candidato) && (
                <div className="muted mono" style={{ fontSize: 11, marginTop: 2 }}>
                  {it.estado ? `estado ${it.estado}` : ""}
                  {it.estado && it.candidato ? " · " : ""}
                  {it.candidato ? `cand ${it.candidato}` : ""}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {error && <div className="card" style={{ borderColor: "var(--err)" }}>⚠ {error}</div>}
      {cargando && <div className="card">Cargando…</div>}
      {detalle?.tipo === "manual" && <DetalleManual detalle={detalle} />}
      {detalle?.tipo === "block" && <DetalleBloque data={detalle.data} />}
      {!detalle && !cargando && <p className="muted">Selecciona un resultado.</p>}
    </div>
  );
}

function DetalleManual({ detalle }) {
  const d = detalle.data;
  return (
    <>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>{d.estrategia || "Resultado"}</h3>
        <div className="muted" style={{ fontSize: 13, marginBottom: 8 }}>
          dataset {d.dataset} · estado {d.estado_inicial} · candidato {d.sistema_candidato}
          {" "}· alcance {d.alcance} · mecanismo {d.mecanismo}
        </div>
        <div className="particion">{d.particion}</div>
      </div>

      {detalle.metricas && (
        <>
          <div className="card">
            <h4 style={{ marginTop: 0 }}>Métricas</h4>
            <MetricCards metricas={detalle.metricas} />
          </div>
          <div className="card">
            <h4 style={{ marginTop: 0 }}>Distribución: subsistema vs partición</h4>
            <DistribucionChart cohesion={detalle.metricas.cohesion} />
          </div>
          <div className="card">
            <CohesionTable cohesion={detalle.metricas.cohesion} />
          </div>
        </>
      )}
    </>
  );
}

function DetalleBloque({ data }) {
  const meta = data.meta || {};
  return (
    <div className="card">
      <h4 style={{ marginTop: 0 }}>Lote (XLSX)</h4>
      {(meta.estado || meta.candidato) && (
        <div className="muted mono" style={{ fontSize: 13, marginBottom: 8 }}>
          estado inicial {meta.estado || "—"} · candidato {meta.candidato || "—"}
        </div>
      )}
      <div className="scroll">
        <table>
          <thead>
            <tr>{data.columnas.map((c, i) => <th key={i}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {data.filas.map((f, i) => (
              <tr key={i}>
                {data.columnas.map((c, j) => (
                  <td key={j} className={j === 3 ? "particion" : ""}>
                    {typeof f[c] === "number" ? fmt(f[c]) : f[c] ?? ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
