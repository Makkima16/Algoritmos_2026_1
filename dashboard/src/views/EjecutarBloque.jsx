import React, { useEffect, useMemo, useState } from "react";
import { api, ejecutarBloque } from "../api.js";
import { fmt, pct, bandaColor, formatearTiempo } from "../lib/metrics.js";
import MetricCards from "../components/MetricCards.jsx";
import DistribucionChart from "../components/DistribucionChart.jsx";
import CohesionTable from "../components/CohesionTable.jsx";

export default function Ejecutar() {
  const [modo, setModo] = useState("manual"); // "manual" | "bloque"
  const [algo, setAlgo] = useState("qnodes");
  const [datasets, setDatasets] = useState([]);
  const [dsRuta, setDsRuta] = useState("");
  const [pruebas, setPruebas] = useState([]);
  const [csvRuta, setCsvRuta] = useState("");
  const [estado, setEstado] = useState("");
  const [candidato, setCandidato] = useState("");
  // Solo modo manual:
  const [alcance, setAlcance] = useState("");
  const [mecanismo, setMecanismo] = useState("");
  const [k, setK] = useState("");
  const [vacio, setVacio] = useState(false);

  const [corriendo, setCorriendo] = useState(false);
  const [error, setError] = useState(null);

  // Estado modo bloque:
  const [filas, setFilas] = useState([]);
  const [progreso, setProgreso] = useState({ idx: 0, total: 0 });
  const [resumen, setResumen] = useState(null);

  // Estado modo manual:
  const [manualRes, setManualRes] = useState(null);

  const dsSel = useMemo(() => datasets.find((d) => d.ruta === dsRuta), [datasets, dsRuta]);
  const n = dsSel?.n;

  useEffect(() => {
    api.datasets(algo).then((r) => {
      setDatasets(r.datasets);
      setDsRuta("");
      setPruebas([]);
      setCsvRuta("");
    }).catch((e) => setError(String(e)));
  }, [algo]);

  // Al elegir dataset: precargar máscaras (todo 1s) y CSVs de pruebas de ese N.
  useEffect(() => {
    if (!dsSel?.n) return;
    const unos = "1".repeat(dsSel.n);
    setEstado(unos);
    setCandidato(unos);
    setAlcance(unos);
    setMecanismo(unos);
    api.pruebas(algo, dsSel.n).then((r) => {
      setPruebas(r.pruebas);
      setCsvRuta(r.pruebas[0]?.ruta || "");
    }).catch((e) => setError(String(e)));
  }, [dsRuta]);

  function resetSalidas() {
    setError(null);
    setResumen(null);
    setFilas([]);
    setManualRes(null);
  }

  async function lanzarManual() {
    resetSalidas();
    setCorriendo(true);
    try {
      const r = await api.run({
        algo,
        ruta_tpm: dsRuta,
        estado,
        candidato,
        alcance,
        mecanismo,
        k: k === "" ? null : Number(k),
        permitir_presente_vacio: vacio,
      });
      setManualRes(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setCorriendo(false);
    }
  }

  async function lanzarBloque() {
    resetSalidas();
    setCorriendo(true);
    try {
      await ejecutarBloque(
        {
          algo,
          ruta_tpm: dsRuta,
          ruta_csv: csvRuta,
          estado,
          candidato,
          n,
          k: k === "" ? null : Number(k),
          permitir_presente_vacio: vacio,
        },
        (ev) => {
          if (ev.tipo === "inicio") setProgreso({ idx: 0, total: ev.total });
          else if (ev.tipo === "fila") {
            setProgreso({ idx: ev.idx, total: ev.total });
            setFilas((prev) => [...prev, ev]);
          } else if (ev.tipo === "fin") setResumen(ev);
          else if (ev.tipo === "error") setError(ev.mensaje);
        }
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setCorriendo(false);
    }
  }

  const lanzar = modo === "manual" ? lanzarManual : lanzarBloque;
  const pctProg = progreso.total ? (progreso.idx / progreso.total) * 100 : 0;

  return (
    <div>
      <h2>Ejecutar</h2>
      <p className="muted">
        Los motores ya están en ejecución (modo manual). En <b>Manual</b> se evalúa un solo caso
        (especificas alcance y mecanismo); en <b>Por bloque</b> se orquesta una corrida por fila del
        CSV de pruebas (alcance y mecanismo salen del CSV) y se guarda un XLSX en <code>results/block/</code>.
      </p>

      <div className="card">
        <div className="row">
          <div className="col" style={{ maxWidth: 220 }}>
            <label>Modo de ejecución</label>
            <select value={modo} onChange={(e) => { setModo(e.target.value); resetSalidas(); }}>
              <option value="manual">Manual (un caso)</option>
              <option value="bloque">Por bloque (CSV)</option>
            </select>
          </div>
          <div className="col">
            <label>Algoritmo</label>
            <select value={algo} onChange={(e) => setAlgo(e.target.value)}>
              <option value="qnodes">KQNodes</option>
              <option value="geomip">KGeoMIP</option>
            </select>
          </div>
          <div className="col">
            <label>TPM (dataset)</label>
            <select value={dsRuta} onChange={(e) => setDsRuta(e.target.value)}>
              <option value="">— seleccionar —</option>
              {datasets.map((d) => (
                <option key={d.ruta} value={d.ruta}>
                  {d.archivo} (N{d.n}, {d.tipo})
                </option>
              ))}
            </select>
          </div>
          {modo === "bloque" && (
            <div className="col">
              <label>CSV de pruebas {n ? `(N${n})` : ""}</label>
              <select value={csvRuta} onChange={(e) => setCsvRuta(e.target.value)} disabled={!pruebas.length}>
                {!pruebas.length && <option value="">— sin pruebas para este N —</option>}
                {pruebas.map((p) => (
                  <option key={p.ruta} value={p.ruta}>{p.archivo}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          <div className="col">
            <label>Estado inicial</label>
            <input className="mono" value={estado} onChange={(e) => setEstado(e.target.value)} />
          </div>
          <div className="col">
            <label>Sistema candidato</label>
            <input className="mono" value={candidato} onChange={(e) => setCandidato(e.target.value)} />
          </div>
          {modo === "manual" && (
            <>
              <div className="col">
                <label>Alcance (Purview t+1)</label>
                <input className="mono" value={alcance} onChange={(e) => setAlcance(e.target.value)} />
              </div>
              <div className="col">
                <label>Mecanismo (t)</label>
                <input className="mono" value={mecanismo} onChange={(e) => setMecanismo(e.target.value)} />
              </div>
            </>
          )}
          <div className="col" style={{ maxWidth: 120 }}>
            <label>k (vacío = todas)</label>
            <input value={k} onChange={(e) => setK(e.target.value)} placeholder="auto" />
          </div>
          <div className="col" style={{ maxWidth: 210, alignSelf: "flex-end" }}>
            <label style={{ display: "inline" }}>
              <input type="checkbox" checked={vacio} onChange={(e) => setVacio(e.target.checked)} />{" "}
              Permitir presente vacío (∅)
            </label>
          </div>
        </div>

        <div style={{ marginTop: 14 }}>
          <button
            className="primary"
            disabled={corriendo || !dsRuta || (modo === "bloque" && !csvRuta)}
            onClick={lanzar}
          >
            {corriendo ? "Ejecutando…" : modo === "manual" ? "Ejecutar caso" : "Ejecutar bloque"}
          </button>
        </div>
      </div>

      {error && <div className="card" style={{ borderColor: "var(--err)" }}>⚠ {error}</div>}

      {/* ── Resultado modo manual ───────────────────────────────────────── */}
      {modo === "manual" && manualRes && (
        <>
          <div className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3 style={{ margin: 0 }}>{manualRes.estrategia}</h3>
              <span className="muted">Φ = {fmt(manualRes.perdida_phi)} · {formatearTiempo(manualRes.tiempo_total_segundos)}</span>
            </div>
            <div className="particion" style={{ marginTop: 10 }}>{manualRes.particion}</div>
          </div>
          <div className="card">
            <h4 style={{ marginTop: 0 }}>Métricas</h4>
            <MetricCards metricas={manualRes.metricas} />
          </div>
          <div className="card">
            <h4 style={{ marginTop: 0 }}>Distribución: subsistema vs partición</h4>
            <DistribucionChart cohesion={manualRes.metricas.cohesion} />
          </div>
          <div className="card">
            <CohesionTable cohesion={manualRes.metricas.cohesion} />
          </div>
        </>
      )}

      {/* ── Resultado modo bloque ───────────────────────────────────────── */}
      {modo === "bloque" && (corriendo || filas.length > 0) && (
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <strong>Progreso {progreso.idx}/{progreso.total}</strong>
            {resumen && (
              <span className="muted">
                ✓ {resumen.exitosas} ok · {resumen.con_error} error · búsqueda {resumen.tiempo_busqueda_fmt} · arranque {resumen.tiempo_arranque_fmt}
              </span>
            )}
          </div>
          <div className="progress"><div style={{ width: `${pctProg}%` }} /></div>

          <div className="scroll">
            <table>
              <thead>
                <tr><th>#</th><th>Alcance</th><th>Mecanismo</th><th>Φ</th><th>Φ rel</th><th>Banda</th><th>Tiempo</th><th>Partición</th></tr>
              </thead>
              <tbody>
                {filas.map((f, i) => (
                  <tr key={i} className={f.error ? "err" : ""}>
                    <td>{f.num}</td>
                    <td className="mono">{f.alcance}</td>
                    <td className="mono">{f.mecanismo}</td>
                    <td>{f.error ? "—" : fmt(f.perdida)}</td>
                    <td>{f.error ? "—" : fmt(f.phi_relativo)}</td>
                    <td>{f.error ? "—" : <span className={`pill ${bandaColor(f.banda)}`}>{f.banda}</span>}</td>
                    <td>{f.error ? "—" : (f.tiempo_fmt || formatearTiempo(f.tiempo_seg))}</td>
                    <td className="particion" style={{ maxWidth: 280, overflow: "hidden" }}>
                      {f.error ? `ERROR: ${f.error}` : f.particion}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {resumen && <BloqueMetricas r={resumen} />}
          {resumen && <p className="muted">Guardado en: <span className="mono">{resumen.salida}</span></p>}
        </div>
      )}
    </div>
  );
}

// Métricas agregadas del lote: calidad de Φ, tiempos (incluido arranque/warmup
// separado del de búsqueda) y tasa de éxito.
function BloqueMetricas({ r }) {
  const b = r.bandas || { baja: 0, media: 0, alta: 0 };
  return (
    <div style={{ marginTop: 14 }}>
      <h4 style={{ margin: "6px 0" }}>Métricas del lote</h4>
      <div className="metric-grid">
        <div className="metric">
          <div className="v">{pct(r.tasa_exito, 0)}</div>
          <div className="l">Tasa de éxito ({r.exitosas}/{r.total})</div>
        </div>
        <div className="metric">
          <div className="v">{fmt(r.phi_promedio)}</div>
          <div className="l">Φ promedio (Σ {fmt(r.phi_total)})</div>
        </div>
        <div className="metric">
          <div className="v">{pct(r.phi_relativo_promedio)}</div>
          <div className="l">Pérdida relativa promedio (Φ / masa)</div>
        </div>
        <div className="metric">
          <div className="v">
            <span className="pill baja">{b.baja}</span>{" "}
            <span className="pill media">{b.media}</span>{" "}
            <span className="pill alta">{b.alta}</span>
          </div>
          <div className="l">Pruebas por banda (baja / media / alta)</div>
        </div>
        <div className="metric">
          <div className="v">{r.tiempo_busqueda_fmt}</div>
          <div className="l">Tiempo de búsqueda (Σ pruebas)</div>
        </div>
        <div className="metric">
          <div className="v">{r.tiempo_promedio_fmt}</div>
          <div className="l">Tardanza promedio por prueba</div>
        </div>
        <div className="metric">
          <div className="v">{r.tiempo_arranque_fmt}</div>
          <div className="l">Arranque del motor (warmup)</div>
        </div>
        <div className="metric">
          <div className="v">{r.tiempo_total_fmt}</div>
          <div className="l">Tiempo total (wall-clock)</div>
        </div>
      </div>
    </div>
  );
}
