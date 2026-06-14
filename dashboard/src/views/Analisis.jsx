import React, { useEffect, useMemo, useState } from "react";
import { api, ejecutarBloque } from "../api.js";
import { fmt, formatearTiempo } from "../lib/metrics.js";
import ComparacionPanel from "../components/ComparacionPanel.jsx";

const ALGOS = ["qnodes", "geomip"];
const NOMBRE = { qnodes: "KQNodes", geomip: "KGeoMIP" };

export default function Analisis() {
  const [modo, setModo] = useState("vivo_manual");

  return (
    <div>
      <h2>Análisis comparativo · KQNodes vs KGeoMIP</h2>
      <div className="card">
        <label>Modo de comparación</label>
        <select value={modo} onChange={(e) => setModo(e.target.value)} style={{ maxWidth: 320 }}>
          <option value="vivo_manual">En vivo — un caso (manual)</option>
          <option value="vivo_bloque">En vivo — lote (por bloque / CSV)</option>
          <option value="guardados">Comparar resultados guardados</option>
        </select>
        <p className="muted" style={{ marginBottom: 0 }}>
          {modo === "vivo_manual" && "Ejecuta el MISMO caso en ambos algoritmos y compara sus métricas."}
          {modo === "vivo_bloque" && "Corre el MISMO CSV de pruebas en ambos algoritmos y compara fila a fila."}
          {modo === "guardados" && "Elige un resultado guardado de cada algoritmo y compáralos."}
        </p>
      </div>

      {modo === "vivo_manual" && <VivoManual />}
      {modo === "vivo_bloque" && <VivoBloque />}
      {modo === "guardados" && <Guardados />}
    </div>
  );
}

// ── Hook: datasets compartidos (data/) — mismos para ambos algoritmos.
// Se identifican por su ruta (única); el mismo archivo puede existir como
// binaria y no_binaria, por eso no se usa el nombre como clave.
function useDatasets() {
  const [datasets, setDatasets] = useState([]);
  useEffect(() => {
    api.datasets("qnodes").then((r) => setDatasets(r.datasets)).catch(() => setDatasets([]));
  }, []);
  return datasets;
}

function normalizarRun(resp, label) {
  return {
    label,
    phi: resp.perdida_phi,
    tiempo: resp.tiempo_total_segundos,
    particion: resp.particion,
    metricas: resp.metricas,
  };
}

// ── Modo 1: comparar un caso en vivo ───────────────────────────────────────
function VivoManual() {
  const datasets = useDatasets();
  const [ruta, setRuta] = useState("");
  const [estado, setEstado] = useState("");
  const [candidato, setCandidato] = useState("");
  const [alcance, setAlcance] = useState("");
  const [mecanismo, setMecanismo] = useState("");
  const [k, setK] = useState("");
  const [vacio, setVacio] = useState(false);
  const [corriendo, setCorriendo] = useState(false);
  const [comp, setComp] = useState(null);
  const [error, setError] = useState(null);

  const n = datasets.find((d) => d.ruta === ruta)?.n;

  useEffect(() => {
    if (!n) return;
    const unos = "1".repeat(n);
    setEstado(unos); setCandidato(unos); setAlcance(unos); setMecanismo(unos);
  }, [ruta]);

  async function comparar() {
    setError(null); setComp(null); setCorriendo(true);
    try {
      const comun = { ruta_tpm: ruta, estado, candidato, alcance, mecanismo, k: k === "" ? null : Number(k), permitir_presente_vacio: vacio };
      const [rq, rg] = await Promise.all(ALGOS.map((a) => api.run({ algo: a, ...comun })));
      setComp({ a: normalizarRun(rq, "KQNodes"), b: normalizarRun(rg, "KGeoMIP") });
    } catch (e) {
      setError(String(e));
    } finally {
      setCorriendo(false);
    }
  }

  return (
    <>
      <div className="card">
        <div className="row">
          <div className="col">
            <label>Dataset (TPM, data/)</label>
            <select value={ruta} onChange={(e) => setRuta(e.target.value)}>
              <option value="">— seleccionar —</option>
              {datasets.map((d) => <option key={d.ruta} value={d.ruta}>{d.archivo} (N{d.n}, {d.tipo})</option>)}
            </select>
          </div>
          <div className="col"><label>Estado</label><input className="mono" value={estado} onChange={(e) => setEstado(e.target.value)} /></div>
          <div className="col"><label>Candidato</label><input className="mono" value={candidato} onChange={(e) => setCandidato(e.target.value)} /></div>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <div className="col"><label>Alcance (t+1)</label><input className="mono" value={alcance} onChange={(e) => setAlcance(e.target.value)} /></div>
          <div className="col"><label>Mecanismo (t)</label><input className="mono" value={mecanismo} onChange={(e) => setMecanismo(e.target.value)} /></div>
          <div className="col" style={{ maxWidth: 110 }}><label>k</label><input value={k} onChange={(e) => setK(e.target.value)} placeholder="auto" /></div>
          <div className="col" style={{ maxWidth: 210, alignSelf: "flex-end" }}>
            <label style={{ display: "inline" }}>
              <input type="checkbox" checked={vacio} onChange={(e) => setVacio(e.target.checked)} /> Permitir presente vacío (∅)
            </label>
          </div>
        </div>
        <div style={{ marginTop: 14 }}>
          <button className="primary" disabled={corriendo || !ruta} onClick={comparar}>
            {corriendo ? "Ejecutando ambos…" : "Comparar en vivo"}
          </button>
        </div>
      </div>
      {error && <div className="card" style={{ borderColor: "var(--err)" }}>⚠ {error}</div>}
      {comp && <ComparacionPanel a={comp.a} b={comp.b} />}
    </>
  );
}

// ── Modo 2: comparar un lote (CSV) en vivo ─────────────────────────────────
function VivoBloque() {
  const datasets = useDatasets();
  const [ruta, setRuta] = useState("");
  const [csv, setCsv] = useState(null);
  const [estado, setEstado] = useState("");
  const [candidato, setCandidato] = useState("");
  const [k, setK] = useState("3");
  const [vacio, setVacio] = useState(false);
  const [corriendo, setCorriendo] = useState(false);
  const [progreso, setProgreso] = useState("");
  const [filasQ, setFilasQ] = useState({});
  const [filasG, setFilasG] = useState({});
  const [error, setError] = useState(null);

  const n = datasets.find((d) => d.ruta === ruta)?.n;

  useEffect(() => {
    if (!n) return;
    setEstado("1".repeat(n)); setCandidato("1".repeat(n));
    api.pruebas("qnodes", n).then((r) => setCsv(r.pruebas[0]?.ruta || null));
  }, [ruta]);

  async function correrLote(algo, setFilas) {
    const acc = {};
    await ejecutarBloque(
      { algo, ruta_tpm: ruta, ruta_csv: csv, estado, candidato, n, k: k === "" ? null : Number(k), permitir_presente_vacio: vacio },
      (ev) => {
        if (ev.tipo === "fila") {
          acc[ev.num] = ev;
          setFilas({ ...acc });
          setProgreso(`${NOMBRE[algo]}: ${ev.idx}/${ev.total}`);
        }
      }
    );
  }

  async function comparar() {
    setError(null); setFilasQ({}); setFilasG({}); setCorriendo(true);
    try {
      await correrLote("qnodes", setFilasQ);
      await correrLote("geomip", setFilasG);
      setProgreso("Completado");
    } catch (e) {
      setError(String(e));
    } finally {
      setCorriendo(false);
    }
  }

  const nums = useMemo(() => {
    const s = new Set([...Object.keys(filasQ), ...Object.keys(filasG)]);
    return [...s].sort((a, b) => Number(a) - Number(b));
  }, [filasQ, filasG]);

  const resumen = useMemo(() => {
    const phis = (obj) => Object.values(obj).filter((f) => !f.error).map((f) => f.perdida);
    const avg = (xs) => (xs.length ? xs.reduce((a, c) => a + c, 0) / xs.length : null);
    return { q: avg(phis(filasQ)), g: avg(phis(filasG)) };
  }, [filasQ, filasG]);

  return (
    <>
      <div className="card">
        <div className="row">
          <div className="col">
            <label>Dataset (data/)</label>
            <select value={ruta} onChange={(e) => setRuta(e.target.value)}>
              <option value="">— seleccionar —</option>
              {datasets.map((d) => <option key={d.ruta} value={d.ruta}>{d.archivo} (N{d.n}, {d.tipo})</option>)}
            </select>
          </div>
          <div className="col"><label>Estado</label><input className="mono" value={estado} onChange={(e) => setEstado(e.target.value)} /></div>
          <div className="col"><label>Candidato</label><input className="mono" value={candidato} onChange={(e) => setCandidato(e.target.value)} /></div>
          <div className="col" style={{ maxWidth: 110 }}><label>k</label><input value={k} onChange={(e) => setK(e.target.value)} placeholder="auto" /></div>
          <div className="col" style={{ maxWidth: 210, alignSelf: "flex-end" }}>
            <label style={{ display: "inline" }}>
              <input type="checkbox" checked={vacio} onChange={(e) => setVacio(e.target.checked)} /> Presente vacío (∅)
            </label>
          </div>
        </div>
        <p className="muted">Se ejecuta el CSV de pruebas N{n || "?"} primero en KQNodes y luego en KGeoMIP (puede tardar).</p>
        <button className="primary" disabled={corriendo || !ruta || !csv} onClick={comparar}>
          {corriendo ? `Ejecutando… ${progreso}` : "Comparar lote"}
        </button>
      </div>

      {error && <div className="card" style={{ borderColor: "var(--err)" }}>⚠ {error}</div>}

      {nums.length > 0 && (
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h4 style={{ margin: 0 }}>Φ por prueba</h4>
            <span className="muted">
              Φ media — KQNodes: {fmt(resumen.q)} · KGeoMIP: {fmt(resumen.g)}
            </span>
          </div>
          <div className="scroll">
            <table>
              <thead>
                <tr><th>#</th><th>Alcance</th><th>Mecanismo</th><th>Φ KQNodes</th><th>Φ KGeoMIP</th><th>Mejor</th></tr>
              </thead>
              <tbody>
                {nums.map((num) => {
                  const q = filasQ[num], g = filasG[num];
                  const pq = q && !q.error ? q.perdida : null;
                  const pg = g && !g.error ? g.perdida : null;
                  const mejor = pq == null || pg == null ? "—" : pq <= pg ? "KQNodes" : "KGeoMIP";
                  return (
                    <tr key={num}>
                      <td>{num}</td>
                      <td className="mono">{(q || g)?.alcance}</td>
                      <td className="mono">{(q || g)?.mecanismo}</td>
                      <td>{pq == null ? "—" : fmt(pq)}</td>
                      <td>{pg == null ? "—" : fmt(pg)}</td>
                      <td><span className={`pill ${mejor === "KQNodes" ? "qnodes" : mejor === "KGeoMIP" ? "geomip" : ""}`}>{mejor}</span></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}

// ── Modo 3: comparar resultados guardados (manual o por bloque) ────────────
function Guardados() {
  const [tipo, setTipo] = useState("manual"); // "manual" | "block"
  const [listas, setListas] = useState({ qnodes: [], geomip: [] });
  const [sel, setSel] = useState({ qnodes: "", geomip: "" });
  const [comp, setComp] = useState(null);     // comparación manual (ComparacionPanel)
  const [bloque, setBloque] = useState(null); // comparación de lotes (tabla por #prueba)
  const [error, setError] = useState(null);

  useEffect(() => {
    setSel({ qnodes: "", geomip: "" });
    setComp(null);
    setBloque(null);
    Promise.all(ALGOS.map((a) => api.results(a, tipo))).then(([q, g]) =>
      setListas({ qnodes: q.resultados, geomip: g.resultados })
    ).catch((e) => setError(String(e)));
  }, [tipo]);

  function normalizarDetalle(det, label) {
    const d = det.data;
    return {
      label,
      phi: d.perdida_phi,
      tiempo: d.tiempo_total_segundos,
      particion: d.particion,
      metricas: det.metricas,
      meta: `${d.dataset} · estado ${d.estado_inicial ?? "?"} · cand ${d.sistema_candidato ?? "?"} · alc ${d.alcance} · mec ${d.mecanismo} · k ${d.k_solicitado ?? "?"}`,
    };
  }

  // Extrae {num -> {alcance, mecanismo, perdida}} de un XLSX de bloque.
  function filasBloque(det) {
    const cols = det.data.columnas;
    const colNum = cols.find((c) => /prueba/i.test(c)) || cols[0];
    const colAlc = cols.find((c) => /alcance|purview/i.test(c)) || cols[1];
    const colMec = cols.find((c) => /mecanismo/i.test(c)) || cols[2];
    const colPer = cols.find((c) => /perdida|pérdida/i.test(c));
    const out = {};
    for (const f of det.data.filas) {
      const num = String(f[colNum] ?? "").trim();
      if (!num || !/^\d+$/.test(num)) continue; // omite fila "Tiempo Total Lote"
      out[num] = {
        alcance: f[colAlc],
        mecanismo: f[colMec],
        perdida: typeof f[colPer] === "number" ? f[colPer] : null,
      };
    }
    return out;
  }

  async function comparar() {
    setError(null); setComp(null); setBloque(null);
    try {
      const [dq, dg] = await Promise.all(ALGOS.map((a) => api.resultDetail(sel[a])));
      if (tipo === "manual") {
        setComp({ a: normalizarDetalle(dq, "KQNodes"), b: normalizarDetalle(dg, "KGeoMIP") });
      } else {
        setBloque({
          q: filasBloque(dq), g: filasBloque(dg),
          metaQ: dq.data.meta || {}, metaG: dg.data.meta || {},
        });
      }
    } catch (e) {
      setError(String(e));
    }
  }

  const nums = bloque
    ? [...new Set([...Object.keys(bloque.q), ...Object.keys(bloque.g)])].sort((a, b) => Number(a) - Number(b))
    : [];
  const media = (obj) => {
    const xs = Object.values(obj).map((r) => r.perdida).filter((x) => x != null);
    return xs.length ? xs.reduce((a, c) => a + c, 0) / xs.length : null;
  };

  return (
    <>
      <div className="card">
        <label>Tipo de resultado</label>
        <select value={tipo} onChange={(e) => setTipo(e.target.value)} style={{ maxWidth: 260 }}>
          <option value="manual">Manual (JSON, un caso)</option>
          <option value="block">Bloque (XLSX, lote)</option>
        </select>
        <div className="row" style={{ marginTop: 12 }}>
          {ALGOS.map((a) => (
            <div className="col" key={a}>
              <label>{NOMBRE[a]} — resultado {tipo === "manual" ? "manual" : "de bloque"}</label>
              <select value={sel[a]} onChange={(e) => setSel({ ...sel, [a]: e.target.value })}>
                <option value="">— seleccionar —</option>
                {listas[a].map((it) => (
                  <option key={it.ruta} value={it.ruta}>
                    {it.archivo}{it.subcarpeta ? ` · ${it.subcarpeta}` : ""}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>
        <button className="primary" style={{ marginTop: 14 }} disabled={!sel.qnodes || !sel.geomip} onClick={comparar}>
          Comparar seleccionados
        </button>
      </div>

      {error && <div className="card" style={{ borderColor: "var(--err)" }}>⚠ {error}</div>}

      {/* Comparación manual */}
      {comp && (
        <>
          <div className="card muted" style={{ fontSize: 13 }}>
            <div>KQNodes: {comp.a.meta}</div>
            <div>KGeoMIP: {comp.b.meta}</div>
          </div>
          <ComparacionPanel a={comp.a} b={comp.b} />
        </>
      )}

      {/* Comparación por bloque (fila a fila) */}
      {bloque && (
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h4 style={{ margin: 0 }}>Φ por prueba</h4>
            <span className="muted">
              Φ media — KQNodes: {fmt(media(bloque.q))} · KGeoMIP: {fmt(media(bloque.g))}
            </span>
          </div>
          {(bloque.metaQ?.estado || bloque.metaQ?.candidato || bloque.metaG?.estado) && (
            <div className="muted mono" style={{ fontSize: 12, marginBottom: 6 }}>
              KQNodes: estado {bloque.metaQ?.estado || "—"} · cand {bloque.metaQ?.candidato || "—"}
              {" │ "}
              KGeoMIP: estado {bloque.metaG?.estado || "—"} · cand {bloque.metaG?.candidato || "—"}
            </div>
          )}
          <div className="scroll">
            <table>
              <thead>
                <tr><th>#</th><th>Alcance</th><th>Mecanismo</th><th>Φ KQNodes</th><th>Φ KGeoMIP</th><th>Mejor</th></tr>
              </thead>
              <tbody>
                {nums.map((num) => {
                  const q = bloque.q[num], g = bloque.g[num];
                  const pq = q?.perdida ?? null, pg = g?.perdida ?? null;
                  const mejor = pq == null || pg == null ? "—" : pq === pg ? "Ambos" : pq < pg ? "KQNodes" : "KGeoMIP";
                  return (
                    <tr key={num}>
                      <td>{num}</td>
                      <td className="mono">{(q || g)?.alcance}</td>
                      <td className="mono">{(q || g)?.mecanismo}</td>
                      <td>{pq == null ? "—" : fmt(pq)}</td>
                      <td>{pg == null ? "—" : fmt(pg)}</td>
                      <td><span className={`pill ${mejor === "KQNodes" ? "qnodes" : mejor === "KGeoMIP" ? "geomip" : ""}`}>{mejor}</span></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
