import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { fmt, pct } from "../lib/metrics.js";

// Panel de comparación lado a lado entre dos resultados normalizados.
// Cada resultado: { label, color, phi, tiempo, particion, metricas }
export default function ComparacionPanel({ a, b }) {
  if (!a || !b) return null;

  const mejor = a.phi == null || b.phi == null ? "—" : a.phi === b.phi ? "Ambos" : a.phi < b.phi ? a.label : b.label;
  const filas = [
    ["Φ (pérdida)", fmt(a.phi), fmt(b.phi), "menor"],
    ["Φ relativa", pct(a.metricas?.perdida_relativa?.phi_relativo), pct(b.metricas?.perdida_relativa?.phi_relativo), "menor"],
    ["Δ L1 distribución", fmt(a.metricas?.crecimiento?.delta_l1), fmt(b.metricas?.crecimiento?.delta_l1), "menor"],
    ["Max drift", fmt(a.metricas?.crecimiento?.max_drift), fmt(b.metricas?.crecimiento?.max_drift), "menor"],
    ["Cohesión media", pct(a.metricas?.cohesion?.cohesion_media_global), pct(b.metricas?.cohesion?.cohesion_media_global), "mayor"],
    ["Tiempo", `${fmt(a.tiempo, 3)} s`, `${fmt(b.tiempo, 3)} s`, "menor"],
  ];

  const chartData = [
    { m: "Φ", [a.label]: r(a.phi), [b.label]: r(b.phi) },
    { m: "Φ rel", [a.label]: r(a.metricas?.perdida_relativa?.phi_relativo), [b.label]: r(b.metricas?.perdida_relativa?.phi_relativo) },
    { m: "Δ L1", [a.label]: r(a.metricas?.crecimiento?.delta_l1), [b.label]: r(b.metricas?.crecimiento?.delta_l1) },
    { m: "Cohesión", [a.label]: r(a.metricas?.cohesion?.cohesion_media_global), [b.label]: r(b.metricas?.cohesion?.cohesion_media_global) },
  ];

  return (
    <>
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <h4 style={{ margin: 0 }}>Comparación</h4>
          <span>
            Mejor Φ: <span className={`pill ${mejor === a.label ? "qnodes" : mejor === b.label ? "geomip" : ""}`}>{mejor}</span>
          </span>
        </div>
        <table style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th>Métrica</th>
              <th>{a.label}</th>
              <th>{b.label}</th>
              <th>Mejor</th>
            </tr>
          </thead>
          <tbody>
            {filas.map(([nombre, va, vb, criterio], i) => {
              const na = num(va), nb = num(vb);
              let win = "—";
              if (na != null && nb != null && na !== nb)
                win = (criterio === "menor" ? na < nb : na > nb) ? a.label : b.label;
              return (
                <tr key={i}>
                  <td>{nombre}</td>
                  <td>{va}</td>
                  <td>{vb}</td>
                  <td className="muted">{win}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="card">
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
              <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
              <XAxis dataKey="m" tick={{ fill: "#94a3b8", fontSize: 12 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155" }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey={a.label} fill="#38bdf8" />
              <Bar dataKey={b.label} fill="#a78bfa" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="row">
        <div className="col card">
          <h4 style={{ marginTop: 0 }}>{a.label} — partición</h4>
          <div className="particion">{a.particion}</div>
        </div>
        <div className="col card">
          <h4 style={{ marginTop: 0 }}>{b.label} — partición</h4>
          <div className="particion">{b.particion}</div>
        </div>
      </div>
    </>
  );
}

const r = (x) => (x == null ? 0 : Math.round(x * 10000) / 10000);
const num = (s) => {
  const v = parseFloat(String(s).replace("%", "").replace("s", "").trim());
  return Number.isNaN(v) ? null : v;
};
