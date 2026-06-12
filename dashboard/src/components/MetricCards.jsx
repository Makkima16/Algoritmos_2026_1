import React from "react";
import { fmt, pct, bandaColor } from "../lib/metrics.js";

// Tarjetas resumen de las 3 familias de métricas por-resultado.
export default function MetricCards({ metricas }) {
  if (!metricas) return null;
  const pr = metricas.perdida_relativa;
  const cr = metricas.crecimiento;
  const co = metricas.cohesion;

  return (
    <div className="metric-grid">
      <div className="metric">
        <div className="v">{fmt(pr.phi)}</div>
        <div className="l">Φ (pérdida absoluta)</div>
      </div>
      <div className="metric">
        <div className="v">
          {pct(pr.phi_relativo)}{" "}
          <span className={`pill ${bandaColor(pr.banda)}`}>{pr.banda}</span>
        </div>
        <div className="l">Pérdida relativa (Φ / masa)</div>
      </div>
      <div className="metric">
        <div className="v">{fmt(cr.delta_l1)}</div>
        <div className="l">Δ L1 (desvío total de la distribución)</div>
      </div>
      <div className="metric">
        <div className="v">{fmt(cr.max_drift)}</div>
        <div className="l">Máximo drift por columna</div>
      </div>
      <div className="metric">
        <div className="v">
          {cr.columnas_cambiadas}/{cr.n_columnas} ({pct(cr.pct_columnas_cambiadas, 0)})
        </div>
        <div className="l">Columnas que cambian</div>
      </div>
      <div className="metric">
        <div className="v">{pct(co.cohesion_media_global)}</div>
        <div className="l">Cohesión media global</div>
      </div>
    </div>
  );
}
