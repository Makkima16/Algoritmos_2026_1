import React from "react";
import { fmt, pct } from "../lib/metrics.js";

// Ranking de cohesión por nodo/columna + cohesión agregada por grupo.
export default function CohesionTable({ cohesion }) {
  if (!cohesion?.columnas?.length) return null;
  const ordenadas = [...cohesion.columnas].sort((a, b) => b.cohesion - a.cohesion);

  return (
    <div className="row">
      <div className="col">
        <h4>Cohesión por nodo (alta = mejor conservado)</h4>
        <div className="scroll">
          <table>
            <thead>
              <tr><th>Nodo</th><th>Cohesión</th><th>Δ columna</th><th></th></tr>
            </thead>
            <tbody>
              {ordenadas.map((c) => (
                <tr key={c.columna}>
                  <td className="mono">{c.label || `c${c.columna}`}</td>
                  <td>{pct(c.cohesion)}</td>
                  <td>{fmt(c.delta)}</td>
                  <td style={{ width: 90 }}>
                    <div className="bar-wrap">
                      <div className="bar" style={{ width: `${Math.max(0, c.cohesion) * 100}%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {cohesion.grupos?.length > 0 && (
        <div className="col">
          <h4>Cohesión por grupo de la partición</h4>
          <table>
            <thead>
              <tr><th>Grupo</th><th>Tamaño</th><th>Cohesión media</th><th>Δ total</th></tr>
            </thead>
            <tbody>
              {cohesion.grupos.map((g, i) => (
                <tr key={i}>
                  <td className="mono">{g.labels.join(",")}</td>
                  <td>{g.tamano}</td>
                  <td>{pct(g.cohesion_media)}</td>
                  <td>{fmt(g.delta_total)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
