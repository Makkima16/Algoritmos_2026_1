import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";

// Barras comparando la distribución del subsistema vs la de la partición, por columna.
export default function DistribucionChart({ cohesion }) {
  if (!cohesion?.columnas?.length) return null;
  const data = cohesion.columnas.map((c) => ({
    col: c.label || `c${c.columna}`,
    Subsistema: round(c.subsistema),
    Particion: round(c.particion),
  }));

  return (
    <div style={{ width: "100%", height: 280 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis dataKey="col" tick={{ fill: "#94a3b8", fontSize: 12 }} />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", color: "#e2e8f0" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="Subsistema" fill="#38bdf8" />
          <Bar dataKey="Particion" fill="#a78bfa" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const round = (x) => (x == null ? 0 : Math.round(x * 10000) / 10000);
