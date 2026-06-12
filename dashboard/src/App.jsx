import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import EjecutarBloque from "./views/EjecutarBloque.jsx";
import Resultados from "./views/Resultados.jsx";
import Analisis from "./views/Analisis.jsx";

const VISTAS = {
  ejecutar: { label: "Ejecutar", comp: EjecutarBloque },
  resultados: { label: "Resultados", comp: Resultados },
  analisis: { label: "Análisis", comp: Analisis },
};

export default function App() {
  const [vista, setVista] = useState("ejecutar");
  const [health, setHealth] = useState(null);
  const Comp = VISTAS[vista].comp;

  useEffect(() => {
    const tick = () => api.health().then(setHealth).catch(() => setHealth(null));
    tick();
    const id = setInterval(tick, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>MIP Dashboard</h1>
        <div className="sub">QNodes &amp; GeoMIP · AYDA 2026-1</div>
        {Object.entries(VISTAS).map(([k, v]) => (
          <button
            key={k}
            className={`nav-btn ${vista === k ? "active" : ""}`}
            onClick={() => setVista(k)}
          >
            {v.label}
          </button>
        ))}

        <div className="health">
          <div style={{ marginBottom: 4 }}>Motores</div>
          {["qnodes", "geomip"].map((a) => {
            const w = health?.workers?.[a];
            const on = w?.alive && w?.ready;
            return (
              <div key={a}>
                <span className={`dot ${on ? "on" : "off"}`} />
                {a} {on ? "listo" : w?.alive ? "arrancando…" : "caído"}
              </div>
            );
          })}
        </div>
      </aside>

      <main className="main">
        <Comp />
      </main>
    </div>
  );
}
