// Utilidades de formato y parsing en cliente (las métricas numéricas las calcula
// el backend; aquí solo formateamos y parseamos para la visualización).

export const fmt = (x, d = 4) =>
  x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d);

export const pct = (x, d = 1) => (x == null ? "—" : `${(x * 100).toFixed(d)} %`);

export function bandaColor(banda) {
  return banda === "baja" ? "baja" : banda === "media" ? "media" : "alta";
}

// Parsea los grupos de la primera línea de la notación de partición
// (soporta QNodes "⎛ ⎞" y GeoMIP "| |").
export function parseGrupos(particion) {
  if (!particion) return [];
  const primera = String(particion).trim().split("\n")[0];
  return primera
    .split(/[⎛⎞⎝⎠|]+/)
    .map((ch) =>
      ch
        .split(",")
        .map((t) => t.trim())
        .filter((t) => /^[A-Z]+$/.test(t))
    )
    .filter((g) => g.length > 0);
}

export function formatearTiempo(seg) {
  if (seg == null) return "—";
  if (seg < 60) return `${seg.toFixed(2)} s`;
  if (seg < 3600) return `${Math.floor(seg / 60)} min ${(seg % 60).toFixed(1)} s`;
  const h = Math.floor(seg / 3600);
  const r = seg % 3600;
  return `${h} h ${Math.floor(r / 60)} min`;
}
