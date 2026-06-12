// Helpers de acceso al backend FastAPI (vía proxy /api de Vite).

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

export const api = {
  health: () => jget("/api/health"),
  datasets: (algo) => jget(`/api/datasets?algo=${algo}`),
  pruebas: (algo, n) => jget(`/api/pruebas?algo=${algo}${n != null ? `&n=${n}` : ""}`),
  pruebasContenido: (ruta) => jget(`/api/pruebas/contenido?ruta=${encodeURIComponent(ruta)}`),
  run: (body) => jpost("/api/run", body),
  results: (algo, tipo) => jget(`/api/results?algo=${algo}&tipo=${tipo}`),
  resultDetail: (ruta) => jget(`/api/results/detail?ruta=${encodeURIComponent(ruta)}`),
  metrics: (body) => jpost("/api/metrics", body),
  comparativa: () => jget("/api/comparativa"),
};

// Streaming del bloque: invoca onEvento(obj) por cada evento SSE recibido.
export async function ejecutarBloque(body, onEvento) {
  const r = await fetch("/api/block", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const bloque = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const linea = bloque.split("\n").find((l) => l.startsWith("data:"));
      if (linea) {
        try {
          onEvento(JSON.parse(linea.slice(5).trim()));
        } catch (_) {
          /* ignora líneas parciales */
        }
      }
    }
  }
}
