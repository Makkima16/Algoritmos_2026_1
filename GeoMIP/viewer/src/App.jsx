import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

function Partition({ particion }) {
  return (
    <div className="flex flex-wrap items-center gap-4 mt-4">
      {particion.partes.map((parte, idx) => (
        <div key={idx} className="flex items-center gap-4">

          <div className="bg-white rounded-2xl border shadow-md overflow-hidden min-w-[160px]">

            <div className="bg-slate-900 text-white text-center py-2 font-bold">
              Futuro
            </div>

            <div className="p-4 text-center">
              <div className="text-2xl font-black tracking-wider">
                {parte.futuro.join(", ")}
              </div>
            </div>

            <div className="border-t bg-slate-50 p-4 text-center">
              <div className="text-lg tracking-wider text-slate-600">
                {parte.presente.join(", ")}
              </div>
            </div>
          </div>

          {idx !== particion.partes.length - 1 && (
            <div className="text-4xl font-black text-indigo-500">
              ⊗
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function App() {

  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState(null);

  useEffect(() => {

    async function loadFiles() {

      const response = await fetch("/results/index.json");
      const fileList = await response.json();

      setFiles(fileList);

      if (fileList.length > 0) {

        const first = await fetch(`/results/${fileList[0]}`);
        const data = await first.json();

        setSelected(data);
      }
    }

    loadFiles();

  }, []);

  async function openFile(name) {

    const response = await fetch(`/results/${name}`);
    const data = await response.json();

    setSelected(data);
  }

  if (!selected) {

    return (
      <div className="p-10 text-2xl">
        Cargando resultados...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-100 flex">

      {/* Sidebar */}

      <div className="w-80 bg-slate-900 text-white p-6 overflow-y-auto">

        <h1 className="text-3xl font-black mb-8">
          KGeoMIP
        </h1>

        <div className="space-y-3">

          {files.map((file) => (

            <button
              key={file}
              onClick={() => openFile(file)}
              className="w-full text-left bg-slate-800 hover:bg-indigo-600 transition rounded-xl p-4"
            >
              {file}
            </button>
          ))}
        </div>
      </div>

      {/* Main */}

      <div className="flex-1 p-10 overflow-y-auto">

        <div className="bg-white rounded-3xl shadow-xl p-8">

          <h1 className="text-4xl font-black text-slate-800">
            {selected.dataset}
          </h1>

          <p className="mt-2 text-slate-500">
            {selected.estrategia}
          </p>

          {/* Cards */}

          <div className="grid grid-cols-3 gap-6 mt-8">

            <div className="bg-slate-100 rounded-2xl p-6">
              <div className="text-sm text-slate-500 uppercase">
                Pérdida φ
              </div>

              <div className="text-5xl font-black text-indigo-600 mt-3">
                {selected.perdida_phi.toFixed(6)}
              </div>
            </div>

            <div className="bg-slate-100 rounded-2xl p-6">
              <div className="text-sm text-slate-500 uppercase">
                Tiempo
              </div>

              <div className="text-5xl font-black text-emerald-600 mt-3">
                {selected.tiempo_total.toFixed(3)}s
              </div>
            </div>

            <div className="bg-slate-100 rounded-2xl p-6">
              <div className="text-sm text-slate-500 uppercase">
                Estado inicial
              </div>

              <div className="text-2xl font-black mt-5 tracking-widest">
                {selected.estado_inicial}
              </div>
            </div>
          </div>

          {/* Partición */}

          <div className="mt-12">

            <h2 className="text-3xl font-black">
              Partición Óptima
            </h2>

            <Partition particion={selected.particion} />
          </div>

          {/* Gráfica */}

          <div className="mt-16">

            <h2 className="text-3xl font-black mb-6">
              Evolución de φ
            </h2>

            <div className="bg-slate-50 rounded-2xl p-6">

              <ResponsiveContainer width="100%" height={400}>

                <LineChart
                  data={selected.historico_comparaciones}
                >
                  <CartesianGrid strokeDasharray="3 3" />

                  <XAxis dataKey="k" />

                  <YAxis />

                  <Tooltip />

                  <Line
                    type="monotone"
                    dataKey="perdida"
                    strokeWidth={4}
                  />
                </LineChart>

              </ResponsiveContainer>
            </div>
          </div>

          {/* Histórico */}

          <div className="mt-16">

            <h2 className="text-3xl font-black mb-8">
              Histórico de Particiones
            </h2>

            <div className="space-y-8">

              {selected.historico_comparaciones.map((item) => (

                <div
                  key={item.k}
                  className="bg-slate-50 rounded-3xl p-6 border"
                >

                  <div className="flex items-center justify-between">

                    <div className="text-3xl font-black">
                      k = {item.k}
                    </div>

                    <div className="text-2xl font-black text-rose-600">
                      φ = {item.perdida.toFixed(6)}
                    </div>
                  </div>

                  <Partition particion={item.particion} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}