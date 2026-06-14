# -*- coding: utf-8 -*-
"""Prueba puntual N22 k=4 — KGeoMIP vs KQNodes (driver desechable)."""
import json, sys, time, subprocess, tempfile, os
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

REPO = Path(__file__).resolve().parent.parent
WORKER = Path(__file__).resolve().parent / "_worker_motor.py"
TPM = REPO / "data" / "samples_binary" / "N22A.csv"
N = 22

def bits(etiquetas):
    activos = set(etiquetas.strip().upper())
    return "".join("1" if chr(65 + i) in activos else "0" for i in range(N))

estado    = "1" * N                       # todo 1 (pedido por el usuario)
candidato = bits("ABCDEFGHIJKLMNOPQRSTUV")  # sistema completo (como la hoja)
alcance   = bits("ABDEGHJKMNPQSTV")
mecanismo = bits("ABCDEFGHIJKLMNOPQRSTUV")
KS = [int(x) for x in sys.argv[1:]] or [4, 5]

print(f"estado    = {estado}")
print(f"candidato = {candidato}")
print(f"alcance   = {alcance}  (ABDEGHJKMNPQSTV)")
print(f"mecanismo = {mecanismo}  (todos)")
print(f"K a probar = {KS}\n")

ROOTS = {"geomip": REPO / "KGeoMIP", "qnodes": REPO / "KQNodes"}
RESULT = "@@RESULT@@"

def correr(engine, k):
    cfg = {
        "engine": engine, "root": str(ROOTS[engine]), "tpm": str(TPM),
        "estado": estado, "candidato": candidato,
        "permitir_presente_vacio": True, "k": k,
        "tests": [{"row": 6, "alcance": alcance, "mecanismo": mecanismo}],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(cfg, tf, ensure_ascii=False); lote = tf.name
    t0 = time.time(); res = None
    try:
        proc = subprocess.Popen([sys.executable, str(WORKER), lote],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding="utf-8", bufsize=1)
        for linea in proc.stdout:
            if not linea.startswith(RESULT):
                continue
            payload = json.loads(linea[len(RESULT):])
            if "ready" in payload or "fatal" in payload:
                continue
            res = payload
        proc.wait()
    finally:
        try: os.unlink(lote)
        except OSError: pass
    print(f"===== {engine.upper()} (wall {time.time()-t0:.2f}s) =====")
    if res and res.get("ok"):
        print(f"  Φ (perdida) = {res['perdida']:.6f}")
        print(f"  tiempo busqueda = {res['tiempo']:.4f}s   prep = {res['prep']:.4f}s")
        print(f"  particion:\n{res['particion']}")
    else:
        print(f"  ERROR: {res.get('error') if res else 'sin resultado'}")
    print()
    return res

for k in KS:
    print(f"\n############## K = {k} ##############")
    r_geo = correr("geomip", k)
    r_qno = correr("qnodes", k)
    if r_geo and r_geo.get("ok") and r_qno and r_qno.get("ok"):
        pg, pq = r_geo["perdida"], r_qno["perdida"]
        print(f"===== COMPARACION K={k} =====")
        print(f"  KGeoMIP Φ = {pg:.6f}")
        print(f"  KQNodes Φ = {pq:.6f}")
        if abs(pg - pq) < 1e-9:
            print("  → EMPATE (mismo Φ)")
        elif pg < pq:
            print(f"  → KGeoMIP MEJOR por {pq - pg:.6f}")
        else:
            print(f"  → KQNodes MEJOR por {pg - pq:.6f}")
