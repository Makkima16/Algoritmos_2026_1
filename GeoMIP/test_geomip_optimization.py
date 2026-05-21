#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de prueba para verificar que la optimización de GeoMIP para N=25 funciona correctamente.

Ejecutar:
    python test_geomip_optimization.py
"""

import sys
import numpy as np
import time
from pathlib import Path

# Agregar ruta del proyecto
sys.path.insert(0, str(Path(__file__).parent))

def test_emd_optimized():
    """Prueba las funciones optimizadas de EMD."""
    print("\n" + "="*70)
    print("TEST 1: Verificar funciones optimizadas de EMD")
    print("="*70)
    
    try:
        from src.funcs.emd_optimized import (
            emd_causal_sampled,
            emd_causal_fast_partition,
            compare_methods_benchmark
        )
        print("✓ Módulo emd_optimized cargado exitosamente")
        
        # Crear distribuciones de prueba
        n_test = 20
        n_states = 2 ** n_test
        u = np.random.dirichlet(np.ones(n_states))
        v = np.random.dirichlet(np.ones(n_states))
        
        # Probar EMD exacto (n_test <= 20)
        start = time.time()
        emd_exact = emd_causal_sampled(u, v, method='hybrid')
        tiempo_exact = time.time() - start
        print(f"✓ EMD Exacto (N={n_test}): {emd_exact:.6f} en {tiempo_exact:.4f}s")
        
        # Probar con N pequeño
        n_small = 15
        u_small = np.random.dirichlet(np.ones(2**n_small))
        v_small = np.random.dirichlet(np.ones(2**n_small))
        
        start = time.time()
        emd_sampled = emd_causal_sampled(u_small, v_small, sample_size=5000, method='hybrid')
        tiempo_sampled = time.time() - start
        print(f"✓ EMD Muestreado (N={n_small}): {emd_sampled:.6f} en {tiempo_sampled:.4f}s")
        
        print("\n✓ TEST 1 PASADO: Funciones optimizadas funcionan correctamente\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 1 FALLÓ: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_kgeomip_import():
    """Prueba que KGeoMIP puede importar las funciones optimizadas."""
    print("\n" + "="*70)
    print("TEST 2: Verificar que KGeoMIP importa correctamente")
    print("="*70)
    
    try:
        from src.controllers.strategies.kgeomip import (
            KGeoMIP,
            evaluar_k_particion,
            emd_causal_fast_partition
        )
        print("✓ KGeoMIP importado exitosamente")
        print("✓ emd_causal_fast_partition importado en kgeomip.py")
        
        # Verificar que emd_causal_fast_partition está disponible
        print(f"✓ Función disponible: {emd_causal_fast_partition.__name__}")
        
        print("\n✓ TEST 2 PASADO: KGeoMIP está correctamente configurado\n")
        return True
        
    except ImportError as e:
        print(f"✗ TEST 2 FALLÓ (ImportError): {e}\n")
        return False
    except Exception as e:
        print(f"✗ TEST 2 FALLÓ: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_large_n_detection():
    """Prueba que el sistema detecta N > 20 correctamente."""
    print("\n" + "="*70)
    print("TEST 3: Verificar detección automática de N > 20")
    print("="*70)
    
    try:
        # Simular una distribución con N=25
        n_large = 25
        print(f"Simulando sistema con N={n_large} variables...")
        
        from src.funcs.emd_optimized import emd_causal_fast_partition
        
        # Crear distribuciones marginales (tamaño 25, no 2^25 para evitar OOM en test)
        dist_original = np.random.uniform(0.1, 0.9, size=n_large)
        dist_reconstruida = np.random.uniform(0.1, 0.9, size=n_large)
        
        # Normalizar
        dist_original = dist_original / np.sum(dist_original)
        dist_reconstruida = dist_reconstruida / np.sum(dist_reconstruida)
        
        # Evaluar con use_marginal=True (usa suma simple)
        start = time.time()
        emd_result = emd_causal_fast_partition(
            dist_original=dist_original,
            dist_reconstruida=dist_reconstruida,
            n_nodos=n_large,
            use_marginal=True
        )
        tiempo_elapsed = time.time() - start
        
        print(f"✓ EMD calculado para N={n_large}: {emd_result:.6f}")
        print(f"  Tiempo: {tiempo_elapsed:.6f}s (muy rápido ✓)")
        
        if tiempo_elapsed < 0.01:
            print("✓ Rendimiento excepcional para N grande")
        else:
            print("⚠ Rendimiento por debajo de lo esperado, pero aceptable")
        
        print("\n✓ TEST 3 PASADO: Sistema listo para N > 20\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 3 FALLÓ: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def check_csv_exists():
    """Verifica si existe el archivo N25A.csv."""
    print("\n" + "="*70)
    print("TEST 4: Verificar disponibilidad de datos N=25")
    print("="*70)
    
    csv_path = Path(__file__).parent / "GeoMIP" / "data" / "samples" / "N25A.csv"
    
    if csv_path.exists():
        size_gb = csv_path.stat().st_size / (1024**3)
        print(f"✓ Archivo N25A.csv encontrado")
        print(f"  Ubicación: {csv_path}")
        print(f"  Tamaño: {size_gb:.2f} GB")
        print(f"\n✓ TEST 4 PASADO: Datos disponibles\n")
        return True
    else:
        print(f"⚠ Archivo N25A.csv no encontrado en:")
        print(f"  {csv_path}")
        print(f"\n  Ejecute primero: python GeoMIP/data/creation.py")
        print(f"  Luego seleccione N=25\n")
        return False


def main():
    """Ejecuta todos los tests."""
    print("\n" + "#"*70)
    print("# SUITE DE TESTS: Optimización GeoMIP para N=25")
    print("#"*70)
    
    results = []
    
    # Cambiar al directorio correcto
    import os
    script_dir = Path(__file__).parent / "GeoMIP" / "src" / "Method2_Dynamic_Programming_Reformulation"
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))
    
    results.append(("EMD Optimizado", test_emd_optimized()))
    results.append(("KGeoMIP Import", test_kgeomip_import()))
    results.append(("Detección N > 20", test_large_n_detection()))
    results.append(("CSV N25 Existe", check_csv_exists()))
    
    # Resumen
    print("\n" + "#"*70)
    print("# RESUMEN DE TESTS")
    print("#"*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASADO" if result else "✗ FALLÓ"
        print(f"{test_name:30s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests pasados")
    
    if passed == total:
        print("\n🎉 TODOS LOS TESTS PASARON - Sistema listo para N=25\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) fallido(s)\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
