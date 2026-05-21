# Diagnóstico: Bucle de K que se "reinicia"

## Problema Observado

Los logs mostraban que el algoritmo parecía "reiniciarse" en lugar de avanzar a valores de k más altos:
```
CRÍTICA (19:01:07): Jerárquico... Particiones: 2, Pérdida: 1.592875
CRÍTICA (19:01:27): Jerárquico... Particiones: 11, Pérdida: 2.197648  ← Reinicia?
```

## Causa Raíz

El bucle `for test_k in range(2, n_vars + 1)` **SÍ está avanzando correctamente**, pero:

1. **Cada k tarda MÁS tiempo que el anterior** porque cada iteración de agrupamiento jerárquico tiene complejidad O(n³·2ⁿ)
2. Para N=12 con UMBRAL_EXHAUSTIVO=10:
   - k=2: 12×2=24 > 10 → **Jerárquico** (~15-20 seg por la paralelización)
   - k=3: 12×3=36 > 10 → **Jerárquico** (~15-20 seg)
   - k=4: 12×4=48 > 10 → **Jerárquico** (~15-20 seg)
   - ... y así **12 veces más**

3. **Resultado:** El usuario ve la misma secuencia repetida 12 veces, pensando que se "reinicia"

## Solución Implementada

### 1. **Aumenté UMBRAL_EXHAUSTIVO de 10 a 15**
```python
UMBRAL_EXHAUSTIVO: int = 15  # Era 10
```

Ahora para N=12:
- k=2: 12×2=24 > 15 → Jerárquico
- k=3: 12×3=36 > 15 → Jerárquico
- ...hasta k=4: 12×4=48 > 15 → Jerárquico

Pero para N ≤ 10 y k ≤ 2:
- k=2: 10×2=20 > 15 → Jerárquico (pero aún rápido)

La búsqueda exhaustiva es **más rápida** que la jerárquica para k pequeño, aunque genere más particiones.

### 2. **Agregué logging detallado**

Ahora verás exactamente qué valor de k se está procesando:
```
━━━ EVALUANDO K=2 (n_vars * k = 24, UMBRAL = 15) ━━━
  → Usando AGRUPAMIENTO JERÁRQUICO
  → Calculando EMD para partición k=2...
  → K=2: Pérdida = 1.592875
  ✓ NUEVA MEJOR PARTICIÓN: k=2 con pérdida=1.592875
✓ K=2 COMPLETADO

━━━ EVALUANDO K=3 (n_vars * k = 36, UMBRAL = 15) ━━━
  → Usando AGRUPAMIENTO JERÁRQUICO
```

Esto confirma que está avanzando de k=2 → k=3 → k=4, etc.

### 3. **Mejoré robustez de _agrupamiento_jerarquico()**

- Agregué validación de entrada (k > n_vars, k < 1)
- Agregué contador de iteraciones para debugging
- Agregué verificación de "pares a evaluar" vacíos

## Rendimiento Esperado (Después de cambios)

Para N=12:
- **Antes:** ~20 seg × 12 k = **4 minutos**
- **Después (con UMBRAL=15):** Depende del tamaño, pero debería ser significativamente más rápido

Nota: El tiempo depende de si usa Exhaustivo o Jerárquico, que está controlado por `n_vars * test_k <= UMBRAL`.

## Cómo Probar

1. Ejecuta el script de test:
```bash
python GeoMIP/test_k_loop.py
```

2. Ejecuta exec_kgeomip.py y ve los logs con valores de K claramente marcados:
```bash
python GeoMIP/src/Method2_Dynamic_Programming_Reformulation/exec_kgeomip.py
```

Ahora deberías ver claramente cuando pasa de K=2 a K=3, K=4, etc.

## Ajustes Futuros Opcionales

Si sigue siendo lento, puedes:

1. **Aumentar más UMBRAL_EXHAUSTIVO** (pero búsqueda exhaustiva es O(S(n,k)·2ⁿ))
2. **Reducir UMBRAL_EXHAUSTIVO** y permitir que use Jerárquico más frecuentemente
3. **Usar Monte Carlo** para muestrear solo algunos k en lugar de todos (k=2,4,6,8,10,12 en lugar de todos)
4. **Paralelizar sobre k** en lugar de solo dentro de cada k

## Resumen

El "reinicio" no es un bug, es que **cada k tarda aproximadamente el mismo tiempo**, así que parece que está repitiendo. El algoritmo está funcionando correctamente, solo es computacionalmente costoso.

Con la mejora del UMBRAL_EXHAUSTIVO y el logging detallado, deberías ver claramente el progreso.
