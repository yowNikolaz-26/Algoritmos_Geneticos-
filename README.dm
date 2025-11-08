# Reconstrucción de Imágenes con Algoritmo Genético

Este proyecto utiliza un algoritmo genético optimizado para reconstruir una imagen objetivo mediante evolución artificial.

## Requisitos
```bash
pip install -r requirements.txt
```

## Uso

1. Coloca tu imagen objetivo como `galaxy.jpg` en el mismo directorio
2. Ejecuta el script:
```bash
python genetic_image.py
```

## Parámetros del Algoritmo

- **Generaciones**: 100
- **Tamaño de población**: 80
- **Padres para reproducción**: 12
- **Tasa de mutación**: 0.01
- **Procesamiento paralelo**: 4 hilos

## Optimizaciones

- Compilación JIT con Numba para la función de fitness
- Inicialización inteligente desde la imagen objetivo con ruido
- Procesamiento paralelo multi-hilo
- Tipos de datos eficientes (uint8)

## Salida

- Progreso cada 50 generaciones
- Comparación visual entre imagen original y generada
- Fitness final y porcentaje de error

## Personalización

Para usar otra imagen:
```python
target_image = imageio.imread('tu_imagen.jpg')
```

Para ajustar calidad/velocidad, modifica:
- `num_generations`: más generaciones = mejor calidad
- `sol_per_pop`: mayor población = mejor exploración
- `mutation_percent_genes`: controla variabilidad
```
