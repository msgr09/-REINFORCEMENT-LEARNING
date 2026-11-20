# Aprendizaje por Refuerzo: Q-learning vs SARSA en el entorno Taxi

Implementación en Python de dos algoritmos tabulares de aprendizaje por refuerzo
(**Q-learning** y **SARSA**) para resolver el problema Taxi de Gymnasium. Se comparan
entre sí, contra una política aleatoria y contra la política óptima calculada por
iteración de valor.

## Problema

Un taxi se mueve en una cuadrícula de 5×5 con paredes. Debe recoger a un pasajero en una de
cuatro ubicaciones (R, G, Y, B) y dejarlo en su destino con el menor número de pasos.

| Componente | Definición |
|---|---|
| Estados | 500 (25 posiciones del taxi × 5 ubicaciones del pasajero × 4 destinos) |
| Acciones | 6: sur, norte, este, oeste, recoger, dejar |
| Recompensas | −1 por paso, +20 al entregar, −10 por recoger/dejar de forma ilegal |
| Terminación | Entrega correcta, o truncado a los 200 pasos |

## Contenido del proyecto

```
.
├── taxi_rl.ipynb        # Cuaderno (Jupyter / Google Colab)
├── taxi_rl.py           # Mismo código como script
├── requirements.txt     # Dependencias
├── README.md
└── resultados/          # Se genera al ejecutar
    ├── fig1_curvas_aprendizaje.png
    ├── fig2_zoom_epsilon.png
    ├── fig3_comparacion_final.png
    ├── fig4_sensibilidad.png
    ├── fig5_politica.png
    ├── fig6_distribucion_valores.png
    └── resultados.json
```

## Requisitos

- Python 3.9 o superior (probado con 3.12)
- `gymnasium`, `numpy`, `matplotlib` (ver `requirements.txt`)

## Instalación y ejecución

### Opción A: Google Colab
1. Sube `taxi_rl.ipynb` a [colab.research.google.com](https://colab.research.google.com).
2. En la primera celda de código, descomenta la línea `# !pip install -q gymnasium numpy matplotlib`.
3. Menú *Entorno de ejecución → Ejecutar todo*.

### Opción B: local
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python taxi_rl.py                 # como script
# o bien
jupyter notebook taxi_rl.ipynb    # como cuaderno
```

La ejecución completa tarda alrededor de uno o dos minutos. Las gráficas se guardan en
`resultados/` y las cifras en `resultados/resultados.json`.

> Si ejecutas el script en un servidor sin pantalla, usa `MPLBACKEND=Agg python taxi_rl.py`.

## Qué hace el código

1. **Entorno:** crea `Taxi-v4` (con respaldo a `Taxi-v3` en versiones antiguas de Gymnasium).
2. **Agentes:** `entrenar()` implementa Q-learning y SARSA con política ε-greedy;
   `evaluar()` mide una política greedy en episodios con semillas distintas a las de entrenamiento.
3. **Referencias:** política aleatoria y política óptima por iteración de valor.
4. **Experimento principal:** 2 algoritmos × 5 semillas × 5000 episodios.
5. **Sensibilidad:** Q-learning variando α, γ y el decaimiento de ε (3 semillas × 3000 episodios).
6. **Visualizaciones:** curvas de aprendizaje, comparación final, sensibilidad, política
   aprendida y distribución de retornos/valores.

## Hiperparámetros base

| Parámetro | Valor |
|---|---|
| α (tasa de aprendizaje) | 0.1 |
| γ (factor de descuento) | 0.99 |
| ε inicial / mínimo / decaimiento | 1.0 / 0.01 / 0.999 por episodio |
| Episodios | 5000 |
| Semillas | 0 a 4 |

Se pueden modificar en las celdas de la sección 4 (`SEMILLAS`, `EPISODIOS`) y de la
sección 5 (`BASE`, `GRID`).

## Resultados obtenidos

Evaluación greedy sobre 300 episodios (promedio de 5 semillas):

| Agente | Retorno medio | Pasos medios | Éxito |
|---|---|---|---|
| Aleatorio | −750.7 | 194.9 | 7.5 % |
| Q-learning | 8.00 | 13.0 | 100 % |
| SARSA | 7.78 | 13.2 | 99.93 % |
| Óptima | 8.00 | 13.0 | 100 % |

Q-learning converge antes que SARSA (≈1891 frente a ≈2250 episodios hasta retorno medio > 0),
pero ambos terminan con un desempeño prácticamente igual.

## Reproducibilidad

Todas las semillas están fijadas, por lo que con las mismas versiones de las librerías se
obtienen los mismos resultados. Con versiones distintas de Gymnasium o NumPy pueden aparecer
diferencias pequeñas.

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3–4), 279–292.
- Towers, M., et al. (2024). Gymnasium: A standard interface for reinforcement learning environments. arXiv:2407.17032.
