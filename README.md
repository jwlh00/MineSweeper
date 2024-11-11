# Proyecto Final - Implementación y Entrenamiento de un Agente de Aprendizaje por Refuerzo para el Juego de Minesweeper

Este proyecto utiliza Deep Q-Learning (DQN) y Proximal Policy Optimization (PPO) para entrenar un agente que aprende a jugar Minesweeper. El objetivo es comparar el desempeño de ambos enfoques en un entorno de tablero pequeño con distintas densidades de minas. La implementación está hecha en Python utilizando PyTorch y Gymnasium.

## Contenido

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Limitaciones y Mejora](#limitaciones-y-mejora)
- [Créditos](#créditos)
- [Video](#video)

## Descripción del Proyecto

Este proyecto explora dos enfoques de aprendizaje profundo en entornos de juegos:

1. **DQN**: Implementado desde cero, busca enseñar al agente utilizando Q-learning y una red neuronal profunda.
2. **PPO**: Utiliza el entorno Gymnasium y un modelo Actor-Crítico que facilita la exploración y adaptación en entornos más complejos.

El entorno de entrenamiento es un tablero de 5x5 inspirado en Minesweeper con diferentes densidades de minas.

### Objetivo

El objetivo es que el agente aprenda a identificar y evitar minas, maximizando la tasa de éxito en juegos de dificultad moderada y alta.

## Estructura del Proyecto

### `AIGym`:

---

- **`agent.py`**: Implementación del agente DQN.
- **`minesweeper_env.py`**: Configuración del entorno personalizado de Minesweeper utilizando Gymnasium.
- **`play.py`**: Funciones para evaluar el desempeño de los agentes.

---

### `FromScratch`:

---

- **`agent.py`**: Implementación del agente PPO.
- **`minesweeper_env.py`**: Configuración del entorno hecho desde cero.
- **`play.py`**: Funciones para evaluar el desempeño de los agentes.
- **`README.md`**: Descripción general del proyecto.

## Requisitos

Asegúrate de tener instalado Python 3.8 o superior. Las dependencias principales incluyen:

- **PyTorch**
- **Gymnasium**
- **Numpy**

Para instalar los requisitos, ejecuta:

```bash
pip install -r requirements.txt
```

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tu_usuario/minesweeper-agent.git
   cd minesweeper-agent
   ```

2. Instala los paquetes requeridos:

   ```bash
   pip install -r requirements.txt
   ```

### Evaluación de los Agentes

Después del entrenamiento, puedes evaluar el desempeño de los agentes usando:

```bash
python play.py
```

## Resultados

### Agente DQN

El agente DQN mostró dificultades para mejorar su rendimiento, estabilizando su valor Q y pérdida rápidamente. La tasa de éxito fue baja debido a una falta de variación en el aprendizaje.

### Agente PPO

El agente PPO demostró un mejor desempeño, logrando tasas de éxito entre 50-60% en tableros de 5x5 con densidades de minas moderadas. A mayor densidad de minas, la tasa de éxito disminuyó a un 20-25%, manteniéndose competitiva en configuraciones de dificultad media.

## Limitaciones y Mejora

- **DQN**: Limitado en la capacidad de explorar y adaptarse en entornos complejos. Las restricciones de recursos influyeron en el tamaño del tablero y las variaciones en los estados observados.
- **PPO**: Su estructura Actor-Crítico permitió un mejor aprendizaje, aunque podría beneficiarse de un entrenamiento más prolongado o de redes neuronales más profundas para entornos con densidades de minas aún mayores.

## Créditos

Este proyecto fue desarrollado utilizando [Gymnasium](https://gymnasium.farama.org/) y [PyTorch](https://pytorch.org/).

## Video

https://youtu.be/gX90RePms9E
