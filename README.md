# Achievement-Triggered DAG Reward para Locomoción Humanoide

Implementación simplificada del paper **"From Rolling Over to Walking: Enabling
Humanoid Robots to Develop Complex Motor Skills"** (Meng & Xiao, 2023,
[arXiv:2303.02581](https://arxiv.org/abs/2303.02581)), adaptada para correr en
una laptop con GPU de consumo (RTX 4050) en lugar del clúster de simulación
masivamente paralela (Isaac Gym) que usa el paper original.

Proyecto realizado para una clase de maestría — no busca reproducir el paper
al pie de la letra, sino implementar su idea central (reward DAG con
achievement triggers) a una escala que sea entrenable en hardware de
consumidor, y documentar honestamente qué tan bien funciona a esa escala.

## Idea del paper, en una frase

En vez de una sola función de recompensa para toda la tarea ("ponte de pie y
camina"), se define un **grafo dirigido acíclico (DAG) de habilidades**
encadenadas (p. ej. `lift → standup → upright → walk`), donde el bono de
recompensa de la habilidad *i+1* solo se activa una vez que el agente alcanza
un *passing score* en la habilidad *i*. Esto evita el problema de recompensa
dispersa (sparse reward) de intentar enseñar la tarea completa de una sola vez.

Fórmulas implementadas (idénticas al paper, sección de reward shaping):

```
A_ij,0 = 0
A_ij,t = max(A_ij,t-1,  R_i,t - PS_ij)
R_t    = sum_i sum_j (A_ij,t * R_j,t)
```

## Adaptaciones por limitación de hardware

| Paper original                          | Esta implementación                         |
|------------------------------------------|----------------------------------------------|
| Simulador Isaac Gym, miles de envs paralelos | MuJoCo CPU, 1 solo entorno                |
| Robot iCub, 32 DoF                        | `HumanoidStandup-v4` (Gymnasium), 17 DoF      |
| Decenas de millones de steps              | 1,000,000 steps                              |
| DAG de habilidades completo (incluye caminar) | DAG simplificado de 3 nodos: `lift → standup → upright` |

Contribuciones del paper que sí se implementan:
1. **Achievement-triggered multi-path reward** (el DAG descrito arriba).
2. **Señales CPG** (Central Pattern Generator): 8 senos/cosenos coprimos
   (frecuencias = primeros 8 primos / 4, rango 0.5–5 Hz) concatenados a la
   observación, igual que en el paper.
3. **Action clamping** proporcional al progreso del episodio: las acciones se
   escalan por `min(1, 2*step/episode_max_steps)`, limitando movimientos
   bruscos al inicio de cada episodio.

## Estructura del repositorio

```
dag_reward_humanoid.py        Script principal (wrapper de reward, entrenamiento, demo, ablation)
results.png                   Curvas de entrenamiento — versión final (calibración corregida)
dag_humanoid_model.zip         Modelo PPO entrenado — versión final
train_full.log                Log de stdout del entrenamiento final

results_v1_buggy.png          Curvas del primer entrenamiento (passing score mal calibrado)
dag_humanoid_model_v1_buggy.zip  Modelo de esa primera corrida
train_full_v1_buggy.log       Log de esa primera corrida
```

Los archivos `*_v1_buggy.*` se conservan deliberadamente como evidencia de la
iteración de depuración descrita abajo (útil para el reporte: comparación
antes/después de corregir la calibración del reward).

## Instalación

Requiere Anaconda/Miniconda. Crear y poblar el entorno:

```bash
conda create -n rl_paper python=3.10 -y
conda activate rl_paper
pip install "gymnasium[mujoco]" "stable-baselines3[extra]" matplotlib numpy
```

Verificar:

```bash
python -c "import gymnasium as gym; import mujoco; import stable_baselines3; print('Setup OK')"
```

**Nota Windows 11:** si la verificación falla con un error de DLL del tipo
*"Una directiva de Control de aplicaciones bloqueó este archivo"*, es
**Smart App Control** bloqueando las DLLs de `mujoco` por no tener firma
"reputable". Hay que desactivarlo desde Configuración → Privacidad y
seguridad → Seguridad de Windows → App y control del navegador → Smart App
Control. **Es irreversible sin reinstalar Windows**, así que es una decisión
a tomar con conocimiento de causa.

## Uso

Todos los comandos se ejecutan desde esta carpeta (`C:\rl_paper`):

```bash
# Entrenar (por defecto 500,000 steps; el modelo final usó --steps 1000000)
python dag_reward_humanoid.py --mode train --steps 1000000

# Ver al robot con el modelo ya entrenado (ventana de MuJoCo)
python dag_reward_humanoid.py --mode demo

# Ablation study: compara DAG completo vs sin CPG vs sin action-clamp vs ninguno
python dag_reward_humanoid.py --mode ablation --steps 150000
```

Si `conda activate` no funciona en tu shell, se puede invocar el Python del
entorno directamente:

```
C:\Users\<usuario>\miniconda3\envs\rl_paper\python.exe dag_reward_humanoid.py --mode demo
```

## Resultados y hallazgos

### Iteración 1 — passing score mal calibrado (`*_v1_buggy`)

Con `DEFAULT_PS = {(0,1): 1.0, (1,2): 4.0}` y una métrica `upright` basada en
el quaternion de la raíz del robot, el entrenamiento de 1M steps mostró:

- `standup` en **0.0 durante los 1000 episodios completos** — el robot nunca
  superó z=0.80 ni una sola vez.
- `lift` alcanzó como máximo ~0.5, pero el *passing score* de 1.0 exigía
  `r_lift > 1.0` (z > 0.63) para desbloquear el bono hacia `standup` → **el
  achievement nunca se activó**, así que el agente entrenó casi todo el
  tiempo con la señal escasa de `r_lift` aislada, sin el mecanismo
  "achievement-triggered" del paper realmente funcionando.
- `upright` aparecía saturado en 3.0 desde el episodio 1, lo cual era
  engañoso: en el XML de `HumanoidStandup-v4`, la postura "acostado" se logra
  flexionando cadera/rodilla/abdomen, no rotando el cuerpo raíz, así que el
  quaternion de la raíz permanece ≈identidad sin importar la postura real del
  robot.

Resultado visible: el robot se mueve de forma errática en el piso sin lograr
incorporarse — consistente con los datos, no es un fallo aleatorio.

### Iteración 2 — calibración corregida (`results.png` final)

Cambios aplicados en `dag_reward_humanoid.py` (ver historial de git para el
diff exacto y la justificación completa):

- `DEFAULT_PS (0,1)`: `1.0 → 0.3` (alcanzable dado el rango real de `r_lift`
  observado en la iteración 1).
- `r_upright`: se reemplazó el término del quaternion por
  `clip(z - 1.00, 0, None) * 8.0`, usando altura real en vez de orientación de
  la raíz — consistente con `r_lift` y `r_standup`, que sí usan altura y sí
  demostraron ser señales confiables.

*(Completar tras la corrida: resumen de si el robot logra `standup` con la
calibración corregida, comparando `results.png` vs `results_v1_buggy.png`.)*

## Limitaciones conocidas

- **Presupuesto de muestras bajo**: 1M steps en un solo entorno CPU es órdenes
  de magnitud menor que el entrenamiento masivamente paralelo (miles de envs
  en Isaac Gym) del paper original. Es esperable que el comportamiento
  resultante esté lejos de óptimo incluso con la calibración corregida.
- **DAG simplificado**: solo 3 nodos (`lift → standup → upright`); el paper
  encadena más habilidades hasta llegar a caminar.
- **Robot distinto**: `HumanoidStandup-v4` (17 DoF, Gymnasium/MuJoCo) en vez
  del iCub (32 DoF) del paper.
- **Sin validación estadística**: una sola corrida por configuración; el
  paper reporta promedios sobre múltiples semillas.

## Referencia

Meng, Y., & Xiao, J. (2023). *From Rolling Over to Walking: Enabling Humanoid
Robots to Develop Complex Motor Skills*. arXiv:2303.02581.
https://arxiv.org/abs/2303.02581
