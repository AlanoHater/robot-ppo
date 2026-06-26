# Achievement-Triggered DAG Reward para Locomoción Humanoide

Implementación simplificada del paper **"From Rolling Over to Walking: Enabling
Humanoid Robots to Develop Complex Motor Skills"** (Meng & Xiao, 2023,
[arXiv:2303.02581](https://arxiv.org/abs/2303.02581)), adaptada para correr en
una laptop con GPU de consumo (RTX 4050) en lugar del clúster de simulación
masivamente paralela (Isaac Gym) que usa el paper original.

Proyecto realizado para una clase de maestría — no busca reproducir el paper
al pie de la letra, sino implementar su idea central (reward DAG con
achievement triggers) a una escala que sea entrenable en hardware de
consumidor, y documentar honestamente qué tan bien funciona a esa escala y
qué partes del método sí/no se llevaron fielmente.

## Idea del paper, en una frase

En vez de una sola función de recompensa para toda la tarea ("ponte de pie y
camina"), se define un **grafo dirigido acíclico (DAG) de habilidades**, donde
el bono de recompensa de una habilidad solo se activa una vez que el agente
alcanza un *passing score* en alguna de sus habilidades predecesoras. Esto
evita el problema de recompensa dispersa (sparse reward) de intentar enseñar
la tarea completa de una sola vez.

Fórmulas del paper (idénticas en esta implementación):

```
A_ij,0 = 0
A_ij,t = max(A_ij,t-1,  R_i,t - PS_ij)
R_t    = sum_i sum_j (A_ij,t * R_j,t)
```

## Comparación verificada contra el paper real

El texto y las cifras de abajo se extrajeron leyendo el PDF completo
(arXiv:2303.02581v2), no de memoria — varias suposiciones iniciales sobre el
paper resultaron incorrectas y se corrigieron aquí.

| Aspecto | Paper (iCub, Isaac Gym) | Esta implementación | Estado |
|---|---|---|---|
| Robot | iCub V3, 32 DoF | `HumanoidStandup-v4` (Gymnasium/MuJoCo), 17 DoF | simplificado |
| Simulador | Isaac Gym + PhysX, GPU, miles de envs vectorizados | MuJoCo CPU, hasta 10 envs vía `SubprocVecEnv` | simplificado |
| Algoritmo | A2C continuo (`rl_games`, clipping tipo PPO) | PPO (`stable-baselines3`) | distinto, familia similar |
| Red neuronal | MLP 4 capas **[800, 400, 200, 100]**, activación **ELU** | MLP 4 capas **[400, 300, 200, 100]**, activación **Tanh** (default SB3) | más chica, activación distinta |
| Minibatch / LR | **65,536** / **5e-3** | **256** / **3e-4** | mucho más chico/conservador |
| Steps de entrenamiento | **2,000,000,000** (2e9), ~100 min en RTX 4090 | 1,000,000 → 3,000,000 | ~1000x menos muestras |
| Fórmula del DAG (`A_ij`, `R_t`) | — | idéntica | ✅ verificado |
| Frecuencias CPG `[0.5,0.75,1.25,1.75,2.75,3.25,4.25,4.75]` Hz | — | `primos[2,3,5,7,11,13,17,19]/4` (mismo resultado, recalculado) | ✅ verificado |
| Action clamping (rampa lineal 0→1 en la primera mitad del episodio) | — | `clamp=min(1, 2*step/episode_max)` | ✅ verificado |
| Representación ego-centric (rotar velocidades al frame de la raíz) | contribución #2 del paper | **no implementada en `main`** | rama aparte, ver abajo |
| Topología del DAG | grafo con **bifurcaciones reales** (Crouch y Crawl convergen ambos en Stand) — 6 habilidades (Roll, Kneel, Crouch, Crawl, Stand, Walk) | cadena **lineal** de 3 nodos en `main` (`lift→standup→upright`) | simplificación estructural — ver rama aparte |

La cadena lineal usada en `main` es, estructuralmente, la misma variante que
el paper prueba como *ablation* ("Linear List Reward") y reporta que falla
("el robot solo aprendió un movimiento tipo gorila"). En nuestro caso la tarea
es más simple (solo pararse, no la secuencia completa rodar→caminar), así que
no es necesariamente igual de grave, pero es una diferencia real, no solo de
escala — por eso se diseñó una versión con bifurcaciones reales en una rama
separada (ver sección dedicada).

## ¿Por qué no se usa la GPU?

La RTX 4050 queda libre durante el entrenamiento, y no es un descuido:

1. **MuJoCo en `gymnasium[mujoco]` simula la física en CPU**, no en GPU — el
   ~95% del tiempo de entrenamiento se gasta en la simulación física
   (contactos, colisiones, integración), no en la red neuronal.
2. La red es chica ([400,300,200,100]) y SB3 recolecta datos **un paso a la
   vez por entorno**; mover esos cómputos tan pequeños a GPU agrega más
   overhead de transferencia CPU↔GPU que el cómputo mismo.
3. El paper usa GPU porque **Isaac Gym simula la física directamente en GPU**
   con miles de copias en paralelo (kernels CUDA, no núcleos de CPU) — eso es
   un simulador distinto (PhysX/Isaac Gym vs MuJoCo), no solo "usar PyTorch en
   GPU". Reproducir esa ganancia requeriría reescribir el simulador completo
   (Isaac Gym / MuJoCo MJX / Brax), fuera de alcance para este proyecto.

La palanca real disponible en este stack es **paralelizar en núcleos de CPU**
(siguiente sección), no la GPU.

## Paralelización (SubprocVecEnv)

El entrenamiento original usaba un solo entorno CPU (~600 fps, dejando la CPU
al ~60-65% con 12 núcleos lógicos disponibles). Se agregó soporte para
`SubprocVecEnv` (un proceso de simulación por núcleo):

| Configuración | fps medido |
|---|---|
| 1 entorno (`DummyVecEnv`) | ~600-650 |
| 8 entornos (`SubprocVecEnv`) | ~1300-2600 |
| 10 entornos (`SubprocVecEnv`) | ~1400-1800 |

Esto permitió subir el presupuesto de entrenamiento de 1M a 3M steps sin que
el tiempo de pared creciera proporcionalmente, atacando directamente la
brecha de muestras frente al paper (aunque sigue siendo ~1000x menor).

## Estructura del repositorio

```
dag_reward_humanoid.py             Script principal (rama main: DAG lineal de 3 nodos)
results.png                        Curvas — calibración corregida, 3M steps, 10 envs en paralelo
dag_humanoid_model.zip             Modelo de esa corrida (iteración 3)
train_full.log                     Log de esa corrida

results_v1_buggy.png               Curvas del primer entrenamiento (passing score mal calibrado)
dag_humanoid_model_v1_buggy.zip    Modelo de esa primera corrida
train_full_v1_buggy.log            Log de esa primera corrida
```

Los archivos `*_v1_buggy.*` se conservan deliberadamente como evidencia de la
iteración de depuración descrita abajo (comparación antes/después de corregir
la calibración del reward).

**Rama `feature/multipath-dag-egocentric`** (en GitHub, sin mergear a `main`):
agrega un DAG con bifurcaciones reales (5 nodos, multi-path) y la
representación ego-centric que falta en `main`. Ver sección dedicada.

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
# Entrenar (single-env por defecto)
python dag_reward_humanoid.py --mode train --steps 1000000

# Entrenar en paralelo (10 entornos, ej. en un CPU de 12 núcleos lógicos)
python dag_reward_humanoid.py --mode train --steps 3000000 --n_envs 10

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
el quaternion de la raíz del robot, el entrenamiento de 1M steps (single-env)
mostró:

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

### Iteración 2 — calibración corregida, single-env, 1M steps (`results.png`)

Cambios aplicados en `dag_reward_humanoid.py`:

- `DEFAULT_PS (0,1)`: `1.0 → 0.3` (alcanzable dado el rango real de `r_lift`
  observado en la iteración 1).
- `r_upright`: se reemplazó el término del quaternion por
  `clip(z - 1.00, 0, None) * 8.0`, usando altura real en vez de orientación de
  la raíz — consistente con `r_lift` y `r_standup`, que sí usan altura y sí
  demostraron ser señales confiables.

### Iteración 3 — paralelizado, 3M steps, 10 envs (`results.png` final)

Mismo DAG lineal de 3 nodos y calibración de la iteración 2, pero con 3x más
steps gracias a la paralelización (36 min de pared en vez de las ~4-5 horas
que habría tomado en single-env). Resultado: **progreso de aprendizaje real
y medible, pero sin cruzar el umbral de pararse del todo**.

- `lift` (panel a, rojo) muestra una **tendencia ascendente clara** por
  primera vez: pasa de picos esporádicos de ~0.1 a picos sostenidos de
  ~0.4-0.55 hacia el final del entrenamiento, con el promedio suavizado
  subiendo de ~0 a ~0.3. El agente sí está aprendiendo a levantar la pelvis
  más y de forma más consistente.
- `standup` y `upright` (naranja/verde) **siguen en 0.0 durante las ~2900
  episodios completos** — la altura z del robot nunca superó 0.80 ni una
  sola vez, igual que en las iteraciones 1 y 2. Triplicar el presupuesto de
  steps no fue suficiente para cruzar ese umbral.
- El reward total por episodio (panel b) también muestra una **tendencia
  ascendente clara**, con picos que llegan hasta ~29 hacia el final. Esto
  **no es un salto de recompensa instantánea** — `ep_total` es la suma
  acumulada de reward a lo largo de todo el episodio (hasta 1000 steps), así
  que un pico de 29 es compatible con sostener un `r_lift` modesto (~0.03)
  durante buena parte del episodio, no con alcanzar `standup`. Confirma que
  el agente aprendió a *sostener* más tiempo una pelvis ligeramente elevada,
  no a completarse el pararse.
- Nota técnica sobre el mecanismo DAG: como `lift` sí superó el nuevo
  `PS=0.3` en muchos episodios, el achievement `(0,1)` sí se activó (a
  diferencia de la iteración 1). Pero como `r_standup` (la recompensa cruda,
  antes de cualquier multiplicador) permaneció en 0 todo el entrenamiento
  (z nunca > 0.80), el bono `a01 * r_standup` fue **0 de todas formas** —
  activar el achievement no ayuda si la recompensa que debía amplificar
  nunca deja de ser cero. La mejora visible en este run viene enteramente de
  `r_lift` sostenido, no del mecanismo de bono del DAG.

**Conclusión honesta:** la combinación calibración-corregida + más
paralelismo + 3x más steps sí produjo aprendizaje verificable (algo que la
iteración 1 nunca mostró), pero el robot todavía no logra completar el
pararse dentro de este presupuesto de muestras. Es consistente con la brecha
de ~1000x en steps frente al paper (3M vs 2,000M) documentada en la tabla de
comparación arriba.

## Rama experimental: DAG multi-path + ego-centric

En `feature/multipath-dag-egocentric` (no mergeada a `main`, sin entrenar
todavía) se atacan las dos brechas frente al paper real identificadas en la
tabla de comparación arriba:

**1. DAG con bifurcaciones reales.** Se generalizó `AchievementRewardWrapper`
para soportar cualquier grafo (no solo cadena lineal): los nodos raíz (sin
padre) se detectan automáticamente y su recompensa se suma siempre; el resto
solo se activa si **alguno** de sus padres supera su passing score. Nuevo
grafo de 5 nodos adaptado a `HumanoidStandup-v4` (no son las habilidades
literales del paper — Roll/Kneel/Crouch/Crawl no aplican a este robot/tarea —
sino un análogo propio):

```
pelvis_lift     \
                 > standup --[PS=4.0]--> upright
leg_extension   /                          ^
torso_straighten ---------------------------/
```

**2. Representación ego-centric.** Rota `qvel` (raíz) y `cvel` (los 14
cuerpos) del frame mundo al frame de la raíz usando el quaternion
`qpos[3:7]`, vía rotación por quaternion.

**Verificación rigurosa (no solo "no crashea"):** la primera pasada de
verificación (preservación de norma de los vectores rotados) era insuficiente
— cualquier rotación, incluso con el eje equivocado, preserva la norma. Se
verificó más a fondo:
- Offsets del vector de observación (`obs[22:28]`, `obs[185:269]`)
  confirmados **exactos** contra `d.qvel`/`d.cvel` reales del simulador, no
  solo aritmética de `nq`/`nv`.
- La función de rotación por quaternion se probó con un caso de 90° conocido
  a mano (independiente de la simulación) — correcta.
- Al comparar contra `mj_objectVelocity` (función nativa de MuJoCo) apareció
  una discrepancia que parecía un bug real. Investigando se encontró la causa:
  esa función, para un `BODY`, rota usando el **frame inercial** (`ximat`,
  ejes principales de inercia, que pueden estar desalineados del cuerpo si la
  masa no es simétrica) — no el frame del cuerpo/articulación (`xmat`, que es
  lo que representa `qpos[3:7]` y lo que pide el paper como "root reference
  frame"). Era la referencia de comparación equivocada, no un bug en la
  implementación: se confirmó que la rotación coincide exactamente con
  `xmat.T @ v` para los 4 vectores que usa el wrapper.

**Pendiente:** no se ha entrenado con esta versión. Falta validar
empíricamente si la escala de las nuevas señales raíz (`leg_extension`,
`torso_straighten`) está bien calibrada, y si el ego-centric realmente ayuda
dado el presupuesto de muestras tan por debajo del paper.

## Limitaciones conocidas

- **Presupuesto de muestras bajo**: incluso con 3M steps y 10 envs en
  paralelo, sigue siendo ~3 órdenes de magnitud menor que los 2,000 millones
  de steps del paper (logrados con simulación GPU masivamente paralela).
- **DAG simplificado en `main`**: cadena lineal de 3 nodos, sin bifurcaciones
  (ver rama experimental para la versión con bifurcaciones).
- **Sin representación ego-centric en `main`** (sí está en la rama
  experimental, sin entrenar aún).
- **Robot distinto**: `HumanoidStandup-v4` (17 DoF, Gymnasium/MuJoCo) en vez
  del iCub (32 DoF) del paper.
- **Hiperparámetros y red más chicos** que el paper (ver tabla de
  comparación) — aunque la evidencia recolectada sugiere que el cuello de
  botella principal fue la calibración del reward y el presupuesto de
  muestras, no el tamaño de la red.
- **Sin validación estadística**: una sola corrida por configuración; el
  paper reporta promedios sobre múltiples semillas.

## Referencia

Meng, Y., & Xiao, J. (2023). *From Rolling Over to Walking: Enabling Humanoid
Robots to Develop Complex Motor Skills*. arXiv:2303.02581.
https://arxiv.org/abs/2303.02581
