# Achievement-Triggered DAG Reward para Locomoción Humanoide

Implementación simplificada del paper **"From Rolling Over to Walking: Enabling
Humanoid Robots to Develop Complex Motor Skills"** (Meng & Xiao, 2023,
[arXiv:2303.02581](https://arxiv.org/abs/2303.02581)), adaptada para correr en
una laptop con GPU de consumo (RTX 4050) en lugar del clúster de simulación
masivamente paralela (Isaac Gym) que usa el paper original.

**Clase:** Robótica Cognitiva — Universidad Autónoma del Estado de México (UAEM)
**Autores:** Alan Cabrera y Enrique Albarrán

Proyecto de la clase de Robótica Cognitiva — no busca reproducir el paper
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

## Entregables del proyecto

| Archivo | Qué es |
|---------|--------|
| [`reporte_proyecto.pdf`](reporte_proyecto.pdf) | **Reporte de 3 páginas (español)** — generado con `generate_report.py` |
| [`SLIDES.md`](SLIDES.md) | **Guion de 3 diapositivas** para la presentación |
| [`demo_v3_height.mp4`](demo_v3_height.mp4) | **Video** del robot incorporándose (v3, mejor modelo) |
| [`demo_multipath_egocentric.mp4`](demo_multipath_egocentric.mp4) | Video del robot antes del fix (v2, retorciéndose) |
| `compare_v2_v3.png` / `pose_sequence_v3.png` | Figuras: antes/después y secuencia de incorporación |

Para regenerar el reporte: `python generate_report.py`. Para grabar un video del
modelo: `python dag_reward_humanoid.py --mode video --out demo.mp4`.

## Estructura del repositorio

```
dag_reward_humanoid.py        Script principal (wrapper de reward, train, demo, ablation, video)
generate_report.py            Genera el reporte PDF de 3 páginas
SLIDES.md                     Guion de 3 diapositivas
reporte_proyecto.pdf          Reporte final (3 páginas, español)

dag_humanoid_model.zip        Modelo PPO principal = v3 (mejor resultado: se sienta)
results.png                   Curvas del modelo principal (v3)
demo_v3_height.mp4            Video del modelo v3

# Evidencia de cada iteración de depuración (antes/después):
dag_humanoid_model_v1_buggy.zip      / results_v1_buggy.png       (passing score inalcanzable)
dag_humanoid_model_v2_heightblind.zip / results_v2_heightblind.png (reward de altura ciego)
dag_humanoid_model_v3_seated.zip     / results_v3_height.png      (FIX: rampa de altura → se sienta)
dag_humanoid_model_v4_feettuck.zip   / results_v4_feettuck.png    (feet_tuck: no escapa la pose en L)
```

Los modelos y curvas versionados (`*_v1_buggy`, `*_v2_heightblind`,
`*_v3_seated`, `*_v4_feettuck`) se conservan deliberadamente como evidencia del
proceso de depuración documentado abajo (comparación antes/después de cada fix).

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

### Rama experimental `feature/multipath-dag-egocentric` — DAG con bifurcaciones + ego-centric

Hipótesis a probar: si el cuello de botella era la topología lineal del DAG
(una sola "puerta" `lift→standup→upright`) y la falta de representación
ego-centric, un grafo con **bifurcaciones reales** (dos caminos alternativos
hacia `standup` y hacia `upright`) más velocidades rotadas al frame de la
raíz deberían ayudar al agente a encontrar más rutas hacia ponerse de pie.

Cambios respecto a `main`:
- DAG de 5 nodos con caminos alternativos: `pelvis_lift` y `leg_extension` →
  `standup`; `standup` y `torso_straighten` → `upright` (ver diagrama en el
  docstring de `AchievementRewardWrapper`).
- Representación ego-centric: velocidades lineales/angulares de la raíz y de
  los 14 cuerpos rotadas al frame del torso (verificado contra
  `mujoco.mj_objectVelocity` — ver historial de commits para el detalle).
- Antes de lanzar la corrida completa se encontraron y corrigieron dos bugs
  de diseño de reward por stress-testing con acciones aleatorias de rango
  completo (ver commit `c2b79f4`): las señales `leg_extension` y
  `torso_straighten` daban recompensa casi máxima desde el frame 1 sin que el
  agente hiciera nada, porque asumían una pose inicial (rodillas flexionadas)
  que no es la real en `HumanoidStandup-v4` (rodillas/abdomen arrancan en
  ≈0). Se corrigió midiendo el ángulo real al resetear, usando signo correcto
  y reduciendo la escala ~50-100x para que fueran empujones de exploración
  menores, no una fuente de reward competitiva con la altura.

**Resultado de la corrida completa (3,000,000 steps, 10 entornos en
paralelo, ~41 min):**

- `ep_rew_mean` subió de 32.9 a 67.3 y se estancó ahí — a diferencia de
  `main` (iteración 3), aquí la curva sí sube de forma sostenida en vez de
  quedar plana desde el principio.
- Pero la razón de esa subida **no es que el robot se pare más**: `standup`
  y `upright` (los nodos que dependían del DAG con bifurcaciones) se
  mantienen en 0.0 durante las ~3000 episodios completos, igual que en
  `main`. La altura (`pelvis_lift`, escala ×3, la señal "real" de progreso)
  nunca supera ~0.05 de recompensa (z apenas pasa de 0.30 + 0.017 ≈ 0.317) —
  el robot prácticamente no se levanta del piso en ningún momento.
- Lo que sí sube es la suma de `leg_extension` (satura en su tope de 0.045
  casi de inmediato) + `torso_straighten` (satura en su tope de 0.03) — dos
  señales de ángulo de articulación aisladas, deliberadamente pequeñas y
  acotadas. El agente aprende a mantener esos dos ángulos en su rango
  recompensado y nada más; el ~67 de reward total es básicamente
  `0.045 + 0.03 ≈ 0.075` por step × 1000 steps, no progreso real hacia pararse.
- **La hipótesis no se confirma**: agregar bifurcaciones al DAG y
  representación ego-centric no resolvió el estancamiento de `standup`/
  `upright`. El cuello de botella real parece estar en que la política nunca
  encuentra una secuencia de acciones que levante el centro de masa de forma
  sostenida — ni con la señal de altura aislada (`pelvis_lift`) ni con los
  caminos alternativos. Es decir, el problema no era la forma del grafo, era
  que ninguna señal en este presupuesto de muestras logra que 17 DoF
  coordinen un movimiento de incorporarse desde el suelo.
- Señal de alerta adicional: `std` (desviación de la distribución de
  acciones de PPO) creció sin freno durante todo el entrenamiento (1.0 →
  4.4), y `approx_kl`/`clip_fraction` se mantuvieron anormalmente altos
  (0.25-0.5 / ~0.65-0.77, muy por encima del rango sano <0.02 / <0.3). Esto
  sugiere que la política se volvió progresivamente **más errática**, no más
  estable — consistente con `ent_coef=0.005` (bono de entropía) empujando
  hacia mayor varianza de acción mientras el reward dominante (las dos
  señales de ángulo acotadas) no penaliza esa varianza, porque se puede
  farmear con ruido de torque sin necesidad de coordinación.
- Se encontró además un bug de logging (no de entrenamiento) en
  `DAGRewardCallback`: el acumulador de reward total por episodio era un
  único escalar compartido entre los 10 entornos paralelos en vez de uno por
  entorno, lo que inflaba ~10x el panel "(b) Reward Total por Episodio" de
  `results.png` en 1 de cada 10 episodios (con un patrón de diente de
  sierra). No afectaba el entrenamiento real ni el `ep_rew_mean` que reporta
  SB3 (ese viene de `Monitor`, una fuente independiente) — solo ese gráfico.
  Corregido en el código (acumuladores por entorno) para corridas futuras.

Conclusión (v2): con este presupuesto de muestras (3M steps, MuJoCo CPU),
ni la topología DAG lineal (`main`) ni la de bifurcaciones + ego-centric
lograban que el agente se levantara. **Pero esa conclusión resultó
incompleta** — ver la iteración siguiente, que identificó la verdadera causa
raíz y sí consiguió que el robot se incorporara.

### Iteración v3 — reward de altura continuo (el robot por fin se levanta del piso)

Tras la corrida v2 se hizo una verificación directa (cargar el modelo
entrenado y medir la altura real del torso `z` durante un episodio) en vez de
seguir suponiendo. El resultado fue revelador:

| | valor |
|---|---|
| `z` inicial (acostado) | 0.103 |
| `z` durante todo el episodio v2 | min 0.076 · mean 0.116 · **max 0.152** |
| umbral de `pelvis_lift` (v2) | z > **0.30** |
| % de steps con z > 0.30 | **0.0 %** |

**La causa raíz no era la topología del DAG ni la representación ego-centric:
era el reward de altura.** `clip(z - 0.30, 0, None)` vale exactamente 0 —y su
gradiente es 0— para todo z < 0.30, pero el robot vivía en z ∈ [0.076, 0.152].
Las tres señales de altura (`pelvis_lift`, `standup`, `upright`) estaban
**muertas** en todo el rango alcanzable: el agente nunca recibía la más mínima
señal de "sube un poco", así que lo único que podía optimizar eran las señales
de ángulo, y por eso las farmeaba. Un mínimo local perfecto.

Corrección (commit del fix v3):
- **Reward de altura: umbral → rampa continua.** Rampas solapadas con
  gradiente no-cero *desde el suelo*, de modo que cada centímetro que sube `z`
  dé más recompensa:
  - `pelvis_lift = clip((z-0.10)/0.20, 0, 1)`  (despegar: 0.10→0.30)
  - `standup     = clip((z-0.30)/0.50, 0, 1)`  (sentarse: 0.30→0.80)
  - `upright     = clip((z-0.80)/0.50, 0, 1)`  (pararse:  0.80→1.30)
- **Passing scores** recalibrados al nuevo rango [0,1] (`(0,2)`:0.5, `(2,4)`:0.5).
- **Hiperparámetros PPO** (la inestabilidad observada en v2): `n_epochs`
  10→5, `batch_size` 256→512, `ent_coef` 0.005→0, y se agregó `target_kl=0.03`.

Resultado de la corrida completa v3 (3M steps, 10 envs, ~40 min):

| métrica | v2 (reward roto) | **v3 (rampa de altura)** |
|---|---|---|
| `ep_rew_mean` final | ~67 (plano) | **~850** (sube sostenido) |
| `z` máximo alcanzado | 0.152 | **0.450** (~3×) |
| `z` promedio | 0.116 | **0.291** |
| % tiempo despegado (z>0.30) | 0 % | **42 %** |
| `standup` (skill) | 0.0 plano | **~0.25** (despega) |
| `std` de PPO | 4.4 explotando | **0.94 estable** |
| `approx_kl` / `clip_fraction` | 0.40 / 0.70 | **0.035 / 0.21** |

El robot pasó de **retorcerse acostado sin despegar del piso** (v2) a hacer un
**esfuerzo coordinado de incorporarse**: dobla las piernas, arquea la pelvis y
empuja el torso hacia arriba hasta z≈0.45, pasando ~42 % del episodio con el
cuerpo despegado del suelo. No llega a sentarse del todo (z>0.55: 0 %) ni a
pararse (`upright` sigue en 0), pero es un progreso real, medible y
visualmente claro. Evidencia en video: `demo_v3_height.mp4` (v3, se levanta)
vs `demo_multipath_egocentric.mp4` (v2, plano en el piso).

Lección para el reporte: el reward shaping del paper funciona, pero es
extremadamente sensible a que cada señal tenga gradiente *en la región donde
el agente realmente opera*. Un umbral mal colocado convierte una señal densa
en una sparse de facto, y el agente —racionalmente— explota lo que sí da
recompensa. El bug no se veía en las métricas de PPO (que parecían "aprender")
ni en el reward total; solo apareció al medir la cantidad física (`z`) que se
suponía debía optimizar.

### Iteración v4 — escapar la pose en "L" con `feet_tuck` (warm-start)  *(en progreso)*

Tras v3 el robot se sienta erguido pero se estanca ahí. Midiendo la geometría
de la pose final se identificó por qué: el robot quedó en **"sentado en L"**,
con las piernas extendidas al frente (distancia horizontal pie→torso ≈ 0.80) y
los pies sin plantar. Es un óptimo local estable pero **un dead-end
geométrico para pararse**: con los pies extendidos al frente no puede meterlos
bajo el centro de masa para empujarse hacia arriba. Por eso `upright` se quedó
en 0 pese a que su rampa de reward sí tenía gradiente.

Intervención (commit de v4):
- **Nodo nuevo en el DAG: `feet_tuck`** — premia recoger los pies hacia la
  proyección horizontal del torso (pose de cuclillas), `feet_under =
  clip(1 - dist/0.5, 0, 1)`, **multiplicado por** `pelvis_lift` para que no
  sea farmeable ovillándose acostado (solo cuenta con altura). Es nodo raíz
  (se suma siempre) **y** padre de `upright` — semánticamente, "recoger los
  pies desbloquea pararse". Verificado: da 0 en la pose en L (empuja a
  cambiarla) y no es explotable con acciones aleatorias.
- **Warm-start**: se reanuda desde el modelo v3 (que ya sabe sentarse) en vez
  de re-entrenar desde cero — nueva opción `--resume_from`.
- **Exploración reactivada** (`ent_coef` 0 → 0.002, pequeño) para que la
  política pruebe la transición arriesgada de sentado a cuclillas sin volver a
  desestabilizarse como en v2.

Resultado de la corrida (warm-start 3M steps, 10 envs) — **la intervención no
funcionó**:

| medida | v3 (sentado en L) | v4 (feet_tuck) |
|---|---|---|
| `z` promedio | 0.291 | **0.209** (bajó) |
| `z` máximo | 0.450 | 0.429 |
| dist. pie→torso | 0.80 | 0.68 (mejoró poco) |
| `feet_tuck` promedio | — | **0.004** (~0) |
| % steps con pies recogidos | — | 9 % |
| % steps parado (z>0.80) | 0 % | **0 %** |

El warm-start con `feet_tuck` **no sacó al robot de la pose en L**; la altura
promedio incluso bajó un poco. Los picos de `feet_tuck` en las curvas son
transitorios — en promedio es ~0, el robot recoge los pies solo 9 % del tiempo
y no de forma sostenida.

Por qué falló: la pose en L es un **atractor demasiado fuerte**. Recoger los
pies exige una transición que temporalmente baja la altura y desestabiliza, y
un bonus pequeño (`feet_tuck`) con exploración baja (`ent_coef=0.002`) no
compensa abandonar un óptimo local estable. Detalle revelador: el reward total
*subió* (~900 vs ~850) por los `feet_tuck` ocasionales, **pero el objetivo
real (altura) empeoró** — un ejemplo claro de optimizar el proxy de
recompensa sin mejorar la meta. El modelo v3 (sentado erguido) se conserva
como el mejor resultado; v4 se preserva como `*_v4_feettuck.*` para
documentar el experimento negativo.

Lecciones para el reporte (ambas valiosas aunque el experimento "fallara"):
1. Sacar a una política de un óptimo local estable requiere más que un bonus
   suave: o una penalización activa de la pose no deseada, o exploración
   mucho mayor, o un curriculum que inicialice desde poses variadas — no basta
   con añadir una recompensa positiva tímida.
2. El reward total no es un buen indicador de éxito por sí solo; hay que medir
   la cantidad física objetivo (aquí `z` y la geometría de la pose).

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
