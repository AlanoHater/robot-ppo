"""
Achievement-Triggered Multi-Path Reward for Humanoid Locomotion
===============================================================
Adaptación de: "From Rolling Over to Walking" (Meng & Xiao, 2023)
              arXiv:2303.02581

Limitación de hardware: RTX 4050 laptop (8GB VRAM) →
  - Simulador: MuJoCo CPU (sin Isaac Gym)
  - Robot: HumanoidStandup-v4 (17 DoF vs 32 DoF del iCub)
  - Escala: entrenamiento single-env, ~500K steps

Contribuciones implementadas (rama feature/multipath-dag-egocentric):
  1. Achievement-triggered multi-path reward (DAG con bifurcaciones reales,
     no solo cadena lineal — ver AchievementRewardWrapper)
  2. Representación ego-centric (velocidades rotadas al frame de la raíz)
  3. CPG signals (coprime sine waves)
  4. Action clamping proporcional al progreso del episodio

Instalación:
  pip install gymnasium[mujoco] stable-baselines3 matplotlib numpy
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # backend no-interactivo: plt.show() no bloquea ni
                        # abre ventana (evita que el proceso se cuelgue al
                        # final del entrenamiento). results.png se guarda
                        # igual con savefig; ábrelo como imagen.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback


# ================================================================
# 1.  ACHIEVEMENT-TRIGGERED DAG REWARD WRAPPER
# ================================================================

class AchievementRewardWrapper(Wrapper):
    """
    Implementa el grafo dirigido acíclico (DAG) de recompensas del paper,
    con soporte genérico para bifurcaciones (varios padres por nodo), como
    en el grafo real del paper (Crouch y Crawl convergen ambos en Stand).

    Grafo de habilidades (versión multi-path para HumanoidStandup-v4):

        pelvis_lift  --[PS]--\
                               > standup --[PS]--\
        leg_extension --[PS]--/                   > upright
        torso_straighten --------------[PS]-------/

    'pelvis_lift', 'leg_extension' y 'torso_straighten' son señales raíz
    (sin padres) propuestas como proxy de las distintas estrategias con que
    un humanoide de 17 DoF puede empezar a incorporarse — no son las
    habilidades literales del paper (Roll/Kneel/Crouch/Crawl, pensadas para
    el iCub partiendo de posición supina), sino un análogo adaptado a
    HumanoidStandup-v4. Cualquier nodo puede tener 0 o más padres; los
    nodos sin padre (raíz) se suman siempre, los demás solo se activan
    cuando ALGUNO de sus padres supera su passing score (OR ponderado).

    Fórmulas del paper:
        A_ij,0 = 0
        A_ij,t = max(A_ij,t-1,  R_i,t - PS_ij)
        R_t    = sum_i sum_j (A_ij,t * R_j,t)
    """

    SKILL_NAMES = ['pelvis_lift', 'leg_extension', 'standup', 'torso_straighten', 'upright', 'feet_tuck']
    # Passing scores recalibrados a las rampas de altura [0,1] (ver
    # _compute_stage_rewards). Un nodo de altura se desbloquea cuando su
    # padre ya avanzó parte de su tramo:
    DEFAULT_PS  = {
        (0, 2): 0.5,    # pelvis_lift>0.5 (z>0.20) desbloquea standup
        (1, 2): 0.02,   # leg_extension -> standup  (camino alternativo débil)
        (2, 4): 0.5,    # standup>0.5 (z>0.55) desbloquea upright
        (3, 4): 0.015,  # torso_straighten -> upright (camino alternativo débil)
        (5, 4): 0.3,    # feet_tuck>0.3 (pies recogidos bajo el cuerpo) desbloquea upright
    }

    def __init__(self, env, passing_scores=None, use_cpg=True,
                 use_action_clamp=True, use_egocentric=True, episode_max_steps=1000):
        super().__init__(env)

        self.passing_scores    = passing_scores or self.DEFAULT_PS
        self.use_cpg           = use_cpg
        self.use_action_clamp  = use_action_clamp
        self.use_egocentric    = use_egocentric
        self.episode_max_steps = episode_max_steps

        # Nodos raíz: los que nunca aparecen como destino (j) de ninguna arista.
        targets    = {j for (_, j) in self.passing_scores}
        n_skills   = len(self.SKILL_NAMES)
        self.roots = [i for i in range(n_skills) if i not in targets]

        # CPG: primeros 8 primos / 4 → rango 0.5–5 Hz (igual que el paper)
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        self.cpg_freqs = np.array(primes, dtype=np.float32) / 4.0

        obs_dim = env.observation_space.shape[0]
        # Offsets del vector de obs de HumanoidStandup-v4 (nq=24, nv=23):
        #   qpos[2:]  -> 22   (0:22)
        #   qvel      -> 23   (22:45)   raíz libre: lineal qvel[22:25], angular qvel[25:28]
        #   cinert    -> 140  (45:185)
        #   cvel      -> 84   (185:269) 14 cuerpos * 6 (angular(3)+lineal(3)) en frame mundo
        #   qfrc_act. -> 23   (269:292)
        #   cfrc_ext  -> 84   (292:376)
        assert obs_dim == 376, f"offsets ego-centric asumen obs_dim=376, recibido {obs_dim}"
        self._qvel_root_slice = slice(22, 28)
        self._cvel_slice      = slice(185, 269)
        self._n_bodies        = 14

        if self.use_cpg:
            n_cpg = len(self.cpg_freqs) * 2
            low  = np.concatenate([env.observation_space.low,  np.full(n_cpg, -1.0, dtype=np.float32)])
            high = np.concatenate([env.observation_space.high, np.full(n_cpg,  1.0, dtype=np.float32)])
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Body-ids para la señal feet_tuck (pies bajo el centro de masa).
        import mujoco
        m = env.unwrapped.model
        self._bid_torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        self._bid_rfoot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_foot')
        self._bid_lfoot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'left_foot')

    @staticmethod
    def _rotate_by_inv_quat(v, quat):
        """Rota el vector 3D v por la inversa del quaternion (w,x,y,z)."""
        w, x, y, z = quat
        # Quaternion inverso (unitario): conjugado (w,-x,-y,-z)
        qi = np.array([w, -x, -y, -z], dtype=np.float64)
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)

        def qmul(a, b):
            aw, ax, ay, az = a
            bw, bx, by, bz = b
            return np.array([
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
            ])

        q_conj = np.array([qi[0], -qi[1], -qi[2], -qi[3]])
        res = qmul(qmul(qi, qv), q_conj)
        return res[1:]

    def _egocentrize_obs(self, obs, root_quat):
        """
        Transforma las velocidades de qvel (raíz) y cvel (todos los cuerpos)
        del frame mundo al frame de la raíz del robot, según la
        representación ego-centric del paper (Sección III.B).
        """
        obs = obs.copy()

        lin = obs[22:25]
        ang = obs[25:28]
        obs[22:25] = self._rotate_by_inv_quat(lin, root_quat)
        obs[25:28] = self._rotate_by_inv_quat(ang, root_quat)

        cvel = obs[self._cvel_slice].reshape(self._n_bodies, 6)
        for b in range(self._n_bodies):
            cvel[b, 0:3] = self._rotate_by_inv_quat(cvel[b, 0:3], root_quat)
            cvel[b, 3:6] = self._rotate_by_inv_quat(cvel[b, 3:6], root_quat)
        obs[self._cvel_slice] = cvel.reshape(-1)

        return obs

    def _process_obs(self, obs):
        if self.use_egocentric:
            root_quat = self.env.unwrapped.data.qpos[3:7]
            obs = self._egocentrize_obs(obs, root_quat)
        if self.use_cpg:
            obs = np.concatenate([obs, self._cpg_signal()]).astype(np.float32)
        return obs

    def _reset_episode_state(self):
        self.achievements = {k: 0.0 for k in self.passing_scores}
        self.step_count   = 0
        self.t            = 0.0

    def _cpg_signal(self):
        cpg0  = np.sin(2 * np.pi * self.t * self.cpg_freqs)
        cpg_p = np.sin(2 * np.pi * self.t * self.cpg_freqs - np.pi)
        return np.concatenate([cpg0, cpg_p]).astype(np.float32)

    def _compute_stage_rewards(self):
        data = self.env.unwrapped.data
        z    = float(data.qpos[2])

        # --- Señales de altura: RAMPAS CONTINUAS, no umbrales -------------
        # Bug encontrado en la corrida v2 (verificado empíricamente): el
        # robot vive en z ∈ [0.076, 0.152] (acostado ~0.10, de pie ~1.4) y
        # los umbrales anteriores —clip(z-0.30,...), clip(z-0.80,...)— daban
        # EXACTAMENTE 0 (y gradiente 0) en todo ese rango: 0% de los steps
        # superaban z=0.30. Las señales de altura estaban muertas, así que el
        # agente solo podía farmear las de ángulo y nunca tenía incentivo de
        # subir. La corrección: rampas solapadas con gradiente no-cero DESDE
        # el suelo, de modo que cada centímetro que sube z dé más recompensa.
        #   pelvis_lift: despegar  0.10 -> 0.30
        #   standup    : sentarse  0.30 -> 0.80
        #   upright    : pararse   0.80 -> 1.30
        r_pelvis_lift = np.clip((z - 0.10) / 0.20, 0.0, 1.0)
        r_standup     = np.clip((z - 0.30) / 0.50, 0.0, 1.0)
        r_upright     = np.clip((z - 0.80) / 0.50, 0.0, 1.0)

        # --- Señales de ángulo: empujones de exploración pequeños ---------
        # qpos[13]=right_knee, qpos[17]=left_knee, qpos[8]=abdomen_y. Ambos
        # empiezan ~0 al resetear; ángulo con signo y escala chica (×0.02)
        # para que NO compitan con las rampas de altura (que llegan a 1.0).
        # Máximos: leg_extension ≤ 0.03, torso_straighten ≤ 0.02.
        knee_now    = (float(data.qpos[13]) + float(data.qpos[17])) / 2.0
        abdomen_now = float(data.qpos[8])
        r_leg_extension   = np.clip(-knee_now - 0.05, 0.0, 1.5) * 0.02
        r_torso_straight  = np.clip(-abdomen_now - 0.05, 0.0, 1.0) * 0.02

        # --- feet_tuck: pies bajo el centro de masa (pose de cuclillas) ----
        # Diagnóstico v3: el robot quedó atrapado en "sentado en L" (pies
        # extendidos al frente, dist horiz ~0.80 del torso) — un dead-end
        # geométrico para pararse, porque no puede meter los pies bajo el CoM
        # para empujarse. Esta señal premia recoger los pies hacia la
        # proyección horizontal del torso. feet_under: 1 si pies bajo el
        # cuerpo (dist≤0), 0 si extendidos (dist≥0.5). Se MULTIPLICA por
        # r_pelvis_lift (rampa de altura desde el suelo) para que NO se pueda
        # farmear ovillándose acostado: solo cuenta cuando ya hay altura.
        # Escala ×1.0 para que sea competitiva con las señales de altura y
        # realmente saque a la política del atractor de la pose en L.
        torso_xy = data.xpos[self._bid_torso][:2]
        rfoot_xy = data.xpos[self._bid_rfoot][:2]
        lfoot_xy = data.xpos[self._bid_lfoot][:2]
        dist = (np.linalg.norm(rfoot_xy - torso_xy) + np.linalg.norm(lfoot_xy - torso_xy)) / 2.0
        feet_under  = np.clip(1.0 - dist / 0.5, 0.0, 1.0)
        r_feet_tuck = r_pelvis_lift * feet_under

        return [r_pelvis_lift, r_leg_extension, r_standup, r_torso_straight, r_upright, r_feet_tuck]

    def _dag_reward(self, stage_rewards):
        total = sum(stage_rewards[r] for r in self.roots)
        for (i, j), a in self.achievements.items():
            if a > 0:
                total += a * stage_rewards[j]
        return total

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_episode_state()
        return self._process_obs(obs), info

    def step(self, action):
        if self.use_action_clamp:
            clamp  = min(1.0, 2.0 * self.step_count / self.episode_max_steps)
            action = action * clamp

        obs, _, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self.t          += 1.0 / 60.0

        stage_rewards = self._compute_stage_rewards()

        for (i, j), ps in self.passing_scores.items():
            self.achievements[(i, j)] = max(self.achievements[(i, j)], stage_rewards[i] - ps)

        total_reward = self._dag_reward(stage_rewards)

        info.update({'stage_rewards': stage_rewards,
                     'achievements' : dict(self.achievements),
                     'dag_reward'   : total_reward})

        return self._process_obs(obs), float(total_reward), terminated, truncated, info


# ================================================================
# 2.  CALLBACK DE LOGGING
# ================================================================

class DAGRewardCallback(BaseCallback):
    # _ep_skill_max/_ep_total deben ser UNA entrada por entorno paralelo: con
    # n_envs>1 (SubprocVecEnv) los 10 entornos truncan a la vez cada 1000
    # steps, así que un único acumulador compartido se llena con los pasos de
    # los 10 entornos antes del primer reset del lote y solo el primer env (i=0)
    # del lote se queda con esa suma ~10x inflada; los otros 9 quedan con la
    # recompensa de un solo paso. Verificado: explica el patrón de "diente de
    # sierra" (picos ~10x ep_rew_mean intercalados con valores casi 0) que
    # aparecía en el panel (b) de results.png. No afectaba a ep_rew_mean de SB3
    # (logueado correctamente por Monitor, fuente independiente) ni al modelo
    # entrenado — solo a este gráfico de diagnóstico.
    def __init__(self, n_skills=None, verbose=0, n_envs=1):
        super().__init__(verbose)
        self.skill_names    = AchievementRewardWrapper.SKILL_NAMES
        self.n_skills       = n_skills or len(self.skill_names)
        self.ep_max_skill   = []
        self.ep_total       = []
        self._ep_skill_max  = [[0.0] * self.n_skills for _ in range(n_envs)]
        self._ep_total      = [0.0] * n_envs

    def _on_step(self) -> bool:
        infos   = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones   = self.locals.get('dones', [False])

        for i, info in enumerate(infos):
            sr = info.get('stage_rewards', [0.0] * self.n_skills)
            for k in range(self.n_skills):
                self._ep_skill_max[i][k] = max(self._ep_skill_max[i][k], sr[k])
            self._ep_total[i] += rewards[i] if i < len(rewards) else 0.0
            if dones[i]:
                self.ep_max_skill.append(list(self._ep_skill_max[i]))
                self.ep_total.append(self._ep_total[i])
                self._ep_skill_max[i] = [0.0] * self.n_skills
                self._ep_total[i]     = 0.0
        return True


# ================================================================
# 3.  PLOTS
# ================================================================

SKILL_COLORS = ['#e06c75', '#d19a66', '#e5c07b', '#61afef', '#98c379', '#c678dd']

def smooth(x, w=5):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')

def plot_training_curves(callback, save_path='results.png'):
    if not callback.ep_max_skill:
        print("Sin datos suficientes para graficar.")
        return

    data   = np.array(callback.ep_max_skill)
    totals = np.array(callback.ep_total)
    n_ep   = len(data)
    w      = max(3, n_ep // 10)

    fig = plt.figure(figsize=(12, 5), facecolor='#1e1e2e')
    fig.suptitle(
        'Achievement-Triggered DAG Reward — HumanoidStandup-v4\n'
        'Adaptación de Meng & Xiao (2023)  ·  RTX 4050 Laptop',
        color='white', fontsize=11
    )
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in (ax1, ax2):
        ax.set_facecolor('#282a36')
        ax.tick_params(colors='#cdd6f4')
        ax.xaxis.label.set_color('#cdd6f4')
        ax.yaxis.label.set_color('#cdd6f4')
        for sp in ax.spines.values():
            sp.set_edgecolor('#45475a')

    for k, sname in enumerate(AchievementRewardWrapper.SKILL_NAMES):
        col = SKILL_COLORS[k]
        ax1.plot(data[:, k], alpha=0.2, color=col)
        if n_ep >= w:
            s = smooth(data[:, k], w)
            ax1.plot(np.arange(len(s)) + w//2, s, color=col, label=sname, lw=2)

    ax1.set_xlabel('Episodio'); ax1.set_ylabel('Max reward (skill)')
    ax1.set_title('(a) Max Reward por Habilidad', color='#cdd6f4')
    ax1.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='white', fontsize=8)

    ax2.plot(totals, alpha=0.2, color='#89b4fa')
    if n_ep >= w:
        s = smooth(totals, w)
        ax2.plot(np.arange(len(s)) + w//2, s, color='#89b4fa', lw=2)
    ax2.axhline(0, color='#45475a', linestyle='--', lw=0.8)
    ax2.set_xlabel('Episodio'); ax2.set_ylabel('Reward DAG total')
    ax2.set_title('(b) Reward Total por Episodio', color='#cdd6f4')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1e1e2e')
    print(f"[OK] Figura guardada: {save_path}")
    plt.show()


# ================================================================
# 4.  ENTRENAMIENTO
# ================================================================

def make_env():
    base = gym.make('HumanoidStandup-v4')
    env  = AchievementRewardWrapper(base, use_cpg=True, use_action_clamp=True, use_egocentric=True)
    return Monitor(env)

def train(total_timesteps=500_000, save_path='dag_humanoid_model', n_envs=1,
          resume_from=None, ent_coef=0.0):
    print("=" * 60)
    print("DAG Reward Humanoid  —  Meng & Xiao (2023) adaptation")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Entornos paralelos: {n_envs}")
    if resume_from:
        print(f"Warm-start desde: {resume_from}.zip  (ent_coef={ent_coef})")
    print("=" * 60)

    if n_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env])

    # Hiperparámetros revisados tras la corrida v2: aquella mostraba
    # approx_kl≈0.4 (sano <0.02), clip_fraction≈0.7 (sano <0.3) y std de las
    # acciones explotando a 4.4 — política cada vez más errática en vez de
    # converger. Causa: demasiados updates agresivos por rollout + ent_coef
    # empujando la varianza sin freno. Correcciones:
    #   n_epochs 10->5, batch_size 256->512  (menos updates por rollout)
    #   ent_coef 0.005->0.0                  (no inflar std; el reward denso
    #                                          ya guía la exploración)
    #   + target_kl=0.03                     (corta epochs si el KL se dispara)
    if resume_from and os.path.exists(resume_from + '.zip'):
        # Warm-start: continuar desde un modelo ya entrenado (p.ej. v3, que
        # aprendió a sentarse) en vez de re-aprender desde cero. El obs/action
        # space no cambió con feet_tuck, así que el modelo carga tal cual; el
        # value function se readapta al nuevo reward (que ahora incluye tuck).
        # ent_coef>0 reactiva algo de exploración para escapar el óptimo local
        # de la pose en "L".
        model = PPO.load(resume_from, env=env)
        model.ent_coef = ent_coef
    else:
        model = PPO(
            policy        = 'MlpPolicy',
            env           = env,
            n_steps       = 2048,
            batch_size    = 512,
            n_epochs      = 5,
            learning_rate = 3e-4,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = ent_coef,
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            target_kl     = 0.03,
            policy_kwargs = dict(net_arch=[400, 300, 200, 100]),
            verbose       = 1,
        )

    callback = DAGRewardCallback(n_envs=n_envs)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True,
                reset_num_timesteps=(resume_from is None))
    model.save(save_path)
    print(f"\n[OK] Modelo guardado: {save_path}.zip")
    return model, callback

def demo(model_path='dag_humanoid_model'):
    model = PPO.load(model_path)
    env   = gym.make('HumanoidStandup-v4', render_mode='human')
    env   = AchievementRewardWrapper(env, use_cpg=True, use_action_clamp=False, use_egocentric=True)
    obs, _ = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

def record_video(model_path='dag_humanoid_model', out_path='demo.mp4', n_episodes=2, max_steps=1000):
    import imageio

    model = PPO.load(model_path)
    env   = gym.make('HumanoidStandup-v4', render_mode='rgb_array')
    fps   = env.metadata.get('render_fps', 33)
    env   = AchievementRewardWrapper(env, use_cpg=True, use_action_clamp=False, use_egocentric=True)

    writer = imageio.get_writer(out_path, fps=fps)
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            writer.append_data(env.render())
            if terminated or truncated:
                break
        print(f"Episodio {ep+1}/{n_episodes}: reward total = {ep_reward:.1f}")
    writer.close()
    env.close()
    print(f"[OK] Video guardado: {out_path}")


def run_ablation(timesteps=150_000):
    configs = {
        'DAG completo'      : dict(use_cpg=True,  use_action_clamp=True,  use_egocentric=True),
        'Sin CPG'           : dict(use_cpg=False, use_action_clamp=True,  use_egocentric=True),
        'Sin action clamp'  : dict(use_cpg=True,  use_action_clamp=False, use_egocentric=True),
        'Sin ego-centric'   : dict(use_cpg=True,  use_action_clamp=True,  use_egocentric=False),
        'Ninguno'           : dict(use_cpg=False, use_action_clamp=False, use_egocentric=False),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"\n--- {name} ---")
        env = DummyVecEnv([lambda c=cfg: Monitor(
            AchievementRewardWrapper(gym.make('HumanoidStandup-v4'), **c)
        )])
        model = PPO('MlpPolicy', env, n_steps=2048, batch_size=256,
                    learning_rate=3e-4, verbose=0)
        cb = DAGRewardCallback()
        model.learn(total_timesteps=timesteps, callback=cb)
        results[name] = np.array(cb.ep_total)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#1e1e2e')
    ax.set_facecolor('#282a36')
    colors = ['#89b4fa', '#f38ba8', '#a6e3a1', '#f9e2af', '#cba6f7']
    for (name, vals), col in zip(results.items(), colors):
        w = max(3, len(vals) // 8)
        ax.plot(vals, alpha=0.15, color=col)
        if len(vals) >= w:
            s = smooth(vals, w)
            ax.plot(np.arange(len(s)) + w//2, s, color=col, label=name, lw=2)
    ax.set_xlabel('Episodio', color='#cdd6f4')
    ax.set_ylabel('Reward total DAG', color='#cdd6f4')
    ax.set_title('Ablation Study — Componentes del método', color='white')
    ax.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='white', fontsize=9)
    ax.tick_params(colors='#cdd6f4')
    for sp in ax.spines.values(): sp.set_edgecolor('#45475a')
    plt.tight_layout()
    plt.savefig('ablation.png', dpi=150, bbox_inches='tight', facecolor='#1e1e2e')
    print("[OK] Ablation guardada: ablation.png")
    plt.show()
    return results


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'demo', 'ablation', 'video'], default='train')
    parser.add_argument('--steps', type=int, default=500_000)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--out', type=str, default='demo.mp4')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Ruta (sin .zip) de un modelo para warm-start')
    parser.add_argument('--ent_coef', type=float, default=0.0,
                        help='Coef. de entropía (subir para reactivar exploración en warm-start)')
    args = parser.parse_args()

    if args.mode == 'train':
        model, cb = train(total_timesteps=args.steps, n_envs=args.n_envs,
                          resume_from=args.resume_from, ent_coef=args.ent_coef)
        plot_training_curves(cb)
    elif args.mode == 'demo':
        demo()
    elif args.mode == 'ablation':
        run_ablation(timesteps=args.steps)
    elif args.mode == 'video':
        record_video(out_path=args.out, n_episodes=args.episodes)
