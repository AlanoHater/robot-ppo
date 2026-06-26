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

    SKILL_NAMES = ['pelvis_lift', 'leg_extension', 'standup', 'torso_straighten', 'upright']
    DEFAULT_PS  = {
        (0, 2): 0.3,   # pelvis_lift   -> standup
        (1, 2): 0.5,   # leg_extension -> standup  (camino alternativo)
        (2, 4): 4.0,   # standup       -> upright
        (3, 4): 0.3,   # torso_straighten -> upright (camino alternativo)
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

        self._reset_episode_state()

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

        # qpos[13]=right_knee, qpos[17]=left_knee (range real ~[-2.79, -0.035] rad)
        knee_ext  = (float(data.qpos[13]) + float(data.qpos[17])) / 2.0 + 2.79
        # qpos[8]=abdomen_y (flexión/extensión de columna, range ~[-1.31, 0.52] rad)
        abdomen_y = float(data.qpos[8])

        r_pelvis_lift     = np.clip(z - 0.30, 0.0, None) * 3.0
        r_leg_extension   = np.clip(knee_ext, 0.0, None) * 1.0
        r_standup         = np.clip(z - 0.80, 0.0, None) * 8.0
        r_torso_straight  = np.clip(1.31 - abs(abdomen_y), 0.0, None) * 1.5
        r_upright         = np.clip(z - 1.00, 0.0, None) * 8.0

        return [r_pelvis_lift, r_leg_extension, r_standup, r_torso_straight, r_upright]

    def _dag_reward(self, stage_rewards):
        total = sum(stage_rewards[r] for r in self.roots)
        for (i, j), a in self.achievements.items():
            if a > 0:
                total += a * stage_rewards[j]
        return total

    def reset(self, **kwargs):
        self._reset_episode_state()
        obs, info = self.env.reset(**kwargs)
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
    def __init__(self, n_skills=None, verbose=0):
        super().__init__(verbose)
        self.skill_names    = AchievementRewardWrapper.SKILL_NAMES
        self.n_skills       = n_skills or len(self.skill_names)
        self.ep_max_skill   = []
        self.ep_total       = []
        self._ep_skill_max  = [0.0] * self.n_skills
        self._ep_total      = 0.0

    def _on_step(self) -> bool:
        infos   = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones   = self.locals.get('dones', [False])

        for i, info in enumerate(infos):
            sr = info.get('stage_rewards', [0.0] * self.n_skills)
            for k in range(self.n_skills):
                self._ep_skill_max[k] = max(self._ep_skill_max[k], sr[k])
            self._ep_total += rewards[i] if i < len(rewards) else 0.0
            if dones[i]:
                self.ep_max_skill.append(list(self._ep_skill_max))
                self.ep_total.append(self._ep_total)
                self._ep_skill_max = [0.0] * self.n_skills
                self._ep_total     = 0.0
        return True


# ================================================================
# 3.  PLOTS
# ================================================================

SKILL_COLORS = ['#e06c75', '#d19a66', '#e5c07b', '#61afef', '#98c379']

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

def train(total_timesteps=500_000, save_path='dag_humanoid_model', n_envs=1):
    print("=" * 60)
    print("DAG Reward Humanoid  —  Meng & Xiao (2023) adaptation")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Entornos paralelos: {n_envs}")
    print("=" * 60)

    if n_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env])

    model = PPO(
        policy        = 'MlpPolicy',
        env           = env,
        n_steps       = 2048,
        batch_size    = 256,
        n_epochs      = 10,
        learning_rate = 3e-4,
        gamma         = 0.99,
        clip_range    = 0.2,
        ent_coef      = 0.005,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = dict(net_arch=[400, 300, 200, 100]),
        verbose       = 1,
    )

    callback = DAGRewardCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
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
    parser.add_argument('--mode', choices=['train', 'demo', 'ablation'], default='train')
    parser.add_argument('--steps', type=int, default=500_000)
    parser.add_argument('--n_envs', type=int, default=1)
    args = parser.parse_args()

    if args.mode == 'train':
        model, cb = train(total_timesteps=args.steps, n_envs=args.n_envs)
        plot_training_curves(cb)
    elif args.mode == 'demo':
        demo()
    elif args.mode == 'ablation':
        run_ablation(timesteps=args.steps)
