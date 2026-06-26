"""
Achievement-Triggered Multi-Path Reward for Humanoid Locomotion
===============================================================
Adaptación de: "From Rolling Over to Walking" (Meng & Xiao, 2023)
              arXiv:2303.02581

Limitación de hardware: RTX 4050 laptop (8GB VRAM) →
  - Simulador: MuJoCo CPU (sin Isaac Gym)
  - Robot: HumanoidStandup-v4 (17 DoF vs 32 DoF del iCub)
  - Escala: entrenamiento single-env, ~500K steps

Contribución principal implementada:
  1. Achievement-triggered multi-path reward (DAG)
  2. CPG signals (coprime sine waves)
  3. Action clamping proporcional al progreso del episodio

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
    Implementa el grafo dirigido acíclico (DAG) de recompensas del paper.

    Grafo de habilidades (versión simplificada para HumanoidStandup-v4):

        Lift --[PS=1.0]--> StandUp --[PS=4.0]--> Upright

    Fórmulas del paper:
        A_ij,0 = 0
        A_ij,t = max(A_ij,t-1,  R_i,t - PS_ij)
        R_t    = sum_i sum_j (A_ij,t * R_j,t)
    """

    SKILL_NAMES = ['lift', 'standup', 'upright']
    DEFAULT_PS  = {(0, 1): 0.3, (1, 2): 4.0}

    def __init__(self, env, passing_scores=None, use_cpg=True,
                 use_action_clamp=True, episode_max_steps=1000):
        super().__init__(env)

        self.passing_scores    = passing_scores or self.DEFAULT_PS
        self.use_cpg           = use_cpg
        self.use_action_clamp  = use_action_clamp
        self.episode_max_steps = episode_max_steps

        # CPG: primeros 8 primos / 4 → rango 0.5–5 Hz (igual que el paper)
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        self.cpg_freqs = np.array(primes, dtype=np.float32) / 4.0

        if self.use_cpg:
            n_cpg = len(self.cpg_freqs) * 2
            low  = np.concatenate([env.observation_space.low,  np.full(n_cpg, -1.0, dtype=np.float32)])
            high = np.concatenate([env.observation_space.high, np.full(n_cpg,  1.0, dtype=np.float32)])
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self._reset_episode_state()

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

        r_lift    = np.clip(z - 0.30, 0.0, None) * 3.0
        r_standup = np.clip(z - 0.80, 0.0, None) * 8.0
        r_upright = np.clip(z - 1.00, 0.0, None) * 8.0

        return [r_lift, r_standup, r_upright]

    def _dag_reward(self, stage_rewards):
        total = stage_rewards[0]
        a01   = self.achievements.get((0, 1), 0.0)
        if a01 > 0:
            total += a01 * stage_rewards[1]
        a12 = self.achievements.get((1, 2), 0.0)
        if a12 > 0:
            total += a12 * stage_rewards[2]
        return total

    def reset(self, **kwargs):
        self._reset_episode_state()
        obs, info = self.env.reset(**kwargs)
        if self.use_cpg:
            obs = np.concatenate([obs, self._cpg_signal()]).astype(np.float32)
        return obs, info

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

        if self.use_cpg:
            obs = np.concatenate([obs, self._cpg_signal()]).astype(np.float32)

        return obs, float(total_reward), terminated, truncated, info


# ================================================================
# 2.  CALLBACK DE LOGGING
# ================================================================

class DAGRewardCallback(BaseCallback):
    def __init__(self, n_skills=3, verbose=0):
        super().__init__(verbose)
        self.skill_names    = AchievementRewardWrapper.SKILL_NAMES
        self.ep_max_skill   = []
        self.ep_total       = []
        self._ep_skill_max  = [0.0] * n_skills
        self._ep_total      = 0.0

    def _on_step(self) -> bool:
        infos   = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones   = self.locals.get('dones', [False])

        for i, info in enumerate(infos):
            sr = info.get('stage_rewards', [0.0, 0.0, 0.0])
            for k in range(3):
                self._ep_skill_max[k] = max(self._ep_skill_max[k], sr[k])
            self._ep_total += rewards[i] if i < len(rewards) else 0.0
            if dones[i]:
                self.ep_max_skill.append(list(self._ep_skill_max))
                self.ep_total.append(self._ep_total)
                self._ep_skill_max = [0.0, 0.0, 0.0]
                self._ep_total     = 0.0
        return True


# ================================================================
# 3.  PLOTS
# ================================================================

SKILL_COLORS = ['#e06c75', '#e5c07b', '#98c379']

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
    env  = AchievementRewardWrapper(base, use_cpg=True, use_action_clamp=True)
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
    env   = AchievementRewardWrapper(env, use_cpg=True, use_action_clamp=False)
    obs, _ = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

def run_ablation(timesteps=150_000):
    configs = {
        'DAG completo'    : dict(use_cpg=True,  use_action_clamp=True),
        'Sin CPG'         : dict(use_cpg=False, use_action_clamp=True),
        'Sin action clamp': dict(use_cpg=True,  use_action_clamp=False),
        'Sin CPG ni clamp': dict(use_cpg=False, use_action_clamp=False),
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
    colors = ['#89b4fa', '#f38ba8', '#a6e3a1', '#f9e2af']
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
