"""
Обучение ANN-модели для управления роботом-гуманоидом в среде Humanoid-v4.

Скрипт позволяет обучать две конфигурации полносвязной ANN:
- ANN-128: два скрытых слоя по 128 нейронов;
- ANN-256: два скрытых слоя по 256 нейронов.

Алгоритм обучения: PPO (Proximal Policy Optimization).
"""

import argparse
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ============================================================================

TOTAL_TIMESTEPS = 3_000_000

LEARNING_RATE = 0.0001
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 4

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.03
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 25_000
N_EVAL_EPISODES = 5
REWARD_THRESHOLD = 5000

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================================
# CALLBACK'И
# ============================================================================

class SaveVecNormalizeCallback(BaseCallback):
    """
    Сохраняет параметры нормализации наблюдений и наград вместе с чекпоинтами.
    Без этих данных модель нельзя корректно восстановить после обучения.
    """

    def __init__(self, save_freq: int, save_path: Path, env, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.env = env

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            filepath = self.save_path / f"vec_normalize_{step}.pkl"
            self.env.save(str(filepath))

            if self.verbose > 0:
                print(f"\n[INFO] Сохранена нормализация: {filepath}")

        return True


class SaveBestVecNormalizeCallback(BaseCallback):
    """
    Сохраняет параметры нормализации в момент, когда EvalCallback
    находит новую лучшую модель.
    """

    def __init__(self, save_path: Path, env, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.env = env

    def _on_step(self) -> bool:
        filepath = self.save_path / "vec_normalize_best.pkl"
        self.env.save(str(filepath))

        if self.verbose > 0:
            print(f"\n[INFO] Сохранена нормализация лучшей модели: {filepath}")

        return True


# ============================================================================
# СОЗДАНИЕ СРЕДЫ
# ============================================================================

def make_env():
    env = gym.make(
        "Humanoid-v4",
        exclude_current_positions_from_observation=True,
    )
    env = Monitor(env)
    return env


def create_training_env():
    env = DummyVecEnv([make_env])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    return env


def create_eval_env():
    env = DummyVecEnv([make_env])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    return env


# ============================================================================
# СОЗДАНИЕ МОДЕЛИ И CALLBACK'ОВ
# ============================================================================

def build_model(training_env, hidden_size: int, log_dir: Path, device: str):
    net_arch = [hidden_size, hidden_size]

    policy_kwargs = {
        "net_arch": net_arch,
        "activation_fn": torch.nn.Tanh,
        "ortho_init": True,
    }

    model = PPO(
        policy="MlpPolicy",
        env=training_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=device,
    )

    return model


def build_callbacks(
    training_env,
    eval_env,
    hidden_size: int,
    checkpoint_dir: Path,
    best_model_dir: Path,
    log_dir: Path,
):
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(checkpoint_dir),
        name_prefix=f"ann_{hidden_size}",
        verbose=1,
    )

    normalize_callback = SaveVecNormalizeCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=checkpoint_dir,
        env=training_env,
        verbose=1,
    )

    save_best_normalize_callback = SaveBestVecNormalizeCallback(
        save_path=best_model_dir,
        env=training_env,
        verbose=1,
    )

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESHOLD,
        verbose=1,
    )

    callback_on_new_best = CallbackList([
        save_best_normalize_callback,
        stop_callback,
    ])

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_new_best,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        n_eval_episodes=N_EVAL_EPISODES,
        verbose=1,
    )

    return [checkpoint_callback, normalize_callback, eval_callback]


# ============================================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение ANN-модели для Humanoid-v4."
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        choices=[128, 256],
        default=128,
        help="Размер скрытых слоев ANN: 128 или 256. По умолчанию: 128.",
    )

    return parser.parse_args()


def create_directories(*directories: Path):
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ЗАПУСК ОБУЧЕНИЯ
# ============================================================================

def main():
    args = parse_args()
    hidden_size = args.hidden_size
    model_name = f"ann_{hidden_size}"

    log_dir = BASE_DIR / "logs" / model_name
    checkpoint_dir = BASE_DIR / "checkpoints" / model_name
    best_model_dir = BASE_DIR / "models" / model_name

    create_directories(
        log_dir,
        checkpoint_dir,
        best_model_dir,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"ОБУЧЕНИЕ {model_name.upper()} ДЛЯ HUMANOID-V4")
    print("=" * 60)
    print(f"[INFO] Корень проекта: {BASE_DIR}")
    print(f"[INFO] Устройство: {device}")
    print(f"[INFO] Архитектура: [{hidden_size}, {hidden_size}]")
    print(f"[INFO] Логи TensorBoard: {log_dir}")
    print(f"[INFO] Чекпоинты: {checkpoint_dir}")
    print(f"[INFO] Модель: {best_model_dir}")

    training_env = create_training_env()
    eval_env = create_eval_env()

    model = build_model(
        training_env=training_env,
        hidden_size=hidden_size,
        log_dir=log_dir,
        device=device,
    )

    callbacks = build_callbacks(
        training_env=training_env,
        eval_env=eval_env,
        hidden_size=hidden_size,
        checkpoint_dir=checkpoint_dir,
        best_model_dir=best_model_dir,
        log_dir=log_dir,
    )

    try:
        print("\n[INFO] Запуск обучения...")

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
        )

        model.save(str(best_model_dir / "final_model"))
        training_env.save(str(best_model_dir / "vec_normalize_final.pkl"))

        print("\n[INFO] Обучение завершено.")
        print(f"[INFO] Финальная модель сохранена: {best_model_dir / 'final_model.zip'}")
        print(f"[INFO] Нормализация сохранена: {best_model_dir / 'vec_normalize_final.pkl'}")

    except KeyboardInterrupt:
        print("\n[INFO] Обучение прервано пользователем. Сохраняем последнее состояние как final_model...")

        model.save(str(best_model_dir / "final_model"))
        training_env.save(str(best_model_dir / "vec_normalize_final.pkl"))

        print(f"[INFO] Финальная модель сохранена: {best_model_dir / 'final_model.zip'}")
        print(f"[INFO] Нормализация сохранена: {best_model_dir / 'vec_normalize_final.pkl'}")

    finally:
        training_env.close()
        eval_env.close()
        print("[INFO] Завершение работы.")


if __name__ == "__main__":
    main()