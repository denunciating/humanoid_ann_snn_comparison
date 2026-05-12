"""
Обучение гибридной SNN-модели для управления роботом-гуманоидом в среде Humanoid-v4.

В гибридной архитектуре актор построен на основе LIF-нейронов,
а критик остается обычной полносвязной ANN-моделью.

Алгоритм обучения: PPO (Proximal Policy Optimization).
"""

from pathlib import Path

import gymnasium as gym
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ============================================================================

TOTAL_TIMESTEPS = 3_000_000

LEARNING_RATE = 0.0002
N_STEPS = 1024
BATCH_SIZE = 512
N_EPOCHS = 5

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.15
ENT_COEF = 0.0005
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.03

NUM_ENVS = 2

NET_ARCH = [128, 128]

SNN_STEPS = 8
SNN_LEAK = 0.85
SNN_THRESHOLD = 1.0
SNN_SURROGATE = "fast_sigmoid"
SNN_SURROGATE_SCALE = 25.0
SNN_RESET_MECHANISM = "subtract"
SNN_READOUT = "membrane"
SNN_LEARN_LEAK = False
SNN_LEARN_THRESHOLD = False

CHECKPOINT_FREQ = 16_384
EVAL_FREQ = 16_384
N_EVAL_EPISODES = 3
REWARD_THRESHOLD = 10_000

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "hybrid_snn"

LOG_DIR = BASE_DIR / "logs" / MODEL_NAME
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / MODEL_NAME
BEST_MODEL_DIR = BASE_DIR / "models" / MODEL_NAME


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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ SNN
# ============================================================================

def build_spike_grad(surrogate_name: str, scale: float):
    if surrogate_name == "atan":
        return surrogate.atan(alpha=scale)

    if surrogate_name == "fast_sigmoid":
        return surrogate.fast_sigmoid(slope=scale)

    raise ValueError(f"Неподдерживаемая суррогатная функция: {surrogate_name}")

    # ============================================================================
# SNN-АКТОР
# ============================================================================

class SnnTorchActorNetwork(nn.Module):
    """
    Актор на основе LIF-нейронов.
    Поддерживает подсчет спайков при тестировании.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_sizes: list[int],
        *,
        snn_steps: int,
        snn_leak: float,
        snn_threshold: float,
        snn_surrogate: str,
        snn_surrogate_scale: float,
        snn_reset_mechanism: str,
        snn_readout: str,
        snn_learn_leak: bool,
        snn_learn_threshold: bool,
    ) -> None:
        super().__init__()

        self.snn_steps = snn_steps
        self.snn_readout = snn_readout

        self.layers = nn.ModuleList()
        self.neurons = nn.ModuleList()

        last_dim = feature_dim
        spike_grad = build_spike_grad(
            surrogate_name=snn_surrogate,
            scale=snn_surrogate_scale,
        )

        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(last_dim, hidden_size))
            self.neurons.append(
                snn.Leaky(
                    beta=snn_leak,
                    threshold=snn_threshold,
                    spike_grad=spike_grad,
                    learn_beta=snn_learn_leak,
                    learn_threshold=snn_learn_threshold,
                    reset_mechanism=snn_reset_mechanism,
                )
            )
            last_dim = hidden_size

        self.output_dim = last_dim

        self.record_spikes = False
        self.total_spikes = 0

    def reset_spike_counter(self):
        self.total_spikes = 0

    def enable_spike_recording(self, enable: bool = True):
        self.record_spikes = enable

        if enable:
            self.reset_spike_counter()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.layers:
            return features

        batch_size = features.shape[0]
        device = features.device

        mem_states = [
            torch.zeros(
                batch_size,
                layer.out_features,
                device=device,
            )
            for layer in self.layers
        ]

        readout_accumulator = torch.zeros(
            batch_size,
            self.output_dim,
            device=device,
        )

        static_input_current = self.layers[0](features)
        spikes_this_forward = 0

        for _ in range(self.snn_steps):
            current = static_input_current

            for idx, (layer, neuron) in enumerate(zip(self.layers, self.neurons)):
                spikes, mem_states[idx] = neuron(current, mem_states[idx])

                if self.record_spikes:
                    spikes_this_forward += spikes.sum().item()

                current = spikes

                if idx + 1 < len(self.layers):
                    current = self.layers[idx + 1](current)

            if self.snn_readout == "membrane":
                readout_accumulator += mem_states[-1]
            else:
                readout_accumulator += current

        if self.record_spikes:
            self.total_spikes += spikes_this_forward

        return readout_accumulator / float(self.snn_steps)


# ============================================================================
# АКТОР-КРИТИК ДЛЯ PPO
# ============================================================================

class SnnTorchActorMlpExtractor(nn.Module):
    """
    Извлекатель признаков для PPO:
    - актор: SNN;
    - критик: обычная MLP.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: dict[str, list[int]],
        activation_fn: type[nn.Module],
        *,
        snn_steps: int,
        snn_leak: float,
        snn_threshold: float,
        snn_surrogate: str,
        snn_surrogate_scale: float,
        snn_reset_mechanism: str,
        snn_readout: str,
        snn_learn_leak: bool,
        snn_learn_threshold: bool,
        device: torch.device | str = "auto",
    ) -> None:
        super().__init__()

        self.policy_net = SnnTorchActorNetwork(
            feature_dim,
            net_arch["pi"],
            snn_steps=snn_steps,
            snn_leak=snn_leak,
            snn_threshold=snn_threshold,
            snn_surrogate=snn_surrogate,
            snn_surrogate_scale=snn_surrogate_scale,
            snn_reset_mechanism=snn_reset_mechanism,
            snn_readout=snn_readout,
            snn_learn_leak=snn_learn_leak,
            snn_learn_threshold=snn_learn_threshold,
        )

        value_layers = []
        last_dim = feature_dim

        for hidden_size in net_arch["vf"]:
            value_layers.append(nn.Linear(last_dim, hidden_size))
            value_layers.append(activation_fn())
            last_dim = hidden_size

        self.value_net = nn.Sequential(*value_layers)

        self.latent_dim_pi = self.policy_net.output_dim
        self.latent_dim_vf = last_dim

        self.to(get_device(device))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class SnnTorchActorCriticPolicy(ActorCriticPolicy):
    """
    PPO-политика для гибридной SNN:
    - актор реализован как SNN;
    - критик реализован как MLP.
    """

    def __init__(
        self,
        *args,
        snn_steps: int = SNN_STEPS,
        snn_leak: float = SNN_LEAK,
        snn_threshold: float = SNN_THRESHOLD,
        snn_surrogate: str = SNN_SURROGATE,
        snn_surrogate_scale: float = SNN_SURROGATE_SCALE,
        snn_reset_mechanism: str = SNN_RESET_MECHANISM,
        snn_readout: str = SNN_READOUT,
        snn_learn_leak: bool = SNN_LEARN_LEAK,
        snn_learn_threshold: bool = SNN_LEARN_THRESHOLD,
        **kwargs,
    ):
        self.snn_steps = snn_steps
        self.snn_leak = snn_leak
        self.snn_threshold = snn_threshold
        self.snn_surrogate = snn_surrogate
        self.snn_surrogate_scale = snn_surrogate_scale
        self.snn_reset_mechanism = snn_reset_mechanism
        self.snn_readout = snn_readout
        self.snn_learn_leak = snn_learn_leak
        self.snn_learn_threshold = snn_learn_threshold

        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SnnTorchActorMlpExtractor(
            self.features_dim,
            net_arch={
                "pi": list(self.net_arch["pi"]),
                "vf": list(self.net_arch["vf"]),
            },
            activation_fn=self.activation_fn,
            snn_steps=self.snn_steps,
            snn_leak=self.snn_leak,
            snn_threshold=self.snn_threshold,
            snn_surrogate=self.snn_surrogate,
            snn_surrogate_scale=self.snn_surrogate_scale,
            snn_reset_mechanism=self.snn_reset_mechanism,
            snn_readout=self.snn_readout,
            snn_learn_leak=self.snn_learn_leak,
            snn_learn_threshold=self.snn_learn_threshold,
            device=self.device,
        )

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                snn_steps=self.snn_steps,
                snn_leak=self.snn_leak,
                snn_threshold=self.snn_threshold,
                snn_surrogate=self.snn_surrogate,
                snn_surrogate_scale=self.snn_surrogate_scale,
                snn_reset_mechanism=self.snn_reset_mechanism,
                snn_readout=self.snn_readout,
                snn_learn_leak=self.snn_learn_leak,
                snn_learn_threshold=self.snn_learn_threshold,
            )
        )

        return data
    
    # ============================================================================
# СОЗДАНИЕ СРЕДЫ
# ============================================================================

def make_env(seed=None):
    env = gym.make(
        "Humanoid-v4",
        exclude_current_positions_from_observation=True,
    )
    env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)

    return env


def create_training_env():
    env_factories = [
        lambda i=i: make_env(seed=i)
        for i in range(NUM_ENVS)
    ]

    env = DummyVecEnv(env_factories)

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

def build_policy_kwargs():
    return {
        "net_arch": {
            "pi": NET_ARCH,
            "vf": NET_ARCH,
        },
        "activation_fn": nn.Tanh,
        "ortho_init": True,
        "log_std_init": 0.0,
        "snn_steps": SNN_STEPS,
        "snn_leak": SNN_LEAK,
        "snn_threshold": SNN_THRESHOLD,
        "snn_surrogate": SNN_SURROGATE,
        "snn_surrogate_scale": SNN_SURROGATE_SCALE,
        "snn_reset_mechanism": SNN_RESET_MECHANISM,
        "snn_readout": SNN_READOUT,
        "snn_learn_leak": SNN_LEARN_LEAK,
        "snn_learn_threshold": SNN_LEARN_THRESHOLD,
    }


def build_model(training_env, device: str):
    model = PPO(
        policy=SnnTorchActorCriticPolicy,
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
        target_kl=TARGET_KL,
        policy_kwargs=build_policy_kwargs(),
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device=device,
        seed=0,
    )

    return model


def build_callbacks(training_env, eval_env):
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(CHECKPOINT_DIR),
        name_prefix=MODEL_NAME,
        verbose=1,
    )

    normalize_callback = SaveVecNormalizeCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        env=training_env,
        verbose=1,
    )

    save_best_normalize_callback = SaveBestVecNormalizeCallback(
        save_path=BEST_MODEL_DIR,
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
        best_model_save_path=str(BEST_MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        n_eval_episodes=N_EVAL_EPISODES,
        verbose=1,
    )

    return [checkpoint_callback, normalize_callback, eval_callback]


# ============================================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================================

def create_directories(*directories: Path):
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ЗАПУСК ОБУЧЕНИЯ
# ============================================================================

def main():
    create_directories(
        LOG_DIR,
        CHECKPOINT_DIR,
        BEST_MODEL_DIR,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("ОБУЧЕНИЕ HYBRID_SNN ДЛЯ HUMANOID-V4")
    print("=" * 60)
    print(f"[INFO] Корень проекта: {BASE_DIR}")
    print(f"[INFO] Устройство: {device}")
    print(f"[INFO] Архитектура актора: {NET_ARCH}")
    print(f"[INFO] num_steps: {SNN_STEPS}")
    print(f"[INFO] beta: {SNN_LEAK}")
    print(f"[INFO] threshold: {SNN_THRESHOLD}")
    print(f"[INFO] surrogate_grad: {SNN_SURROGATE}")
    print(f"[INFO] surrogate_scale: {SNN_SURROGATE_SCALE}")
    print(f"[INFO] reset_mechanism: {SNN_RESET_MECHANISM}")
    print(f"[INFO] readout: {SNN_READOUT}")
    print(f"[INFO] Логи TensorBoard: {LOG_DIR}")
    print(f"[INFO] Чекпоинты: {CHECKPOINT_DIR}")
    print(f"[INFO] Модель: {BEST_MODEL_DIR}")

    training_env = create_training_env()
    eval_env = create_eval_env()

    model = build_model(
        training_env=training_env,
        device=device,
    )

    callbacks = build_callbacks(
        training_env=training_env,
        eval_env=eval_env,
    )

    try:
        print("\n[INFO] Запуск обучения гибридной SNN...")

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
        )

        model.save(str(BEST_MODEL_DIR / "final_model"))
        training_env.save(str(BEST_MODEL_DIR / "vec_normalize_final.pkl"))

        print("\n[INFO] Обучение завершено.")
        print(f"[INFO] Финальная модель сохранена: {BEST_MODEL_DIR / 'final_model.zip'}")
        print(f"[INFO] Нормализация сохранена: {BEST_MODEL_DIR / 'vec_normalize_final.pkl'}")

    except KeyboardInterrupt:
        print("\n[INFO] Обучение прервано пользователем. Сохраняем последнее состояние как final_model...")

        model.save(str(BEST_MODEL_DIR / "final_model"))
        training_env.save(str(BEST_MODEL_DIR / "vec_normalize_final.pkl"))

        print(f"[INFO] Финальная модель сохранена: {BEST_MODEL_DIR / 'final_model.zip'}")
        print(f"[INFO] Нормализация сохранена: {BEST_MODEL_DIR / 'vec_normalize_final.pkl'}")

    finally:
        training_env.close()
        eval_env.close()
        print("[INFO] Завершение работы.")


if __name__ == "__main__":
    main()