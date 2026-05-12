"""
Тестирование конвертированной SNN-модели для управления роботом-гуманоидом
в среде Humanoid-v4.

Конвертированная SNN получается из обученной ANN-256 без дополнительного обучения.

Во время тестирования:
- загружается сохраненный SNN-актор;
- загружается конфигурация SNN;
- загружаются параметры VecNormalize от исходной ANN-256;
- выполняется 100 тестовых эпизодов;
- рассчитываются награды и длины эпизодов;
- подсчитываются спайки;
- сохраняется текстовый файл с результатами;
- строятся графики наград и длин эпизодов.

MAC для конвертированной SNN в этом скрипте не рассчитываются.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ============================================================================

N_EPISODES = 100
MAX_STEPS = 1000
PRINT_EVERY_EPISODES = 10

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "converted_snn"
SOURCE_MODEL_NAME = "ann_256"


# ============================================================================
# АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Тестирование конвертированной SNN-модели для Humanoid-v4."
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Включить визуализацию MuJoCo.",
    )

    return parser.parse_args()


# ============================================================================
# ПУТИ
# ============================================================================

def get_paths():
    model_dir = BASE_DIR / "models" / MODEL_NAME
    source_model_dir = BASE_DIR / "models" / SOURCE_MODEL_NAME

    results_dir = BASE_DIR / "results" / MODEL_NAME
    graphs_dir = BASE_DIR / "graphs" / MODEL_NAME

    model_path = model_dir / "snn_actor_state.pth"
    config_path = model_dir / "config.json"

    normalize_path = source_model_dir / "vec_normalize_best.pkl"

    results_path = results_dir / "test_results.txt"
    reward_graph_path = graphs_dir / "rewards.png"
    length_graph_path = graphs_dir / "lengths.png"

    return {
        "model_dir": model_dir,
        "source_model_dir": source_model_dir,
        "results_dir": results_dir,
        "graphs_dir": graphs_dir,
        "model_path": model_path,
        "config_path": config_path,
        "normalize_path": normalize_path,
        "results_path": results_path,
        "reward_graph_path": reward_graph_path,
        "length_graph_path": length_graph_path,
    }


def create_directories(*directories: Path):
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# КЛАСС КОНВЕРТИРОВАННОГО SNN-АКТОРА
# ============================================================================

class SNNActorWithSpikeCount(nn.Module):
    """
    SNN-актор, повторяющий архитектуру конвертированной ANN.

    Линейные слои загружаются из сохраненного state_dict.
    Между линейными слоями, кроме выходного, используются LIF-нейроны.
    Выход ограничивается функцией tanh диапазоном [-1, 1].

    Дополнительно ведется подсчет спайков.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        num_steps: int,
        beta: float,
        threshold: float,
        spike_grad,
        reset_mechanism: str,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.num_steps = num_steps

        self.snn_linears = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.snn_linears.append(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )

        self.lifs = nn.ModuleList()
        for _ in range(len(self.snn_linears) - 1):
            self.lifs.append(
                snn.Leaky(
                    beta=beta,
                    threshold=threshold,
                    spike_grad=spike_grad,
                    reset_mechanism=reset_mechanism,
                )
            )

        self.tanh = nn.Tanh()

        self.total_spikes = 0
        self.num_forward_calls = 0

    def reset_spike_counter(self):
        self.total_spikes = 0
        self.num_forward_calls = 0

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        device = obs.device

        mems = []
        for i, linear_layer in enumerate(self.snn_linears):
            if i < len(self.lifs):
                mems.append(
                    torch.zeros(
                        batch_size,
                        linear_layer.out_features,
                        device=device,
                    )
                )

        obs_t = obs.unsqueeze(0).repeat(self.num_steps, 1, 1)

        outputs = []
        spikes_this_forward = 0

        for t in range(self.num_steps):
            x = obs_t[t]

            for i, linear_layer in enumerate(self.snn_linears):
                x = linear_layer(x)

                if i < len(self.lifs):
                    x, mems[i] = self.lifs[i](x, mems[i])
                    spikes_this_forward += x.sum().item()

            outputs.append(x)

        mean_actions = torch.stack(outputs).mean(dim=0)
        actions = self.tanh(mean_actions)

        self.total_spikes += spikes_this_forward
        self.num_forward_calls += 1

        return actions
    

# ============================================================================
# СОЗДАНИЕ СРЕДЫ
# ============================================================================

def make_env(render_mode):
    return gym.make(
        "Humanoid-v4",
        exclude_current_positions_from_observation=True,
        render_mode=render_mode,
    )


def create_test_env(normalize_path: Path, render_mode):
    env = DummyVecEnv([
        lambda: make_env(render_mode=render_mode)
    ])

    env = VecNormalize.load(str(normalize_path), env)
    env.training = False
    env.norm_reward = False

    return env


# ============================================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================================

def build_spike_grad(config: dict):
    spike_grad_name = config.get("spike_grad", "atan")

    if spike_grad_name == "atan":
        return surrogate.atan()

    if spike_grad_name == "fast_sigmoid":
        return surrogate.fast_sigmoid()

    raise ValueError(f"Неподдерживаемая surrogate-функция: {spike_grad_name}")


def load_config(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_snn_actor(model_path: Path, config: dict, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"SNN-модель не найдена: {model_path}")

    reset_mechanism = config.get("reset_mechanism", "zero")

    snn_actor = SNNActorWithSpikeCount(
        layer_sizes=config["layer_sizes"],
        num_steps=config["num_steps"],
        beta=config["beta"],
        threshold=config["threshold"],
        spike_grad=build_spike_grad(config),
        reset_mechanism=reset_mechanism,
    )

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    snn_actor.load_state_dict(state_dict)
    snn_actor.to(device)
    snn_actor.eval()

    return snn_actor


def load_model_and_env(paths, render_mode):
    if not paths["normalize_path"].exists():
        raise FileNotFoundError(
            f"Файл нормализации не найден: {paths['normalize_path']}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(paths["config_path"])

    snn_actor = load_snn_actor(
        model_path=paths["model_path"],
        config=config,
        device=device,
    )

    env = create_test_env(
        normalize_path=paths["normalize_path"],
        render_mode=render_mode,
    )

    return snn_actor, env, config, device


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

def run_test_episodes(snn_actor: SNNActorWithSpikeCount, env, device: torch.device):
    results = []

    for episode in range(1, N_EPISODES + 1):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        snn_actor.reset_spike_counter()

        while not done and steps < MAX_STEPS:
            obs_tensor = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                action = snn_actor(obs_tensor).cpu().numpy()

            obs, reward, done, _ = env.step(action)

            if hasattr(reward, "__getitem__"):
                reward = reward[0]

            total_reward += reward
            steps += 1

        spikes = snn_actor.total_spikes

        results.append({
            "episode": episode,
            "steps": steps,
            "reward": float(total_reward),
            "spikes": float(spikes),
        })

        if episode % PRINT_EVERY_EPISODES == 0:
            print(
                f"[INFO] Эпизод {episode}/{N_EPISODES}: "
                f"reward = {total_reward:.2f}, "
                f"steps = {steps}, "
                f"spikes = {spikes:.0f}"
            )

    return results


# ============================================================================
# СТАТИСТИКА
# ============================================================================

def calculate_statistics(results):
    rewards = np.array([item["reward"] for item in results])
    lengths = np.array([item["steps"] for item in results])
    spikes = np.array([item["spikes"] for item in results])

    statistics = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "median_reward": np.median(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "median_length": np.median(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "mean_spikes": np.mean(spikes),
        "std_spikes": np.std(spikes),
        "median_spikes": np.median(spikes),
        "min_spikes": np.min(spikes),
        "max_spikes": np.max(spikes),
        "spikes_per_step": np.mean(spikes) / np.mean(lengths),
    }

    return statistics


def print_statistics(statistics):
    print("\n" + "=" * 60)
    print(f"ИТОГОВАЯ СТАТИСТИКА ЗА {N_EPISODES} ЭПИЗОДОВ")
    print("=" * 60)

    print(
        f"Средняя награда:               "
        f"{statistics['mean_reward']:.2f} ± {statistics['std_reward']:.2f}"
    )
    print(f"Медианная награда:             {statistics['median_reward']:.2f}")
    print(
        f"Минимальная / максимальная:     "
        f"{statistics['min_reward']:.2f} / {statistics['max_reward']:.2f}"
    )

    print()

    print(
        f"Средняя длина эпизода:          "
        f"{statistics['mean_length']:.1f} ± {statistics['std_length']:.1f}"
    )
    print(f"Медианная длина:               {statistics['median_length']:.1f}")
    print(
        f"Минимальная / максимальная:     "
        f"{statistics['min_length']} / {statistics['max_length']}"
    )

    print()

    print(
        f"Среднее количество спайков:     "
        f"{statistics['mean_spikes']:.0f} ± {statistics['std_spikes']:.0f}"
    )
    print(f"Медианное количество спайков:   {statistics['median_spikes']:.0f}")
    print(
        f"Минимум / максимум спайков:     "
        f"{statistics['min_spikes']:.0f} / {statistics['max_spikes']:.0f}"
    )
    print(f"Спайков на шаг:                 {statistics['spikes_per_step']:.2f}")

    print("=" * 60)


# ============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def save_results(
    results_path: Path,
    results,
    statistics,
    config: dict,
    model_path: Path,
    normalize_path: Path,
    render_enabled: bool,
):
    with open(results_path, "w", encoding="utf-8") as file:
        file.write("MODEL INFO\n")
        file.write(f"model_name,{MODEL_NAME}\n")
        file.write(f"source_model,{SOURCE_MODEL_NAME}\n")
        file.write(f"model_path,{model_path}\n")
        file.write(f"normalize_path,{normalize_path}\n")
        file.write(f"n_episodes,{N_EPISODES}\n")
        file.write(f"max_steps,{MAX_STEPS}\n")
        file.write(f"render,{render_enabled}\n")

        file.write("\nSNN CONFIG\n")
        file.write(f"layer_sizes,{config.get('layer_sizes')}\n")
        file.write(f"num_steps,{config.get('num_steps')}\n")
        file.write(f"beta,{config.get('beta')}\n")
        file.write(f"threshold,{config.get('threshold')}\n")
        file.write(f"reset_mechanism,{config.get('reset_mechanism')}\n")
        file.write(f"spike_grad,{config.get('spike_grad')}\n")
        file.write(f"output_activation,tanh\n")

        file.write("\nEPISODE RESULTS\n")
        file.write("episode,steps,reward,spikes\n")

        for item in results:
            file.write(
                f"{item['episode']},"
                f"{item['steps']},"
                f"{item['reward']:.2f},"
                f"{item['spikes']:.0f}\n"
            )

        file.write("\nSUMMARY\n")
        file.write(f"mean_reward,{statistics['mean_reward']:.2f}\n")
        file.write(f"std_reward,{statistics['std_reward']:.2f}\n")
        file.write(f"median_reward,{statistics['median_reward']:.2f}\n")
        file.write(f"min_reward,{statistics['min_reward']:.2f}\n")
        file.write(f"max_reward,{statistics['max_reward']:.2f}\n")

        file.write(f"mean_length,{statistics['mean_length']:.2f}\n")
        file.write(f"std_length,{statistics['std_length']:.2f}\n")
        file.write(f"median_length,{statistics['median_length']:.2f}\n")
        file.write(f"min_length,{statistics['min_length']}\n")
        file.write(f"max_length,{statistics['max_length']}\n")

        file.write(f"mean_spikes,{statistics['mean_spikes']:.2f}\n")
        file.write(f"std_spikes,{statistics['std_spikes']:.2f}\n")
        file.write(f"median_spikes,{statistics['median_spikes']:.2f}\n")
        file.write(f"min_spikes,{statistics['min_spikes']:.2f}\n")
        file.write(f"max_spikes,{statistics['max_spikes']:.2f}\n")
        file.write(f"spikes_per_step,{statistics['spikes_per_step']:.2f}\n")


# ============================================================================
# ГРАФИКИ
# ============================================================================

def plot_rewards(results, output_path: Path):
    rewards = [item["reward"] for item in results]
    episodes = [item["episode"] for item in results]
    mean_reward = np.mean(rewards)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(rewards, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Награда за эпизод")
    plt.ylabel("Частота")
    plt.title("Распределение наград (converted_snn)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(
        episodes,
        rewards,
        marker="o",
        markersize=2,
        linestyle="-",
        linewidth=0.5,
    )
    plt.axhline(
        y=mean_reward,
        linestyle="--",
        label=f"Среднее = {mean_reward:.1f}",
    )
    plt.xlabel("Номер эпизода")
    plt.ylabel("Награда")
    plt.title("Награда по эпизодам (converted_snn)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_lengths(results, output_path: Path):
    lengths = [item["steps"] for item in results]
    episodes = [item["episode"] for item in results]
    mean_length = np.mean(lengths)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Длина эпизода, шагов")
    plt.ylabel("Частота")
    plt.title("Распределение длин (converted_snn)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(
        episodes,
        lengths,
        marker="o",
        markersize=2,
        linestyle="-",
        linewidth=0.5,
    )
    plt.axhline(
        y=mean_length,
        linestyle="--",
        label=f"Среднее = {mean_length:.1f}",
    )
    plt.xlabel("Номер эпизода")
    plt.ylabel("Длина эпизода, шагов")
    plt.title("Длина эпизода по эпизодам (converted_snn)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_graphs(results, reward_graph_path: Path, length_graph_path: Path):
    plot_rewards(
        results=results,
        output_path=reward_graph_path,
    )

    plot_lengths(
        results=results,
        output_path=length_graph_path,
    )


# ============================================================================
# ЗАПУСК ТЕСТИРОВАНИЯ
# ============================================================================

def main():
    args = parse_args()

    render_enabled = args.render
    render_mode = "human" if render_enabled else None

    paths = get_paths()

    create_directories(
        paths["results_dir"],
        paths["graphs_dir"],
    )

    print("=" * 60)
    print("ТЕСТИРОВАНИЕ CONVERTED_SNN ДЛЯ HUMANOID-V4")
    print("=" * 60)
    print(f"[INFO] Корень проекта: {BASE_DIR}")
    print(f"[INFO] Модель: {MODEL_NAME}")
    print(f"[INFO] Исходная модель: {SOURCE_MODEL_NAME}")
    print(f"[INFO] Файл SNN-модели: {paths['model_path']}")
    print(f"[INFO] Файл конфигурации: {paths['config_path']}")
    print(f"[INFO] Файл нормализации: {paths['normalize_path']}")
    print(f"[INFO] Визуализация: {render_enabled}")
    print(f"[INFO] Результаты: {paths['results_path']}")
    print(f"[INFO] Графики: {paths['graphs_dir']}")

    snn_actor, env, config, device = load_model_and_env(
        paths=paths,
        render_mode=render_mode,
    )

    print("\n" + "=" * 60)
    print("ПАРАМЕТРЫ CONVERTED_SNN")
    print("=" * 60)
    print(f"[INFO] Устройство: {device}")
    print(f"[INFO] Архитектура: {config.get('layer_sizes')}")
    print(f"[INFO] num_steps: {config.get('num_steps')}")
    print(f"[INFO] beta: {config.get('beta')}")
    print(f"[INFO] threshold: {config.get('threshold')}")
    print(f"[INFO] reset_mechanism: {config.get('reset_mechanism')}")
    print(f"[INFO] spike_grad: {config.get('spike_grad')}")
    print(f"[INFO] output_activation: tanh")
    print("=" * 60)

    if render_enabled:
        input("\nНажмите Enter для начала тестирования с визуализацией...\n")
    else:
        print("\n[INFO] Тестирование без визуализации...\n")

    try:
        results = run_test_episodes(
            snn_actor=snn_actor,
            env=env,
            device=device,
        )

    finally:
        env.close()

    statistics = calculate_statistics(results)
    print_statistics(statistics)

    save_results(
        results_path=paths["results_path"],
        results=results,
        statistics=statistics,
        config=config,
        model_path=paths["model_path"],
        normalize_path=paths["normalize_path"],
        render_enabled=render_enabled,
    )

    save_graphs(
        results=results,
        reward_graph_path=paths["reward_graph_path"],
        length_graph_path=paths["length_graph_path"],
    )

    print(f"\n[INFO] Результаты сохранены: {paths['results_path']}")
    print(f"[INFO] График наград сохранен: {paths['reward_graph_path']}")
    print(f"[INFO] График длин сохранен: {paths['length_graph_path']}")
    print("[INFO] Тестирование завершено.")


if __name__ == "__main__":
    main()