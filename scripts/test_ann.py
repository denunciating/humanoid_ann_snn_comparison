"""
Тестирование обученной ANN-модели для управления роботом-гуманоидом в среде Humanoid-v4.

Скрипт позволяет тестировать ANN-128 или ANN-256,
а также выбирать версию модели: best_model или final_model.

Во время тестирования:
- загружается обученная PPO-модель;
- загружаются сохраненные параметры VecNormalize;
- выполняется 100 тестовых эпизодов;
- рассчитываются награды и длины эпизодов;
- рассчитывается число MAC-операций для актора;
- сохраняется текстовый файл с результатами;
- строятся графики наград и длин эпизодов.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ============================================================================

N_EPISODES = 100
MAX_STEPS = 1000
PRINT_EVERY_EPISODES = 10

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================================
# АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Тестирование ANN-модели для Humanoid-v4."
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        choices=[128, 256],
        required=True,
        help="Размер скрытых слоев тестируемой ANN: 128 или 256.",
    )

    parser.add_argument(
        "--model-version",
        type=str,
        choices=["best", "final"],
        required=True,
        help="Версия тестируемой модели: best или final.",
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

def get_paths(hidden_size: int, model_version: str):
    model_name = f"ann_{hidden_size}"

    model_dir = BASE_DIR / "models" / model_name
    results_dir = BASE_DIR / "results" / model_name
    graphs_dir = BASE_DIR / "graphs" / model_name

    model_path = model_dir / f"{model_version}_model.zip"
    normalize_path = model_dir / f"vec_normalize_{model_version}.pkl"

    results_path = results_dir / f"test_{model_version}_results.txt"
    reward_graph_path = graphs_dir / f"rewards_{model_version}.png"
    length_graph_path = graphs_dir / f"lengths_{model_version}.png"

    return {
        "model_name": model_name,
        "model_dir": model_dir,
        "results_dir": results_dir,
        "graphs_dir": graphs_dir,
        "model_path": model_path,
        "normalize_path": normalize_path,
        "results_path": results_path,
        "reward_graph_path": reward_graph_path,
        "length_graph_path": length_graph_path,
    }


def create_directories(*directories: Path):
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


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

def load_model_and_env(model_path: Path, normalize_path: Path, render_mode):
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    if not normalize_path.exists():
        raise FileNotFoundError(f"Файл нормализации не найден: {normalize_path}")

    env = create_test_env(
        normalize_path=normalize_path,
        render_mode=render_mode,
    )

    model = PPO.load(str(model_path))

    return model, env


# ============================================================================
# ПОДСЧЕТ MAC ДЛЯ ANN-АКТОРА
# ============================================================================

def count_actor_mac(model: PPO):
    """
    Считает количество MAC-операций для актора ANN за один выбор действия.

    Учитываются только слои, необходимые для инференса действия:
    - policy_net;
    - action_net.

    Критик не учитывается.
    """

    policy = model.policy
    policy_net = policy.mlp_extractor.policy_net
    action_net = policy.action_net

    linear_layers = [
        layer for layer in policy_net
        if isinstance(layer, torch.nn.Linear)
    ]

    if not linear_layers:
        raise RuntimeError("В policy_net не найдены линейные слои.")

    layer_details = []
    total_mac = 0

    for layer in linear_layers:
        mac = layer.in_features * layer.out_features
        total_mac += mac

        layer_details.append({
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "mac": mac,
        })

    action_mac = action_net.in_features * action_net.out_features
    total_mac += action_mac

    layer_details.append({
        "in_features": action_net.in_features,
        "out_features": action_net.out_features,
        "mac": action_mac,
    })

    return total_mac, layer_details


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

def run_test_episodes(model: PPO, env):
    results = []

    for episode in range(1, N_EPISODES + 1):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

            if hasattr(reward, "__getitem__"):
                reward = reward[0]

            total_reward += reward
            steps += 1

        results.append({
            "episode": episode,
            "steps": steps,
            "reward": float(total_reward),
        })

        if episode % PRINT_EVERY_EPISODES == 0:
            print(
                f"[INFO] Эпизод {episode}/{N_EPISODES}: "
                f"reward = {total_reward:.2f}, steps = {steps}"
            )

    return results


# ============================================================================
# СТАТИСТИКА
# ============================================================================

def calculate_statistics(results):
    rewards = np.array([item["reward"] for item in results])
    lengths = np.array([item["steps"] for item in results])

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

    print("=" * 60)


# ============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def save_results(
    results_path: Path,
    results,
    statistics,
    model_name: str,
    model_version: str,
    model_path: Path,
    normalize_path: Path,
    render_enabled: bool,
    actor_mac: int,
    layer_details,
):
    with open(results_path, "w", encoding="utf-8") as file:
        file.write("MODEL INFO\n")
        file.write(f"model_name,{model_name}\n")
        file.write(f"model_version,{model_version}\n")
        file.write(f"model_path,{model_path}\n")
        file.write(f"normalize_path,{normalize_path}\n")
        file.write(f"n_episodes,{N_EPISODES}\n")
        file.write(f"max_steps,{MAX_STEPS}\n")
        file.write(f"render,{render_enabled}\n")

        file.write("\nCOMPUTATIONAL COST\n")
        file.write(f"actor_MAC_per_action,{actor_mac}\n")

        file.write("\nACTOR LAYERS\n")
        file.write("layer,in_features,out_features,mac\n")

        for i, layer in enumerate(layer_details, start=1):
            file.write(
                f"{i},"
                f"{layer['in_features']},"
                f"{layer['out_features']},"
                f"{layer['mac']}\n"
            )

        file.write("\nEPISODE RESULTS\n")
        file.write("episode,steps,reward\n")

        for item in results:
            file.write(
                f"{item['episode']},"
                f"{item['steps']},"
                f"{item['reward']:.2f}\n"
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


# ============================================================================
# ГРАФИКИ
# ============================================================================

def plot_rewards(results, output_path: Path, model_name: str, model_version: str):
    rewards = [item["reward"] for item in results]
    episodes = [item["episode"] for item in results]
    mean_reward = np.mean(rewards)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(rewards, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Награда за эпизод")
    plt.ylabel("Частота")
    plt.title(f"Распределение наград ({model_name}, {model_version})")
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
    plt.title(f"Награда по эпизодам ({model_name}, {model_version})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_lengths(results, output_path: Path, model_name: str, model_version: str):
    lengths = [item["steps"] for item in results]
    episodes = [item["episode"] for item in results]
    mean_length = np.mean(lengths)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Длина эпизода, шагов")
    plt.ylabel("Частота")
    plt.title(f"Распределение длин ({model_name}, {model_version})")
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
    plt.title(f"Длина эпизода по эпизодам ({model_name}, {model_version})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_graphs(
    results,
    reward_graph_path: Path,
    length_graph_path: Path,
    model_name: str,
    model_version: str,
):
    plot_rewards(
        results=results,
        output_path=reward_graph_path,
        model_name=model_name,
        model_version=model_version,
    )

    plot_lengths(
        results=results,
        output_path=length_graph_path,
        model_name=model_name,
        model_version=model_version,
    )


# ============================================================================
# ЗАПУСК ТЕСТИРОВАНИЯ
# ============================================================================

def main():
    args = parse_args()

    hidden_size = args.hidden_size
    model_version = args.model_version
    render_enabled = args.render
    render_mode = "human" if render_enabled else None

    paths = get_paths(
        hidden_size=hidden_size,
        model_version=model_version,
    )

    create_directories(
        paths["results_dir"],
        paths["graphs_dir"],
    )

    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ANN-МОДЕЛИ ДЛЯ HUMANOID-V4")
    print("=" * 60)
    print(f"[INFO] Корень проекта: {BASE_DIR}")
    print(f"[INFO] Модель: {paths['model_name']}")
    print(f"[INFO] Версия модели: {model_version}")
    print(f"[INFO] Файл модели: {paths['model_path']}")
    print(f"[INFO] Файл нормализации: {paths['normalize_path']}")
    print(f"[INFO] Визуализация: {render_enabled}")
    print(f"[INFO] Результаты: {paths['results_path']}")
    print(f"[INFO] Графики: {paths['graphs_dir']}")

    model, env = load_model_and_env(
        model_path=paths["model_path"],
        normalize_path=paths["normalize_path"],
        render_mode=render_mode,
    )

    actor_mac, layer_details = count_actor_mac(model)

    print("\n" + "=" * 60)
    print("MAC ДЛЯ ANN-АКТОРА")
    print("=" * 60)

    for i, layer in enumerate(layer_details, start=1):
        print(
            f"Слой {i}: "
            f"{layer['in_features']} x {layer['out_features']} "
            f"= {layer['mac']} MAC"
        )

    print(f"Итого MAC за один выбор действия: {actor_mac}")
    print("=" * 60)

    if render_enabled:
        input("\nНажмите Enter для начала тестирования с визуализацией...\n")
    else:
        print("\n[INFO] Тестирование без визуализации...\n")

    try:
        results = run_test_episodes(
            model=model,
            env=env,
        )

    finally:
        env.close()

    statistics = calculate_statistics(results)
    print_statistics(statistics)

    save_results(
        results_path=paths["results_path"],
        results=results,
        statistics=statistics,
        model_name=paths["model_name"],
        model_version=model_version,
        model_path=paths["model_path"],
        normalize_path=paths["normalize_path"],
        render_enabled=render_enabled,
        actor_mac=actor_mac,
        layer_details=layer_details,
    )

    save_graphs(
        results=results,
        reward_graph_path=paths["reward_graph_path"],
        length_graph_path=paths["length_graph_path"],
        model_name=paths["model_name"],
        model_version=model_version,
    )

    print(f"\n[INFO] Результаты сохранены: {paths['results_path']}")
    print(f"[INFO] График наград сохранен: {paths['reward_graph_path']}")
    print(f"[INFO] График длин сохранен: {paths['length_graph_path']}")
    print("[INFO] Тестирование завершено.")


if __name__ == "__main__":
    main()