"""
Конверсия обученной ANN-модели в SNN для среды Humanoid-v4.

Скрипт загружает обученную ANN-модель, извлекает веса актора,
создает SNN-актор с аналогичной структурой линейных слоев,
копирует веса ANN в SNN и сохраняет полученную SNN-модель.

По умолчанию используется лучшая ANN-256:
models/ann_256/best_model.zip

Результат сохраняется в:
models/converted_snn/
"""

import json
from pathlib import Path

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate
from stable_baselines3 import PPO


# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_MODEL_NAME = "ann_256"
TARGET_MODEL_NAME = "converted_snn"

MLP_MODEL_PATH = BASE_DIR / "models" / SOURCE_MODEL_NAME / "best_model.zip"
SAVE_DIR = BASE_DIR / "models" / TARGET_MODEL_NAME

NUM_STEPS = 8
BETA = 0.9
THRESHOLD = 1.0
RESET_MECHANISM = "zero"
SPIKE_GRAD = surrogate.atan()


# ============================================================================
# КЛАСС SNN-АКТОРА
# ============================================================================

class SNNActor(nn.Module):
    """
    Спайковый актор, повторяющий архитектуру актора исходной ANN.

    Линейные слои копируются из обученной ANN.
    Между линейными слоями, кроме выходного, используются LIF-нейроны.
    Выход ограничивается функцией tanh диапазоном [-1, 1].
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
        for t in range(self.num_steps):
            x = obs_t[t]

            for i, linear_layer in enumerate(self.snn_linears):
                x = linear_layer(x)

                if i < len(self.lifs):
                    x, mems[i] = self.lifs[i](x, mems[i])

            outputs.append(x)

        mean_actions = torch.stack(outputs).mean(dim=0)
        actions = self.tanh(mean_actions)

        return actions


# ============================================================================
# ФУНКЦИИ КОНВЕРСИИ
# ============================================================================

def extract_actor_layers(mlp_model: PPO) -> tuple[list[int], list[nn.Linear], nn.Linear]:
    """
    Извлекает линейные слои актора ANN и формирует список размерностей.
    """

    hidden_layers = mlp_model.policy.mlp_extractor.policy_net
    action_net = mlp_model.policy.action_net

    linear_layers = [
        layer for layer in hidden_layers
        if isinstance(layer, nn.Linear)
    ]

    if not linear_layers:
        raise RuntimeError("В policy_net не найдены линейные слои.")

    layer_sizes = [linear_layers[0].in_features]

    for linear_layer in linear_layers:
        layer_sizes.append(linear_layer.out_features)

    layer_sizes.append(action_net.out_features)

    return layer_sizes, linear_layers, action_net


def build_snn_actor(layer_sizes: list[int]) -> SNNActor:
    """
    Создает SNN-актор с заданной архитектурой.
    """

    snn_actor = SNNActor(
        layer_sizes=layer_sizes,
        num_steps=NUM_STEPS,
        beta=BETA,
        threshold=THRESHOLD,
        spike_grad=SPIKE_GRAD,
        reset_mechanism=RESET_MECHANISM,
    )

    snn_actor.eval()

    return snn_actor


def copy_weights_from_ann(
    snn_actor: SNNActor,
    linear_layers: list[nn.Linear],
    action_net: nn.Linear,
) -> None:
    """
    Копирует веса и смещения из ANN-актора в SNN-актор.
    """

    for i, linear_layer in enumerate(linear_layers):
        snn_actor.snn_linears[i].weight.data = linear_layer.weight.data.clone()

        if linear_layer.bias is not None:
            snn_actor.snn_linears[i].bias.data = linear_layer.bias.data.clone()

    snn_actor.snn_linears[-1].weight.data = action_net.weight.data.clone()

    if action_net.bias is not None:
        snn_actor.snn_linears[-1].bias.data = action_net.bias.data.clone()


def create_directories(*directories: Path):
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_snn_model(snn_actor: SNNActor, layer_sizes: list[int]) -> None:
    """
    Сохраняет веса SNN-актора и конфигурацию конверсии.
    """

    create_directories(SAVE_DIR)

    model_path = SAVE_DIR / "snn_actor_state.pth"
    config_path = SAVE_DIR / "config.json"

    torch.save(snn_actor.state_dict(), model_path)

    config = {
        "source_model": SOURCE_MODEL_NAME,
        "layer_sizes": layer_sizes,
        "num_steps": NUM_STEPS,
        "beta": BETA,
        "threshold": THRESHOLD,
        "reset_mechanism": RESET_MECHANISM,
        "spike_grad": "atan",
        "output_activation": "tanh",
    }

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    print(f"[INFO] SNN-модель сохранена: {model_path}")
    print(f"[INFO] Конфигурация сохранена: {config_path}")


# ============================================================================
# ЗАПУСК КОНВЕРСИИ
# ============================================================================

def main():
    print("=" * 60)
    print("КОНВЕРСИЯ ANN -> SNN")
    print("=" * 60)
    print(f"[INFO] Корень проекта: {BASE_DIR}")
    print(f"[INFO] Исходная ANN-модель: {MLP_MODEL_PATH}")
    print(f"[INFO] Директория сохранения SNN: {SAVE_DIR}")
    print(f"[INFO] num_steps: {NUM_STEPS}")
    print(f"[INFO] beta: {BETA}")
    print(f"[INFO] threshold: {THRESHOLD}")
    print(f"[INFO] reset_mechanism: {RESET_MECHANISM}")

    if not MLP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ANN-модель не найдена: {MLP_MODEL_PATH}\n"
            f"Сначала обучите ANN командой:\n"
            f"python scripts/train_ann.py"
        )

    print("\n[INFO] Загрузка ANN-модели...")
    mlp_model = PPO.load(str(MLP_MODEL_PATH))

    layer_sizes, linear_layers, action_net = extract_actor_layers(mlp_model)

    print(f"[INFO] Архитектура актора ANN: {layer_sizes}")

    snn_actor = build_snn_actor(layer_sizes)

    print("[INFO] Копирование весов ANN -> SNN...")
    copy_weights_from_ann(
        snn_actor=snn_actor,
        linear_layers=linear_layers,
        action_net=action_net,
    )

    print("[INFO] Веса скопированы.")

    save_snn_model(
        snn_actor=snn_actor,
        layer_sizes=layer_sizes,
    )

    print("[INFO] Конверсия завершена.")


if __name__ == "__main__":
    main()