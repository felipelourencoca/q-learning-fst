"""
Módulo de Configuração Experimental

Define os hiperparâmetros padronizados para os experimentos de comparação
entre os métodos de cobertura (All States e All Transitions), além de
utilitários para controle de reprodutibilidade.

Uso:
    from src.experiment_config import ExperimentConfig, set_seed

    config = ExperimentConfig()
    set_seed(42)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


def set_seed(seed: int) -> None:
    """
    Define a semente aleatória do NumPy para reprodutibilidade.

    Deve ser chamada antes de cada execução de treinamento para garantir
    que os resultados sejam determinísticos para uma dada semente.

    Args:
        seed: Valor inteiro da semente.
    """
    np.random.seed(seed)


def _default_seeds(n: int) -> List[int]:
    """Gera uma lista de sementes determinísticas a partir do número de repetições."""
    return list(range(1, n + 1))


def _default_fsm_files() -> List[str]:
    """Retorna a lista padrão de FSMs para experimentação."""
    return [
        "fsm/_01_LightSwitch_flattened.json",
        "fsm/_02_DimmableLightSwitch_flattened.json",
        "fsm/_03_MotionLightSwitch_flattened.json",
    ]


@dataclass
class ExperimentConfig:
    """
    Configuração centralizada para os experimentos de cobertura.

    Contém todos os hiperparâmetros do Q-Learning e parâmetros do
    protocolo experimental. Garante que ambos os métodos (All States
    e All Transitions) operem sob condições idênticas.

    Attributes:
        alpha: Taxa de aprendizado (learning rate)
        gamma: Fator de desconto (discount factor)
        epsilon: Probabilidade inicial de exploração (ε-greedy)
        epsilon_min: Limite inferior para epsilon
        epsilon_decay: Multiplicador de decay para epsilon a cada episódio
        n_episodes: Número máximo de episódios de treinamento
        max_steps: Número máximo de passos por episódio
        n_repetitions: Número de repetições por (FSM, método)
        seeds: Lista de sementes aleatórias para cada repetição
        fsm_files: Lista de caminhos dos arquivos JSON das FSMs
    """

    # === Hiperparâmetros Q-Learning (idênticos para ambos os métodos) ===
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99
    n_episodes: int = 500
    max_steps: int = 50

    # === Parâmetros do protocolo experimental ===
    n_repetitions: int = 30
    seeds: List[int] = field(default_factory=lambda: _default_seeds(30))
    fsm_files: List[str] = field(default_factory=_default_fsm_files)

    def __post_init__(self):
        """Ajusta a lista de seeds se n_repetitions foi alterado."""
        if len(self.seeds) != self.n_repetitions:
            self.seeds = _default_seeds(self.n_repetitions)

    def to_dict(self) -> dict:
        """
        Converte a configuração para um dicionário serializável.

        Returns:
            Dicionário com todos os parâmetros da configuração.
        """
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "n_episodes": self.n_episodes,
            "max_steps": self.max_steps,
            "n_repetitions": self.n_repetitions,
            "seeds": self.seeds,
            "fsm_files": self.fsm_files,
        }
