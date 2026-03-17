"""
Módulo do Ambiente FSM (Máquina de Estados Finitos)

Modela um autômato finito como um ambiente de Reinforcement Learning,
onde o agente deve aprender a navegar dos estados iniciais até um
estado objetivo de forma ótima.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set


class FSMEnvironment:
    """
    Ambiente de Máquina de Estados Finitos para Reinforcement Learning.

    O autômato é representado como um grafo direcionado onde:
    - Nós = estados
    - Arestas = transições (ações)
    - Cada transição tem uma recompensa associada

    Attributes:
        states: Lista de todos os estados do autômato
        actions: Lista de todas as ações possíveis
        transitions: Dicionário de transições {(estado, ação): próximo_estado}
        rewards: Dicionário de recompensas {(estado, ação): recompensa}
        initial_state: Estado inicial do autômato
        goal_states: Conjunto de estados objetivo
        current_state: Estado atual do agente
        max_steps: Número máximo de passos por episódio
    """

    def __init__(
        self,
        states: List[str],
        actions: List[str],
        transitions: Dict[Tuple[str, str], str],
        rewards: Dict[Tuple[str, str], float],
        initial_state: str,
        goal_states: Set[str],
        max_steps: int = 50,
    ):
        """
        Inicializa o ambiente FSM.

        Args:
            states: Lista de nomes dos estados
            actions: Lista de nomes das ações
            transitions: Mapa (estado, ação) -> próximo_estado
            rewards: Mapa (estado, ação) -> recompensa
            initial_state: Estado onde o agente começa
            goal_states: Conjunto de estados que encerram o episódio com sucesso
            max_steps: Limite de passos por episódio (evita loops infinitos)
        """
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.initial_state = initial_state
        self.goal_states = goal_states
        self.max_steps = max_steps

        # Mapear nomes para índices (útil para a Q-Table)
        self.state_to_idx: Dict[str, int] = {s: i for i, s in enumerate(states)}
        self.action_to_idx: Dict[str, int] = {a: i for i, a in enumerate(actions)}
        self.idx_to_state: Dict[int, str] = {i: s for i, s in enumerate(states)}
        self.idx_to_action: Dict[int, str] = {i: a for i, a in enumerate(actions)}

        # Estado interno
        self.current_state: str = initial_state
        self.steps_taken: int = 0

        # Pré-computar ações válidas por estado
        self._valid_actions: Dict[str, List[str]] = {}
        for state in states:
            self._valid_actions[state] = [
                action for action in actions
                if (state, action) in transitions
            ]

    @property
    def n_states(self) -> int:
        """Retorna o número de estados."""
        return len(self.states)

    @property
    def n_actions(self) -> int:
        """Retorna o número de ações."""
        return len(self.actions)

    def reset(self) -> str:
        """
        Reinicia o ambiente para o estado inicial.

        Returns:
            O estado inicial do autômato.
        """
        self.current_state = self.initial_state
        self.steps_taken = 0
        return self.current_state

    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """
        Executa uma ação no ambiente.

        Args:
            action: A ação a ser executada.

        Returns:
            Tupla (próximo_estado, recompensa, terminado, info)
            - próximo_estado: estado resultante da transição
            - recompensa: recompensa recebida
            - terminado: True se o episódio acabou
            - info: dicionário com informações adicionais
        """
        self.steps_taken += 1
        info = {"steps": self.steps_taken}

        # Verificar se a ação é válida no estado atual
        if (self.current_state, action) not in self.transitions:
            # Ação inválida: penalidade e permanece no mesmo estado
            reward = -10.0
            done = self.steps_taken >= self.max_steps
            info["invalid_action"] = True
            return self.current_state, reward, done, info

        # Transição válida
        next_state = self.transitions[(self.current_state, action)]
        reward = self.rewards.get((self.current_state, action), -1.0)

        self.current_state = next_state

        # Verificar condições de término
        reached_goal = self.current_state in self.goal_states
        exceeded_steps = self.steps_taken >= self.max_steps
        done = reached_goal or exceeded_steps

        info["reached_goal"] = reached_goal
        info["exceeded_steps"] = exceeded_steps

        return next_state, reward, done, info

    def get_valid_actions(self, state: Optional[str] = None) -> List[str]:
        """
        Retorna as ações válidas para um dado estado.

        Args:
            state: Estado para consultar. Se None, usa o estado atual.

        Returns:
            Lista de ações válidas.
        """
        if state is None:
            state = self.current_state
        return self._valid_actions.get(state, [])

    def get_transition_matrix(self) -> np.ndarray:
        """
        Retorna a matriz de transições como um array NumPy.

        Returns:
            Matriz (n_states x n_actions) onde cada célula contém
            o índice do próximo estado, ou -1 se a transição é inválida.
        """
        matrix = np.full((self.n_states, self.n_actions), -1, dtype=int)
        for (state, action), next_state in self.transitions.items():
            s_idx = self.state_to_idx[state]
            a_idx = self.action_to_idx[action]
            ns_idx = self.state_to_idx[next_state]
            matrix[s_idx, a_idx] = ns_idx
        return matrix

    def __repr__(self) -> str:
        return (
            f"FSMEnvironment(\n"
            f"  estados={self.states},\n"
            f"  ações={self.actions},\n"
            f"  estado_inicial='{self.initial_state}',\n"
            f"  estados_objetivo={self.goal_states},\n"
            f"  transições={len(self.transitions)}\n"
            f")"
        )


def load_fsm_from_json(
    filepath: str,
    goal_states: Optional[Set[str]] = None,
    max_steps: int = 50,
) -> FSMEnvironment:
    """
    Carrega uma Máquina de Estados Finitos a partir de um arquivo JSON.

    O formato JSON esperado é:
        {
            "initial": "nome_estado_inicial",
            "states": [
                {
                    "state": "nome_estado",
                    "transitions": [
                        {
                            "input": "nome_acao",
                            "output": [...],
                            "target": "nome_estado_destino"
                        }
                    ]
                }
            ]
        }

    Args:
        filepath: Caminho para o arquivo JSON da FSM.
        goal_states: Conjunto de estados objetivo. Se None, usa conjunto vazio.
        max_steps: Limite de passos por episódio.

    Returns:
        Instância de FSMEnvironment configurada.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    initial_state = data["initial"]

    # Extrair estados e transições
    states = []
    transitions: Dict[Tuple[str, str], str] = {}
    actions_set: set = set()

    for state_obj in data["states"]:
        state_name = state_obj["state"]
        states.append(state_name)

        for trans in state_obj.get("transitions", []):
            action = trans["input"]
            target = trans["target"]
            actions_set.add(action)
            transitions[(state_name, action)] = target

    actions = sorted(actions_set)

    # Recompensas padrão: -1.0 para todas as transições válidas
    rewards: Dict[Tuple[str, str], float] = {
        key: -1.0 for key in transitions
    }

    if goal_states is None:
        goal_states = set()

    return FSMEnvironment(
        states=states,
        actions=actions,
        transitions=transitions,
        rewards=rewards,
        initial_state=initial_state,
        goal_states=goal_states,
        max_steps=max_steps,
    )

