"""
Módulo do Agente Q-Learning para Cobertura de Transições

Adapta o algoritmo Q-Learning para gerar sequências de teste que
maximizem a cobertura de transições de uma Máquina de Estados Finitos.

Critério de cobertura: Transition Coverage (Cobertura de Transições)
- O agente recebe recompensa positiva ao exercitar transições ainda não cobertas
- O objetivo é cobrir 100% das transições definidas na FSM
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from .fsm_environment import FSMEnvironment


class CoverageQLearningAgent:
    """
    Agente Q-Learning adaptado para gerar testes com cobertura de transições.

    Diferenças em relação ao QLearningAgent original:
    - A recompensa é baseada em cobertura (transição nova = +50, já coberta = -1)
    - Mantém tracking global das transições cobertas ao longo dos episódios
    - Cada episódio gera uma "sequência de teste" independente
    - Pode gerar uma suíte de testes mínima que cobre todas as transições

    Attributes:
        env: O ambiente FSM
        all_transitions: Conjunto de todas as transições possíveis da FSM
        covered_transitions: Conjunto de transições já cobertas
        coverage_history: Histórico de % de cobertura por episódio
    """

    # Recompensas para cobertura
    REWARD_NEW_TRANSITION = 50.0     # Transição nova coberta
    REWARD_OLD_TRANSITION = -1.0     # Transição já coberta
    REWARD_INVALID_ACTION = -10.0    # Ação inválida

    def __init__(
        self,
        env: FSMEnvironment,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        """
        Inicializa o agente de cobertura.

        Args:
            env: Ambiente FSM para interação
            alpha: Taxa de aprendizado
            gamma: Fator de desconto
            epsilon: Probabilidade inicial de exploração
            epsilon_min: Limite inferior para epsilon
            epsilon_decay: Multiplicador de decay para epsilon
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-Table
        self.q_table = np.zeros((env.n_states, env.n_actions))

        # === Tracking de cobertura ===
        # Todas as transições definidas na FSM: {(estado, ação, próximo_estado)}
        self.all_transitions: Set[Tuple[str, str, str]] = set()
        for (state, action), next_state in env.transitions.items():
            self.all_transitions.add((state, action, next_state))

        self.total_transitions = len(self.all_transitions)
        self.covered_transitions: Set[Tuple[str, str, str]] = set()

        # Históricos
        self.coverage_history: List[float] = []          # % cobertura por episódio
        self.rewards_history: List[float] = []
        self.steps_history: List[int] = []
        self.epsilon_history: List[float] = []
        self.new_transitions_per_episode: List[int] = [] # transições novas por episódio
        self.episode_sequences: List[List[Tuple[str, str, str]]] = []  # sequências geradas

        # Episódio onde atingiu 100% de cobertura
        self.full_coverage_episode: Optional[int] = None

    @property
    def coverage_percentage(self) -> float:
        """Retorna a porcentagem atual de cobertura de transições."""
        if self.total_transitions == 0:
            return 100.0
        return (len(self.covered_transitions) / self.total_transitions) * 100.0

    @property
    def uncovered_transitions(self) -> Set[Tuple[str, str, str]]:
        """Retorna o conjunto de transições ainda não cobertas."""
        return self.all_transitions - self.covered_transitions

    def choose_action(self, state: str) -> str:
        """
        Seleciona uma ação usando a política ε-greedy.

        Args:
            state: Estado atual do agente.

        Returns:
            A ação escolhida.
        """
        valid_actions = self.env.get_valid_actions(state)

        if not valid_actions:
            return self.env.actions[0]

        if np.random.random() < self.epsilon:
            # Exploração: ação aleatória entre as válidas
            return np.random.choice(valid_actions)
        else:
            # Explotação: melhor ação pela Q-Table
            state_idx = self.env.state_to_idx[state]
            valid_action_indices = [
                self.env.action_to_idx[a] for a in valid_actions
            ]
            q_values = self.q_table[state_idx, valid_action_indices]
            best_local_idx = np.argmax(q_values)
            best_action_idx = valid_action_indices[best_local_idx]
            return self.env.idx_to_action[best_action_idx]

    def _compute_coverage_reward(
        self, state: str, action: str, next_state: str, is_valid: bool
    ) -> float:
        """
        Calcula a recompensa baseada em cobertura de transições.

        Args:
            state: Estado atual
            action: Ação executada
            next_state: Estado resultante
            is_valid: Se a transição é válida na FSM

        Returns:
            Recompensa calculada.
        """
        if not is_valid:
            return self.REWARD_INVALID_ACTION

        transition = (state, action, next_state)
        reward = 0.0

        if transition not in self.covered_transitions:
            # Transição nova! Alta recompensa
            reward += self.REWARD_NEW_TRANSITION
            self.covered_transitions.add(transition)
        else:
            # Já coberta
            reward += self.REWARD_OLD_TRANSITION

        return reward

    def update_q_table(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool,
    ) -> float:
        """
        Atualiza a Q-Table usando a equação de Bellman.

        Args:
            state: Estado atual
            action: Ação executada
            reward: Recompensa recebida
            next_state: Próximo estado
            done: Se o episódio terminou

        Returns:
            O TD error da atualização.
        """
        s_idx = self.env.state_to_idx[state]
        a_idx = self.env.action_to_idx[action]
        ns_idx = self.env.state_to_idx[next_state]

        current_q = self.q_table[s_idx, a_idx]

        if done:
            max_future_q = 0.0
        else:
            valid_next_actions = self.env.get_valid_actions(next_state)
            if valid_next_actions:
                valid_next_indices = [
                    self.env.action_to_idx[a] for a in valid_next_actions
                ]
                max_future_q = np.max(self.q_table[ns_idx, valid_next_indices])
            else:
                max_future_q = 0.0

        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q
        self.q_table[s_idx, a_idx] = current_q + self.alpha * td_error

        return td_error

    def decay_epsilon(self):
        """Aplica o decaimento de epsilon após cada episódio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        n_episodes: int = 500,
        verbose: bool = True,
        print_interval: int = 50,
    ) -> Dict[str, List]:
        """
        Executa o treinamento focado em cobertura de transições.

        Cada episódio gera uma sequência de teste. A cobertura é
        acumulada globalmente entre os episódios.

        Args:
            n_episodes: Número de episódios de treinamento
            verbose: Se True, imprime progresso
            print_interval: Intervalo para impressão

        Returns:
            Dicionário com histórico de treinamento.
        """
        if verbose:
            print("=" * 65)
            print("  Q-LEARNING PARA COBERTURA DE TRANSIÇÕES")
            print("=" * 65)
            print(f"  Total de transições na FSM: {self.total_transitions}")
            print(f"  Episódios: {n_episodes}")
            print(f"  α={self.alpha}  γ={self.gamma}  ε₀={self.epsilon}")
            print("=" * 65)

        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            episode_sequence = []
            coverage_before = len(self.covered_transitions)

            while not done:
                action = self.choose_action(state)

                # Executar ação no ambiente
                next_state, _, env_done, info = self.env.step(action)
                is_valid = not info.get("invalid_action", False)

                # Recompensa baseada em cobertura
                reward = self._compute_coverage_reward(
                    state, action, next_state, is_valid
                )

                # Atualizar Q-Table
                self.update_q_table(state, action, reward, next_state, env_done)

                # Registrar a transição na sequência do episódio
                if is_valid:
                    episode_sequence.append((state, action, next_state))

                state = next_state
                total_reward += reward
                steps += 1
                done = env_done

            # Decair epsilon
            self.decay_epsilon()

            # Métricas
            new_trans = len(self.covered_transitions) - coverage_before
            coverage_pct = self.coverage_percentage

            self.rewards_history.append(total_reward)
            self.steps_history.append(steps)
            self.epsilon_history.append(self.epsilon)
            self.coverage_history.append(coverage_pct)
            self.new_transitions_per_episode.append(new_trans)
            self.episode_sequences.append(episode_sequence)

            # Marcar episódio de 100% de cobertura
            if self.full_coverage_episode is None and coverage_pct >= 100.0:
                self.full_coverage_episode = episode + 1

            # Progresso
            if verbose and (episode + 1) % print_interval == 0:
                print(
                    f"  Ep {episode + 1:5d}/{n_episodes} | "
                    f"Cobertura: {coverage_pct:5.1f}% "
                    f"({len(self.covered_transitions)}/{self.total_transitions}) | "
                    f"Novas: {new_trans} | "
                    f"ε: {self.epsilon:.4f}"
                )

            # Parar cedo se já cobriu tudo e treinou o suficiente depois
            if (self.full_coverage_episode is not None
                    and episode + 1 >= self.full_coverage_episode + 50):
                if verbose:
                    print(f"\n  [!] 100% cobertura atingida no episódio "
                          f"{self.full_coverage_episode}. "
                          f"Encerrando treinamento no episódio {episode + 1}.")
                break

        if verbose:
            print("=" * 65)
            print("  TREINAMENTO DE COBERTURA CONCLUÍDO!")
            print(f"  Cobertura final: {self.coverage_percentage:.1f}% "
                  f"({len(self.covered_transitions)}/{self.total_transitions})")
            if self.full_coverage_episode:
                print(f"  100% atingido no episódio: {self.full_coverage_episode}")
            print("=" * 65)

        return {
            "rewards": self.rewards_history,
            "steps": self.steps_history,
            "epsilons": self.epsilon_history,
            "coverage": self.coverage_history,
            "new_transitions": self.new_transitions_per_episode,
        }

    def generate_test_suite(self) -> List[List[Tuple[str, str, str]]]:
        """
        Gera uma suíte de testes mínima a partir dos episódios de treinamento.

        Seleciona o menor subconjunto de episódios cujas sequências,
        juntas, cobrem todas as transições da FSM (greedy set cover).

        Returns:
            Lista de sequências de teste, onde cada sequência é uma
            lista de tuplas (estado, ação, próximo_estado).
        """
        # Transições cobertas por cada episódio
        episode_coverage = []
        for seq in self.episode_sequences:
            transitions_in_ep = set()
            for (s, a, ns) in seq:
                transitions_in_ep.add((s, a, ns))
            episode_coverage.append(transitions_in_ep)

        # Greedy set cover
        remaining = set(self.covered_transitions)
        selected_indices = []

        while remaining:
            # Encontrar episódio que cobre mais transições restantes
            best_idx = -1
            best_count = 0
            for i, cov in enumerate(episode_coverage):
                if i in selected_indices:
                    continue
                overlap = len(cov & remaining)
                if overlap > best_count:
                    best_count = overlap
                    best_idx = i

            if best_idx == -1 or best_count == 0:
                break

            selected_indices.append(best_idx)
            remaining -= episode_coverage[best_idx]

        # Ordenar por índice do episódio
        selected_indices.sort()

        return [self.episode_sequences[i] for i in selected_indices]

    def get_coverage_report(self) -> str:
        """
        Gera um relatório textual de cobertura de transições.

        Returns:
            String formatada com o relatório.
        """
        lines = []
        lines.append("=" * 65)
        lines.append("  RELATÓRIO DE COBERTURA DE TRANSIÇÕES")
        lines.append("=" * 65)
        lines.append(f"  Total de transições na FSM: {self.total_transitions}")
        lines.append(f"  Transições cobertas:        {len(self.covered_transitions)}")
        lines.append(f"  Cobertura:                  {self.coverage_percentage:.1f}%")
        lines.append("")

        # Listar transições cobertas
        lines.append("  ✔ Transições cobertas:")
        for (s, a, ns) in sorted(self.covered_transitions):
            lines.append(f"     {s} --[{a}]--> {ns}")

        # Listar transições não cobertas
        uncovered = self.uncovered_transitions
        if uncovered:
            lines.append("")
            lines.append("  ✘ Transições NÃO cobertas:")
            for (s, a, ns) in sorted(uncovered):
                lines.append(f"     {s} --[{a}]--> {ns}")
        else:
            lines.append("")
            lines.append("  ✔ TODAS as transições foram cobertas!")

        if self.full_coverage_episode:
            lines.append(f"\n  100% cobertura atingida no episódio: "
                         f"{self.full_coverage_episode}")

        lines.append("=" * 65)
        return "\n".join(lines)

    def get_q_table_formatted(self) -> str:
        """Retorna a Q-Table formatada como string legível."""
        header = f"{'Estado':<10}" + "".join(
            f"{'Ação ' + a:>14}" for a in self.env.actions
        )
        lines = [header, "-" * len(header)]
        for i, state in enumerate(self.env.states):
            values = "".join(
                f"{self.q_table[i, j]:14.4f}" for j in range(self.env.n_actions)
            )
            lines.append(f"{state:<10}{values}")
        return "\n".join(lines)
