"""
Módulo do Agente Q-Learning

Implementa o algoritmo Q-Learning para aprender a política ótima
de navegação em uma Máquina de Estados Finitos.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .fsm_environment import FSMEnvironment


class QLearningAgent:
    """
    Agente que utiliza o algoritmo Q-Learning para aprender a navegar
    em um ambiente de Máquina de Estados Finitos.

    O Q-Learning é um algoritmo off-policy de aprendizado por reforço que
    atualiza a Q-Table usando a equação de Bellman:

        Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') - Q(s, a)]

    Attributes:
        env: O ambiente FSM
        alpha: Taxa de aprendizado (learning rate)
        gamma: Fator de desconto (discount factor)
        epsilon: Taxa de exploração inicial (ε-greedy)
        epsilon_min: Valor mínimo de epsilon
        epsilon_decay: Fator de decaimento de epsilon por episódio
        q_table: Tabela Q de valores estado-ação
    """

    def __init__(
        self,
        env: FSMEnvironment,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """
        Inicializa o agente Q-Learning.

        Args:
            env: Ambiente FSM para interação
            alpha: Taxa de aprendizado (0 < α ≤ 1)
            gamma: Fator de desconto (0 ≤ γ ≤ 1)
            epsilon: Probabilidade inicial de exploração (0 ≤ ε ≤ 1)
            epsilon_min: Limite inferior para epsilon
            epsilon_decay: Multiplicador de decay para epsilon a cada episódio
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Inicializar Q-Table com zeros
        self.q_table = np.zeros((env.n_states, env.n_actions))

        # Histórico de treinamento
        self.rewards_history: List[float] = []
        self.steps_history: List[int] = []
        self.epsilon_history: List[float] = []

    def choose_action(self, state: str) -> str:
        """
        Seleciona uma ação usando a política ε-greedy.

        Com probabilidade ε, escolhe uma ação aleatória (exploração).
        Com probabilidade (1-ε), escolhe a melhor ação conhecida (exploração).

        Args:
            state: Estado atual do agente.

        Returns:
            A ação escolhida.
        """
        valid_actions = self.env.get_valid_actions(state)

        if not valid_actions:
            # Se não há ações válidas, retorna uma ação qualquer
            return self.env.actions[0]

        if np.random.random() < self.epsilon:
            # Exploração: ação aleatória entre as válidas
            return np.random.choice(valid_actions)
        else:
            # Explotação: melhor ação conhecida (pela Q-Table)
            state_idx = self.env.state_to_idx[state]
            valid_action_indices = [
                self.env.action_to_idx[a] for a in valid_actions
            ]

            # Selecionar a ação válida com maior valor Q
            q_values = self.q_table[state_idx, valid_action_indices]
            best_local_idx = np.argmax(q_values)
            best_action_idx = valid_action_indices[best_local_idx]

            return self.env.idx_to_action[best_action_idx]

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

        Q(s,a) ← Q(s,a) + α [r + γ · max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Estado atual
            action: Ação executada
            reward: Recompensa recebida
            next_state: Próximo estado
            done: Se o episódio terminou

        Returns:
            O erro temporal (TD error) da atualização.
        """
        s_idx = self.env.state_to_idx[state]
        a_idx = self.env.action_to_idx[action]
        ns_idx = self.env.state_to_idx[next_state]

        # Valor Q atual
        current_q = self.q_table[s_idx, a_idx]

        # Valor Q futuro máximo
        if done:
            max_future_q = 0.0
        else:
            # Considerar apenas ações válidas no próximo estado
            valid_next_actions = self.env.get_valid_actions(next_state)
            if valid_next_actions:
                valid_next_indices = [
                    self.env.action_to_idx[a] for a in valid_next_actions
                ]
                max_future_q = np.max(self.q_table[ns_idx, valid_next_indices])
            else:
                max_future_q = 0.0

        # Equação de Bellman (TD Update)
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q
        self.q_table[s_idx, a_idx] = current_q + self.alpha * td_error

        return td_error

    def decay_epsilon(self):
        """Aplica o decaimento de epsilon após cada episódio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        n_episodes: int = 1000,
        verbose: bool = True,
        print_interval: int = 100,
    ) -> Dict[str, List]:
        """
        Executa o loop de treinamento do Q-Learning.

        Args:
            n_episodes: Número de episódios de treinamento
            verbose: Se True, imprime progresso durante o treinamento
            print_interval: Intervalo de episódios para impressão

        Returns:
            Dicionário com histórico de treinamento:
            - 'rewards': recompensa total por episódio
            - 'steps': passos por episódio
            - 'epsilons': valor de epsilon por episódio
        """
        if verbose:
            print("=" * 60)
            print("  TREINAMENTO Q-LEARNING PARA FSM")
            print("=" * 60)
            print(f"  Episódios: {n_episodes}")
            print(f"  α (learning rate): {self.alpha}")
            print(f"  γ (discount factor): {self.gamma}")
            print(f"  ε (exploração inicial): {self.epsilon}")
            print(f"  ε decay: {self.epsilon_decay}")
            print(f"  ε mínimo: {self.epsilon_min}")
            print("=" * 60)

        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                # 1. Escolher ação (ε-greedy)
                action = self.choose_action(state)

                # 2. Executar ação no ambiente
                next_state, reward, done, info = self.env.step(action)

                # 3. Atualizar Q-Table
                self.update_q_table(state, action, reward, next_state, done)

                # 4. Avançar
                state = next_state
                total_reward += reward
                steps += 1

            # Decair epsilon
            self.decay_epsilon()

            # Registrar histórico
            self.rewards_history.append(total_reward)
            self.steps_history.append(steps)
            self.epsilon_history.append(self.epsilon)

            # Imprimir progresso
            if verbose and (episode + 1) % print_interval == 0:
                avg_reward = np.mean(self.rewards_history[-print_interval:])
                avg_steps = np.mean(self.steps_history[-print_interval:])
                print(
                    f"  Episódio {episode + 1:5d}/{n_episodes} | "
                    f"Recompensa média: {avg_reward:8.2f} | "
                    f"Passos médios: {avg_steps:5.1f} | "
                    f"ε: {self.epsilon:.4f}"
                )

        if verbose:
            print("=" * 60)
            print("  TREINAMENTO CONCLUÍDO!")
            print("=" * 60)

        return {
            "rewards": self.rewards_history,
            "steps": self.steps_history,
            "epsilons": self.epsilon_history,
        }

    def get_best_path(self, max_steps: int = 20) -> List[Tuple[str, str, str]]:
        """
        Extrai o melhor caminho aprendido seguindo a política greedy.

        Args:
            max_steps: Limite de passos para evitar loops infinitos.

        Returns:
            Lista de tuplas (estado, ação, próximo_estado) representando
            o caminho ótimo aprendido.
        """
        path = []
        state = self.env.reset()
        visited = set()

        for _ in range(max_steps):
            if state in self.env.goal_states:
                break

            if state in visited:
                # Evita loop infinito
                break
            visited.add(state)

            # Salvar epsilon e forçar greedy
            old_epsilon = self.epsilon
            self.epsilon = 0.0
            action = self.choose_action(state)
            self.epsilon = old_epsilon

            next_state, _, done, _ = self.env.step(action)
            path.append((state, action, next_state))
            state = next_state

            if done:
                break

        # Resetar o ambiente
        self.env.reset()
        return path

    def get_q_table_formatted(self) -> str:
        """
        Retorna a Q-Table formatada como string legível.

        Returns:
            String formatada representando a Q-Table.
        """
        header = f"{'Estado':<10}" + "".join(
            f"{'Ação ' + a:>14}" for a in self.env.actions
        )
        lines = [header, "-" * len(header)]

        for i, state in enumerate(self.env.states):
            values = "".join(f"{self.q_table[i, j]:14.4f}" for j in range(self.env.n_actions))
            lines.append(f"{state:<10}{values}")

        return "\n".join(lines)
