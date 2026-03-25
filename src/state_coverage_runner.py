"""
Módulo Runner de Cobertura de Estados

Orquestra a geração de suítes de teste com cobertura de estados
usando o StateCoverageQLearningAgent.
"""

from typing import List, Tuple, Optional
from .fsm_environment import FSMEnvironment
from .state_coverage_agent import StateCoverageQLearningAgent


class StateCoverageRunner:
    """
    Executa o agente de cobertura de estados e gera relatórios e suítes de teste.

    Attributes:
        env: O ambiente FSM
        agent: O agente de cobertura de estados Q-Learning
        test_suite: A suíte de testes gerada após o treinamento
    """

    def __init__(
        self,
        env: FSMEnvironment,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99,
        n_episodes: int = 500,
    ):
        """
        Inicializa o runner de cobertura de estados.

        Args:
            env: Ambiente FSM para teste
            alpha: Taxa de aprendizado
            gamma: Fator de desconto
            epsilon: Taxa de exploração inicial
            epsilon_min: Exploração mínima
            epsilon_decay: Decaimento de epsilon
            n_episodes: Número de episódios de treinamento
        """
        self.env = env
        self.n_episodes = n_episodes
        self.agent = StateCoverageQLearningAgent(
            env=env,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )
        self.test_suite: List[List[Tuple[str, str, str]]] = []

    def run(self, verbose: bool = True) -> dict:
        """
        Executa o treinamento e gera a suíte de testes.

        Args:
            verbose: Se True, imprime progresso

        Returns:
            Dicionário com histórico de treinamento.
        """
        # 1. Treinar o agente
        history = self.agent.train(
            n_episodes=self.n_episodes,
            verbose=verbose,
        )

        # 2. Gerar suíte de testes mínima
        self.test_suite = self.agent.generate_test_suite()

        return history

    def print_test_suite(self):
        """Imprime a suíte de testes gerada de forma legível."""
        print("\n" + "=" * 65)
        print("  SUÍTE DE TESTES GERADA (Cobertura de Estados)")
        print("=" * 65)

        if not self.test_suite:
            print("  [!] Nenhuma suíte de teste gerada. Execute run() primeiro.")
            return

        total_states_covered = set()

        for tc_idx, sequence in enumerate(self.test_suite):
            print(f"\n  ── Caso de Teste {tc_idx + 1} ──")
            print(f"  Início: {self.env.initial_state}")

            states_in_tc = set()
            for step_idx, (s, a, ns) in enumerate(sequence):
                is_new_src = s not in total_states_covered
                is_new_dst = ns not in total_states_covered
                markers = []
                if is_new_src:
                    markers.append(f"{s} ★")
                if is_new_dst:
                    markers.append(f"{ns} ★")
                marker_str = f" NOVO({', '.join(markers)})" if markers else ""
                print(f"    Passo {step_idx + 1}: {s} --[{a}]--> {ns}{marker_str}")
                states_in_tc.add(s)
                states_in_tc.add(ns)
                total_states_covered.add(s)
                total_states_covered.add(ns)

            print(f"  Estados neste caso: {len(states_in_tc)}")

        print(f"\n  ── Resumo ──")
        print(f"  Total de casos de teste: {len(self.test_suite)}")
        print(f"  Total de estados cobertos: "
              f"{len(total_states_covered)}/{self.agent.total_states}")
        print(f"  Cobertura: {self.agent.coverage_percentage:.1f}%")
        print("=" * 65)

    def print_coverage_report(self):
        """Imprime o relatório de cobertura de estados."""
        print(self.agent.get_coverage_report())

    def print_test_suite_as_actions(self):
        """
        Imprime a suíte de testes como sequências de ações
        (formato mais compacto e prático).
        """
        print("\n" + "=" * 65)
        print("  SEQUÊNCIAS DE TESTE (formato de ações)")
        print("=" * 65)

        for tc_idx, sequence in enumerate(self.test_suite):
            actions = [a for (_, a, _) in sequence]
            states = [sequence[0][0]] + [ns for (_, _, ns) in sequence]
            path_str = " → ".join(states)
            actions_str = ", ".join(f'"{a}"' for a in actions)
            print(f"\n  TC{tc_idx + 1}: [{actions_str}]")
            print(f"       Caminho: {path_str}")

        print("=" * 65)
