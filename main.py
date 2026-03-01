"""
Q-Learning para Máquina de Estados Finitos (FSM)

Script principal que executa o treinamento do agente Q-Learning
em um autômato finito e exibe os resultados.

Uso:
    python main.py
"""

import sys
import os

# Configurar encoding UTF-8 para o console do Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from fsm_environment import FSMEnvironment, create_default_fsm
from q_learning_agent import QLearningAgent
from visualization import (
    plot_training_results,
    plot_fsm_diagram,
    plot_q_table_heatmap,
    print_q_table,
    print_best_path,
)


def main():
    """Função principal: treina o agente e exibe resultados."""

    # =====================================================
    # 1. CRIAR O AMBIENTE FSM
    # =====================================================
    print("\n[*] Criando ambiente FSM...")
    env = create_default_fsm()
    print(env)

    # =====================================================
    # 2. CRIAR O AGENTE Q-LEARNING
    # =====================================================
    print("\n[*] Criando agente Q-Learning...")
    agent = QLearningAgent(
        env=env,
        alpha=0.1,         # Taxa de aprendizado
        gamma=0.95,        # Fator de desconto
        epsilon=1.0,       # Exploração inicial (100%)
        epsilon_min=0.01,  # Exploração mínima (1%)
        epsilon_decay=0.995,  # Decaimento por episódio
    )

    # =====================================================
    # 3. TREINAR O AGENTE
    # =====================================================
    print("\n[>] Iniciando treinamento...\n")
    history = agent.train(
        n_episodes=1000,
        verbose=True,
        print_interval=100,
    )

    # =====================================================
    # 4. EXIBIR RESULTADOS
    # =====================================================

    # Q-Table final
    print_q_table(agent)

    # Melhor caminho aprendido
    best_path = agent.get_best_path()
    print_best_path(best_path, env)

    # =====================================================
    # 5. GRÁFICOS
    # =====================================================
    print("\n[+] Gerando visualizacoes...")

    # Gráfico de convergência do treinamento
    plot_training_results(agent, window_size=50, save_path="training_results.png")

    # Heatmap da Q-Table
    plot_q_table_heatmap(agent, save_path="q_table_heatmap.png")

    # Diagrama do autômato com caminho ótimo
    plot_fsm_diagram(env, best_path=best_path, save_path="fsm_diagram.png")

    print("\n[OK] Execucao concluida com sucesso!")
    print("   Arquivos gerados:")
    print("   - training_results.png  (graficos de convergencia)")
    print("   - q_table_heatmap.png   (heatmap da Q-Table)")
    print("   - fsm_diagram.png       (diagrama do automato)")


if __name__ == "__main__":
    main()
