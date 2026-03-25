"""
Q-Learning para Máquina de Estados Finitos (FSM)

Script principal que executa o treinamento do agente Q-Learning
em um autômato finito e exibe os resultados.

Uso:
    python main.py <caminho_fsm.json> [--goal-states S1 S2] [--max-steps 50]
"""

import sys
import os
import argparse

# Configurar encoding UTF-8 para o console do Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.fsm_environment import load_fsm_from_json
from src.q_learning_agent import QLearningAgent
from src.visualization import (
    plot_training_results,
    plot_fsm_diagram,
    plot_q_table_heatmap,
    print_q_table,
    print_best_path,
)


def main():
    """Função principal: treina o agente e exibe resultados."""

    # =====================================================
    # 0. ARGUMENTOS DA LINHA DE COMANDO
    # =====================================================
    parser = argparse.ArgumentParser(
        description="Q-Learning para Máquina de Estados Finitos (FSM)"
    )
    parser.add_argument(
        "fsm_file",
        help="Caminho para o arquivo JSON da FSM",
    )
    parser.add_argument(
        "--goal-states",
        nargs="+",
        default=None,
        help="Estados objetivo da FSM (ex: --goal-states S5 S6)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Número máximo de passos por episódio (padrão: 50)",
    )
    args = parser.parse_args()

    goal_states = set(args.goal_states) if args.goal_states else None

    # =====================================================
    # 1. CRIAR O AMBIENTE FSM
    # =====================================================
    print(f"\n[*] Carregando FSM de: {args.fsm_file}")
    env = load_fsm_from_json(
        args.fsm_file,
        goal_states=goal_states,
        max_steps=args.max_steps,
    )
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

    os.makedirs("results", exist_ok=True)

    # Gráfico de convergência do treinamento
    plot_training_results(agent, window_size=50, save_path="results/training_results.png")

    # Heatmap da Q-Table
    plot_q_table_heatmap(agent, save_path="results/q_table_heatmap.png")

    # Diagrama do autômato com caminho ótimo
    plot_fsm_diagram(env, best_path=best_path, save_path="results/fsm_diagram.png")

    print("\n[OK] Execucao concluida com sucesso!")
    print("   Arquivos gerados:")
    print("   - results/training_results.png  (graficos de convergencia)")
    print("   - results/q_table_heatmap.png   (heatmap da Q-Table)")
    print("   - results/fsm_diagram.png       (diagrama do automato)")


if __name__ == "__main__":
    main()

