"""
Q-Learning para Cobertura de Estados em FSM

Script principal que executa o agente Q-Learning adaptado para
gerar sequências de teste com critério de cobertura de estados.

Critério: State Coverage (Cobertura de Estados / All States)
Objetivo: Cobrir 100% dos estados alcançáveis da Máquina de Estados Finitos

Uso:
    python main_state_coverage.py <caminho_fsm.json> [--max-steps 50]
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
from src.state_coverage_runner import StateCoverageRunner
from src.visualization import (
    plot_state_coverage_progress,
    plot_fsm_state_coverage_diagram,
    plot_q_table_heatmap,
)


def main():
    """Função principal: treina o agente de cobertura de estados e exibe resultados."""

    # =====================================================
    # 0. ARGUMENTOS DA LINHA DE COMANDO
    # =====================================================
    parser = argparse.ArgumentParser(
        description="Q-Learning para Cobertura de Estados em FSM"
    )
    parser.add_argument(
        "fsm_file",
        help="Caminho para o arquivo JSON da FSM",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Número máximo de passos por episódio (padrão: 50)",
    )
    args = parser.parse_args()

    # =====================================================
    # 1. CRIAR O AMBIENTE FSM
    # =====================================================
    print(f"\n[*] Carregando FSM de: {args.fsm_file}")
    env = load_fsm_from_json(
        args.fsm_file,
        max_steps=args.max_steps,
    )
    print(env)

    # Mostrar todos os estados da FSM
    print(f"\n[*] Estados da FSM ({len(env.states)} total):")
    for state in env.states:
        print(f"    {state}")

    # =====================================================
    # 2. EXECUTAR Q-LEARNING PARA COBERTURA DE ESTADOS
    # =====================================================
    print("\n[>] Iniciando Q-Learning para cobertura de estados...\n")
    runner = StateCoverageRunner(
        env=env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        n_episodes=500,
    )

    history = runner.run(verbose=True)

    # =====================================================
    # 3. RELATÓRIO DE COBERTURA
    # =====================================================
    runner.print_coverage_report()

    # =====================================================
    # 4. SUÍTE DE TESTES GERADA
    # =====================================================
    runner.print_test_suite()
    runner.print_test_suite_as_actions()

    # =====================================================
    # 5. GRÁFICOS
    # =====================================================
    print("\n[+] Gerando visualizações...")

    os.makedirs("results", exist_ok=True)

    # Progresso da cobertura ao longo dos episódios
    plot_state_coverage_progress(
        runner.agent,
        save_path="results/state_coverage_progress.png"
    )

    # Diagrama do autômato com cobertura de estados
    plot_fsm_state_coverage_diagram(
        env,
        covered_states=runner.agent.covered_states,
        all_reachable_states=runner.agent.all_reachable_states,
        test_suite=runner.test_suite,
        save_path="results/fsm_state_coverage.png"
    )

    print("\n[OK] Execução concluída com sucesso!")
    print("   Arquivos gerados:")
    print("   - results/state_coverage_progress.png  (evolução da cobertura de estados)")
    print("   - results/fsm_state_coverage.png       (diagrama com estados cobertos)")


if __name__ == "__main__":
    main()
