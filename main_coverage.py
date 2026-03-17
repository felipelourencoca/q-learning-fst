"""
Q-Learning para Cobertura de Transições em FSM

Script principal que executa o agente Q-Learning adaptado para
gerar sequências de teste com critério de cobertura de transições.

Critério: Transition Coverage (Cobertura de Transições)
Objetivo: Cobrir 100% das transições definidas na Máquina de Estados Finitos

Uso:
    python main_coverage.py <caminho_fsm.json> [--max-steps 50]
"""

import sys
import os
import argparse

# Configurar encoding UTF-8 para o console do Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from fsm_environment import load_fsm_from_json
from coverage_runner import CoverageRunner
from visualization import (
    plot_coverage_progress,
    plot_fsm_coverage_diagram,
    plot_q_table_heatmap,
)


def main():
    """Função principal: treina o agente de cobertura e exibe resultados."""

    # =====================================================
    # 0. ARGUMENTOS DA LINHA DE COMANDO
    # =====================================================
    parser = argparse.ArgumentParser(
        description="Q-Learning para Cobertura de Transições em FSM"
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

    # Mostrar todas as transições da FSM
    print(f"\n[*] Transições da FSM ({len(env.transitions)} total):")
    for (state, action), next_state in sorted(env.transitions.items()):
        print(f"    {state} --[{action}]--> {next_state}")

    # =====================================================
    # 2. EXECUTAR Q-LEARNING PARA COBERTURA
    # =====================================================
    print("\n[>] Iniciando Q-Learning para cobertura de transições...\n")
    runner = CoverageRunner(
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

    # Progresso da cobertura ao longo dos episódios
    plot_coverage_progress(
        runner.agent,
        save_path="coverage_progress.png"
    )

    # Diagrama do autômato com cobertura
    plot_fsm_coverage_diagram(
        env,
        covered_transitions=runner.agent.covered_transitions,
        test_suite=runner.test_suite,
        save_path="fsm_coverage.png"
    )

    print("\n[OK] Execução concluída com sucesso!")
    print("   Arquivos gerados:")
    print("   - coverage_progress.png  (evolução da cobertura)")
    print("   - fsm_coverage.png       (diagrama com transições cobertas)")


if __name__ == "__main__":
    main()

