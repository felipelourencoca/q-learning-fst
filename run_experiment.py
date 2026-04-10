"""
Script Experimental Unificado — Comparação All States vs All Transitions

Executa os dois métodos de cobertura (All States e All Transitions) sobre
as mesmas FSMs, com os mesmos hiperparâmetros, por múltiplas repetições
com sementes diferentes, coletando resultados em formato estruturado.

Uso:
    python run_experiment.py
    python run_experiment.py --n-repetitions 5 --fsm fsm/_01_LightSwitch_flattened.json
    python run_experiment.py --n-episodes 300 --max-steps 100

Saída:
    results/experiment_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── summary.csv
    └── <fsm_name>/
        ├── all_transitions_seed_01.json
        ├── all_states_seed_01.json
        └── ...
"""

import sys
import os
import csv
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Configurar encoding UTF-8 para o console do Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.fsm_environment import load_fsm_from_json
from src.coverage_runner import CoverageRunner
from src.state_coverage_runner import StateCoverageRunner
from src.experiment_config import ExperimentConfig
from src.analyze_results import analyze_experiment


def extract_fsm_name(filepath: str) -> str:
    """Extrai um nome legível a partir do caminho do arquivo FSM."""
    return Path(filepath).stem


def run_single_experiment(
    method: str,
    fsm_file: str,
    seed: int,
    config: ExperimentConfig,
) -> dict:
    """
    Executa uma única instância de experimento (1 método, 1 FSM, 1 seed).

    Args:
        method: 'all_transitions' ou 'all_states'
        fsm_file: Caminho para o arquivo JSON da FSM
        seed: Semente aleatória
        config: Configuração experimental

    Returns:
        Dicionário com métricas coletadas.
    """
    env = load_fsm_from_json(fsm_file, max_steps=config.max_steps)

    runner_kwargs = dict(
        env=env,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        n_episodes=config.n_episodes,
    )

    if method == "all_transitions":
        runner = CoverageRunner(**runner_kwargs)
    elif method == "all_states":
        runner = StateCoverageRunner(**runner_kwargs)
    else:
        raise ValueError(f"Método desconhecido: {method}")

    start_time = time.time()
    runner.run(verbose=False, seed=seed)
    elapsed = time.time() - start_time

    agent = runner.agent

    # Coletar métricas
    result = {
        "method": method,
        "fsm_file": fsm_file,
        "fsm_name": extract_fsm_name(fsm_file),
        "seed": seed,
        # Hiperparâmetros usados (rastreabilidade)
        "hyperparameters": {
            "alpha": config.alpha,
            "gamma": config.gamma,
            "epsilon": config.epsilon,
            "epsilon_min": config.epsilon_min,
            "epsilon_decay": config.epsilon_decay,
            "n_episodes": config.n_episodes,
            "max_steps": config.max_steps,
        },
        # Métricas de resultado
        "coverage_final": agent.coverage_percentage,
        "episode_full_coverage": agent.full_coverage_episode,
        "total_episodes_run": len(agent.coverage_history),
        "total_steps": sum(agent.steps_history),
        "test_suite_size": len(runner.test_suite),
        "execution_time_s": round(elapsed, 4),
        # Históricos completos
        "coverage_history": agent.coverage_history,
        "steps_history": agent.steps_history,
        "rewards_history": agent.rewards_history,
    }

    # Métricas específicas por método + campo unificado coverage_target
    if method == "all_transitions":
        result["total_transitions"] = agent.total_transitions
        result["covered_transitions"] = len(agent.covered_transitions)
        result["coverage_target"] = agent.total_transitions
        result["coverage_achieved"] = len(agent.covered_transitions)
        result["new_per_episode"] = agent.new_transitions_per_episode
    elif method == "all_states":
        result["total_states"] = agent.total_states
        result["covered_states"] = len(agent.covered_states)
        result["coverage_target"] = agent.total_states
        result["coverage_achieved"] = len(agent.covered_states)
        result["new_per_episode"] = agent.new_states_per_episode

    return result


def save_individual_result(result: dict, output_dir: str):
    """Salva o resultado individual de uma execução como JSON."""
    fsm_name = result["fsm_name"]
    method = result["method"]
    seed = result["seed"]

    fsm_dir = os.path.join(output_dir, fsm_name)
    os.makedirs(fsm_dir, exist_ok=True)

    filename = f"{method}_seed_{seed:02d}.json"
    filepath = os.path.join(fsm_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def run_experiment(config: ExperimentConfig):
    """
    Executa o experimento completo conforme a configuração fornecida.

    Para cada combinação (FSM, método, seed), executa o treinamento e
    coleta métricas. Gera summary.csv e JSONs individuais.

    Args:
        config: Configuração experimental centralizada.
    """
    # Criar diretório de saída com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Salvar configuração
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

    methods = ["all_transitions", "all_states"]
    total_runs = len(config.fsm_files) * len(methods) * len(config.seeds)

    print("=" * 70)
    print("  EXPERIMENTO COMPARATIVO — All States vs All Transitions")
    print("=" * 70)
    print(f"  FSMs:        {len(config.fsm_files)}")
    print(f"  Métodos:     {', '.join(methods)}")
    print(f"  Repetições:  {config.n_repetitions}")
    print(f"  Total:       {total_runs} execuções")
    print(f"  Saída:       {output_dir}")
    print("=" * 70)

    # CSV summary
    csv_path = os.path.join(output_dir, "summary.csv")
    csv_fields = [
        "fsm", "method", "seed", "coverage_final",
        "episode_full_coverage", "total_episodes",
        "total_steps", "test_suite_size", "execution_time_s",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        run_count = 0

        for fsm_file in config.fsm_files:
            fsm_name = extract_fsm_name(fsm_file)

            # Verificar que o arquivo FSM existe
            if not os.path.exists(fsm_file):
                print(f"\n  [!] AVISO: FSM não encontrada: {fsm_file} — pulando")
                continue

            print(f"\n  ── FSM: {fsm_name} ──")

            for method in methods:
                for seed in config.seeds:
                    run_count += 1

                    try:
                        result = run_single_experiment(
                            method=method,
                            fsm_file=fsm_file,
                            seed=seed,
                            config=config,
                        )

                        # Escrever linha no CSV
                        csv_row = {
                            "fsm": fsm_name,
                            "method": method,
                            "seed": seed,
                            "coverage_final": result["coverage_final"],
                            "episode_full_coverage": result["episode_full_coverage"] or "",
                            "total_episodes": result["total_episodes_run"],
                            "total_steps": result["total_steps"],
                            "test_suite_size": result["test_suite_size"],
                            "execution_time_s": result["execution_time_s"],
                        }
                        writer.writerow(csv_row)
                        csvfile.flush()

                        # Salvar JSON individual
                        save_individual_result(result, output_dir)

                        # Log de progresso
                        cov = result["coverage_final"]
                        ep100 = result["episode_full_coverage"] or "-"
                        t = result["execution_time_s"]
                        print(
                            f"    [{run_count:4d}/{total_runs}] "
                            f"{method:20s} seed={seed:2d} | "
                            f"cov={cov:5.1f}% ep100={str(ep100):>4s} "
                            f"t={t:.2f}s"
                        )

                    except Exception as e:
                        print(
                            f"    [{run_count:4d}/{total_runs}] "
                            f"{method:20s} seed={seed:2d} | "
                            f"ERRO: {e}"
                        )
                        # Escrever linha de erro no CSV
                        csv_row = {
                            "fsm": fsm_name,
                            "method": method,
                            "seed": seed,
                            "coverage_final": "ERRO",
                            "episode_full_coverage": "",
                            "total_episodes": "",
                            "total_steps": "",
                            "test_suite_size": "",
                            "execution_time_s": "",
                        }
                        writer.writerow(csv_row)
                        csvfile.flush()

    print("\n" + "=" * 70)
    print("  EXPERIMENTO CONCLUÍDO!")
    print("=" * 70)
    print(f"  Execuções:    {run_count}/{total_runs}")
    print(f"  Configuração: {config_path}")
    print(f"  Resumo CSV:   {csv_path}")
    print(f"  Resultados:   {output_dir}")
    print("=" * 70)

    # Executar análise estatística automaticamente
    print("\n")
    analyze_experiment(output_dir)

    return output_dir


def main():
    """Ponto de entrada com argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Experimento Comparativo: All States vs All Transitions"
    )
    parser.add_argument(
        "--n-repetitions", type=int, default=30,
        help="Número de repetições por (FSM, método) (padrão: 30)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=500,
        help="Número máximo de episódios de treinamento (padrão: 500)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Número máximo de passos por episódio (padrão: 50)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Taxa de aprendizado (padrão: 0.1)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95,
        help="Fator de desconto (padrão: 0.95)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="Exploração inicial (padrão: 1.0)",
    )
    parser.add_argument(
        "--epsilon-min", type=float, default=0.05,
        help="Exploração mínima (padrão: 0.05)",
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.99,
        help="Decaimento de epsilon (padrão: 0.99)",
    )
    parser.add_argument(
        "--fsm", nargs="+", default=None,
        help="Caminhos dos arquivos FSM (padrão: FSMs 01-03)",
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        n_repetitions=args.n_repetitions,
    )

    if args.fsm is not None:
        config.fsm_files = args.fsm

    run_experiment(config)


if __name__ == "__main__":
    main()
