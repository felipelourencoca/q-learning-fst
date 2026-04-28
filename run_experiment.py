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
    ├── <fsm_name>/
    │   ├── all_transitions_seed_01.json
    │   ├── all_states_seed_01.json
    │   └── ...
    ├── q_tables/
    │   └── {method}_{fsm}_rep{NN}_seed{NN}_q_table.json
    ├── logs/
    │   └── {method}_{fsm}_rep{NN}_seed{NN}.json
    └── test_sequences/
        └── {method}_{fsm}_rep{NN}_seed{NN}_sequences.json
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
    repetition: int = 1,
) -> dict:
    """
    Executa uma única instância de experimento (1 método, 1 FSM, 1 seed).

    Args:
        method: 'all_transitions' ou 'all_states'
        fsm_file: Caminho para o arquivo JSON da FSM
        seed: Semente aleatória
        config: Configuração experimental
        repetition: Número da repetição (1-based)

    Returns:
        Dicionário com métricas coletadas e referência ao runner.
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
        "repetition": repetition,
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
        "cumulative_reward": round(sum(agent.rewards_history), 4),
        # Históricos completos
        "coverage_history": agent.coverage_history,
        "steps_history": agent.steps_history,
        "rewards_history": agent.rewards_history,
    }

    # Métricas de estados e transições (unificadas para ambos os métodos)
    if method == "all_transitions":
        result["total_transitions"] = agent.total_transitions
        result["covered_transitions"] = len(agent.covered_transitions)
        result["total_states"] = env.n_states
        result["covered_states"] = None
        result["coverage_target"] = agent.total_transitions
        result["coverage_achieved"] = len(agent.covered_transitions)
        result["new_per_episode"] = agent.new_transitions_per_episode
    elif method == "all_states":
        result["total_states"] = agent.total_states
        result["covered_states"] = len(agent.covered_states)
        result["total_transitions"] = len(agent.env.transitions)
        result["covered_transitions"] = None
        result["coverage_target"] = agent.total_states
        result["coverage_achieved"] = len(agent.covered_states)
        result["new_per_episode"] = agent.new_states_per_episode

    # Anexar referência ao runner para acesso à Q-table e test_suite
    result["_runner"] = runner

    return result


def save_individual_result(result: dict, output_dir: str) -> str:
    """Salva o resultado individual de uma execução como JSON."""
    fsm_name = result["fsm_name"]
    method = result["method"]
    seed = result["seed"]

    fsm_dir = os.path.join(output_dir, fsm_name)
    os.makedirs(fsm_dir, exist_ok=True)

    filename = f"{method}_seed_{seed:02d}.json"
    filepath = os.path.join(fsm_dir, filename)

    # Excluir referência interna ao runner antes de serializar
    serializable = {k: v for k, v in result.items() if not k.startswith("_")}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    return filepath


def save_q_table(result: dict, output_dir: str) -> str:
    """
    Salva a Q-table do agente em arquivo JSON com metadados completos.

    Returns:
        Caminho do arquivo salvo.
    """
    runner = result["_runner"]
    agent = runner.agent
    env = agent.env

    q_dir = os.path.join(output_dir, "q_tables")
    os.makedirs(q_dir, exist_ok=True)

    method = result["method"]
    fsm_name = result["fsm_name"]
    rep = result["repetition"]
    seed = result["seed"]

    filename = f"{method}_{fsm_name}_rep{rep:02d}_seed{seed:02d}_q_table.json"
    filepath = os.path.join(q_dir, filename)

    q_data = {
        "method": method,
        "fsm_name": fsm_name,
        "repetition": rep,
        "seed": seed,
        "hyperparameters": result["hyperparameters"],
        "state_index_map": env.state_to_idx,
        "action_index_map": env.action_to_idx,
        "index_to_state": {str(v): k for k, v in env.state_to_idx.items()},
        "index_to_action": {str(v): k for k, v in env.action_to_idx.items()},
        "q_table": agent.q_table.tolist(),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(q_data, f, indent=2, ensure_ascii=False)

    return filepath


def save_test_sequences(result: dict, output_dir: str) -> str:
    """
    Salva as sequências de teste geradas pelo agente.

    Returns:
        Caminho do arquivo salvo.
    """
    runner = result["_runner"]
    agent = runner.agent

    seq_dir = os.path.join(output_dir, "test_sequences")
    os.makedirs(seq_dir, exist_ok=True)

    method = result["method"]
    fsm_name = result["fsm_name"]
    rep = result["repetition"]
    seed = result["seed"]

    filename = f"{method}_{fsm_name}_rep{rep:02d}_seed{seed:02d}_sequences.json"
    filepath = os.path.join(seq_dir, filename)

    # Suíte de testes mínima (greedy set cover)
    test_suite_data = []
    for tc_idx, sequence in enumerate(runner.test_suite):
        steps = [
            {"state": s, "action": a, "next_state": ns}
            for s, a, ns in sequence
        ]
        tc = {
            "test_case_index": tc_idx + 1,
            "num_steps": len(steps),
            "steps": steps,
        }
        test_suite_data.append(tc)

    # Todas as sequências por episódio (compactas)
    episode_sequences_data = []
    for ep_idx, sequence in enumerate(agent.episode_sequences):
        if not sequence:
            continue
        steps = [
            {"state": s, "action": a, "next_state": ns}
            for s, a, ns in sequence
        ]
        ep_data = {
            "episode": ep_idx + 1,
            "num_steps": len(steps),
            "coverage_at_end": agent.coverage_history[ep_idx] if ep_idx < len(agent.coverage_history) else None,
            "reward": agent.rewards_history[ep_idx] if ep_idx < len(agent.rewards_history) else None,
            "steps": steps,
        }
        episode_sequences_data.append(ep_data)

    seq_data = {
        "method": method,
        "fsm_name": fsm_name,
        "repetition": rep,
        "seed": seed,
        "coverage_criterion": "All Transitions" if method == "all_transitions" else "All States",
        "coverage_final": result["coverage_final"],
        "test_suite": test_suite_data,
        "episode_sequences": episode_sequences_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(seq_data, f, indent=2, ensure_ascii=False)

    return filepath


def save_execution_log(
    result: dict,
    output_dir: str,
    run_start_time: str,
    generated_files: list,
    error: str = None,
) -> str:
    """
    Salva um log estruturado de execução como JSON.

    Returns:
        Caminho do arquivo salvo.
    """
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    method = result["method"]
    fsm_name = result["fsm_name"]
    rep = result["repetition"]
    seed = result["seed"]

    filename = f"{method}_{fsm_name}_rep{rep:02d}_seed{seed:02d}.json"
    filepath = os.path.join(log_dir, filename)

    agent = result["_runner"].agent

    # Resumo compacto de episódios (a cada 10% dos episódios executados)
    total_ep = result["total_episodes_run"]
    sample_interval = max(1, total_ep // 10)
    episode_summary = []
    for ep_idx in range(0, total_ep, sample_interval):
        ep_entry = {
            "episode": ep_idx + 1,
            "coverage": agent.coverage_history[ep_idx] if ep_idx < len(agent.coverage_history) else None,
            "steps": agent.steps_history[ep_idx] if ep_idx < len(agent.steps_history) else None,
            "reward": round(agent.rewards_history[ep_idx], 4) if ep_idx < len(agent.rewards_history) else None,
        }
        episode_summary.append(ep_entry)
    # Sempre inclui o último episódio
    if total_ep > 0 and (total_ep - 1) % sample_interval != 0:
        last = total_ep - 1
        episode_summary.append({
            "episode": last + 1,
            "coverage": agent.coverage_history[last] if last < len(agent.coverage_history) else None,
            "steps": agent.steps_history[last] if last < len(agent.steps_history) else None,
            "reward": round(agent.rewards_history[last], 4) if last < len(agent.rewards_history) else None,
        })

    log_data = {
        "method": method,
        "fsm_name": fsm_name,
        "fsm_file": result["fsm_file"],
        "repetition": rep,
        "seed": seed,
        "start_time": run_start_time,
        "end_time": datetime.now().isoformat(),
        "execution_time_s": result["execution_time_s"],
        "hyperparameters": result["hyperparameters"],
        "results_summary": {
            "coverage_final": result["coverage_final"],
            "episode_full_coverage": result["episode_full_coverage"],
            "total_episodes_run": result["total_episodes_run"],
            "total_steps": result["total_steps"],
            "test_suite_size": result["test_suite_size"],
            "cumulative_reward": result.get("cumulative_reward"),
            "total_states": result.get("total_states"),
            "total_transitions": result.get("total_transitions"),
            "coverage_target": result.get("coverage_target"),
            "coverage_achieved": result.get("coverage_achieved"),
        },
        "episode_summary": episode_summary,
        "generated_files": generated_files,
        "error": error,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    return filepath


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
                for seed_idx, seed in enumerate(config.seeds):
                    run_count += 1
                    repetition = seed_idx + 1
                    run_start_time = datetime.now().isoformat()

                    try:
                        result = run_single_experiment(
                            method=method,
                            fsm_file=fsm_file,
                            seed=seed,
                            config=config,
                            repetition=repetition,
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

                        # Salvar JSON individual (existente)
                        generated_files = []
                        json_path = save_individual_result(result, output_dir)
                        generated_files.append(json_path)

                        # Salvar Q-table
                        qt_path = save_q_table(result, output_dir)
                        generated_files.append(qt_path)

                        # Salvar sequências de teste
                        seq_path = save_test_sequences(result, output_dir)
                        generated_files.append(seq_path)

                        # Salvar log de execução
                        log_path = save_execution_log(
                            result, output_dir, run_start_time, generated_files
                        )

                        # Log de progresso (stdout)
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
        help="Caminhos dos arquivos FSM (padrão: FSMs 01-05)",
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
    else:
        config.fsm_files = [
            "fsm/_01_LightSwitch_flattened.json",
            "fsm/_02_DimmableLightSwitch_flattened.json",
            "fsm/_03_MotionLightSwitch_flattened.json",
            "fsm/_04_LightAndMotionSensingLightSwitch_flattened.json",
            "fsm/_05_PresenceSimulationLightSwitch_flattened.json",
        ]

    run_experiment(config)


if __name__ == "__main__":
    main()
