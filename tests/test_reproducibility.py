"""Teste de reprodutibilidade end-to-end."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiment_config import ExperimentConfig
from run_experiment import run_single_experiment

config = ExperimentConfig(n_repetitions=1)

# Execução A
r_a = run_single_experiment("all_transitions", "fsm/_02_DimmableLightSwitch_flattened.json", seed=42, config=config)
# Execução B
r_b = run_single_experiment("all_transitions", "fsm/_02_DimmableLightSwitch_flattened.json", seed=42, config=config)

checks = {
    "coverage_history": r_a["coverage_history"] == r_b["coverage_history"],
    "steps_history": r_a["steps_history"] == r_b["steps_history"],
    "coverage_final": r_a["coverage_final"] == r_b["coverage_final"],
    "episode_full_coverage": r_a["episode_full_coverage"] == r_b["episode_full_coverage"],
    "test_suite_size": r_a["test_suite_size"] == r_b["test_suite_size"],
}

all_ok = True
for k, v in checks.items():
    status = "OK" if v else "FALHOU"
    print(f"  {status}: {k}")
    if not v:
        all_ok = False

assert all_ok, "Reprodutibilidade falhou!"
print("\nREPRODUTIBILIDADE END-TO-END OK")
