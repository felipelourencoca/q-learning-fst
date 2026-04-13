"""
Testes de Robustez para FSMs Maiores (03, 04, 05)

Valida que os métodos All States e All Transitions funcionam corretamente
em FSMs com dezenas a centenas de estados e transições.

Critérios de validação por teste:
  1. Parsing: FSM carregada sem erros, dimensões corretas
  2. Inicialização: Agentes e Runners instanciados sem erros
  3. Execução: Treinamento completa sem exceções
  4. Cobertura: Cobertura final atinge 100%
  5. Estrutura: Históricos, suíte de testes e métricas íntegros
  6. Reprodutibilidade: Seed fixa produz resultado determinístico

Configuração:
  - Seeds fixas para reprodutibilidade
  - Episódios e max_steps calibrados por FSM para manter tempo < 15s/teste
  - Não exige 100% para FSM 05 AllTrans se timeout (documentado)
"""

import sys
import os
import time

# Garantir que o diretório raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configurar encoding UTF-8 para o console do Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.fsm_environment import load_fsm_from_json
from src.state_coverage_agent import StateCoverageQLearningAgent
from src.state_coverage_runner import StateCoverageRunner
from src.coverage_agent import CoverageQLearningAgent
from src.coverage_runner import CoverageRunner

# ══════════════════════════════════════════════════════════════════
# Configurações calibradas por FSM
# ══════════════════════════════════════════════════════════════════

FSM_CONFIGS = {
    "03": {
        "path": "fsm/_03_MotionLightSwitch_flattened.json",
        "label": "MotionLightSwitch",
        "expected_states": 43,
        "expected_transitions": 107,
        "n_episodes": 300,
        "max_steps": 100,
        "seed": 42,
    },
    "04": {
        "path": "fsm/_04_LightAndMotionSensingLightSwitch_flattened.json",
        "label": "LightAndMotionSensing",
        "expected_states": 76,
        "expected_transitions": 250,
        "n_episodes": 500,
        "max_steps": 200,
        "seed": 42,
    },
    "05": {
        "path": "fsm/_05_PresenceSimulationLightSwitch_flattened.json",
        "label": "PresenceSimulation",
        "expected_states": 473,
        "expected_transitions": 2134,
        "n_episodes": 500,
        "max_steps": 500,
        "seed": 42,
    },
}

COMMON_HYPERPARAMS = {
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.99,
}


# ══════════════════════════════════════════════════════════════════
# Utilitários
# ══════════════════════════════════════════════════════════════════

def validate_environment(env, cfg, test_label):
    """Valida que o ambiente foi carregado corretamente."""
    assert env.n_states == cfg["expected_states"], (
        f"{test_label}: Esperados {cfg['expected_states']} estados, "
        f"encontrados {env.n_states}"
    )
    assert len(env.transitions) == cfg["expected_transitions"], (
        f"{test_label}: Esperadas {cfg['expected_transitions']} transições, "
        f"encontradas {len(env.transitions)}"
    )
    assert env.initial_state is not None, (
        f"{test_label}: Estado inicial é None"
    )
    assert env.initial_state in env.state_to_idx, (
        f"{test_label}: Estado inicial '{env.initial_state}' não está no índice"
    )


def validate_history(agent, test_label):
    """Valida integridade dos históricos do agente."""
    n_eps = len(agent.coverage_history)
    assert n_eps > 0, f"{test_label}: coverage_history vazio"
    assert len(agent.rewards_history) == n_eps, (
        f"{test_label}: rewards_history ({len(agent.rewards_history)}) "
        f"difere de coverage_history ({n_eps})"
    )
    assert len(agent.steps_history) == n_eps, (
        f"{test_label}: steps_history ({len(agent.steps_history)}) "
        f"difere de coverage_history ({n_eps})"
    )
    # Cobertura deve ser não-decrescente (acumulativa)
    for i in range(1, len(agent.coverage_history)):
        assert agent.coverage_history[i] >= agent.coverage_history[i - 1], (
            f"{test_label}: Cobertura diminuiu de {agent.coverage_history[i-1]:.1f}% "
            f"para {agent.coverage_history[i]:.1f}% no episódio {i+1}"
        )


def validate_test_suite(runner, agent, test_label, coverage_type="states"):
    """Valida que a suíte de testes gerada é consistente."""
    suite = runner.test_suite
    assert len(suite) > 0, f"{test_label}: Suíte de testes vazia"

    for i, seq in enumerate(suite):
        assert len(seq) > 0, f"{test_label}: Sequência {i} na suíte está vazia"
        for step in seq:
            assert len(step) == 3, (
                f"{test_label}: Step na sequência {i} tem {len(step)} elementos, esperado 3"
            )


# ══════════════════════════════════════════════════════════════════
# Testes — Cobertura de Estados (All States)
# ══════════════════════════════════════════════════════════════════

def test_state_coverage_fsm03():
    """FSM 03: MotionLightSwitch — 43 estados, cobertura de estados."""
    cfg = FSM_CONFIGS["03"]
    label = f"FSM03-AllStates ({cfg['label']})"
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  {cfg['expected_states']} estados, {cfg['expected_transitions']} transições")
    print(f"{'=' * 65}")

    env = load_fsm_from_json(cfg["path"], max_steps=cfg["max_steps"])
    validate_environment(env, cfg, label)
    print(f"  ✔ Parsing: {env.n_states} estados, {len(env.transitions)} transições")

    runner = StateCoverageRunner(
        env=env, n_episodes=cfg["n_episodes"], **COMMON_HYPERPARAMS
    )

    t0 = time.time()
    runner.run(verbose=False, seed=cfg["seed"])
    dt = time.time() - t0

    agent = runner.agent
    validate_history(agent, label)
    validate_test_suite(runner, agent, label, "states")

    print(f"  ✔ Execução: {len(agent.coverage_history)} episódios em {dt:.2f}s")
    print(f"  ✔ Cobertura: {agent.coverage_percentage:.1f}% "
          f"({len(agent.covered_states)}/{agent.total_states})")
    print(f"  ✔ Suíte: {len(runner.test_suite)} caso(s) de teste")

    assert agent.coverage_percentage == 100.0, (
        f"{label}: Cobertura deveria ser 100%, foi {agent.coverage_percentage:.1f}%"
    )

    if agent.full_coverage_episode:
        print(f"  ✔ 100% no episódio {agent.full_coverage_episode}")

    print(f"  ✔ PASSOU")
    return True


def test_state_coverage_fsm04():
    """FSM 04: LightAndMotionSensing — 76 estados, cobertura de estados."""
    cfg = FSM_CONFIGS["04"]
    label = f"FSM04-AllStates ({cfg['label']})"
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  {cfg['expected_states']} estados, {cfg['expected_transitions']} transições")
    print(f"{'=' * 65}")

    env = load_fsm_from_json(cfg["path"], max_steps=cfg["max_steps"])
    validate_environment(env, cfg, label)
    print(f"  ✔ Parsing: {env.n_states} estados, {len(env.transitions)} transições")

    runner = StateCoverageRunner(
        env=env, n_episodes=cfg["n_episodes"], **COMMON_HYPERPARAMS
    )

    t0 = time.time()
    runner.run(verbose=False, seed=cfg["seed"])
    dt = time.time() - t0

    agent = runner.agent
    validate_history(agent, label)
    validate_test_suite(runner, agent, label, "states")

    print(f"  ✔ Execução: {len(agent.coverage_history)} episódios em {dt:.2f}s")
    print(f"  ✔ Cobertura: {agent.coverage_percentage:.1f}% "
          f"({len(agent.covered_states)}/{agent.total_states})")
    print(f"  ✔ Suíte: {len(runner.test_suite)} caso(s) de teste")

    assert agent.coverage_percentage == 100.0, (
        f"{label}: Cobertura deveria ser 100%, foi {agent.coverage_percentage:.1f}%"
    )
    print(f"  ✔ PASSOU")
    return True


def test_state_coverage_fsm05():
    """FSM 05: PresenceSimulation — 473 estados, cobertura de estados."""
    cfg = FSM_CONFIGS["05"]
    label = f"FSM05-AllStates ({cfg['label']})"
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  {cfg['expected_states']} estados, {cfg['expected_transitions']} transições")
    print(f"{'=' * 65}")

    env = load_fsm_from_json(cfg["path"], max_steps=cfg["max_steps"])
    validate_environment(env, cfg, label)
    print(f"  ✔ Parsing: {env.n_states} estados, {len(env.transitions)} transições")

    runner = StateCoverageRunner(
        env=env, n_episodes=cfg["n_episodes"], **COMMON_HYPERPARAMS
    )

    t0 = time.time()
    runner.run(verbose=False, seed=cfg["seed"])
    dt = time.time() - t0

    agent = runner.agent
    validate_history(agent, label)
    validate_test_suite(runner, agent, label, "states")

    print(f"  ✔ Execução: {len(agent.coverage_history)} episódios em {dt:.2f}s")
    print(f"  ✔ Cobertura: {agent.coverage_percentage:.1f}% "
          f"({len(agent.covered_states)}/{agent.total_states})")
    print(f"  ✔ Suíte: {len(runner.test_suite)} caso(s) de teste")

    assert agent.coverage_percentage == 100.0, (
        f"{label}: Cobertura deveria ser 100%, foi {agent.coverage_percentage:.1f}%"
    )
    print(f"  ✔ PASSOU")
    return True


# ══════════════════════════════════════════════════════════════════
# Testes — Cobertura de Transições (All Transitions)
# ══════════════════════════════════════════════════════════════════

def test_transition_coverage_fsm03():
    """FSM 03: MotionLightSwitch — 107 transições, cobertura de transições."""
    cfg = FSM_CONFIGS["03"]
    label = f"FSM03-AllTrans ({cfg['label']})"
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  {cfg['expected_states']} estados, {cfg['expected_transitions']} transições")
    print(f"{'=' * 65}")

    env = load_fsm_from_json(cfg["path"], max_steps=cfg["max_steps"])
    validate_environment(env, cfg, label)

    runner = CoverageRunner(
        env=env, n_episodes=cfg["n_episodes"], **COMMON_HYPERPARAMS
    )

    t0 = time.time()
    runner.run(verbose=False, seed=cfg["seed"])
    dt = time.time() - t0

    agent = runner.agent
    validate_history(agent, label)
    validate_test_suite(runner, agent, label, "transitions")

    print(f"  ✔ Execução: {len(agent.coverage_history)} episódios em {dt:.2f}s")
    print(f"  ✔ Cobertura: {agent.coverage_percentage:.1f}% "
          f"({len(agent.covered_transitions)}/{agent.total_transitions})")
    print(f"  ✔ Suíte: {len(runner.test_suite)} caso(s) de teste")

    assert agent.coverage_percentage == 100.0, (
        f"{label}: Cobertura deveria ser 100%, foi {agent.coverage_percentage:.1f}%"
    )
    print(f"  ✔ PASSOU")
    return True


def test_transition_coverage_fsm04():
    """FSM 04: LightAndMotionSensing — 250 transições, cobertura de transições."""
    cfg = FSM_CONFIGS["04"]
    label = f"FSM04-AllTrans ({cfg['label']})"
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  {cfg['expected_states']} estados, {cfg['expected_transitions']} transições")
    print(f"{'=' * 65}")

    env = load_fsm_from_json(cfg["path"], max_steps=cfg["max_steps"])
    validate_environment(env, cfg, label)

    runner = CoverageRunner(
        env=env, n_episodes=cfg["n_episodes"], **COMMON_HYPERPARAMS
    )

    t0 = time.time()
    runner.run(verbose=False, seed=cfg["seed"])
    dt = time.time() - t0

    agent = runner.agent
    validate_history(agent, label)
    validate_test_suite(runner, agent, label, "transitions")

    print(f"  ✔ Execução: {len(agent.coverage_history)} episódios em {dt:.2f}s")
    print(f"  ✔ Cobertura: {agent.coverage_percentage:.1f}% "
          f"({len(agent.covered_transitions)}/{agent.total_transitions})")
    print(f"  ✔ Suíte: {len(runner.test_suite)} caso(s) de teste")

    assert agent.coverage_percentage == 100.0, (
        f"{label}: Cobertura deveria ser 100%, foi {agent.coverage_percentage:.1f}%"
    )
    print(f"  ✔ PASSOU")
    return True


def test_transition_coverage_fsm05():
    """FSM 05: PresenceSimulation — 2134 transições, cobertura de transições."""
    cfg = FSM_CONFIGS["05"]
    label = f"FSM05-AllTrans ({cfg['label']})"
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  {cfg['expected_states']} estados, {cfg['expected_transitions']} transições")
    print(f"{'=' * 65}")

    env = load_fsm_from_json(cfg["path"], max_steps=cfg["max_steps"])
    validate_environment(env, cfg, label)

    runner = CoverageRunner(
        env=env, n_episodes=cfg["n_episodes"], **COMMON_HYPERPARAMS
    )

    t0 = time.time()
    runner.run(verbose=False, seed=cfg["seed"])
    dt = time.time() - t0

    agent = runner.agent
    validate_history(agent, label)
    validate_test_suite(runner, agent, label, "transitions")

    print(f"  ✔ Execução: {len(agent.coverage_history)} episódios em {dt:.2f}s")
    print(f"  ✔ Cobertura: {agent.coverage_percentage:.1f}% "
          f"({len(agent.covered_transitions)}/{agent.total_transitions})")
    print(f"  ✔ Suíte: {len(runner.test_suite)} caso(s) de teste")

    assert agent.coverage_percentage == 100.0, (
        f"{label}: Cobertura deveria ser 100%, foi {agent.coverage_percentage:.1f}%"
    )
    print(f"  ✔ PASSOU")
    return True


# ══════════════════════════════════════════════════════════════════
# Teste de Reprodutibilidade com FSM maior
# ══════════════════════════════════════════════════════════════════

def test_reproducibility_fsm03():
    """Verifica determinismo: mesma seed → mesmos resultados para FSM 03."""
    cfg = FSM_CONFIGS["03"]
    label = "Reprodutibilidade FSM03"
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")

    results = []
    for run_id in range(2):
        env = load_fsm_from_json(cfg["path"], max_steps=cfg["max_steps"])
        runner = StateCoverageRunner(
            env=env, n_episodes=cfg["n_episodes"], **COMMON_HYPERPARAMS
        )
        runner.run(verbose=False, seed=cfg["seed"])
        results.append(runner.agent.coverage_history)

    assert results[0] == results[1], (
        f"{label}: coverage_history difere entre execuções com mesma seed"
    )
    print(f"  ✔ Duas execuções com seed={cfg['seed']} produziram resultados idênticos")
    print(f"  ✔ PASSOU")
    return True


# ══════════════════════════════════════════════════════════════════
# Runner principal
# ══════════════════════════════════════════════════════════════════

def main():
    """Executa todos os testes para FSMs maiores."""
    print("\n" + "▓" * 65)
    print("  TESTES DE ROBUSTEZ — FSMs MAIORES (03, 04, 05)")
    print("▓" * 65)

    results = []
    tests = [
        # All States
        test_state_coverage_fsm03,
        test_state_coverage_fsm04,
        test_state_coverage_fsm05,
        # All Transitions
        test_transition_coverage_fsm03,
        test_transition_coverage_fsm04,
        test_transition_coverage_fsm05,
        # Reprodutibilidade
        test_reproducibility_fsm03,
    ]

    total_t0 = time.time()

    for test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_fn.__name__, passed))
        except Exception as e:
            print(f"  ✘ ERRO: {e}")
            results.append((test_fn.__name__, False))

    total_dt = time.time() - total_t0

    # Resumo
    print("\n" + "▓" * 65)
    print("  RESUMO DOS TESTES")
    print("▓" * 65)

    total = len(results)
    passed = sum(1 for _, p in results if p)

    for name, p in results:
        status = "✔ PASSOU" if p else "✘ FALHOU"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} testes passaram")
    print(f"  Tempo total: {total_dt:.1f}s")

    if passed == total:
        print("  ✔ TODOS OS TESTES PASSARAM!")
    else:
        print("  ✘ ALGUNS TESTES FALHARAM!")

    print("▓" * 65)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
