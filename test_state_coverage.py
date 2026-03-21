"""
Testes para Q-Learning com Cobertura de Estados

Verifica que:
1. A cobertura de estados funciona corretamente
2. Os modos existentes (target state, transition coverage) continuam funcionando
"""

import sys
import os

# Configurar encoding UTF-8 para o console do Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from fsm_environment import load_fsm_from_json
from state_coverage_agent import StateCoverageQLearningAgent
from state_coverage_runner import StateCoverageRunner
from coverage_agent import CoverageQLearningAgent
from q_learning_agent import QLearningAgent


def test_state_coverage_simple():
    """
    Teste 1: FSM simples (LightSwitch - 2 estados).
    Deve atingir 100% de cobertura de estados rapidamente.
    """
    print("\n" + "=" * 65)
    print("  TESTE 1: Cobertura de Estados — LightSwitch (2 estados)")
    print("=" * 65)

    env = load_fsm_from_json("fsm/_01_LightSwitch_flattened.json", max_steps=50)

    agent = StateCoverageQLearningAgent(
        env=env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
    )

    agent.train(n_episodes=100, verbose=False)

    print(f"  Estados alcançáveis: {agent.all_reachable_states}")
    print(f"  Estados cobertos:    {agent.covered_states}")
    print(f"  Cobertura:           {agent.coverage_percentage:.1f}%")

    assert agent.coverage_percentage == 100.0, (
        f"FALHOU: Cobertura deveria ser 100%, mas foi {agent.coverage_percentage:.1f}%"
    )
    assert agent.covered_states == agent.all_reachable_states, (
        "FALHOU: Nem todos os estados alcançáveis foram cobertos"
    )
    print("  ✔ PASSOU: 100% de cobertura de estados atingida!")
    return True


def test_state_coverage_medium():
    """
    Teste 2: FSM média (DimmableLightSwitch - 21 estados).
    Deve cobrir todos os estados alcançáveis.
    """
    print("\n" + "=" * 65)
    print("  TESTE 2: Cobertura de Estados — DimmableLightSwitch (21 estados)")
    print("=" * 65)

    env = load_fsm_from_json(
        "fsm/_02_DimmableLightSwitch_flattened.json",
        max_steps=100,
    )

    runner = StateCoverageRunner(
        env=env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        n_episodes=500,
    )

    runner.run(verbose=False)

    agent = runner.agent
    print(f"  Total de estados alcançáveis: {agent.total_states}")
    print(f"  Estados cobertos:             {len(agent.covered_states)}")
    print(f"  Cobertura:                    {agent.coverage_percentage:.1f}%")

    if agent.full_coverage_episode:
        print(f"  100% atingido no episódio:     {agent.full_coverage_episode}")

    assert agent.coverage_percentage == 100.0, (
        f"FALHOU: Cobertura deveria ser 100%, mas foi {agent.coverage_percentage:.1f}%"
    )

    # Verificar que a suíte de testes foi gerada
    test_suite = runner.test_suite
    assert len(test_suite) > 0, "FALHOU: Suíte de testes vazia"

    # Verificar que a suíte realmente cobre todos os estados
    states_in_suite = set()
    for seq in test_suite:
        for (s, a, ns) in seq:
            states_in_suite.add(s)
            states_in_suite.add(ns)
    # Adicionar o estado inicial (sempre presente)
    states_in_suite.add(env.initial_state)

    uncovered = agent.all_reachable_states - states_in_suite
    assert len(uncovered) == 0, (
        f"FALHOU: A suíte de testes não cobre os estados: {uncovered}"
    )

    print(f"  Casos de teste na suíte:       {len(test_suite)}")
    print("  ✔ PASSOU: 100% de cobertura + suíte de testes válida!")
    return True


def test_reachable_states_computation():
    """
    Teste 3: Verifica que o cálculo de estados alcançáveis é correto.
    """
    print("\n" + "=" * 65)
    print("  TESTE 3: Cálculo de Estados Alcançáveis")
    print("=" * 65)

    env = load_fsm_from_json("fsm/_01_LightSwitch_flattened.json", max_steps=50)
    agent = StateCoverageQLearningAgent(env=env)

    # LightSwitch tem 2 estados: main_Off__ e main_On__, ambos alcançáveis
    expected_states = {"main_Off__", "main_On__"}
    assert agent.all_reachable_states == expected_states, (
        f"FALHOU: Estados alcançáveis deveria ser {expected_states}, "
        f"mas foi {agent.all_reachable_states}"
    )
    print(f"  Estados alcançáveis calculados: {agent.all_reachable_states}")
    print("  ✔ PASSOU: Cálculo de BFS correto!")
    return True


def test_transition_coverage_no_regression():
    """
    Teste 4: Verifica que o modo Transition Coverage continua funcionando.
    """
    print("\n" + "=" * 65)
    print("  TESTE 4: Regressão — Cobertura de Transições")
    print("=" * 65)

    env = load_fsm_from_json("fsm/_01_LightSwitch_flattened.json", max_steps=50)

    agent = CoverageQLearningAgent(
        env=env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
    )

    agent.train(n_episodes=100, verbose=False)

    print(f"  Transições totais: {agent.total_transitions}")
    print(f"  Transições cobertas: {len(agent.covered_transitions)}")
    print(f"  Cobertura: {agent.coverage_percentage:.1f}%")

    assert agent.coverage_percentage == 100.0, (
        f"REGRESSÃO: Cobertura de transições deveria ser 100%, "
        f"mas foi {agent.coverage_percentage:.1f}%"
    )
    print("  ✔ PASSOU: Cobertura de transições continua funcionando!")
    return True


def test_target_state_no_regression():
    """
    Teste 5: Verifica que o modo Target State continua funcionando.
    """
    print("\n" + "=" * 65)
    print("  TESTE 5: Regressão — Q-Learning com Estado Alvo")
    print("=" * 65)

    env = load_fsm_from_json(
        "fsm/_01_LightSwitch_flattened.json",
        goal_states={"main_On__"},
        max_steps=50,
    )

    agent = QLearningAgent(
        env=env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    agent.train(n_episodes=200, verbose=False)

    # Verificar que aprendeu o caminho
    best_path = agent.get_best_path()
    print(f"  Melhor caminho: {best_path}")

    assert len(best_path) > 0, "REGRESSÃO: Nenhum caminho aprendido"

    last_state = best_path[-1][2]
    assert last_state in env.goal_states, (
        f"REGRESSÃO: Último estado ({last_state}) não é estado objetivo"
    )
    print("  ✔ PASSOU: Q-Learning com estado alvo continua funcionando!")
    return True


def main():
    """Executa todos os testes."""
    print("\n" + "▓" * 65)
    print("  TESTES DE COBERTURA DE ESTADOS — Q-LEARNING FSM")
    print("▓" * 65)

    results = []
    tests = [
        test_state_coverage_simple,
        test_state_coverage_medium,
        test_reachable_states_computation,
        test_transition_coverage_no_regression,
        test_target_state_no_regression,
    ]

    for test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_fn.__name__, passed))
        except Exception as e:
            print(f"  ✘ ERRO: {e}")
            results.append((test_fn.__name__, False))

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

    if passed == total:
        print("  ✔ TODOS OS TESTES PASSARAM!")
    else:
        print("  ✘ ALGUNS TESTES FALHARAM!")

    print("▓" * 65)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
