"""
Módulo de Visualização

Funções para visualizar os resultados do treinamento Q-Learning,
incluindo gráficos de convergência, Q-Table e diagrama do autômato.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional, Dict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .fsm_environment import FSMEnvironment
from .q_learning_agent import QLearningAgent


def plot_training_results(
    agent: QLearningAgent,
    window_size: int = 50,
    save_path: Optional[str] = None,
):
    """
    Plota os resultados do treinamento em subplots.

    Exibe 3 gráficos:
    1. Recompensa total por episódio (com média móvel)
    2. Passos por episódio (com média móvel)
    3. Evolução do epsilon (exploração)

    Args:
        agent: Agente treinado com histórico
        window_size: Janela para a média móvel
        save_path: Caminho para salvar a imagem (se None, exibe na tela)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Resultados do Treinamento Q-Learning", fontsize=16, fontweight="bold")

    episodes = range(1, len(agent.rewards_history) + 1)

    # --- Gráfico 1: Recompensas ---
    ax1 = axes[0]
    ax1.plot(episodes, agent.rewards_history, alpha=0.3, color="#4A90D9", linewidth=0.8, label="Por episódio")
    if len(agent.rewards_history) >= window_size:
        moving_avg = _moving_average(agent.rewards_history, window_size)
        ax1.plot(
            range(window_size, len(agent.rewards_history) + 1),
            moving_avg,
            color="#E74C3C",
            linewidth=2,
            label=f"Média móvel ({window_size} ep.)",
        )
    ax1.set_ylabel("Recompensa Total")
    ax1.set_title("Recompensa por Episódio")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Gráfico 2: Passos ---
    ax2 = axes[1]
    ax2.plot(episodes, agent.steps_history, alpha=0.3, color="#27AE60", linewidth=0.8, label="Por episódio")
    if len(agent.steps_history) >= window_size:
        moving_avg_steps = _moving_average(agent.steps_history, window_size)
        ax2.plot(
            range(window_size, len(agent.steps_history) + 1),
            moving_avg_steps,
            color="#E67E22",
            linewidth=2,
            label=f"Média móvel ({window_size} ep.)",
        )
    ax2.set_ylabel("Número de Passos")
    ax2.set_title("Passos por Episódio")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # --- Gráfico 3: Epsilon ---
    ax3 = axes[2]
    ax3.plot(episodes, agent.epsilon_history, color="#8E44AD", linewidth=2)
    ax3.set_xlabel("Episódio")
    ax3.set_ylabel("Epsilon (ε)")
    ax3.set_title("Decaimento da Taxa de Exploração")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  [+] Grafico salvo em: {save_path}")
    else:
        plt.show()


def plot_fsm_diagram(
    env: FSMEnvironment,
    best_path: Optional[List[Tuple[str, str, str]]] = None,
    save_path: Optional[str] = None,
):
    """
    Visualiza o autômato como um grafo dirigido, destacando o caminho ótimo.

    Args:
        env: Ambiente FSM
        best_path: Lista de tuplas (estado, ação, próximo_estado) do caminho ótimo
        save_path: Caminho para salvar a imagem (se None, exibe na tela)
    """
    if not HAS_NETWORKX:
        print("\n  [!] networkx nao instalado. Pulando diagrama do automato.")
        print("     Instale com: pip install networkx")
        return

    G = nx.MultiDiGraph()

    # Adicionar nós
    for state in env.states:
        G.add_node(state)

    # Adicionar arestas com labels
    edge_labels = {}
    for (state, action), next_state in env.transitions.items():
        reward = env.rewards.get((state, action), 0)
        G.add_edge(state, next_state, action=action, reward=reward)
        key = (state, next_state)
        label = f"{action} (r={reward:.0f})"
        if key in edge_labels:
            edge_labels[key] += f"\n{label}"
        else:
            edge_labels[key] = label

    # Determinar arestas do caminho ótimo
    optimal_edges = set()
    optimal_nodes = set()
    if best_path:
        for state, action, next_state in best_path:
            optimal_edges.add((state, next_state))
            optimal_nodes.add(state)
            optimal_nodes.add(next_state)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_title(
        "Diagrama do Autômato (FSM) com Caminho Ótimo",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Cores dos nós
    node_colors = []
    for state in G.nodes():
        if state in env.goal_states:
            node_colors.append("#2ECC71")       # Verde para objetivo
        elif state == env.initial_state:
            node_colors.append("#3498DB")       # Azul para início
        elif state in optimal_nodes:
            node_colors.append("#F39C12")       # Laranja para caminho ótimo
        else:
            node_colors.append("#BDC3C7")       # Cinza para outros

    # Desenhar nós
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2000,
        edgecolors="#2C3E50",
        linewidths=2,
    )

    # Desenhar labels dos nós
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=14,
        font_weight="bold",
        font_color="#2C3E50",
    )

    # Desenhar arestas
    all_edges = list(G.edges())
    edge_colors_list = []
    edge_widths = []
    edge_styles = []

    for edge in all_edges:
        if (edge[0], edge[1]) in optimal_edges:
            edge_colors_list.append("#E74C3C")
            edge_widths.append(3.0)
            edge_styles.append("solid")
        else:
            edge_colors_list.append("#95A5A6")
            edge_widths.append(1.5)
            edge_styles.append("dashed")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=all_edges,
        edge_color=edge_colors_list,
        width=edge_widths,
        style=edge_styles,
        arrows=True,
        arrowsize=25,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.15",
        min_source_margin=25,
        min_target_margin=25,
    )

    # Desenhar labels das arestas
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=9,
        font_color="#2C3E50",
        label_pos=0.3,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    )

    # Legenda
    legend_elements = [
        mpatches.Patch(color="#3498DB", label="Estado Inicial"),
        mpatches.Patch(color="#2ECC71", label="Estado Objetivo"),
        mpatches.Patch(color="#F39C12", label="Caminho Ótimo"),
        mpatches.Patch(color="#BDC3C7", label="Outros Estados"),
        plt.Line2D([0], [0], color="#E74C3C", linewidth=3, label="Transição Ótima"),
        plt.Line2D([0], [0], color="#95A5A6", linewidth=1.5, linestyle="--", label="Outra Transição"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=10,
        framealpha=0.9,
    )

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [+] Diagrama salvo em: {save_path}")
    else:
        plt.show()


def print_q_table(agent: QLearningAgent):
    """
    Imprime a Q-Table formatada no terminal.

    Args:
        agent: Agente treinado com Q-Table preenchida.
    """
    print("\n" + "=" * 60)
    print("  Q-TABLE FINAL")
    print("=" * 60)
    print(agent.get_q_table_formatted())
    print("=" * 60)


def print_best_path(path: List[Tuple[str, str, str]], env: FSMEnvironment):
    """
    Imprime o melhor caminho aprendido de forma visual.

    Args:
        path: Lista de tuplas (estado, ação, próximo_estado)
        env: Ambiente FSM para informações adicionais
    """
    print("\n" + "=" * 60)
    print("  MELHOR CAMINHO APRENDIDO")
    print("=" * 60)

    if not path:
        print("  [!] Nenhum caminho encontrado!")
        return

    total_reward = 0.0
    for i, (state, action, next_state) in enumerate(path):
        reward = env.rewards.get((state, action), 0)
        total_reward += reward
        reached_goal = "  >>> OBJETIVO!" if next_state in env.goal_states else ""
        print(f"  Passo {i + 1}: {state} --[{action}]--> {next_state}  (r={reward:+.1f}){reached_goal}")

    print(f"\n  [+] Recompensa total do caminho: {total_reward:.1f}")
    print(f"  [+] Numero de passos: {len(path)}")

    if path and path[-1][2] in env.goal_states:
        print("  [OK] O agente encontrou o objetivo!")
    else:
        print("  [X] O agente NAO alcancou o objetivo.")

    print("=" * 60)


def plot_q_table_heatmap(
    agent: QLearningAgent,
    save_path: Optional[str] = None,
):
    """
    Plota a Q-Table como um heatmap.

    Args:
        agent: Agente treinado
        save_path: Caminho para salvar (se None, exibe)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(agent.q_table, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(agent.env.n_actions))
    ax.set_xticklabels([f"Ação '{a}'" for a in agent.env.actions], fontsize=12)
    ax.set_yticks(range(agent.env.n_states))
    ax.set_yticklabels(agent.env.states, fontsize=12)

    # Adicionar valores nas células
    for i in range(agent.env.n_states):
        for j in range(agent.env.n_actions):
            value = agent.q_table[i, j]
            color = "white" if abs(value) > np.max(np.abs(agent.q_table)) * 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11, color=color, fontweight="bold")

    ax.set_title("Q-Table (Heatmap)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Ações", fontsize=13)
    ax.set_ylabel("Estados", fontsize=13)

    plt.colorbar(im, ax=ax, label="Valor Q")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [+] Heatmap salvo em: {save_path}")
    else:
        plt.show()


def _moving_average(data: List[float], window: int) -> List[float]:
    """Calcula a média móvel de uma lista de valores."""
    cumsum = np.cumsum(data)
    cumsum = np.insert(cumsum, 0, 0)
    return list((cumsum[window:] - cumsum[:-window]) / window)


# =====================================================================
#  FUNÇÕES DE VISUALIZAÇÃO PARA COBERTURA DE TRANSIÇÕES
# =====================================================================

def plot_coverage_progress(
    agent,
    save_path: Optional[str] = None,
):
    """
    Plota a evolução da cobertura de transições ao longo dos episódios.

    Exibe 2 gráficos:
    1. Porcentagem de cobertura acumulada por episódio
    2. Novas transições descobertas por episódio

    Args:
        agent: CoverageQLearningAgent treinado
        save_path: Caminho para salvar a imagem
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        "Progresso da Cobertura de Transições",
        fontsize=16, fontweight="bold"
    )

    episodes = range(1, len(agent.coverage_history) + 1)

    # --- Gráfico 1: Cobertura acumulada ---
    ax1 = axes[0]
    ax1.plot(
        episodes, agent.coverage_history,
        color="#2ECC71", linewidth=2.5, label="Cobertura (%)"
    )
    ax1.axhline(y=100, color="#E74C3C", linestyle="--", alpha=0.7, label="100% alvo")

    if agent.full_coverage_episode:
        ax1.axvline(
            x=agent.full_coverage_episode,
            color="#F39C12", linestyle=":", linewidth=2,
            label=f"100% no ep. {agent.full_coverage_episode}"
        )

    ax1.set_ylabel("Cobertura (%)")
    ax1.set_title("Cobertura de Transições Acumulada")
    ax1.set_ylim(-5, 110)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Gráfico 2: Novas transições por episódio ---
    ax2 = axes[1]
    ax2.bar(
        episodes, agent.new_transitions_per_episode,
        color="#3498DB", alpha=0.7, width=1.0
    )
    ax2.set_xlabel("Episódio")
    ax2.set_ylabel("Novas Transições")
    ax2.set_title("Transições Novas Descobertas por Episódio")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [+] Gráfico de cobertura salvo em: {save_path}")
    else:
        plt.show()


def plot_fsm_coverage_diagram(
    env: FSMEnvironment,
    covered_transitions,
    test_suite=None,
    save_path: Optional[str] = None,
):
    """
    Visualiza o autômato com transições cobertas vs não cobertas.

    Args:
        env: Ambiente FSM
        covered_transitions: Set de tuplas (estado, ação, próx_estado) cobertas
        test_suite: Lista de sequências de teste (opcional, para legenda)
        save_path: Caminho para salvar a imagem
    """
    if not HAS_NETWORKX:
        print("\n  [!] networkx não instalado. Pulando diagrama.")
        return

    G = nx.MultiDiGraph()

    for state in env.states:
        G.add_node(state)

    # Construir set de arestas cobertas para lookup
    covered_edges = set()
    for (s, a, ns) in covered_transitions:
        covered_edges.add((s, a, ns))

    all_transitions = set()
    for (state, action), next_state in env.transitions.items():
        all_transitions.add((state, action, next_state))

    edge_labels = {}
    for (state, action), next_state in env.transitions.items():
        is_covered = (state, action, next_state) in covered_edges
        G.add_edge(state, next_state, action=action, covered=is_covered)
        key = (state, next_state)
        status = "✔" if is_covered else "✘"
        label = f"{action} [{status}]"
        if key in edge_labels:
            edge_labels[key] += f"\n{label}"
        else:
            edge_labels[key] = label

    # Nós visitados
    visited_states = set()
    for (s, a, ns) in covered_transitions:
        visited_states.add(s)
        visited_states.add(ns)

    pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    total = len(all_transitions)
    covered = len(covered_edges & all_transitions)
    pct = (covered / total * 100) if total > 0 else 0

    ax.set_title(
        f"Cobertura de Transições: {covered}/{total} ({pct:.0f}%)",
        fontsize=16, fontweight="bold", pad=20,
    )

    # Cores dos nós
    node_colors = []
    for state in G.nodes():
        if state in env.goal_states:
            node_colors.append("#2ECC71")
        elif state == env.initial_state:
            node_colors.append("#3498DB")
        elif state in visited_states:
            node_colors.append("#F39C12")
        else:
            node_colors.append("#E74C3C")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2000,
        edgecolors="#2C3E50",
        linewidths=2,
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=14,
        font_weight="bold",
        font_color="#2C3E50",
    )

    # Separar arestas cobertas e não cobertas
    all_graph_edges = list(G.edges(data=True))
    covered_edgelist = []
    uncovered_edgelist = []

    for u, v, data in all_graph_edges:
        if data.get("covered", False):
            covered_edgelist.append((u, v))
        else:
            uncovered_edgelist.append((u, v))

    # Desenhar arestas cobertas (verde, sólido)
    if covered_edgelist:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=covered_edgelist,
            edge_color="#27AE60",
            width=3.0,
            style="solid",
            arrows=True,
            arrowsize=25,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.15",
            min_source_margin=25,
            min_target_margin=25,
        )

    # Desenhar arestas não cobertas (vermelho, tracejado)
    if uncovered_edgelist:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=uncovered_edgelist,
            edge_color="#E74C3C",
            width=2.0,
            style="dashed",
            arrows=True,
            arrowsize=25,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.15",
            min_source_margin=25,
            min_target_margin=25,
        )

    # Labels das arestas
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=9,
        font_color="#2C3E50",
        label_pos=0.3,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.8
        ),
    )

    # Legenda
    legend_elements = [
        mpatches.Patch(color="#3498DB", label="Estado Inicial"),
        mpatches.Patch(color="#2ECC71", label="Estado Objetivo"),
        mpatches.Patch(color="#F39C12", label="Estado Visitado"),
        mpatches.Patch(color="#E74C3C", label="Estado Não Visitado"),
        plt.Line2D([0], [0], color="#27AE60", linewidth=3,
                   label="Transição Coberta ✔"),
        plt.Line2D([0], [0], color="#E74C3C", linewidth=2, linestyle="--",
                   label="Transição Não Coberta ✘"),
    ]

    if test_suite:
        legend_elements.append(
            mpatches.Patch(
                color="none",
                label=f"Casos de teste: {len(test_suite)}"
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=10,
        framealpha=0.9,
    )

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [+] Diagrama de cobertura salvo em: {save_path}")
    else:
        plt.show()


# =====================================================================
#  FUNÇÕES DE VISUALIZAÇÃO PARA COBERTURA DE ESTADOS
# =====================================================================

def plot_state_coverage_progress(
    agent,
    save_path: Optional[str] = None,
):
    """
    Plota a evolução da cobertura de estados ao longo dos episódios.

    Exibe 2 gráficos:
    1. Porcentagem de cobertura de estados acumulada por episódio
    2. Novos estados descobertos por episódio

    Args:
        agent: StateCoverageQLearningAgent treinado
        save_path: Caminho para salvar a imagem
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        "Progresso da Cobertura de Estados",
        fontsize=16, fontweight="bold"
    )

    episodes = range(1, len(agent.coverage_history) + 1)

    # --- Gráfico 1: Cobertura acumulada ---
    ax1 = axes[0]
    ax1.plot(
        episodes, agent.coverage_history,
        color="#3498DB", linewidth=2.5, label="Cobertura (%)"
    )
    ax1.axhline(y=100, color="#E74C3C", linestyle="--", alpha=0.7, label="100% alvo")

    if agent.full_coverage_episode:
        ax1.axvline(
            x=agent.full_coverage_episode,
            color="#F39C12", linestyle=":", linewidth=2,
            label=f"100% no ep. {agent.full_coverage_episode}"
        )

    ax1.set_ylabel("Cobertura (%)")
    ax1.set_title("Cobertura de Estados Acumulada")
    ax1.set_ylim(-5, 110)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Gráfico 2: Novos estados por episódio ---
    ax2 = axes[1]
    ax2.bar(
        episodes, agent.new_states_per_episode,
        color="#8E44AD", alpha=0.7, width=1.0
    )
    ax2.set_xlabel("Episódio")
    ax2.set_ylabel("Novos Estados")
    ax2.set_title("Estados Novos Descobertos por Episódio")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [+] Gráfico de cobertura de estados salvo em: {save_path}")
    else:
        plt.show()


def plot_fsm_state_coverage_diagram(
    env: FSMEnvironment,
    covered_states,
    all_reachable_states,
    test_suite=None,
    save_path: Optional[str] = None,
):
    """
    Visualiza o autômato com estados cobertos vs não cobertos.

    Args:
        env: Ambiente FSM
        covered_states: Set de nomes dos estados cobertos
        all_reachable_states: Set de nomes dos estados alcançáveis
        test_suite: Lista de sequências de teste (opcional, para legenda)
        save_path: Caminho para salvar a imagem
    """
    if not HAS_NETWORKX:
        print("\n  [!] networkx não instalado. Pulando diagrama.")
        return

    G = nx.MultiDiGraph()

    for state in env.states:
        G.add_node(state)

    edge_labels = {}
    for (state, action), next_state in env.transitions.items():
        G.add_edge(state, next_state, action=action)
        key = (state, next_state)
        label = action
        if key in edge_labels:
            edge_labels[key] += f"\n{label}"
        else:
            edge_labels[key] = label

    pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    total = len(all_reachable_states)
    covered = len(covered_states & all_reachable_states)
    pct = (covered / total * 100) if total > 0 else 0

    ax.set_title(
        f"Cobertura de Estados: {covered}/{total} ({pct:.0f}%)",
        fontsize=16, fontweight="bold", pad=20,
    )

    # Cores dos nós baseadas em cobertura de estados
    node_colors = []
    for state in G.nodes():
        if state not in all_reachable_states:
            # Estado não alcançável
            node_colors.append("#BDC3C7")  # Cinza
        elif state == env.initial_state:
            node_colors.append("#3498DB")  # Azul para início
        elif state in covered_states:
            node_colors.append("#2ECC71")  # Verde para coberto
        else:
            node_colors.append("#E74C3C")  # Vermelho para não coberto

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2000,
        edgecolors="#2C3E50",
        linewidths=2,
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=14,
        font_weight="bold",
        font_color="#2C3E50",
    )

    # Desenhar todas as arestas
    all_edges = list(G.edges())
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=all_edges,
        edge_color="#95A5A6",
        width=1.5,
        style="solid",
        arrows=True,
        arrowsize=25,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.15",
        min_source_margin=25,
        min_target_margin=25,
    )

    # Labels das arestas
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=9,
        font_color="#2C3E50",
        label_pos=0.3,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.8
        ),
    )

    # Legenda
    legend_elements = [
        mpatches.Patch(color="#3498DB", label="Estado Inicial"),
        mpatches.Patch(color="#2ECC71", label="Estado Coberto ✔"),
        mpatches.Patch(color="#E74C3C", label="Estado Não Coberto ✘"),
        mpatches.Patch(color="#BDC3C7", label="Estado Não Alcançável"),
    ]

    if test_suite:
        legend_elements.append(
            mpatches.Patch(
                color="none",
                label=f"Casos de teste: {len(test_suite)}"
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=10,
        framealpha=0.9,
    )

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [+] Diagrama de cobertura de estados salvo em: {save_path}")
    else:
        plt.show()
