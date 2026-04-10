"""
Módulo de Análise Estatística dos Resultados Experimentais

Carrega os resultados estruturados (summary.csv) gerados por run_experiment.py
e produz análise estatística comparativa entre All States e All Transitions.

Cálculos realizados:
- Estatísticas descritivas: média, desvio padrão, mediana, mín, máx
- Intervalo de confiança de 95% (t de Student)
- Teste de significância: Mann-Whitney U (não paramétrico)

Justificativa do teste Mann-Whitney U:
- As distribuições de métricas de cobertura (episódios até 100%, passos totais,
  tamanho da suíte) tipicamente não seguem distribuição normal
- O teste é robusto para amostras pequenas (n ≥ 5)
- Não assume homogeneidade de variância
- É o teste recomendado para comparação de dois grupos independentes
  quando a normalidade não pode ser garantida

Uso:
    python -m src.analyze_results results/experiment_YYYYMMDD_HHMMSS/
    python -m src.analyze_results results/experiment_YYYYMMDD_HHMMSS/ --output report.txt
"""

import csv
import json
import math
import os
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Configurar encoding UTF-8 para o console do Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'


def _mean(values: List[float]) -> float:
    """Calcula a média aritmética."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    """Calcula o desvio padrão amostral."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _median(values: List[float]) -> float:
    """Calcula a mediana."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def _t_critical_95(df: int) -> float:
    """
    Retorna o valor crítico t para IC 95% (bicaudal) para graus de liberdade comuns.
    Tabela pré-computada para evitar dependência de scipy.
    """
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        35: 2.030, 40: 2.021, 45: 2.014, 50: 2.009, 60: 2.000,
        70: 1.994, 80: 1.990, 90: 1.987, 100: 1.984,
    }
    if df in t_table:
        return t_table[df]
    # Interpolar para valores não exatos
    keys = sorted(t_table.keys())
    if df < keys[0]:
        return t_table[keys[0]]
    if df > keys[-1]:
        return 1.960  # Aproximação z para df grande
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            # Interpolação linear
            t1, t2 = t_table[keys[i]], t_table[keys[i + 1]]
            frac = (df - keys[i]) / (keys[i + 1] - keys[i])
            return t1 + frac * (t2 - t1)
    return 1.960


def _confidence_interval_95(values: List[float]) -> Tuple[float, float]:
    """
    Calcula o intervalo de confiança de 95% usando distribuição t de Student.

    Returns:
        Tupla (limite_inferior, limite_superior) do IC 95%.
    """
    n = len(values)
    if n < 2:
        m = _mean(values)
        return (m, m)
    m = _mean(values)
    s = _std(values)
    df = n - 1
    t_crit = _t_critical_95(df)
    margin = t_crit * s / math.sqrt(n)
    return (m - margin, m + margin)


def _mann_whitney_u(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Teste de Mann-Whitney U (Wilcoxon rank-sum) para dois grupos independentes.

    Hipótese nula (H0): As duas populações têm a mesma distribuição.
    Hipótese alternativa (H1): As distribuições diferem.

    Implementação sem dependência de scipy.

    Args:
        x: Valores do grupo 1
        y: Valores do grupo 2

    Returns:
        Tupla (U_statistic, p_value_approx)
        O p-valor é calculado pela aproximação normal (válida para n >= 8).
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)

    # Combinar e rankear
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])

    # Atribuir ranks (com tratamento de empates)
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Soma dos ranks para grupo x (grupo 0)
    r1 = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == 0)

    # Estatística U
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)

    # Aproximação normal para p-valor (para n >= 8)
    mu = n1 * n2 / 2
    n = n1 + n2

    # Correção para empates
    tie_counts = defaultdict(int)
    for v, _ in combined:
        tie_counts[v] += 1
    tie_correction = sum(t ** 3 - t for t in tie_counts.values()) / (n * (n - 1))

    sigma_squared = n1 * n2 * ((n + 1) / 12.0 - tie_correction)
    if sigma_squared <= 0:
        # Todos os valores são idênticos — não há diferença
        return (u_stat, 1.0)
    sigma = math.sqrt(sigma_squared)

    if sigma == 0:
        return (u_stat, 1.0)

    z = (u_stat - mu) / sigma
    # Aproximação do p-valor bicaudal via função erro (sem scipy)
    p_value = 2 * _normal_cdf(-abs(z))

    return (u_stat, p_value)


def _normal_cdf(z: float) -> float:
    """Aproximação da CDF da distribuição normal padrão (Abramowitz & Stegun)."""
    if z < -8:
        return 0.0
    if z > 8:
        return 1.0
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if z >= 0 else -1
    z_abs = abs(z)
    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs / 2)
    return 0.5 * (1.0 + sign * y)


def load_summary_csv(csv_path: str) -> List[dict]:
    """Carrega o summary.csv como lista de dicionários."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("coverage_final") == "ERRO":
                continue
            # Converter tipos
            parsed = {
                "fsm": row["fsm"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "coverage_final": float(row["coverage_final"]),
                "episode_full_coverage": int(row["episode_full_coverage"]) if row["episode_full_coverage"] else None,
                "total_episodes": int(row["total_episodes"]),
                "total_steps": int(row["total_steps"]),
                "test_suite_size": int(row["test_suite_size"]),
                "execution_time_s": float(row["execution_time_s"]),
            }
            rows.append(parsed)
    return rows


def group_by(rows: List[dict], *keys) -> Dict[tuple, List[dict]]:
    """Agrupa linhas por uma ou mais chaves."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups[key] = groups.get(key, [])
        groups[key].append(row)
    return groups


def compute_descriptive_stats(values: List[float]) -> dict:
    """Calcula estatísticas descritivas para uma lista de valores."""
    ci_lo, ci_hi = _confidence_interval_95(values)
    return {
        "n": len(values),
        "mean": round(_mean(values), 4),
        "std": round(_std(values), 4),
        "median": round(_median(values), 4),
        "min": round(min(values), 4) if values else 0,
        "max": round(max(values), 4) if values else 0,
        "ci_95_lower": round(ci_lo, 4),
        "ci_95_upper": round(ci_hi, 4),
    }


def analyze_experiment(experiment_dir: str, output_path: Optional[str] = None):
    """
    Executa a análise estatística completa sobre um diretório de experimento.

    Args:
        experiment_dir: Caminho para o diretório do experimento
        output_path: Caminho para salvar o relatório (se None, imprime no stdout)
    """
    csv_path = os.path.join(experiment_dir, "summary.csv")
    if not os.path.exists(csv_path):
        print(f"ERRO: summary.csv não encontrado em {experiment_dir}")
        return

    rows = load_summary_csv(csv_path)
    if not rows:
        print("ERRO: Nenhum resultado válido encontrado no CSV.")
        return

    # Carregar config se disponível
    config_path = os.path.join(experiment_dir, "config.json")
    config_info = ""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config_info = (
            f"  α={config['alpha']}  γ={config['gamma']}  "
            f"ε={config['epsilon']}→{config['epsilon_min']}  "
            f"decay={config['epsilon_decay']}  "
            f"episódios={config['n_episodes']}  "
            f"max_steps={config['max_steps']}"
        )

    lines = []

    def w(text=""):
        lines.append(text)

    w("=" * 75)
    w("  RELATÓRIO DE ANÁLISE ESTATÍSTICA")
    w("  Comparação: All States vs All Transitions")
    w("=" * 75)
    if config_info:
        w(f"\n  Configuração experimental:")
        w(config_info)
    w(f"  Diretório: {experiment_dir}")
    w(f"  Total de execuções válidas: {len(rows)}")

    # Métricas para análise
    metrics = [
        ("episode_full_coverage", "Episódio de 100% cobertura"),
        ("total_steps", "Total de passos"),
        ("test_suite_size", "Tamanho da suíte de testes"),
        ("execution_time_s", "Tempo de execução (s)"),
        ("coverage_final", "Cobertura final (%)"),
    ]

    # Agrupar por FSM
    fsms = sorted(set(r["fsm"] for r in rows))
    methods = ["all_transitions", "all_states"]

    # ──────────────────────────────────
    # Análise POR FSM
    # ──────────────────────────────────
    all_stats = {}  # Para JSON export

    for fsm in fsms:
        w(f"\n{'─' * 75}")
        w(f"  FSM: {fsm}")
        w(f"{'─' * 75}")

        fsm_rows = [r for r in rows if r["fsm"] == fsm]
        fsm_stats = {}

        for metric_key, metric_label in metrics:
            w(f"\n  ▸ {metric_label}")

            method_values = {}
            for method in methods:
                values = [
                    r[metric_key] for r in fsm_rows
                    if r["method"] == method and r[metric_key] is not None
                ]
                if not values:
                    w(f"    {method:20s}: sem dados")
                    continue

                stats = compute_descriptive_stats(values)
                method_values[method] = values

                method_label = method.replace("_", " ").title()
                w(f"    {method_label:20s}: "
                  f"média={stats['mean']:.4f}  "
                  f"dp={stats['std']:.4f}  "
                  f"IC95%=[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]  "
                  f"med={stats['median']:.4f}  "
                  f"n={stats['n']}")

                fsm_stats.setdefault(metric_key, {})[method] = stats

            # Teste Mann-Whitney U se ambos os métodos têm dados
            if len(method_values) == 2:
                vals_a = method_values.get("all_transitions", [])
                vals_b = method_values.get("all_states", [])
                if vals_a and vals_b:
                    u_stat, p_value = _mann_whitney_u(vals_a, vals_b)
                    sig = "SIM" if p_value < 0.05 else "NÃO"
                    w(f"    Mann-Whitney U:    U={u_stat:.1f}  p={p_value:.6f}  "
                      f"significativo(α=0.05): {sig}")

                    fsm_stats.setdefault(metric_key, {})["mann_whitney"] = {
                        "U": round(u_stat, 4),
                        "p_value": round(p_value, 6),
                        "significant_005": p_value < 0.05,
                    }

        all_stats[fsm] = fsm_stats

    # ──────────────────────────────────
    # Análise AGREGADA (todas as FSMs)
    # ──────────────────────────────────
    if len(fsms) > 1:
        w(f"\n{'═' * 75}")
        w(f"  AGREGADO (todas as FSMs)")
        w(f"{'═' * 75}")

        agg_stats = {}

        for metric_key, metric_label in metrics:
            w(f"\n  ▸ {metric_label}")

            method_values = {}
            for method in methods:
                values = [
                    r[metric_key] for r in rows
                    if r["method"] == method and r[metric_key] is not None
                ]
                if not values:
                    continue

                stats = compute_descriptive_stats(values)
                method_values[method] = values

                method_label = method.replace("_", " ").title()
                w(f"    {method_label:20s}: "
                  f"média={stats['mean']:.4f}  "
                  f"dp={stats['std']:.4f}  "
                  f"IC95%=[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]  "
                  f"n={stats['n']}")

                agg_stats.setdefault(metric_key, {})[method] = stats

            if len(method_values) == 2:
                vals_a = method_values.get("all_transitions", [])
                vals_b = method_values.get("all_states", [])
                if vals_a and vals_b:
                    u_stat, p_value = _mann_whitney_u(vals_a, vals_b)
                    sig = "SIM" if p_value < 0.05 else "NÃO"
                    w(f"    Mann-Whitney U:    U={u_stat:.1f}  p={p_value:.6f}  "
                      f"significativo(α=0.05): {sig}")

                    agg_stats.setdefault(metric_key, {})["mann_whitney"] = {
                        "U": round(u_stat, 4),
                        "p_value": round(p_value, 6),
                        "significant_005": p_value < 0.05,
                    }

        all_stats["_aggregated"] = agg_stats

    # ──────────────────────────────────
    # Notas metodológicas
    # ──────────────────────────────────
    w(f"\n{'═' * 75}")
    w("  NOTAS METODOLÓGICAS")
    w(f"{'═' * 75}")
    w("  • Teste de significância: Mann-Whitney U (não paramétrico, bicaudal)")
    w("  • Justificativa: não assume normalidade; robusto para amostras pequenas")
    w("  • H0: as distribuições dos dois métodos são iguais")
    w("  • H1: as distribuições diferem")
    w("  • Nível de significância: α = 0.05")
    w("  • IC 95%: calculado via distribuição t de Student")
    w("  • Funções de recompensa: padronizadas (+50 novo, -1 repetido, -10 inválido)")
    w("=" * 75)

    report_text = "\n".join(lines)

    # Salvar relatório texto
    if output_path is None:
        output_path = os.path.join(experiment_dir, "statistical_report.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Salvar estatísticas como JSON
    stats_json_path = os.path.join(experiment_dir, "statistical_results.json")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(report_text)
    print(f"\n  [+] Relatório salvo em: {output_path}")
    print(f"  [+] Estatísticas JSON: {stats_json_path}")


def main():
    """Ponto de entrada para execução via linha de comando."""
    parser = argparse.ArgumentParser(
        description="Análise estatística dos resultados experimentais"
    )
    parser.add_argument(
        "experiment_dir",
        help="Caminho para o diretório do experimento (contendo summary.csv)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Caminho para salvar o relatório (padrão: dentro do experiment_dir)",
    )
    args = parser.parse_args()

    analyze_experiment(args.experiment_dir, args.output)


if __name__ == "__main__":
    main()
