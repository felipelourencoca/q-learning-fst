# Q-Learning para Máquina de Estados Finitos (FSM)

Implementação de um agente **Q-Learning** que aprende a navegar de forma ótima em uma **Máquina de Estados Finitos** (FST/FSM).

## 📖 Sobre

O **Q-Learning** é um algoritmo de **Aprendizado por Reforço** (*Reinforcement Learning*) que permite a um agente aprender, por tentativa e erro, qual ação tomar em cada estado para maximizar a recompensa acumulada.

Neste projeto, o ambiente é modelado como um **autômato finito** carregado a partir de um **arquivo JSON**, onde:

- **Estados** = Nós do autômato
- **Ações** = Inputs das transições (extraídos automaticamente do JSON)
- **Transições** = Mapeamento `(estado, ação) → próximo_estado`
- **Objetivo** = Alcançar um estado alvo no menor número de passos (modo Q-Learning) ou cobrir todas as transições (modo cobertura)

### Formato do arquivo JSON da FSM

```json
{
  "initial": "nome_estado_inicial",
  "states": [
    {
      "state": "nome_estado",
      "transitions": [
        {
          "input": "nome_acao",
          "output": [],
          "target": "nome_estado_destino"
        }
      ]
    }
  ]
}
```

A pasta `fsm/` contém exemplos de FSMs neste formato.

### Equação de Bellman (Q-Update)

```
Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') - Q(s, a)]
```

Onde:
- `α` = Taxa de aprendizado (*learning rate*)
- `γ` = Fator de desconto (*discount factor*)
- `r` = Recompensa imediata
- `s'` = Próximo estado
- `max_a' Q(s', a')` = Melhor valor Q futuro estimado

## 🚀 Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Modo Q-Learning (caminho ótimo)

```bash
python main.py <caminho_fsm.json> --goal-states <estado_alvo> [--max-steps 50]
```

**Exemplos:**

```bash
# LightSwitch — objetivo: chegar ao estado "ligado"
python main.py fsm/_01_LightSwitch_flattened.json --goal-states main_On__

# DimmableLightSwitch — objetivo: brilho máximo
python main.py fsm/_02_DimmableLightSwitch_flattened.json --goal-states main_On__brightness_10

# Múltiplos estados alvo
python main.py fsm/_02_DimmableLightSwitch_flattened.json --goal-states main_On__brightness_9 main_On__brightness_10
```

**Saída:**
1. Treina o agente por 1000 episódios
2. Exibe a **Q-Table** final
3. Mostra o **melhor caminho** aprendido
4. Gera gráficos:
   - `training_results.png` — Convergência das recompensas
   - `q_table_heatmap.png` — Heatmap da Q-Table
   - `fsm_diagram.png` — Diagrama do autômato com caminho ótimo

### 3. Modo Cobertura de Transições

```bash
python main_coverage.py <caminho_fsm.json> [--max-steps 50]
```

**Exemplos:**

```bash
# FSM simples (2 estados, 2 transições)
python main_coverage.py fsm/_01_LightSwitch_flattened.json

# FSM média (21 estados, 31 transições)
python main_coverage.py fsm/_02_DimmableLightSwitch_flattened.json

# FSM maior com mais passos por episódio
python main_coverage.py fsm/_03_MotionLightSwitch_flattened.json --max-steps 200
```

**Saída:**
1. Treina o agente com foco em cobertura
2. Exibe **relatório de cobertura** (transições cobertas/não cobertas)
3. Gera **suíte de testes mínima** (greedy set cover)
4. Gera gráficos:
   - `coverage_progress.png` — Evolução da cobertura ao longo dos episódios
   - `fsm_coverage.png` — Diagrama do autômato com transições cobertas vs não cobertas

## 🧪 Cobertura de Transições

**Transition Coverage (Cobertura de Transições)**: cada transição `(estado, ação) → próximo_estado` definida na FSM deve ser exercitada pelo menos uma vez pela suíte de testes gerada.

A função de recompensa é adaptada:
- **+50** por exercitar uma transição **ainda não coberta**
- **-1** por revisitar uma transição **já coberta**
- **-10** por ação inválida

Assim, o Q-Learning aprende a explorar transições novas, gerando sequências de teste com cobertura máxima.

## ⚙️ Hiperparâmetros

| Parâmetro | Valor Padrão | Descrição |
|-----------|-------------|-----------|
| `α` (alpha) | 0.1 | Taxa de aprendizado |
| `γ` (gamma) | 0.95 | Fator de desconto |
| `ε` (epsilon) | 1.0 → 0.01 | Taxa de exploração (com decay) |
| `ε decay` | 0.995 | Decaimento por episódio |
| Episódios | 1000 | Número de episódios de treino |

## 📁 Estrutura do Projeto

```
Q-learning-fst/
├── fsm/                      # FSMs em formato JSON
│   ├── _01_LightSwitch_flattened.json
│   ├── _02_DimmableLightSwitch_flattened.json
│   ├── _03_MotionLightSwitch_flattened.json
│   ├── _04_LightAndMotionSensingLightSwitch_flattened.json
│   └── _05_PresenceSimulationLightSwitch_flattened.json
├── fsm_environment.py        # Ambiente FSM + carregamento de JSON
├── q_learning_agent.py       # Agente Q-Learning clássico (caminho ótimo)
├── coverage_agent.py         # Agente Q-Learning para cobertura de transições
├── coverage_runner.py        # Orquestrador da geração de testes
├── visualization.py          # Gráficos e visualizações (ambos os modos)
├── main.py                   # Script principal (modo caminho ótimo)
├── main_coverage.py          # Script principal (modo cobertura de transições)
├── requirements.txt          # Dependências
└── README.md                 # Este arquivo
```

"# q-learning-fst" 
