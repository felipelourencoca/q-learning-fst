# Q-Learning para Máquina de Estados Finitos (FSM)

Implementação de um agente **Q-Learning** que aprende a navegar de forma ótima em uma **Máquina de Estados Finitos** (FST/FSM).

## 📖 Sobre

O **Q-Learning** é um algoritmo de **Aprendizado por Reforço** (*Reinforcement Learning*) que permite a um agente aprender, por tentativa e erro, qual ação tomar em cada estado para maximizar a recompensa acumulada.

Neste projeto, o ambiente é modelado como um **autômato finito** onde:

- **Estados** = Nós do autômato (`S0`, `S1`, ..., `S5`)
- **Ações** = Transições possíveis (`a`, `b`)
- **Recompensas** = Feedback numérico por cada transição
- **Objetivo** = Alcançar o estado `S5` no menor número de passos

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

## 🏗️ Estrutura do autômato de exemplo

```
S0 --a--> S1 --a--> S3 --b--> S5 (OBJETIVO)
|         |                     ^
b         b                     |
v         v                     a
S2 --a--> S4 --------a-------> S5
```

## 🚀 Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar

```bash
python main.py
```

### Saída esperada

O script irá:
1. Treinar o agente por 1000 episódios
2. Exibir a **Q-Table** final
3. Mostrar o **melhor caminho** aprendido
4. Gerar gráficos:
   - `training_results.png` — Convergência das recompensas
   - `q_table_heatmap.png` — Heatmap da Q-Table
   - `fsm_diagram.png` — Diagrama do autômato com caminho ótimo

## 📁 Estrutura do Projeto

```
Q-learning-fst/
├── fsm_environment.py    # Ambiente FSM (estados, transições, recompensas)
├── q_learning_agent.py   # Agente Q-Learning (Q-Table, ε-greedy, treino)
├── visualization.py      # Gráficos e visualizações
├── main.py               # Script principal
├── requirements.txt      # Dependências
└── README.md             # Este arquivo
```

## ⚙️ Hiperparâmetros

| Parâmetro | Valor Padrão | Descrição |
|-----------|-------------|-----------|
| `α` (alpha) | 0.1 | Taxa de aprendizado |
| `γ` (gamma) | 0.95 | Fator de desconto |
| `ε` (epsilon) | 1.0 → 0.01 | Taxa de exploração (com decay) |
| `ε decay` | 0.995 | Decaimento por episódio |
| Episódios | 1000 | Número de episódios de treino |

## 🔧 Personalização

Você pode definir seu próprio autômato editando as transições em `fsm_environment.py` ou criando uma nova instância de `FSMEnvironment` em `main.py`:

```python
from fsm_environment import FSMEnvironment

env = FSMEnvironment(
    states=["A", "B", "C"],
    actions=["x", "y"],
    transitions={
        ("A", "x"): "B",
        ("A", "y"): "C",
        ("B", "x"): "C",
    },
    rewards={
        ("A", "x"): -1.0,
        ("A", "y"): -1.0,
        ("B", "x"): 100.0,
    },
    initial_state="A",
    goal_states={"C"},
)
```

## 🧪 Modo: Cobertura de Transições (Test Coverage)

Além do modo clássico (encontrar caminho ótimo), o projeto inclui um **modo de cobertura de transições** que usa Q-Learning para **gerar sequências de teste** que cobrem todas as transições da FSM.

### Critério de cobertura

**Transition Coverage (Cobertura de Transições)**: cada transição `(estado, ação) → próximo_estado` definida na FSM deve ser exercitada pelo menos uma vez pela suíte de testes gerada.

### Como funciona

A função de recompensa é adaptada:
- **+50** por exercitar uma transição **ainda não coberta**
- **-1** por revisitar uma transição **já coberta**
- **-10** por ação inválida

Assim, o Q-Learning aprende a explorar transições novas, gerando sequências de teste com cobertura máxima.

### Executar modo cobertura

```bash
python main_coverage.py
```

### Saída esperada

1. Treina o agente com foco em cobertura
2. Exibe **relatório de cobertura** (transições cobertas/não cobertas)
3. Gera **suíte de testes mínima** (greedy set cover)
4. Gera gráficos:
   - `coverage_progress.png` — Evolução da cobertura ao longo dos episódios
   - `fsm_coverage.png` — Diagrama do autômato com transições cobertas vs não cobertas

## 📁 Estrutura do Projeto

```
Q-learning-fst/
├── fsm_environment.py    # Ambiente FSM (estados, transições, recompensas)
├── q_learning_agent.py   # Agente Q-Learning clássico (caminho ótimo)
├── coverage_agent.py     # Agente Q-Learning para cobertura de transições
├── coverage_runner.py    # Orquestrador da geração de testes
├── visualization.py      # Gráficos e visualizações (ambos os modos)
├── main.py               # Script principal (modo caminho ótimo)
├── main_coverage.py      # Script principal (modo cobertura de transições)
├── requirements.txt      # Dependências
└── README.md             # Este arquivo
```

"# q-learning-fst" 
