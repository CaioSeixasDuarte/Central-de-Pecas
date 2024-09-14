import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import plotly.graph_objs as go

# Configurando a página do Streamlit
st.set_page_config(page_title='Central de Peças', layout='wide')

# Função para tratamento de erros ao computar
def safe_compute(simulation):
    try:
        simulation.compute()
        return True
    except Exception as e:
        st.error(f"Erro ao calcular a saída: {e}")
        return False

# Definindo as variáveis linguísticas de entrada
# Tempo médio de espera (m) em minutos (0 a 120)
m = ctrl.Antecedent(np.arange(0, 121, 1), 'tempo_espera')
m['muito_pequeno'] = fuzz.trapmf(m.universe, [0, 0, 10, 30])
m['pequeno'] = fuzz.trimf(m.universe, [10, 30, 50])
m['medio'] = fuzz.trimf(m.universe, [40, 60, 90])
m['grande'] = fuzz.trapmf(m.universe, [80, 100, 120, 120])

# Fator de utilização de reparo (p)
p = ctrl.Antecedent(np.arange(0, 1.2, 0.001), 'fator_utilizacao')
p['baixo'] = fuzz.trapmf(p.universe, [0, 0, 0.2, 0.4])
p['medio'] = fuzz.trimf(p.universe, [0.3, 0.5, 0.7])
p['alto'] = fuzz.trapmf(p.universe, [0.6, 0.8, 1, 1])

# Número de funcionários (s) entre 0 e 100
s = ctrl.Antecedent(np.arange(0, 101, 1), 'numero_funcionarios')
s['pequeno'] = fuzz.trapmf(s.universe, [0, 0, 20, 40])
s['medio'] = fuzz.trimf(s.universe, [30, 50, 70])
s['grande'] = fuzz.trapmf(s.universe, [60, 80, 100, 100])

# Definindo a variável de saída (n) - número de peças extras (0 a 500)
n = ctrl.Consequent(np.arange(0, 501, 1), 'numero_pecas')
n['muito_pequeno'] = fuzz.trapmf(n.universe, [0, 0, 50, 150])
n['pequeno'] = fuzz.trimf(n.universe, [50, 150, 250])
n['pouco_pequeno'] = fuzz.trimf(n.universe, [125, 175, 225])
n['medio'] = fuzz.trimf(n.universe, [150, 250, 350])
n['pouco_grande'] = fuzz.trimf(n.universe, [275, 325, 375])
n['grande'] = fuzz.trimf(n.universe, [300, 400, 500])
n['muito_grande'] = fuzz.trapmf(n.universe, [350, 450, 500, 500])

# Definindo as regras fuzzy
rule1 = ctrl.Rule(m['muito_pequeno'] & s['pequeno'], n['muito_grande'])
rule2 = ctrl.Rule(m['pequeno'] & s['grande'], n['pequeno'])
rule3 = ctrl.Rule(p['baixo'], n['pequeno'])
rule4 = ctrl.Rule(p['alto'], n['grande'])
rule5 = ctrl.Rule(m['grande'], n['muito_grande'])  

# Criando o sistema de controle
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
sim = ctrl.ControlSystemSimulation(system)

# Interface com Streamlit
st.title('Sistema de Inferência Nebulosa para Central de Peças')

# Sliders para entrada de dados
tempo_espera = st.slider('Tempo de Espera (em minutos)', 0, 120, 30, step=1)
fator_utilizacao = st.slider('Fator de Utilização (0.0 a 1.0)', 0.0, 1.0, 0.3, step=0.01)
numero_funcionarios = st.slider('Número de Funcionários', 0, 100, 30, step=1)

# Passando as entradas para o sistema fuzzy
sim.input['tempo_espera'] = tempo_espera
sim.input['fator_utilizacao'] = fator_utilizacao
sim.input['numero_funcionarios'] = numero_funcionarios


if safe_compute(sim):
    
    numero_pecas = round(sim.output["numero_pecas"])
    st.write(f'Número de peças extras recomendadas: {numero_pecas}')

# Função para plotar as funções de pertinência com Plotly
def plot_fuzzy_var(var, var_name, input_value=None, output_value=None, medians=[]):
    traces = []
    for label in var.terms:
        trace = go.Scatter(
            x=var.universe,
            y=var[label].mf,
            mode='lines',
            name=label
        )
        traces.append(trace)

    layout = go.Layout(
        title=f'Função de Pertinência - {var_name}',
        xaxis=dict(
            title=var_name,
            tickvals=np.linspace(min(var.universe), max(var.universe), 10),
            ticktext=[f'{i:.0f}' for i in np.linspace(min(var.universe), max(var.universe), 10)]
        ),
        yaxis=dict(title='Pertinência'),
    )

    fig = go.Figure(data=traces, layout=layout)

    # Adicionando linhas medianas
    for median in medians:
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=median, y0=0,
                x1=median, y1=1,
                line=dict(color="LightGray", dash="dash"),
            )
        )

    if input_value is not None:
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=input_value, y0=0,
                x1=input_value, y1=1,
                line=dict(color="Red", dash="dashdot"),
            )
        )

    if output_value is not None:
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=output_value, y0=0,
                x1=output_value, y1=1,
                line=dict(color="Blue", dash="dashdot"),
            )
        )

    return fig

# Exibindo gráficos das funções de pertinência
st.subheader('Funções de Pertinência')
st.plotly_chart(plot_fuzzy_var(m, 'Tempo de Espera (min)', input_value=tempo_espera, medians=[10, 30, 60, 100]))
st.plotly_chart(plot_fuzzy_var(p, 'Fator de Utilização', input_value=fator_utilizacao, medians=[0.2, 0.5, 0.8]))
st.plotly_chart(plot_fuzzy_var(s, 'Número de Funcionários', input_value=numero_funcionarios, medians=[20, 50, 80]))
st.plotly_chart(plot_fuzzy_var(n, 'Número de Peças Extras', output_value=sim.output["numero_pecas"], medians=[100, 200, 300, 400]))
