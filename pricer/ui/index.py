from functools import partial

from dash import Dash, dcc, html, Input, Output

from pricer.option import models

app = Dash(__name__)


OPTION_SPECS = {
    'p': {
        'label': 'Call/Put',
        'input': lambda id: dcc.RadioItems(['C', 'P'], id=id)
    },
    'S': {
        'label': 'Spot',
        'input': lambda id: dcc.Input(100, id=id, type='number', debounce=True)
    },
    'S2': {
        'label': 'Spot2',
        'input': lambda id: dcc.Input(100, id=id, type='number', debounce=True)
    },
    'K': {
        'label': 'Strike',
        'input': lambda id: dcc.Input(100, id=id, type='number', debounce=True)
    },
    'T': {
        'label': 'Maturity',
        'input': lambda id: dcc.Input(0.5, id=id, min=0, max=50, step=0.1, \
                            type='number', debounce=True)
    },
    'sigma': {
        'label': 'Sigma',
        'input': lambda id: dcc.Input(0.3, id=id, min=0, max=1, step=0.1, \
                                  type='number', debounce=True)
    },
    'sigma2': {
        'label': 'Sigma2',
        'input': lambda id: dcc.Input(0.3, id=id, min=0, max=1, step=0.1, \
                                  type='number', debounce=True)
    },
    'corr': {
        'label': 'Correlation',
        'input': lambda id: dcc.Input(0.3, id=id, min=-1, max=1, step=0.1, \
                                  type='number', debounce=True)
    },
    'r': {
        'label': 'Free Rate',
        'input': lambda id: dcc.Input(0.05, id=id, type='number', debounce=True)
    },
    's': {
        'label': 'Steps',
        'input': lambda id: dcc.Input(10, id=id, min=10, max=100, step=10, \
                                      type='number', debounce=True)
    },
    'n': {
        'label': '# observations',
        'input': lambda id: dcc.Input(10, id=id, min=5, max=50, step=1, \
                                      type='number', debounce=True)
    },
    'M': {
        'label': 'Simulations',
        'input': lambda id: dcc.Input(1000, id=id, min=10, max=1e6, step=1e2, \
                                      type='number', debounce=True)
    }
}

OPTION_TYPES = {
    'European (Black-Scholes)': 
        {
            'id': 'eu_bs',
            'model':models.European,
            'input': ['p', 'S', 'K', 'T', 'sigma', 'r']
        },
    'European (Binomial Tree)':
        {
            'id': 'eu_bino',
            'model': models.BinomialTree,
            'input': ['p', 'S', 'K', 'T', 'sigma', 'r', 's']
        },
    'Asian Geometric (Black Scholes)':
        {
            'id': 'as_geo_bs',
            'model': models.AsianGeometric,
            'input': ['p', 'S', 'n', 'K', 'T', 'sigma', 'r']
        },
    'Asian Geometric (Monte Carlo)':
        {
            'id': 'as_geo_mc',
            'model': partial(models.Asian, type_="geometric"),
            'input': ['p', 'S', 'n', 'K', 'T', 'sigma', 'r', 'M']
        },
    'Asian Arithmetic (Monte Carlo)':
        {
            'id': 'as_ari_mc',
            'model': partial(models.Asian, type_="arithmetic"),
            'input': ['p', 'S', 'K', 'T', 'sigma', 'sigma2', 'r', 'M']
        },
    'Basket Geometric (Black Scholes)':
        {
            'id': 'basket_geo_bs',
            'model': models.GeometricBasketWithTwoAssets,
            'input': ['p', 'S', 'S2', 'K', 'T', 'sigma', 'sigma2', 'r'],
        },
    'Basket Arithmetic (Monte Carlo)':
        {
            'id': 'basket_ari_mc',
            'model': models.ArithmeticBasketWithTwoAssets,
            'input': ['p', 'S', 'S2', 'K', 'T', 'sigma', 'sigma2', 'r', 'M']
        }
}

def build_price_input():
    tables = {}
    for (name, spec) in OPTION_TYPES.items():
        ID = spec['id']
        tbody = []
        for arg_name in spec['input']:
            spec = OPTION_SPECS[arg_name]
            tr = html.Tr([
                html.Td(html.Label(spec['label'])),
                html.Td(spec['input'](f'{ID}_{arg_name}'))
            ])
            tbody.append(tr)
        table = html.Div([
                html.Table(html.Tbody(tbody), style={'width': '500px'}),
                html.Div(id=f'{ID}_px')
            ],
            id=f'{ID}_input',
            hidden=False)
        tables[name] = table
    return tables
    
PRICE_INPUTS = build_price_input()

def build_layout():
    return html.Div([
            html.H2("Option Pricer"),
            dcc.Dropdown(list(OPTION_TYPES.keys()), id='dropdown', style={'width': '500px'}),
            html.Div([], id="price_input"),
            html.Br(),
            html.Div(id='px')
        ])

def pricer_by_id(name):
    def price(_type, *args):
        if _type != name: return None
        print(f'pricing {name} with {args}')
        if not _type: return 'nyi'
        model = OPTION_TYPES[_type]
        option = model['model'](*args)
        px = option.price()
        if isinstance(px, (tuple, list)):
            px, l, u = px
        return px
    return price

def bind_callbacks():
    for (name, spec) in OPTION_TYPES.items():
        ID = spec['id']
        output = Output(component_id=f'{ID}_px', component_property='children')
        inputs = [Input(component_id=f'dropdown', component_property='value'),]
        for arg_name in spec['input']:
            input = Input(component_id=f'{ID}_{arg_name}', component_property='value'),
            inputs.append(input)
        app.callback(output, *inputs)(pricer_by_id(name))

@app.callback(
    Output(component_id='price_input', component_property='children'),
    Input(component_id='dropdown', component_property='value')
)
def switch(type_):
    return PRICE_INPUTS[type_]

def start(port):
    app.layout = build_layout()
    bind_callbacks()
    app.run_server(host="0.0.0.0", port=port, debug=False)
