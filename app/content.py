
from dash import dcc, html
import dash_bootstrap_components as dbc
from globals import *

#####################
####### tab 1 #######
#####################
tab1 = dbc.Tab(label="Stocks", children=[
    intro_message,
            dbc.Row([
                dbc.Col([
                            # html.I(
                            #     'Statistics',
                            #     style={'font-size': '12px'}
                            # ),
                            popovers
                         ],
                        width={"size": 2, "offset": 7},
                        style = {
                            'margin-bottom': '25px', 
                            # 'margin-top': '-5px'
                        }
                    ),
                dbc.Col([
                            html.I(
                                'Select Company',
                                style={'font-size': '12px'}
                            ),
                            dropdown
                        ], 
                        width={"size": 3, "offset": 0},
                        style = {
                            'margin-bottom': '25px', 
                            # 'margin-top': '-5px'
                        }
                        ),
                
            # dbc.Col(
            #     [
                    dcc.Graph(
                        id='line-chart', 
                        style={'margin-top': '-30px', 'height': 800}),
            #     ],
            #     width = 12
            # ),
            ]),
            dbc.Col([wordCloud]),
        ])

#####################
####### tab 2 #######
#####################
tab2 = dbc.Tab(
    html.Div([
    dcc.Textarea(
        id='prediction',
        value="Hell yeahhh ~ I'm feeling extra bullish today.\nLooks like them stock prices are shooting to the moon 😍",
        style={'width': '100%', 'height': 200},
    ),
    html.Button('Classify Text', id='sentiment-prediction-button', n_clicks=0),
    html.Div(id='sentiment-prediction-output', style={'whiteSpace': 'pre-line'})
]), label="Sentiment Prediction")

#####################
####### tab 3 #######
#####################
tab3 = dbc.Tab(id='topic_tab', label="Sentiment Topics", children=[
    dbc.Row([
        dbc.Col([
            topic_title, *sent_topics,
        ], width=12),
    ])
])

#####################
####### tab 4 #######
#####################
tab4 = dbc.Tab(label="Table", children=[
    table
])

# change active tab when done testing
tabs = dbc.Card(
    dbc.Tabs([tab1, tab2, tab3, tab4])
    )

########################
####### home tab #######
########################
app_layout = dbc.Container(
    [
        
        dbc.Row(
            [
                dbc.Col(
                    
                    [tabs], width=12
                    ),
# ThemeChangerAIO(aio_id="theme")
            ]
        ),
        
        
    ],
    fluid=True,
    className="dbc",
)