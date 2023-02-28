
from dash import dcc, html
import dash_bootstrap_components as dbc
from globals import *

#####################
####### tab 1 #######
#####################
tab1 = dbc.Tab(label="Stocks", children=[
            dbc.Row([
                dbc.Col(
                        [], width=9
                    ),
                dbc.Col(
                        [dropdown], width=3
                        ),
                ]),
            dbc.Col(
                [
                    dcc.Graph(id='line-chart', style={'margin-top': '-30px', 'height': 600}),#'width': '80vh', 'height': 1100}),
                ],
                width = 12
            # dcc.Graph(id='scatter-chart')
            ),
        ])

#####################
####### tab 2 #######
#####################
tab2 = dbc.Tab(html.Div([
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
            topic_title, *topics,
        ], width=12),
        # dbc.Col([
        #     topics, topics,
        # ], width=3)
    ])
#     html.Div(
#     [html.Button("Download Text", id="btn_txt"), dcc.Download(id="download-text-index")]
# )
])

#####################
####### tab 4 #######
#####################
tab4 = dbc.Tab(label="Table", children=[
    table
])

# change active tab when done testing
tabs = dbc.Card(
    dbc.Tabs([tab3, tab2, tab1, tab4])
    )





########################
####### home tab #######
########################
app_layout = dbc.Container(
    [
        
        dbc.Row(
            [
                # dbc.Col(
                #     [nav_menu], width=9
                # ),
                dbc.Col(
                    
                    [tabs], width=12
                    ),
                # dbc.Col(
                #     [
                #         controls,
                #         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #         # When running this app locally, un-comment this line:
                #         # ThemeChangerAIO(aio_id="theme"),
                #         dbc.Row(
                #             [
                                dbc.Col(
                                    [wordCloud_bull]#, wordCloud_bear]
                                ),
                #             ]
                #         ),
                #     ],
                #     width=3,
                # ),
                # dbc.Col([tabs, colors], width=8),
            ]
        ),
        
        
    ],
    fluid=True,
    className="dbc",
)