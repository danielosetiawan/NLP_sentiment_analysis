
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
                        style={'margin-top': '-30px', 'height': 1000}),
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
    dbc.Row([
    dbc.Col([
        html.I('Sentiment Prediction Box'),
        sent_pred, 
        dbc.Col(dbc.Card(pred_summary, color="primary", outline=True), 
),
        ], 
        width={'size': 4}, 
        style={
            'margin-top': '30px',
            'margin-left': '20px'}
    ),
    dbc.Col([
        dcc.Graph(
            id='prediction-chart', 
            style={'margin-top': '0px', 'height': 800, 
                   'width': 1000}),
        ], 
        width={'size': 7}, 
        style={
            'margin-top': '0px',
            'margin-left': '0px'}
    ), 
    ], className="g-0"),
    
    label="Sentiment Prediction")

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
    dbc.Tabs([tab3, tab2, tab1, tab4])
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