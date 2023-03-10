
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
                    stock_chart,
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
        width={'size': 3}, 
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
        width={'size': 8}, 
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
tab4 = dbc.Tab(label="About Project", children=[
    dbc.Row([
        dbc.Col(
         html.Div([
            html.H4("Trading Strategy", className="display-6"),
            html.Hr(className="my-2"),
            html.P([
                html.Div("Conventional RSI Trading Strategy:", style={"font-weight":"bold", "font-size": "20px"}), 
                "- General condition to buy: the stock price is above its 200-day Moving Average ", html.Br(),
                "- If 10-period RSI of the stock is below 30, buy on the next day's open ", html.Br(),
                "- If 10-period RSI is above 40 or after 10 trading days, sell on the next day's open ", html.Br(),
                html.Br(),
                html.Div("Proposed Sentiment Trading Strategy:", style={"font-weight":"bold", "font-size": "20px"}),
                "- Buy at the start of bullish market signaled by the 50-day sentiment MA crossing the 200-day sentiment MA from below ", html.Br(),
                "- Sell at the start of bearish market signaled by the 200-day sentiment MA crossing the 50-day sentiment MA from below "
                ],
                style = {'margin-top': '10px'}
            ),
            ], className="h-100 p-5 bg-light border rounded-3",  
        ), md=6,
    ),
        dbc.Col(
         html.Div([
            html.H4("Future Work", className="display-6"),
            html.Hr(className="my-2"),
            html.P([
                "- Build Kirkland GPT from BERT next-sentence prediction ", html.Br(),
                "- Predict stock price from sentiment (will need accurate representations of correlation, lag and distance matrix) ", html.Br(),
                "- Automate workflow: scrape Twitter -> classify sentiments on Tweets -> alert user when to buy and sell ", html.Br(),
                "- Improve sentiment classification accuracy by adding a neutral class ", html.Br(),
                "- Improve sentiment trading strategy accuracy ", html.Br(),
                "- Add more input varieties like blogs, forums, comments; Add more sentiment topics like Forex, Crypto ", html.Br(),
                "- Find correlation/uncorrelation between topics to diversify investment portfolio "],
                style = {'margin-top': '10px'}
            ),
            ], className="h-100 p-5 bg-light border rounded-3",  
        ), md=6,
    ),
    ]), 
    dbc.Row([
        html.Div([
            html.H4("About This Project", className="display-6"),
            html.Hr(className="my-2"),
            html.P(
                'Data observed in this page utilizes a semi-supervised learning approach. '
                'The initial training involved labeled targets and incorporated the BERT model, '
                'which itself was trained on a vast corpus of textual data. The ultimate testing '
                'phase involved evaluating the trained data on a diverse range of Twitter tweets '
                'that were scraped from various topics. These results were then compared against '
                'real-world data to assess the accuracy of the approach.',
                style = {'margin-top': '10px'}
            ),
            ], className="h-100 p-5 bg-light border rounded-3",  
        )#, md=6,
    ])
])

# change active tab when done testing
tabs = dbc.Card(
    dbc.Tabs([tab4, tab1, tab2, tab3])
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