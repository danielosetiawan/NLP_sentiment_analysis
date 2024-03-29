from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from data import *
# from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.stats import pearsonr
import base64
from statsmodels.tsa.stattools import grangercausalitytests

def transform_image(path):
    img = base64.b64encode(open(path, 'rb').read()).decode('ascii')
    return f'data:image/png;base64,{img}'

def create_logo(image, link):
    return html.A(
        href=link,
        target="_blank",
        children=[
            html.Img(
                src=transform_image(image),
                style={"height": "30px", "width": "30px", "marginRight": "10px"},
            ),
        ],
    )


def topic_data(topic):

    df = topics_stats[topic]

    # volume section
    style = {'font-size': '12px', 'margin-bottom': '-2px', 'margin-top': '-2px'}
    negative = round((df.negative / df.mentions) * 100)
    positive = round((df.positive / df.mentions) * 100)

    if df.mentions > 490000:
        mentions = '> 500,000'
    else:
        mentions = f'{round(df.mentions):,}'

    volume = [
        html.H1(f'{mentions}', style=style),
        html.P(f'Positive: {positive}%', className="text-success", style=style),
        html.P(f'Negative: {negative}%', className="text-danger", style=style),
    ]

    sent_val = df.sentiment

    if sent_val <= 0.2:
        sent_val = 'Very Negative'
        color = 'danger'
    elif sent_val <= 0.4:
        sent_val = 'Negative'
        color = 'danger'
    elif sent_val <= 0.6:
        sent_val = 'Neutral'
        color = 'warning'
    elif sent_val <= 0.8:
        sent_val = 'Positive'
        color = 'success'
    else:
        sent_val = 'Very positive'
        color = 'success'

    sentiment = [
        dbc.Badge(sent_val, color=color, className="me-1"),
    ]

    sent_today = df.sent_today
    sent_yday = df.sent_yday

    if sent_today > sent_yday:
        traj = ' Rising'
        cls = 'fa fa-angle-up'
        clr = 'green'
    else:
        traj = ' Falling'
        cls = 'fa fa-angle-down'
        clr = 'red'

    trajectory =  html.P([
                            html.I(
                                className=cls, 
                                style={'font-size': '1rem', 'color': clr}
                                ), 
                            traj,
                    ], style={'font-size': '1rem', 'color': clr})

    # modal section
    data = topics_df
    fig = make_subplots(
        rows=2, cols=1, 
        row_heights=[10, 10],
        vertical_spacing=0.05,
        x_title='Date',
        shared_xaxes=True,
        specs=[[{"secondary_y": False, 'rowspan': 1}], 
               [{"secondary_y": True, 'rowspan': 1}]]
    )
    
    # plot 1 : twitter sentiment
    STS = data[f'{topic}_sentiment'].rolling(7).mean()
    MTS = data[f'{topic}_sentiment'].rolling(30).mean()
    LTS = data[f'{topic}_sentiment'].rolling(50).mean()

    STM = go.Scatter(x=data['Date'], y=STS, name='Short Term', visible='legendonly')
    MTM = go.Scatter(x=data['Date'], y=MTS, name='Mid Term')
    LTM = go.Scatter(x=data['Date'], y=LTS, name='Long Term')
    
    fig.add_traces([STM, MTM, LTM], rows=1, cols=1)
    fig.update_yaxes(title_text="Sentiment", secondary_y=False, row=1, col=1)
    
    
    # plot 2 : market data
    market_data = topic_dct[topic]
    topic_info = zip(
        market_data['data'], market_data['name'], ['#FEAF16', '#AA0DFE'],
        market_data['yaxis_title'], ['y1', 'y2'], [True, False]
    )
    
        
    for dta, name, color, ax_title, axis, secondY in topic_info:
        fig.add_trace(go.Scatter(
            x=data.Date,
            y=data[dta],
            name=name, 
            connectgaps=True,
            yaxis=axis,
            line=dict(color=color)
        ), row=2, col=1, secondary_y=secondY)

        fig.update_yaxes(
            title_text=ax_title, 
            secondary_y=secondY,
            row=2, col=1
        )
        
        
    fig.update_layout(
        hovermode='x unified',
        template='plotly_white',
        showlegend=False,
        title=f'Twitter Sentiments vs. {topic_dct[topic]["title"]}',
        height=600,
        yaxis2=dict(
            # tickfont=dict(color='#AA0DFE'),
            titlefont=dict(color='#AA0DFE')
        ),
        yaxis3=dict(
            # tickfont=dict(color='#FEAF16'),
            titlefont=dict(color='#FEAF16')
        ),
    )

    fig.update_traces(xaxis='x2')


    modal_title = f'{topic_dct[topic]["title"]} Sentiment Analysis'
    modal_fig = html.Div(
        [
            dbc.Button("Show", id=f"{topic}-button"),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(modal_title)),
                    dbc.ModalBody(dcc.Graph(figure=fig)),
                ],
                id=f"{topic}-modal",
                size='xl'
            ),
        ]
    )

    data_inter = data.interpolate()
    data_inter = data_inter.fillna(method='bfill')
    def correlation_coeff(col1, col2):
            corr, pvalue = pearsonr(col1, col2)
            return round(corr, 2)
    
    def topic_granger_causality(topic, comparison_data):
        coefs = []

        # data = topics_df.copy()
        # data['Date'] = pd.to_datetime(topics_df.Date).dt.strftime('%Y-%m')

        

        try:
            data1 = topics_df[['Date', comparison_data, f'{topic}_sentiment']].groupby(['Date']).mean().dropna()
            data2 = topics_df[['Date', comparison_data, f'{topic}_sentiment']].groupby(['Date']).mean()
            data2 = data2.dropna(how='all').fillna(method='ffill').dropna()

            for data in [data1, data2]:
                results = grangercausalitytests(data, maxlag=2, verbose=False)

                for idx in range(2):
                    pval = [results[i+1][0]['ssr_ftest'][idx] for i in range(2)]
                    granger_causality_coef = 1 - pval[1] / pval[0]
                    coefs.append(granger_causality_coef)
                
            return f'{round(max(coefs), 2):.2f}'
        except:
            return 'N/A'
        
    # corr1 = correlation_coeff(data_inter['IRS Tax'], data_inter['taxes_sentiment'])
    # topic_dct['taxes']['corr'] = corr1

    topic_dct[topic]['corr1'] = topic_granger_causality(topic, topic_dct[topic]['data'][0])
    topic_dct[topic]['corr2'] = topic_granger_causality(topic, topic_dct[topic]['data'][1])

    # corr2 = correlation_coeff(data_inter['bank_loan'], data_inter['loans_sentiment'])
    # topic_dct['loans']['corr'] = corr2
    # corr3 = correlation_coeff(data_inter['inflation'], data_inter['inflation_sentiment'])
    # topic_dct['inflation']['corr'] = corr3
    # corr4 = correlation_coeff(data_inter['total_debt'], data_inter['recession_sentiment'])
    # topic_dct['recession']['corr'] = corr4
    # corr5 = correlation_coeff(data_inter['bonds_issued'], data_inter['bonds_sentiment'])
    # topic_dct['bonds']['corr'] = corr5
    # corr6 = correlation_coeff(data_inter['GDP'], data_inter['economy_sentiment'])
    # topic_dct['economy']['corr'] = corr6
    # corr7 = correlation_coeff(data_inter['unemployment_rate'], data_inter['unemployment_sentiment'])
    # topic_dct['unemployment']['corr'] = corr7
    # corr8 = correlation_coeff(data_inter['mortgage_rates'], data_inter['housing_market_sentiment'])
    # topic_dct['housing_market']['corr'] = corr8
    # corr9 = correlation_coeff(data_inter['federal_funds'], data_inter['interest_rates_sentiment'])
    # topic_dct['interest_rates']['corr'] = corr9

    def topic_logo():
        return html.A(
            target="_blank",
            children=[
                html.Img(
                    src=transform_image(topic_dct[topic]['logo_path']),
                    style={"height": "30px", "width": "30px", "marginRight": "10px"},
                ),
            ],
        )

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5([
                        f'{topic_dct[topic]["title"]} ',
                        html.Br(),
                        topic_logo()
                        # html.I(className="fa fa-dollar", 
                            # style={'font-size': '1rem'}),
                    ])
                ], width = 2),
                dbc.Col([*volume], style = {'margin-left': '-5px'}),
                dbc.Col([*sentiment], width=2),
                dbc.Col([trajectory], width=2),
                dbc.Col(modal_fig, width=2),
                dbc.Col([
                    html.I([
                        f'{topic_dct[topic]["name"][0]}: ',
                        f'{topic_dct[topic]["corr1"]}',
                        html.Br(),
                        f'{topic_dct[topic]["name"][1]}: ',
                        f'{topic_dct[topic]["corr2"]}',
                    ])
                    ], width=2)
                    ])
                ], className='g-0')
            )

    
topic_style = {
    'font-weight': 'bold',
    'font-size': '15px',
}

style = {'margin-top': '-5px', 'margin-bottom': '-5px', 'font-weight': 'bold'}
style2 = {'margin-top': '-5px', 'margin-bottom': '-30px', 
          'font-style': 'italic', 'font-size': '10px'}
topic_title = dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.H4('Topic')], style = style, width = 2),
                dbc.Col([html.H4('Mentions')], style = style, width = 2),
                dbc.Col([html.H4('Sentiment')], style = style, width = 2),
                dbc.Col([html.H4('Direction')], style = style, width = 2),
                dbc.Col([html.H4('Chart')], style = style, width = 2),
                dbc.Col([html.H4('Correlation')], style = style, width = 2)
                ], className='g-0'),
            dbc.Row([
                dbc.Col([], style = style, width = 2),
                dbc.Col([html.P('(past year)')], style = style2, width = 2),
                dbc.Col([html.P('(past 7 days)')], style = style2, width = 2),
                dbc.Col([html.P('(past 7 days)')], style = style2, width = 2),
                dbc.Col([], style = style, width = 2),
                dbc.Col([html.P('(topic and sentiment correlation)')], style = style2, width = 2),
                ], className='g-0'),
            ])
        )

sent_topics = [
            topic_data('taxes'),
            topic_data('loans'),
            topic_data('inflation'),
            topic_data('recession'),
            topic_data('bonds'),
            topic_data('economy'),
            topic_data('unemployment'),
            topic_data('housing_market'),
            topic_data('interest_rates'),
            ]


table = html.Div(
    dash_table.DataTable(
        id="table",
        columns=[{"name": i, "id": i, "deletable": True} for i in tweets_df.columns],
        data=tweets_df.to_dict("records"),
        page_size=10,
        editable=True,
        cell_selectable=True,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        row_selectable="multi",
    ), className="dbc-row-selectable",
)

info_msg = html.Div(
    dbc.Container(
        [
            html.H1("NLP Analysis", className="display-3"),
            html.P(
                'Data observed in this page utilizes a semi-supervised learning approach '
                'The initial training involved labeled targets and incorporated the BERT model, '
                'which itself was trained on a vast corpus of textual data. The ultimate testing '
                'phase involved evaluating the trained data on a diverse range of Twitter tweets '
                'that were scraped from various topics. These results were then compared against '
                'real-world data to assess the accuracy of the approach.',
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                " "
                "larger container."
            ),
            html.P(
                dbc.Button("Learn more", id='intro-modal-button', color="primary"), className="lead"
            ),
        ],
        fluid=True,
        className="py-3",
    ),
    className="p-3 bg-light rounded-3",
)

intro_message = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("A message for our viewers")),
                dbc.ModalBody(info_msg),
            ],
            id="intro-modal",
            size="xl",
            # is_open=True,
        )

# slider = html.Div(
#     [
#         # dbc.Label("Select Years"),
#         html.I(
#             'Years',
#             style={'font-size': '12px'}
#         ),
#         dcc.RangeSlider(
#             tweets_df.Date.dt.year.min(),
#             tweets_df.Date.dt.year.max(),
#             1,
#             id="years",
#             marks=None,
#             tooltip={"placement": "bottom", "always_visible": True},
#             value=[2017, 2020],
#             className="p-0",
#         ),
#     ],
#     className="mb-4",
# )

dropdown = html.Div(
    [
        dcc.Dropdown(
            options=tweets_df['Stock Name'].unique(),
            value='AAPL',
            id="company",
            clearable=False,
            style={
                'position': 'relative', 
                'width': '200px'
            }
        ),
    ], className="mb-4",
)

popovers = html.Div(
    [
        dbc.Button(
            "stats",
            id="stock-stats",
            n_clicks=0,
            style={
                'margin-left': '60px', 
                'margin-top': '15px', 
                'line-height': '30px',
                'width': '90px', 
                'font-size': '13px',
                'textAlign': 'top',
            },
            color='info',
        ),
        dbc.Popover(
            [
                #dbc.PopoverHeader(slider),
                dbc.PopoverBody([
                    # html.P(id='lag-coef-value',
                    #     'Lag Coefficient: ',
                    # ),
                    html.P(id='lag-coef-value'),
                ]),
                
            ], style={
                'width': '500px',
            },
            target="stock-stats",
            trigger="hover",
            placement='bottom'
        ),
    ]
)

# checklist = html.Div(
#     [
#         dbc.Label("Select Analysis"),
#         dbc.Checklist(
#             id="analysis",
#             options=stocks_df.columns,
#             value=stocks_df.columns,
#             inline=True,
#         ),
#     ],
#     className="mb-4",
# )

sent_pred = html.Div([
        dcc.Textarea(
            id='prediction',
            value="Hell yeahhh ~ I'm feeling extra bullish today. \nLooks like them stock prices are shooting to the moon 😍",
            style={'width': '100%', 'height': 200},
        ),
        html.Button('Classify Text', id='sentiment-prediction-button', n_clicks=0),
        html.Div(id='sentiment-prediction-output', style={'whiteSpace': 'pre-line'})
    ])

controls = dbc.Card(
    [dropdown],#, checklist],#, slider],
    body=True,
)

pred_summary = [
    dbc.CardHeader("Prediction Summary"),
    dbc.CardBody(
        [
            html.Div([
                html.Br(),
                'Test Accuracy: 0.8893',
                html.Br(),
                'Support Vector: 8626',
                html.Br(),
                'Training Loss: 0.124',
                html.Br(),
                'Corpus: 5M+ StockTwits Data',
                html.Br(),
                html.Br(),
                html.Br(),
                html.I(
                    'Note: Composition Weight is the weight of the combined sentence.',
                    className="card-title"
                    ),
                html.Br(),
            ])
        ], style={'height': 300}
    ), 
]

wordCloud = dbc.Card([html.Img(id="wordcloud")])

stock_chart = html.Div([
    dcc.Graph(
        id='line-chart', 
        style={'margin-top': '-30px', 'height': 1000}
    ),
])

title_style = {'margin-left': '20px', 'margin-top': '5px'}

left_jumbotron = dbc.Col(
    html.Div([
        html.Div(
            [
                html.Img(src=transform_image('img/daniel.jpeg'),
                         style={'width': 200, 'height': 200}),
                html.Div([
                    html.Br(), html.Br(),
                    create_logo('img/linkedin.png', 'https://www.linkedin.com/in/danielosetiawan/'),
                    create_logo('img/github.png', 'https://github.com/danielosetiawan'),                    
                    html.H4("Daniel S.", className="display-6"),
                    html.I('Data Science Fellow', style={'margin-top': '-200px'}),
                ], style={'margin-left': '20px', 'line-height': 0.5}),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),
            html.Hr(className="my-2"),
            html.P(
                "I'm an alumnus of UCSB with a BS in Chemistry. "
                "I currently work as an R&D engineer at a quantum computing startup, "
                "where I had the opportunity to interface with various electronics (RF/DC) "
                "and code (python, SCPI, Labview) to analyze failure rates, "
                "improve product performance, and develop methods attuned to scalability. "
                "This journey brought me to channel my inner passion for data science, "
                "as there are many ways to creatively tell a story using data.",
                style = {'margin-top': '10px'}
            ),
        ], className="h-100 p-5 bg-light border rounded-3",  
    ), md=6,
)

right_jumbotron = dbc.Col(
    html.Div([
        html.Div(
            [
                html.Img(src=transform_image('img/laurel.jpg'),
                         style={'width': 200, 'height': 200}),
                html.Div([
                    html.Br(), html.Br(),
                    create_logo('img/linkedin.png', 'https://www.linkedin.com/in/cheng-laurel-he-b04a59104/'),
                    create_logo('img/github.png', 'https://github.com/LaurelHe1'),                    
                    html.H4("Laurel H.", className="display-6"),
                    html.I('Data Science Fellow', style={'margin-top': '-200px'}),
                ], style={'margin-left': '20px', 'line-height': 0.5}),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),
            html.Hr(className="my-2"),
            html.P(
                "My undergrad was in Geosystems Engineering and Hydrogeology at the "
                "University of Texas, Austin. I got my Master's in Atmosphere and Energy, "
                "Civil Engineering at Stanford. I'm passionate about nature, "
                "environmental protection and renewable energy. I'm excited about how "
                "machine learning and data analytics are giving us better tools to "
                "understand and fight climate change, and I'm looking forward to kickstart "
                "my career in this exciting field.",
                style = {'margin-top': '10px'}
            ),
        ], className="h-100 p-5 bg-light border rounded-3",  
    ), md=6,
)


jumbotron = dbc.Row(
    [left_jumbotron, right_jumbotron],
    className="align-items-md-stretch",
)