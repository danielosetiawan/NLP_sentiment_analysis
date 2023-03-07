from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
from data import *
# from dash_iconify import DashIconify
# from content import *
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import dash_icons as fas

# icon
# header = html.H4(
#     "Sentiments NLP", className="bg-primary text-white p-2 mb-2 text-center"
# )

# nav_menu = dbc.NavbarSimple(
#     children=[
#         dbc.NavItem(dbc.NavLink("Page 1", href="#")),
#         dbc.DropdownMenu(
#             children=[
#                 dbc.DropdownMenuItem("More pages", header=True),
#                 dbc.DropdownMenuItem("Page 2", href="#"),
#                 dbc.DropdownMenuItem("Page 3", href="#"),
#             ],
#             nav=True,
#             in_navbar=True,
#             label="More",
#         ),
#     ],
#     brand="NavbarSimple",
#     brand_href="#",
#     color="primary",
#     dark=True,
# )


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

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5([
                        f'{topic_dct[topic]["title"]} ',
                        html.I(className="fa fa-dollar", 
                            style={'font-size': '1rem'}),
                    ])
                ], width = 4),
                dbc.Col([*volume], style = {'margin-left': '-5px'}),
                dbc.Col([*sentiment], width=2),
                dbc.Col([trajectory], width=2),
                dbc.Col(modal_fig, width=2),
                ], className='g-0')
            ])
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
                dbc.Col([html.H4('Topic')], style = style, width = 4),
                dbc.Col([html.H4('Mentions')], style = style, width = 2),
                dbc.Col([html.H4('Sentiment')], style = style, width = 2),
                dbc.Col([html.H4('Direction')], style = style, width = 2),
                dbc.Col([html.H4('Chart')], style = style, width = 2),
                ], className='g-0'),
            dbc.Row([
                dbc.Col([], style = style, width = 4),
                dbc.Col([html.P('(past year)')], style = style2, width = 2),
                dbc.Col([html.P('(past 7 days)')], style = style2, width = 2),
                dbc.Col([html.P('(past 7 days)')], style = style2, width = 2),
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

jumbotron = html.Div(
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
                dbc.ModalBody(jumbotron),
            ],
            id="intro-modal",
            size="xl",
            # is_open=True,
        )

slider = html.Div(
    [
        # dbc.Label("Select Years"),
        html.I(
            'Years',
            style={'font-size': '12px'}
        ),
        dcc.RangeSlider(
            tweets_df.Date.dt.year.min(),
            tweets_df.Date.dt.year.max(),
            1,
            id="years",
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            value=[2017, 2020],
            className="p-0",
        ),
    ],
    className="mb-4",
)

dropdown = html.Div(
    [
        dcc.Dropdown(
            options=tweets_df['Stock Name'].unique(),
            value='All',
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
                dbc.PopoverHeader(slider),
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



controls = dbc.Card(
    [dropdown],#, checklist],#, slider],
    body=True,
)


wordCloud = dbc.Card([html.Img(id="wordcloud")])

about_card = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.CardImg(
                        src="daniel.jpeg",
                        # className="img-fluid rounded-start",
                    ),
                    className="col-md-4",
                ),
                dbc.Col(
                    dbc.CardBody(
                        [
                            html.H4("Card title", className="card-title"),
                            html.P(
                                "This is a wider card with supporting text "
                                "below as a natural lead-in to additional "
                                "content. This content is a bit longer.",
                                className="card-text",
                            ),
                            html.Small(
                                "Last updated 3 mins ago",
                                className="card-text text-muted",
                            ),
                        ]
                    ),
                    className="col-md-8",
                ),
            ],
            className="g-0 d-flex align-items-center",
        )
    ],
    className="mb-3",
    style={"maxWidth": "540px"},
)