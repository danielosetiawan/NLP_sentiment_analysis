from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
from data import *
# from dash_iconify import DashIconify
# from content import *
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import plotly.express as px
import plotly.graph_objects as go
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
    # global volume, trajectory, sentiment, modal_fig #create global functions
    # sentiments = globals()[topic]['labels']
    
    df_raw = globals()[f'{topic}_raw']
    df = globals()[f'{topic}']
    df['Datetime'] = pd.to_datetime(df.Datetime)

    # volume section
    style = {'font-size': '12px', 'margin-bottom': '-2px', 'margin-top': '-2px'}
    negative = round((df_raw['labels'].value_counts()['negative'] / len(df_raw)) * 100)
    neutral = round((df_raw['labels'].value_counts()['neutral'] / len(df_raw)) * 100)
    positive = round((df_raw['labels'].value_counts()['positive'] / len(df_raw)) * 100)

    volume = [
        html.H1(f'{len(df_raw):,d}', style=style),
        html.P(f'Positive: {positive}%', className="text-success", style=style),
        html.P(f'Negative: {negative}%', className="text-danger", style=style),
        # html.P(f'Neutral: {neutral}%', className="text-warning", style=style),
    ]

    # sentiment section
    end_date = df.Datetime.max()
    start_date = end_date - pd.Timedelta(days=7)

    sent_val = df[df.Datetime.between(start_date, end_date)]['sentiment'].mean()

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

    # trajectory section
    sent_today = df[df.Datetime == df.Datetime.max()].sentiment.values
    sent_yday = df[df.Datetime == df.Datetime.max()-pd.Timedelta(days=1)].sentiment.values

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
                            style={'font-size': '1rem', 'font-color': clr}
                            ), traj
                    ])

    # modal section
    data = globals()[topic]
    data['30-day MA'] = data['sentiment'].rolling(30).mean()
    data['50-day MA'] = data['sentiment'].rolling(50).mean()
    data['200-day MA'] = data['sentiment'].rolling(200).mean()

    sent = go.Scatter(x=data['Datetime'], y=data['sentiment'], name='Observed', visible='legendonly')
    MA_30 = go.Scatter(x=data['Datetime'], y=data['30-day MA'], name='30-day MA')
    MA_50 = go.Scatter(x=data['Datetime'], y=data['50-day MA'], name='50-day MA')
    
    data = [sent, MA_30, MA_50]
    
    fig = go.Figure(data=data)
    fig.layout.update(showlegend=True, hovermode='x unified', template='plotly_white')



    modal_fig = html.Div(
        [
            dbc.Button("Show", id=f"{topic}-button"),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(f'{topic} Sentiment Analysis')),
                    dbc.ModalBody(dcc.Graph(figure=fig)),
                ],
                id=f"{topic}-modal",
                size='xl'
                # fullscreen=True,
            ),
        ]
    )

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    # html.Div([
                    #     html.P([
                            html.H5([
                                f'{topic.replace("_", " ")} ',
                                html.I(className="fa fa-dollar", 
                                    style={'font-size': '1rem'}),
                        #         ]),
                        # ])
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
    # 'text-decoration': 'underline'
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

# topics = dbc.Card(
#     dbc.CardBody(
topics = [
            topic_data('bonds'),
            topic_data('economy'),
            topic_data('recession'),
            topic_data('unemployment'),
            topic_data('interest_rates'),
            topic_data('cryptocurrency')
            ]
     #   ])#, color='red', outline=True
    # style={"width": "75rem"},
#





table = html.Div(
    dash_table.DataTable(
        id="table",
        columns=[{"name": i, "id": i, "deletable": True} for i in df.columns],
        data=df.to_dict("records"),
        page_size=10,
        editable=True,
        cell_selectable=True,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        row_selectable="multi",
    ),
    className="dbc-row-selectable",
)


dropdown = html.Div(
    [
        dbc.Label("Select Company", style={'top': '-20px'}),
        dcc.Dropdown(
            options=tweets_df['Stock Name'].unique(),
            value='All',
            id="company",
            clearable=False,
            style={'position': 'relative', 'top': '-10px', 'left': '-20px'}
        ),
    ],
    className="mb-4",
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

# slider = html.Div(
#     [
#         dbc.Label("Select Years"),
#         dcc.RangeSlider(
#             stocks_df.Date.dt.year.min(),
#             stocks_df.Date.dt.year.max(),
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

controls = dbc.Card(
    [dropdown],#, checklist],#, slider],
    body=True,
)


wordCloud_bull = dbc.Card(
    
    [
        # dbc.CardImg(id='wordcloud1', top=True),
        # dbc.CardBody(
        #     html.P("Bullish Sentiment Word Cloud", className='card-text')
        # ),
        html.Img(id="wordcloud"),
    ]
)

