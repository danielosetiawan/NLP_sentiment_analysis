from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
from data import *
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url


header = html.H4(
    "Sentiments NLP", className="bg-primary text-white p-2 mb-2 text-center"
)

nav_menu = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="NavbarSimple",
    brand_href="#",
    color="primary",
    dark=True,
)


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
        dbc.Label("Select Company"),
        dcc.Dropdown(
            df['ticker'].unique(),
            'All',
            id="company",
            clearable=False,
        ),
    ],
    className="mb-4",
)

checklist = html.Div(
    [
        dbc.Label("Select Analysis"),
        dbc.Checklist(
            id="analysis",
            options=stocks_df.columns,
            value=stocks_df.columns,
            inline=True,
        ),
    ],
    className="mb-4",
)

slider = html.Div(
    [
        dbc.Label("Select Years"),
        dcc.RangeSlider(
            stocks_df.Date.dt.year.min(),
            stocks_df.Date.dt.year.max(),
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

controls = dbc.Card(
    [dropdown, checklist, slider],
    body=True,
)

theme_colors = [
    "primary",
    "secondary",
    "success",
    "warning",
    "danger",
    "info",
    "light",
    "dark",
    "link",
]

colors = html.Div(
    [dbc.Button(f"{color}", color=f"{color}", size="sm") for color in theme_colors]
)
colors = html.Div(["Theme Colors:", colors], className="mt-2")

tab1 = dbc.Tab(label="Stocks", children=[
            dcc.Graph(id='line-chart', style={'width': '80vh', 'height': 1100}),
            # dcc.Graph(id='scatter-chart')
            ])
tab2 = dbc.Tab(html.Div([
    dcc.Textarea(
        id='sentiment-prediction-text',
        value="Hell yeahhh ~ I'm feeling extra bullish today.\nLooks like them stock prices are shooting to the moon 😍",
        style={'width': '100%', 'height': 200},
    ),
    html.Button('Classify Text', id='sentiment-prediction-button', n_clicks=0),
    html.Div(id='sentiment-prediction-output', style={'whiteSpace': 'pre-line'})
]), label="Sentiment Prediction")
tab3 = dbc.Tab(label="Table", children=[
    table
#     html.Div(
#     [html.Button("Download Text", id="btn_txt"), dcc.Download(id="download-text-index")]
# )
])
tabs = dbc.Card(dbc.Tabs([tab1, tab2, tab3]))

wordCloud_bull = dbc.Card(
    
    [
        dbc.CardImg(id='wordcloud1', top=True),
        dbc.CardBody(
            html.P("Bullish Sentiment Word Cloud", className='card-text')
        ),
    ]
)

wordCloud_bear = dbc.Card(
    [
        dbc.CardImg(id='wordcloud2', top=True),
        dbc.CardBody(
            html.P("Bearish Sentiment Word Cloud", className='card-text')
        ),
    ]
)


app_layout = dbc.Container(
    [
        
        dbc.Row(
            [
                # dbc.Col(
                #     [nav_menu], width=9
                # ),
                dbc.Col(
                    
                    [tabs, colors], width=9
                    ),
                dbc.Col(
                    [
                        controls,
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # When running this app locally, un-comment this line:
                        ThemeChangerAIO(aio_id="theme"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [wordCloud_bull, wordCloud_bear]
                                ),
                            ]
                        ),
                    ],
                    width=3,
                ),
                # dbc.Col([tabs, colors], width=8),
            ]
        ),
        
        
    ],
    fluid=True,
    className="dbc",
)