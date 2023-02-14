"""
****** Important! *******
If you run this app locally, un-comment line 113 to add the ThemeChangerAIO component to the layout
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
import dash_bootstrap_components as dbc
from sentiment_prediction import checkSenti
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url

df = px.data.gapminder()

sent_df = pd.read_csv('./data/balanced_tokenized_cleaned_stocktwits.csv', 
                 parse_dates=['created_at']).drop('body', axis=1)

stocks_df = pd.read_csv('./data/scraped_stock_2015_2022.csv', 
                        parse_dates=['Date']).iloc[:, 1:]


years = df.year.unique()
continents = df.continent.unique()

# find all indexes with tickers
df = sent_df['raw_content'].str.upper().str.extractall(r'\$(\w+)')[0].reset_index()

# remove indexes containing more than one ticker
df = sent_df.drop(df[df['match'] == 1]['level_0'].tolist())

# add ticker 
df['ticker'] = df['raw_content'].str.upper().str.extract(r'\$(\w+)')

# extract date only
df['Date'] = df['created_at'].dt.date

def clean_tickers(ticker):
    if 'AAPL' in ticker:
        ticker = 'AAPL'
    elif 'AMZN' in ticker:
        ticker = 'AMZN'
    elif 'FB' in ticker:
        ticker = 'META'
    else:
        ticker = ticker
        
    return ticker

df['ticker'] = df['ticker'].apply(lambda x: clean_tickers(x))

# get average sentiment per day
df = df.groupby(['ticker', 'Date']).agg({'sentiment': 'mean'}).reset_index()
df['Date'] = pd.to_datetime(df['Date'])

for tick in df['ticker'].unique():
    df1 = df[df['ticker'] == tick]
    df2 = stocks_df[stocks_df['Stock Name'] == tick]
    
    globals()[tick] = pd.merge(df1, df2, on='Date').iloc[:, 1:]
    globals()[tick]['color'] = globals()[tick]['sentiment'].apply(lambda x: 'green' if x > 0.5 else 'red')
    globals()[tick]['label'] = globals()[tick]['sentiment'].apply(lambda x: 'Bullish' if x > 0.5 else 'Bearish')

# stylesheet with the .dbc class
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc_css])

header = html.H4(
    "Sentimentsdklfjsklfjs NLP", className="bg-primary text-white p-2 mb-2 text-center"
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
            "All",
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


controls = dbc.Card(
    [dropdown, checklist, slider],
    body=True,
)

tab1 = dbc.Tab(label="Stocks", children=[
            dcc.Graph(id='line-chart'),
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
tab3 = dbc.Tab([table], label="Table", className="p-4")
tabs = dbc.Card(dbc.Tabs([tab1, tab2, tab3]))

app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(
                    [
                        controls,
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # When running this app locally, un-comment this line:
                        ThemeChangerAIO(aio_id="theme")
                    ],
                    width=4,
                ),
                dbc.Col([tabs, colors], width=8),
            ]
        ),
    ],
    fluid=True,
    className="dbc",
)

@callback(
    Output('sentiment-prediction-output', 'children'),
    Input('sentiment-prediction-button', 'n_clicks'),
    State('sentiment-prediction-text', 'value')
)
def update_output(n_clicks, value):
    # if n_clicks > 0:
    prediction = checkSenti(value)
    color = 'success' if prediction[0] == 'Bullish' else 'danger'
    value2 = f"Based on our models, this text is {round(prediction[1]*100, 2)}% likely to be "

    return dbc.Alert([value2, html.B(prediction[0], className="alert-heading"), '.'], color=color),

@callback(
    Output("line-chart", "figure"),
    # Output("sentiment-prediction", "figure"),
    Output("table", "data"),
    Input("company", "value"),
    Input("analysis", "value"),
    Input("years", "value"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)

def update_line_chart(company, analysis, yrs, theme):
    if analysis == [] or company is 'All':
        return {}, []

    if company == 'All':
        # change this when you're done with testing
        df = globals()['AAPL'][globals()['AAPL'].Date.dt.year.between(yrs[0], yrs[1])]
    else:
        df = globals()[company][globals()[company].Date.dt.year.between(yrs[0], yrs[1])]

    data = df.to_dict("records")

    fig = tools.make_subplots(
        rows=3, cols=1,
        specs=[[{'rowspan': 2}],
            [None],
            [{'rowspan': 1}]],
        vertical_spacing=0.05)

    stock = go.Scatter(x=df['Date'], y=df['Adj Close'], name="Adj. Close")
    MA30 = go.Scatter(x=df['Date'], y=df['High'].rolling(window=30).mean(), name="30 day MA")
    MA50 = go.Scatter(x=df['Date'], y=df['High'].rolling(window=50).mean(), name="50 day MA")
    sentiment = go.Bar(x=df['Date'], y=df['sentiment'], name="Sentiment", marker=dict(color=df['color'], line=dict(width=0)), showlegend=False)
    
    fig.append_trace(stock, row=1, col=1)
    fig.append_trace(MA30, row=1, col=1)
    fig.append_trace(MA50, row=1, col=1)
    fig.append_trace(sentiment, row=3, col=1)

    fig.update_yaxes(title_text='Stock Price', row=1, col=1)
    fig.update_yaxes(title_text='Sentiment', row=3, col=1)
    fig.update_yaxes(tickmode='array',
                 tickvals=[0, 0.5, 1],
                 row=3, col=1)

    fig.layout.update(title=f'{company} Stock Price v. Sentiment',
                     height=600, width=850, showlegend=True, hovermode='closest')

    fig.update_layout(
        template=template_from_url(theme),
        hovermode='x unified', 
        legend=dict(
        x=0,
        y=1.05,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        )))
    fig.update_traces(xaxis='x1')
    
    return fig, data


if __name__ == "__main__":
    app.run_server(debug=True)