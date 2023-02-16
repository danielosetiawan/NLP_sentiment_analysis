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
from globals import *
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url


# stylesheet with the .dbc class
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc_css])

header = html.H4(
    "Sentiments NLP", className="bg-primary text-white p-2 mb-2 text-center"
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


colors = html.Div(
    [dbc.Button(f"{color}", color=f"{color}", size="sm") for color in theme_colors]
)
colors = html.Div(["Theme Colors:", colors], className="mt-2")



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
tab3 = dbc.Tab(label="Table", children=[
    table
#     html.Div(
#     [html.Button("Download Text", id="btn_txt"), dcc.Download(id="download-text-index")]
# )
])
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

# callback for tab3
# @app.callback(Output("download-text-index", "data"), Input("btn_txt", "n_clicks"))
# def func(n_clicks):
#     if n_clicks is None:
#         raise dash.exceptions.PreventUpdate
#     else:
#         return dict(content="Hello world!", filename="hello.txt")

# callback for tab2
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

# callback for tab1
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