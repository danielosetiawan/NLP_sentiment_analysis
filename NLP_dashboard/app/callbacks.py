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
import base64
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS

# callback for tab1
@callback(
    Output("line-chart", "figure"),
    Output("table", "data"),
    Input("company", "value"),
    Input("analysis", "value"),
    Input("years", "value"),
)

def update_line_chart(company, analysis, yrs):
    if analysis == [] or company == 'All':
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
        template='plotly_white',
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

# callback for word cloud
@callback(
    [Output("wordcloud1", "src"),
     Output("wordcloud2", "src")],
    [Input("company", "value")]
)
def make_wordcloud(company):

    # filter by company
    df_comp = cleaned_df[cleaned_df['company'] == company]
    # filter by sentiments
    df_comp_bull = df_comp[df_comp['sentiment'] == 1]
    df_comp_bear = df_comp[df_comp['sentiment'] == 0] 
    text_bull = ' '.join(i for i in df_comp_bull['body'])
    text_bear = ' '.join(i for i in df_comp_bear['body'])

    wc = WordCloud(
        width=500,
        height=500,
        max_words=200,
        background_color="white",
        colormap="plasma",
        stopwords = set(STOPWORDS),
    )
    img1 = BytesIO()
    img2 = BytesIO()
    wc.generate(text_bull).to_image().save(img1, format="png")
    wc.generate(text_bear).to_image().save(img2, format="png")

    return "data:image/png;base64,{}".format(base64.b64encode(img1.getvalue()).decode()), "data:image/png;base64,{}".format(base64.b64encode(img2.getvalue()).decode()),