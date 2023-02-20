import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    #Output("table", "data"),
    Input("company", "value"),
    Input("analysis", "value"),
    Input("years", "value"),
)

def update_line_chart(company, analysis, yrs):
    # if analysis == [] or company == 'All':
    #     return {}, []

    # if company == 'All':
    #     # change this when you're done with testing
    #     df = globals()['AAPL'][globals()['AAPL'].Date.dt.year.between(yrs[0], yrs[1])]
    # else:
    #     df = globals()[company][globals()[company].Date.dt.year.between(yrs[0], yrs[1])]

    # data = df.to_dict("records")

    df_comp = combined_df[combined_df['company'] == company]
    df_comp = df_comp.groupby('date', as_index=False)['sentiment'].mean()
    df_comp['SMA30'] = df_comp['sentiment'].rolling(30).mean()
    df_comp['SMA90'] = df_comp ['sentiment'].rolling(90).mean()
    df_comp = df_comp.rename(columns={'date': 'Date'})

    stock_comp = stocks_df[stocks_df['Stock Name'] == company]
    stock_comp['Date'] = pd.to_datetime(stock_comp['Date'])
    stock_comp['Date'] = stock_comp['Date'].dt.date
    combined = pd.merge(df_comp, stock_comp, how='left', on="Date")
    #data = combined.to_dict("records")
    # Create figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1, specs=[[{"secondary_y": True}], 
                                               [{'rowspan': 1}],
                                               [{'rowspan': 1}]], vertical_spacing=0.2,)

    # Add traces
    fig.add_trace(
        go.Candlestick(x=combined['Date'],
                    open=combined['Open'],
                    high=combined['High'],
                    low=combined['Low'],
                    close=combined['Close'],
                    name="Stock Candlestick Price"),
        secondary_y=True,
    )
    
    combined1 = combined.copy()

    # split data into chunks where averages cross each other
    combined['label'] = np.where(combined['SMA30']>combined['SMA90'], 1, 0)
    combined['group'] = combined['label'].ne(combined['label'].shift()).cumsum()
    combined2 = combined.groupby('group')
    combined_s = []
    for name, data in combined2:
        combined_s.append(data)

    # custom function to set fill color
    def fillcol(label):
        if label >= 1:
            return 'rgba(0,250,0,0.4)'
        else:
            return 'rgba(250,0,0,0.4)'

    for df in combined_s:
        fig.add_traces(go.Scatter(x=df.Date, y = df.SMA30,
                                line = dict(color='rgba(0,0,0,0)')))
        
        fig.add_traces(go.Scatter(x=df.Date, y = df.SMA90,
                                line = dict(color='rgba(0,0,0,0)'),
                                fill='tonexty', 
                                fillcolor = fillcol(df['label'].iloc[0])))

    # include averages
    fig.add_traces(go.Scatter(x=combined1.Date, y = combined1.SMA30,
                            line = dict(color = 'green', width=1), name='MA30'))

    fig.add_traces(go.Scatter(x=combined1.Date, y = combined1.SMA90,
                            line = dict(color = 'red', width=1), name='MA90'))
    df2 = combined_df[combined_df['company'] == company]
    comp_group = df2.groupby(by=["date", "sentiment"], as_index=False).agg(
        count_col=pd.NamedAgg(column="sentiment", aggfunc="count"))

    trace1 = go.Bar(
        x = comp_group['date'],
        y = comp_group['count_col'],
        name=1,
        marker_color='green',
        marker_line_width=0)
    trace2 = go.Bar(
        x = comp_group['date'],
        y = comp_group['count_col'],
        name=-1,
        marker_color='red',
        marker_line_width=0)
    fig.add_traces([trace1, trace2], rows=2, cols=1)
    fig.update_layout(barmode = 'stack')

    stock_vol = go.Bar(
        x = combined['Date'],
        y = combined['Volume'],
        marker_color='blue')
    fig.add_trace(stock_vol, row=3, col=1)

    # Set title
    fig.layout.update(title=f'{company} Stock Price v. Sentiment',
                     height=600, width=850, showlegend=True, hovermode='closest')

    # Set x-axis title
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    # Set y-axes titles
    fig.update_yaxes(title_text="Stock Price", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Stock Sentiment", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Volume", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Stock Volume", secondary_y=False, row=3, col=1)
    
    fig.update_layout(
    width=1250,
    height=1000,
    showlegend=False,
    template='plotly_white',
    legend=dict(
    x=0,
    y=1.05,
    traceorder="normal",
    font=dict(
        family="sans-serif",
        size=12,
        color="black"
    )))
    return fig



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

