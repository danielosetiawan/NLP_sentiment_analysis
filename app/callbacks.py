import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import tools
from PIL import Image

import dash_bootstrap_components as dbc
from sentiment_prediction import checkSenti
from globals import *
from content import *
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import base64
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS


# @callback(
#     Output('inflation-chart', 'figure'),
#     []
# )



# callback for tab1
@callback(
    Output("line-chart", "figure"),
    #Output("table", "data"),
    Input("company", "value"),
    # Input("analysis", "value"),
    # Input("years", "value"),
)

def update_line_chart(company):

    if company == 'All':
        # change this when you're done with testing
        company = 'AAPL'

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
    fig = make_subplots(rows=4, cols=1, specs=[[{"secondary_y": True, 'rowspan': 2}], 
                                               [None],
                                               [{'rowspan': 1}],
                                               [{'rowspan': 1}]], vertical_spacing=0.05)

    # Add traces
    fig.add_trace(
        go.Candlestick(x=combined['Date'],
                    open=combined['Open'],
                    high=combined['High'],
                    low=combined['Low'],
                    close=combined['Close'],
                    name=""),
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
                                line = dict(color='rgba(0,0,0,0)'),
                                hoverinfo='skip'
                                ))
        
        fig.add_traces(go.Scatter(x=df.Date, y = df.SMA90,
                                line = dict(color='rgba(0,0,0,0)'),
                                fill='tonexty', 
                                fillcolor = fillcol(df['label'].iloc[0]),
                                hoverinfo='skip'
                                ))

    # include averages
    fig.add_traces(go.Scatter(x=combined1.Date, y = combined1.SMA30,
                            line = dict(color = 'green', width=1), 
                            name='MA30', hoverinfo='skip'
                            ))

    fig.add_traces(go.Scatter(x=combined1.Date, y = combined1.SMA90,
                            line = dict(color = 'red', width=1), 
                            name='MA90', hoverinfo='skip'
                            ))
    df2 = combined_df[combined_df['company'] == company]
    comp_group = df2.groupby(by=["date", "sentiment"], as_index=False).agg(
        count_col=pd.NamedAgg(column="sentiment", aggfunc="count"))

    # subplot 1: sentiment volume
    trace1 = go.Bar(
        x = comp_group['date'],
        y = comp_group[comp_group['sentiment'] == 1]['count_col'],
        name='Bullish',
        marker_color='green',
        marker_line_width=0
        )
    trace2 = go.Bar(
        x = comp_group['date'],
        y = comp_group[comp_group['sentiment'] == -1]['count_col'],
        name='Bearish',
        marker_color='red',
        marker_line_width=0
        )
    fig.add_traces([trace1, trace2], rows=3, cols=1)
    fig.update_layout(barmode = 'stack')

    # subplot 2: stock volume
    stock_vol = go.Bar(
        x = combined['Date'],
        y = combined['Volume'],
        name = 'Volume',
        marker_color='blue')
    fig.add_trace(stock_vol, row=4, col=1)

    # Set title
    fig.layout.update(title=f'{company} Stock Price v. Sentiment',
                     showlegend=True, hovermode='closest')

    # Set x-axis title
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    # Set y-axes titles
    fig.update_yaxes(title_text="Stock Price", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Stock Sentiment", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Volume", secondary_y=False, row=3, col=1)
    fig.update_yaxes(title_font=dict(size=10), secondary_y=False, row=3, col=1)
    fig.update_yaxes(title_text="Stock Volume", secondary_y=False, row=4, col=1)
    fig.update_yaxes(title_font=dict(size=10), secondary_y=False, row=4, col=1)

    # set y-axes subplots to display only min/max
    # fig.update_layout(yaxis=dict(tickmode='linear', nticks=2, 
    #     range=[min(combined['Volume']), max(combined['Volume'])], row=4, col=1))
    
    fig.update_layout(
    # width=500,
    # height=850,
    showlegend=False,
    hovermode='x unified', 
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
    fig.update_traces(xaxis='x1')
    return fig



# callback for tab2
@callback(
    Output('sentiment-prediction-output', 'children'),
    Input('sentiment-prediction-button', 'n_clicks'),
    State('sentiment-prediction-button', 'n_clicks')
)
def update_output(n_clicks, value):
    # if n_clicks > 0:
    prediction = checkSenti(value)
    color = 'success' if prediction[0] == 'Bullish' else 'danger'
    value2 = f"Based on our models, this text is {round(prediction[1]*100, 2)}% likely to be "

    return dbc.Alert([value2, html.B(prediction[0], className="alert-heading"), '.'], color=color),


# callback for word cloud
@callback(
    Output('wordcloud', 'src'), 
    Input('company', 'value')
)
def plot_wordcloud(company):
    if company == 'All':
        # change this when you're done with testing
        company = 'AAPL'

    df_comp = cleaned_df[cleaned_df['company'] == company]
    # filter by sentiments
    df_comp_bull = df_comp[df_comp['sentiment'] == 1]
    df_comp_bear = df_comp[df_comp['sentiment'] == 0] 
    text_bull = ' '.join(i for i in df_comp_bull['body'])
    text_bear = ' '.join(i for i in df_comp_bear['body'])

    # Load the two images and convert them to numpy arrays
    mask1 = np.array(Image.open('../logos/bull.png'))
    mask2 = np.array(Image.open('../logos/bear.png'))

    # Create the WordCloud objects with the masks
    wc1 = WordCloud(background_color='white', mask=mask1)
    wc2 = WordCloud(background_color='white', mask=mask2)

    # Generate the word clouds
    wc1.generate(text_bull)
    wc2.generate(text_bear)

    # Combine the two images side by side
    combined_width = mask1.shape[1] + mask2.shape[1]
    combined_height = max(mask1.shape[0], mask2.shape[0])

    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
    combined_image.paste(Image.fromarray(wc1.to_array()), (0, 0))
    combined_image.paste(Image.fromarray(wc2.to_array()), (mask1.shape[1], 0))

    img = BytesIO()
    combined_image.save(img, format='PNG')

    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


@callback(
    Output("bonds-modal", "is_open"),
    Input("bonds-button", "n_clicks"),
    State("bonds-modal", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open
    
@callback(
    Output("cryptocurrency-modal", "is_open"),
    Input("cryptocurrency-button", "n_clicks"),
    State("cryptocurrency-modal", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output("economy-modal", "is_open"),
    Input("economy-button", "n_clicks"),
    State("economy-modal", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output("interest_rates-modal", "is_open"),
    Input("interest_rates-button", "n_clicks"),
    State("interest_rates-modal", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output("recession-modal", "is_open"),
    Input("recession-button", "n_clicks"),
    State("recession-modal", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output("unemployment-modal", "is_open"),
    Input("unemployment-button", "n_clicks"),
    State("unemployment-modal", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open