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
        
    data = tweets_df[tweets_df['Stock Name'] == company]

    STM = data['sentiment avg'].rolling(50).mean() #short term sentiment
    MTM = data['sentiment avg'].rolling(200).mean() #long term sentiment
    data['SMA30'] = data['sentiment avg'].rolling(50).mean()
    data['SMA90'] = data['sentiment avg'].rolling(200).mean()
    
    data['label'] = np.where(data['SMA30']>data['SMA90'], 1, 0)
    data['group'] = data['label'].ne(data['label'].shift()).cumsum()

    # create subplot layout
    fig = make_subplots(
        rows=4, cols=1, 
        row_heights=[1.5, 1.5, 1.5, 1.5],
        vertical_spacing=0.15,
        specs=[[{"secondary_y": True, 'rowspan': 2}], 
               [None],
               [{'rowspan': 1}],
               [{'rowspan': 1}]]
    )
    
    
######## subplot 1: sentiment v. stock ########

    # subplot 1A: candlestick trace
    fig.add_trace(
        go.Candlestick(x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=""),
        secondary_y=True)
    
    # subplot 1B: grouping colors by trace crosses
    combined1 = data.copy()
    combined = data.groupby('group')
    combined_s = []
    for _, dta in combined:
        combined_s.append(dta)

    # custom function to set fill colors
    def fillcol(label):
        if label >= 1:
            return 'rgba(0,250,0,0.4)'
        else:
            return 'rgba(250,0,0,0.4)'

    # subplot 1B: make moving average lines transparrent
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

    # subplot 1B: add colors for traces that cross MA
    fig.add_traces(go.Scatter(x=combined1.Date, y = combined1.SMA30,
                            line = dict(color = 'green', width=1), 
                            name='MA30', hoverinfo='skip'
                            ))

    fig.add_traces(go.Scatter(x=combined1.Date, y = combined1.SMA90,
                            line = dict(color = 'red', width=1), 
                            name='MA90', hoverinfo='skip'
                            ))

######## subplot 2: sentiment ########

    trace1 = go.Line(
        x = data['Date'],
        y = STM,
        name='Short Term',
        marker_line_width=0,
        marker_color = 'orange',
        )
    trace2 = go.Line(
        x = data['Date'],
        y = MTM,
        name='Long Term',
        marker_line_width=0,
        marker_color = 'blue',
        )

    fig.add_traces([trace1, trace2], rows=3, cols=1)

######## subplot 3: stock volume ########
    stock_vol = go.Bar(
        x = data['Date'],
        y = data['Volume'],
        name = 'Volume',
        marker_color='black',
    )
    fig.add_trace(stock_vol, row=4, col=1)
    
######## subplot layouts ########

    # Set title
    fig.layout.update(title=f'{company} Stock Price v. Sentiment',
                     showlegend=True, hovermode='closest')

    # Set axis title
    fig.update_xaxes(title_text="Date", row=1, col=1)

    fig.update_yaxes(title_text="Stock Price", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Stock Sentiment", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", secondary_y=False, row=3, col=1)
    fig.update_yaxes(title_text="Volume", secondary_y=False, row=4, col=1)

    # hiding the bottom range window
    fig.update_layout(xaxis_rangeslider_visible=False)

    # updating y-axis ranges for the subplot
    company_dct = {
        'AAPL': {'sentiment': [0.6, 0.8], 'volume': [0, 5e8]},
        'AMZN': {'sentiment': [0.65, 0.85], 'volume': [0, 6e8]},
        'CRM': {'sentiment': [0.7, 0.9], 'volume': [0, 1e7]}, 
        'DIS': {'sentiment': [0.5, 0.8], 'volume': [0, 6e7]},
        'GOOG': {'sentiment': [0.6, 0.8], 'volume': [0, 1.5e8]},
        'KO': {'sentiment': [0.6, 0.8], 'volume': [0, 6e7]},
        'MSFT': {'sentiment': [0.6, 0.9], 'volume': [0, 1.5e8]}, 
        'TSLA': {'sentiment': [0.5, 0.8], 'volume': [0, 8e8]},
        'BA': {'sentiment': [0.6, 0.9], 'volume': [0, 1e7]},
        'BX': {'sentiment': [0.6, 0.9], 'volume': [0, 2e7]},
        'NOC': {'sentiment': [0.6, 0.9], 'volume': [0, 6e6]},
        'NFLX': {'sentiment': [0.5, 0.8], 'volume': [0, 5e7]},
        'TSM': {'sentiment': [0.7, 0.9], 'volume': [0, 5e7]},
        'META': {'sentiment': [0.8, 0.95], 'volume': [0, 1.5e8]},
        'PYPL': {'sentiment': [0.6, 0.9], 'volume': [0, 4e7]},
        'PG': {'sentiment': [0.6, 0.8], 'volume': [0, 5e7]},
        'ZS': {'sentiment': [0.6, 0.9], 'volume': [0, 1e7]},
        'NIO': {'sentiment': [0.7, 0.9], 'volume': [0, 6e8]},
    }
    
    try:
        
        fig.update_yaxes(tickmode='array', tickvals=company_dct[company]['sentiment'], row=3, col=1)
        fig.update_yaxes(tickmode='array', tickvals=company_dct[company]['volume'], row=4, col=1)
        fig.update_yaxes(range=company_dct[company]['volume'], secondary_y=False, row=4, col=1)
        
    except:
        pass
    
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
        )),
    )
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


for topic in topics:
    @callback(
        Output(f"{topic}-modal", "is_open"),
        Input(f"{topic}-button", "n_clicks"),
        State(f"{topic}-modal", "is_open"),
    )
    def toggle_modal(n, is_open):
        if n:
            return not is_open
        return is_open