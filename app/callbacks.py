import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import tools
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colors
import colorsys

import dash_bootstrap_components as dbc
from sentiment_prediction import checkSenti
from globals import *
from content import *
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import base64
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS

from statsmodels.tsa.stattools import grangercausalitytests


# callback for tab1
@callback(
    Output("line-chart", "figure"),
    Output("lag-coef-value", "children"),
    Input("company", "value"),
)

def update_line_chart(company):

    if company == 'All':
        # change this when you're done with testing
        company = 'AAPL'
        
    data = tweets_df[tweets_df['Stock Name'] == company].copy()

    data['SMA30'] = data['sentiment avg'].rolling(50).mean()
    data['SMA90'] = data['sentiment avg'].rolling(90).mean()
    
    data['label'] = np.where(data['SMA30']>data['SMA90'], 1, 0)
    data['group'] = data['label'].ne(data['label'].shift()).cumsum()

    # create subplot layout
    fig = make_subplots(
        rows=4, cols=1, 
        row_heights=[4, 3.5, 3.5, 2],
        vertical_spacing=0.05,
        x_title='Date',
        shared_xaxes=True
    )
######## subplot 1: sentiment ########
    # grouping colors by trace crosses
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

    # make moving average lines transparrent
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

    # add colors for traces that cross MA
    fig.add_traces(go.Scatter(x=combined1.Date, y = combined1.SMA30,
                            line = dict(color = 'green', width=1), 
                            name='Short Term'
                            ))

    fig.add_traces(go.Scatter(x=combined1.Date, y = combined1.SMA90,
                            line = dict(color = 'red', width=1), 
                            name='Long term'
                            ))

######## subplot 2: stock ########

    # subplot 2A: stock price
    price_trace = go.Scatter(x=data['Date'],
                y=data['Adj Close'],
                mode='lines',
                marker_line_width=0,
                marker_color = 'blue',
                name="Adj Close")
    
    # subplot 2B: adding buy/sell signals
    def RSIcalc(df):
        df = df.copy() # or else a dreading warning sign will keep popping up
        df['MA200'] = df['Adj Close'].rolling(window=200).mean()
        df['price change'] = df['Adj Close'].pct_change()
        df['Upmove'] = df['price change'].apply(lambda x: x if x>0 else 0)
        df['Downmove'] = df['price change'].apply(lambda x: abs(x) if x<0 else 0)
        df['avg Up'] = df['Upmove'].ewm(span=19).mean()
        df['avg Down'] = df['Downmove'].ewm(span=19).mean()
        df = df.dropna()
        df['RS'] = df['avg Up']/df['avg Down']
        df['RSI'] = df['RS'].apply(lambda x: 100-(100/(x+1)))
        df.loc[(df['Adj Close'] > df['MA200']) & (df['RSI'] < 30), 'Buy'] = 'Yes'
        df.loc[(df['Adj Close'] < df['MA200']) | (df['RSI'] > 30), 'Buy'] = 'No'
        return df
    
    def getSignals(df):
        Buying_dates=[]
        Selling_dates=[]
        
        for i in range(len(df)):
            if "Yes" in df['Buy'].iloc[i]:
                Buying_dates.append(df.iloc[i+1].name)
                for j in range(1, 11):
                    if df['RSI'].iloc[i+j] > 40:
                        Selling_dates.append(df.iloc[i+j+1].name)
                        break
                    elif j == 10:
                        Selling_dates.append(df.iloc[i+j+1].name)
        return Buying_dates, Selling_dates
    
    frame = RSIcalc(data)
    buy, sell = getSignals(frame)
    buy_trace = go.Scatter(x=pd.to_datetime(frame.loc[buy]['Date']), y = frame.loc[buy]['Adj Close'],
                              marker=dict(symbol='triangle-up', color='green'), 
                              mode = 'markers', name='Buy'
                            )
    sell_trace = go.Scatter(x=pd.to_datetime(frame.loc[sell]['Date']), y = frame.loc[sell]['Adj Close'],
                              marker=dict(symbol='triangle-up', color='red'), 
                              mode = 'markers', name='Sell'
                            )
    fig.add_traces([price_trace, buy_trace, sell_trace], rows=2, cols=1)

######## subplot 3: RSI ########
    
    trace1 = go.Scatter(
        x = pd.to_datetime(frame['Date']),
        y = frame['RSI'],
        name='RSI',
        mode='lines',
        marker_line_width=0,
        marker_color = 'orange',
        )
    trace2 = go.Scatter(x = pd.to_datetime(frame['Date']),
        y = np.repeat(30, len(frame['Date'])),
        name='oversold',
        line=dict(color='green', dash='dash'))
    trace3 = go.Scatter(x = pd.to_datetime(frame['Date']),
        y = np.repeat(70, len(frame['Date'])),
        name='overbought',
        line=dict(color='indianred', dash='dash'))
    fig.add_traces([trace1, trace2, trace3], rows=3, cols=1)

######## subplot 4: stock volume ########
    stock_vol = go.Bar(
        x = data['Date'],
        y = data['Volume'],
        name = 'Stock Volume',
        marker_color='rgb(158,202,225)',
        marker_line_width=0,
    )
    
    fig.add_trace(stock_vol, row=4, col=1)
    
    
######## subplot layouts ########

    # Set title
    fig.layout.update(title=f'{company} Stock Price v. Sentiment',
                     showlegend=True, hovermode='closest')

    # Set axis title
    fig.update_yaxes(title_text="Sentiment", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)

    # hiding the bottom range window
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    fig.update_layout(
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
            )
        ),
    )
    fig.update_traces(xaxis='x4')

    # finding the lag coefficient value
    lag_data = data[['Date', 'sentiment avg', 'Close']].groupby(['Date']).mean()
    results = grangercausalitytests(lag_data, maxlag=2, verbose=False)
    corr_coef = 0.98

    granger_coefs = []
    for idx in range(2):
        pval = [results[i+1][0]['ssr_ftest'][idx] for i in range(2)]
        granger_causality_coef = 1 - pval[1] / pval[0]
        granger_coefs.append(granger_causality_coef)

    sub_style = {
        'font-size': '10px', 
        'margin-top': '-30px', 
        'margin-bottom': '100px'
        }
    stats = html.Div([
        html.P([
            f'Correlation Coefficient: {corr_coef}'
        ], style={'margin-bottom': '-9px'}),
        html.I([
            'Association between Sentiment and Stock Price',
        ], style=sub_style),

        html.P([
            f'Granger Causality Coefficient: {round(max(granger_coefs), 2)}'
        ], style={'margin-bottom': '-9px'}),
        html.I([
            'How well the sentiments predict stock prices',
        ], style=sub_style)
    ])


    return fig, stats


# callback for tab2
@callback(
    Output('sentiment-prediction-output', 'children'),
    Input('sentiment-prediction-button', 'n_clicks'),
    State('prediction', 'value')
)
def update_output(n_clicks, value):
    prediction = checkSenti(value)
    color = 'success' if prediction[0] == 'Bullish' else 'danger'
    value = f"Based on our models, this text is {round(prediction[1]*100, 2)}% likely to be "

    return dbc.Alert([value, html.B(prediction[0], className="alert-heading"), '.'], color=color),


# callback for word cloud
@callback(
    Output('wordcloud', 'src'), 
    Input('company', 'value')
)
def plot_wordcloud(company):
    if company == 'All':
        # change this when you're done with testing
        company = 'AAPL'

    text_bull = wordcloud_df[f'{company}_bull'].dropna().to_dict()
    text_bear = wordcloud_df[f'{company}_bear'].dropna().to_dict()

    # Load the two images and convert them to numpy arrays
    mask1 = np.array(Image.open('../logos/bull.png'))
    mask2 = np.array(Image.open('../logos/bear.png'))

    # Create the WordCloud objects with the masks
    wc1 = WordCloud(background_color='white', mask=mask1)
    wc2 = WordCloud(background_color='white', mask=mask2)

    # Generate the word clouds
    wc1.generate_from_frequencies(text_bull)
    wc2.generate_from_frequencies(text_bear)
    
    # defining function for color func
    def hsl_color_func(word, font_size, position, orientation, random_state = None, **kwargs):
        return(hsl_val % np.random.randint(0,100))
    
    # change color for bear
    color = 'xkcd:blood red'
    r,g,b = colors.to_rgb(color)
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    hsl_val = 'hsl(' + str(h*360) + ', 75%%, %d%%)'
    color_bear = hsl_color_func
    wc2.recolor(color_func=color_bear)

    # Combine the two images side by side
    combined_width = mask1.shape[1] + mask2.shape[1]
    combined_height = max(mask1.shape[0], mask2.shape[0])

    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
    combined_image.paste(Image.fromarray(wc1.to_array()), (0, 100))
    combined_image.paste(Image.fromarray(wc2.to_array()), (mask1.shape[1], 0))
    
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype('Arial.ttf', size=50)

    draw.text((700, combined_height-75), 'Bulls', fill='black', font=font)
    draw.text((mask1.shape[1]+750, combined_height-75), 'Bears', fill='black', font=font)

    img = BytesIO()
    combined_image.save(img, format='PNG')

    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    # return combined_image


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
    
@callback(
    Output("intro-modal", "is_open"),
    Input("intro-modal-button", "n_clicks"),
    State("intro-modal", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output('prediction-chart', 'figure'),
    Input('sentiment-prediction-button', 'n_clicks'),
    State('prediction', 'value')
)
def final_sentpredict(n_clicks, text):
    sent_txt = []
    sent_val = []
    sent_clr = []
    rounded = []
    txt_color = []
    
    def predict_color(value, label):
        if label == 'Bearish':
            return 1-value, 'red', 'white'
        else:
            return value, 'green', 'white'
        
    for txt in text.split(' '):
        
        # assigning variable and color
        label, value = checkSenti(txt)
        value, color, color2 = predict_color(value, label)
        txt_color.append(color2)
        
        # splitting up each word and making it proper
        sent_txt.append(txt.capitalize())
        
        # setting the bar colors
        sent_clr.append(color)
        
        # appending value and rounding it
        sent_val.append(value)
        rounded.append(round(float(value), 2))
    
    # finding the weight of the composition
    comp_weight = value - np.mean(sent_val)
    
    # append to list
    sent_txt.append('Composition Weight')
    sent_val.append(comp_weight)
    sent_clr.append('orange')
    txt_color.append('black')
    rounded.append(round(comp_weight, 2))
        
    # create and return df
    df = pd.DataFrame({
        'text': sent_txt, 'value': sent_val, 
        'color': sent_clr, 'rounded': rounded,
        'txt_color': txt_color
    }).sort_values('value', ascending=False)
    
    # plotting
    trace = go.Bar(
        x=df.value,
        y=df.text,
        orientation='h',
        text = [f'{l} | {r}' for l, r in zip(df.text, df.rounded)],
        textfont=dict(
            color=df.txt_color,
            size=10,
            ),
        marker=dict(
            color=df.color
        )
    )

    layout = go.Layout(
        title='Sentiment Weight of each word',
        template='simple_white',
        xaxis=dict(title='Weight'),
        yaxis=dict(ticktext=[], tickvals=[])
    )

    fig = go.Figure(trace, layout)
    
    return fig


# @callback(
#     Output("page-content", "children"), 
#     Input("url", "pathname")
# )
# def render_stocktab(tab):
#     if tab == 'tab1':
#         return tab1_content
#     elif tab == 'tab2':
#         return tab2_content