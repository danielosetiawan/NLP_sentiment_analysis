"""
****** Important! *******
If you run this app locally, un-comment line 113 to add the ThemeChangerAIO component to the layout
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dash import Dash, dcc, html, dash_table, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
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

# get average sentiment per day
df = df.groupby(['ticker', 'Date']).agg({'sentiment': 'mean'}).reset_index()
df['Date'] = pd.to_datetime(df['Date'])

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

for tick in df['ticker'].unique():
    df1 = df[df['ticker'] == tick]
    df2 = stocks_df[stocks_df['Stock Name'] == tick]
    
    globals()[tick] = pd.merge(df1, df2, on='Date').iloc[:, 1:]

# stylesheet with the .dbc class
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])

header = html.H4(
    "Sentimentsdfsdf NLP", className="bg-primary text-white p-2 mb-2 text-center"
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
        dbc.Label("Select Continents"),
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

tab1 = dbc.Tab(label="Line Chart", children=[
            dcc.Graph(id='line-chart'),
            dcc.Graph(id='scatter-chart')
            ])
# tab2 = dbc.Tab([dcc.Graph(id="scatter-chart")], label="Scatter Chart")
tab3 = dbc.Tab([table], label="Table", className="p-4")
tabs = dbc.Card(dbc.Tabs([tab1, tab3]))

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
    Output("line-chart", "figure"),
    Output("scatter-chart", "figure"),
    Output("table", "data"),
    Input("company", "value"),
    Input("analysis", "value"),
    Input("years", "value"),
    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
)

def update_line_chart(company, analysis, yrs, theme):
    if analysis == [] or company == 'All':
        return {}, {}, []

    dff = stocks_df[stocks_df.Date.dt.year.between(yrs[0], yrs[1])]
    # dff = dff[dff.continent.isin(continent)]
    data = dff.to_dict("records")

    if company == 'All':
        fig_data = stocks_df
    else:
        fig_data = stocks_df[stocks_df['Stock Name'] == company]

    fig = px.line(
        fig_data,
        x='Date',
        y='High',
        # name = 'Original Price',
        title = f'{company} Stock',
        template=template_from_url(theme),
    )

    fig30 = fig_data['High'].rolling(window=30).mean()
    fig50 = fig_data['High'].rolling(window=50).mean()
    # maybe add a for loop here?
    fig.add_trace(go.Scatter(x=fig_data['Date'], y=fig30, name='30 day MA'))
    fig.add_trace(go.Scatter(x=fig_data['Date'], y=fig50, name='50 day MA'))


    fig_scatter = px.line(
        stocks_df,
        x=stocks_df.index,
        y='High',
        height=300,
        # color="continent",
        # line_group="country",
        template=template_from_url(theme),
    )
    # px.scatter(
    #     df.query(f"year=={yrs[1]} & continent=={continent}"),
    #     x="gdpPercap",
    #     y="lifeExp",
    #     size="pop",
    #     color="continent",
    #     log_x=True,
    #     size_max=60,
    #     template=template_from_url(theme),
    #     title="Gapminder %s: %s theme" % (yrs[1], template_from_url(theme)),
    # )

    return fig, fig_scatter, data


if __name__ == "__main__":
    app.run_server(debug=True)