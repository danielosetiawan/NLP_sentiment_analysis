from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from content import *
from globals import *
from callbacks import *


app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.LUX, dbc.icons.FONT_AWESOME, 
        # 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
    ],
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
    # external_stylesheets=[dbc.themes.LUX]
)

server = app.server

# application = app.server

sidebar = html.Div(
    [
        html.Div(
            [
                # width: 3rem ensures the logo is the exact width of the
                # collapsed sidebar (accounting for padding)
                html.Img(src=transform_image('img/sentiment.png'), style={"width": "3rem"}),
                html.H2("NLP Dashboard"),
            ],
            className="sidebar-header",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-home me-2"), html.Span("Home")],
                    href="/",
                    active="exact",
                    
                ),
                
                dbc.NavLink(
                    [
                        html.I(className="fas fa-database me-2"),
                        html.Span("Data"),
                    ],
                    href="/data",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-question me-2"),
                        html.Span("About"),
                    ],
                    href="/about",
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

content = html.Div(id="page-content", className="content")

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# set the content according to the current pathname
@callback(
    Output("page-content", "children"), 
    Input("url", "pathname")
)
def render_page_content(pathname):
    if pathname == "/":
        return app_layout #dbc.Col([tabs, colors], width=8)
    elif pathname == "/data":
        return table
    elif pathname == "/about":
        return jumbotron
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == '__main__':
    # application.run(debug=True, host='0.0.0.0', port=8080)
    app.run_server(debug=False, host='0.0.0.0', port=8080)