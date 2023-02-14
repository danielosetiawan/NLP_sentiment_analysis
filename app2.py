import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

header = html.H4(
    "Sentiments NLP", className="bg-primary text-white p-2 mb-2 text-center"
)

header

if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
