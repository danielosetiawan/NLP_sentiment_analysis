from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

text = "hello world this is a sample text for creating a wordcloud in dash"

wordcloud = WordCloud().generate(text)
buffer = BytesIO()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig(buffer, format="png")
buffer.seek(0)
image = base64.b64encode(buffer.getvalue()).decode()

app = Dash(
    prevent_initial_callbacks=True, suppress_callback_exceptions=True
    # external_stylesheets=[dbc.themes.LUX]
)


app.layout = html.Div(
    [
        html.Img(
            src=f"data:image/png;base64,{image}",
            style={"height": "50vh", "display": "block", "margin": "auto"},
        )
    ]
)
if __name__ == "__main__":
    app.run_server(debug=True)
