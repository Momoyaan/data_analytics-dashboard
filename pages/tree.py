import dash
from dash import dcc, html
import pandas as pd
import dash_ag_grid as dag

dash.register_page(__name__, path='/tree')


layout = html.Div(
    html.Div(
        html.Img(
            className="rounded-xl border border-neutral-800 w-full",
            style={"height": "800px"},
            id="tree-graph",
            src="../assets/tree.svg"
        )
    )
),
