import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html

dash.register_page(__name__, path='/')

data = pd.read_csv("assets/dataset.csv")

df = pd.DataFrame(
    {
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4.2, 1.0, 2.1, 2.32, 4.20, 5.0],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"],
    }
)

fruit_count = df.Fruit.count()
total_amt = df.Amount.sum()
city_count = df.City.count()
variables = df.shape[1]

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig.update_layout(
    paper_bgcolor='rgba(0, 0, 0 , 0)',
    plot_bgcolor='rgba(0, 0, 0 , 0)',
    title='My Plot Title',
    font=dict(
        family="JetBrainsMono, sans-serif",
        size=12,
        color="white"
    ),
    xaxis=dict(
        titlefont=dict(
            color="grey"
        )
    ),
    yaxis=dict(
        titlefont=dict(
            color="grey"
        )
    ),
    legend=dict(
        font=dict(
            size=10,
            color="grey"),
    )
)


fig1 = px.box(df, x="City", y="Amount", color="City")

fig1.update_layout(
    paper_bgcolor='rgba(0, 0, 0 , 0)',
    plot_bgcolor='rgba(0, 0, 0 , 0)',
    title='My Plot Title',
    font=dict(
        family="JetBrainsMono, sans-serif",
        size=12,
        color="white"
    ),
    xaxis=dict(
        titlefont=dict(
            color="grey"
        )
    ),
    yaxis=dict(
        titlefont=dict(
            color="grey"
        )
    ),
    legend=dict(
        font=dict(
            size=10,
            color="grey"),
    )
)

fig2 = px.scatter(df, x="Fruit", y="Amount", color="City")

fig2.update_layout(
    paper_bgcolor='rgba(0, 0, 0 , 0)',
    plot_bgcolor='rgba(0, 0, 0 , 0)',
    title='Scatter Plot',
    font=dict(
        family="JetBrainsMono, sans-serif",
        size=12,
        color="white"
    ),
    xaxis=dict(
        titlefont=dict(
            color="grey"
        )
    ),
    yaxis=dict(
        titlefont=dict(
            color="grey"
        )
    ),
    legend=dict(
        font=dict(
            size=10,
            color="grey"),
    )
)
layout = html.Div([
    html.Div(
        className="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:gap-8",
        children=[
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="example-graph", figure=fig),
            ),
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="example-graph1", figure=fig1),
            ),
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="example-graph2", figure=fig2),
            ),
        ],
    ),
])
