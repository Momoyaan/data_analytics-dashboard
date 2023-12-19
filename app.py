import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html
import dash_table

external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/rippleui@1.12.1/dist/css/styles.css",
]

external_script = [
    "https://tailwindcss.com/",
    {"src": "https://cdn.tailwindcss.com"},
]
app = dash.Dash(
    __name__,
    external_scripts=external_script,
    external_stylesheets=external_stylesheets,
)

app.scripts.config.serve_locally = True

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

Sidebar = html.Div(className="flex h-screen flex-col justify-between", style={"backgroundColor": "#09090b"}, children=[
    html.Div(className="px-4 py-6", children=[
        html.Span(className="grid h-10 w-32 place-content-center rounded-lg text-lg text-white jbm-bold",
                  style={"backgroundColor": "#09090b"}, children="Sixteen"),
        html.Ul(className="mt-6 space-y-1", children=[
            html.Li(children=[
                dcc.Link(
                    "Dashboard", href="", className="btn btn-ghost px-4 py-2 text-sm font-medium text-white"),
                dcc.Link(
                    "Table", href="#target-table", className="scroll-link btn btn-ghost px-4 py-2 text-sm font-medium text-white")
            ]),
            # ... add more list items here ...
        ])
    ]),
])

Table = html.Div(className="flex w-full overflow-x-auto", id="target-table", children=[
    html.Table([
        html.Thead([
            html.Tr([
                html.Th("Type"),
                html.Th("Where"),
                html.Th("Description"),
                html.Th("Amount"),
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Th("1"),
                html.Td("Cy Ganderton"),
                html.Td("Quality Control Specialist"),
                html.Td("Blue"),
            ]),
            html.Tr([
                html.Th("2"),
                html.Td("Hart Hagerty"),
                html.Td("Desktop Support Technician"),
                html.Td("Purple"),
            ]),
            html.Tr([
                html.Th("3"),
                html.Td("Brice Swyre"),
                html.Td("Tax Accountant"),
                html.Td("Red"),
            ]),
            html.Tr([
                html.Th("3"),
                html.Td("Brice Swyre"),
                html.Td("Tax Accountant"),
                html.Td("Red"),
            ]),
            html.Tr([
                html.Th("3"),
                html.Td("Brice Swyre"),
                html.Td("Tax Accountant"),
                html.Td("Red"),
            ]),
        ])
    ], className="table-hover table")
])

app.layout = html.Div(
    html.Div(
        className="flex h-screen overflow-hidden",
        children=[
            Sidebar,
            html.Div(
                className="flex0 relative flex h-full w-full flex-col overflow-y-auto overflow-x-hidden",
                style={"backgroundColor": "#09090b"},
                children=[
                    html.Div(
                        className="navbar sticky top-0 z-50 w-full",
                        children=[
                            html.Div(
                                'Dashboard',
                                className="btn btn-ghost text-xl text-white font-bold",
                            ),
                        ]
                    ),
                    html.Main(
                        className="my-animation-class max-w-screen-2xl p-4 md:p-6 2xl:p-10",
                        style={"backgroundColor": "#09090b"},
                        children=[
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
                            html.Div(Table, className="mt-8")
                        ]
                    )
                ]
            ),
        ]
    )
)


if __name__ == "__main__":
    app.run_server(debug=True)
