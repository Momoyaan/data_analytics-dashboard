import dash
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Output, Input, State

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
    use_pages=True,
)

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

Sidebar = html.Div(className="flex h-screen flex-col justify-between", style={"backgroundColor": "#09090b"}, children=[
    html.Div(className="px-4 py-6", children=[
        html.Span(className="grid h-10 w-32 place-content-center rounded-lg text-lg text-white jbm-bold",
                  style={"backgroundColor": "#09090b"}, children="Sixteen"),
        html.Ul(className="mt-6 space-y-1", children=[
            html.Li(children=[
                dcc.Link(
                    "Dashboard", href="/", className="btn btn-ghost px-4 py-2 text-sm font-medium text-white"),
                dcc.Link(
                    "Table", href="/table", id="table-button", className="btn btn-ghost px-4 py-2 text-sm font-medium text-white")
            ]),
            # ... add more list items here ...
        ])
    ]),
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
                    html.Main(
                        style={"backgroundColor": "#09090b"},
                        className="my-animation-class mx-auto max-w-screen-2xl p-4 md:p-6 2xl:p-10",
                        children=[
                            dash.page_container
                        ]
                    )
                ]
            ),
        ]
    )
)


if __name__ == "__main__":
    app.run_server(debug=True)
