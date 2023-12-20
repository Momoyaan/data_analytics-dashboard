import dash
from dash import dcc, html
import pandas as pd
import dash_ag_grid as dag

dash.register_page(__name__, path='/table')

data = pd.read_csv("assets/dataset.csv")

datatable = html.Div(className="flex w-full overflow-x-auto", style={'width': '1300px', 'overflow': 'auto'}, children=[
    dag.AgGrid(
        className="ag-theme-material",
        id='my_aggrid',
        rowData=data.to_dict('records'),
        columnDefs=[{'headerName': col, 'field': col} for col in data.columns],
        defaultColDef={"sortable": True, "filter": True},
        columnSize="autoSize",
    )
])

layout = html.Div([
    datatable
])
