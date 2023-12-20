import dash
from dash import Dash, dcc, html, Input, Output, callback
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
dash.register_page(__name__, path='/')

data = pd.read_csv("assets/dataset.csv")
X = data[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
          'tenure', 'InternetService', 'OnlineSecurity']]
y = data['Churn']
X = pd.get_dummies(X, columns=['gender', 'Partner',
                   'Dependents', 'InternetService', 'OnlineSecurity'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

colorscale = [[0, 'red'], [0.5, 'blue'], [1, 'green']]

logreg_fig = ff.create_annotated_heatmap(cm, colorscale=colorscale,
                                         x=['Predicted 0', 'Predicted 1'],
                                         y=['Actual 0', 'Actual 1'])

logreg.plot = html.Div([
    dcc.Graph(figure=logreg_fig)
])

logreg_fig.update_layout(
    paper_bgcolor='rgba(0, 0, 0 , 0)',
    plot_bgcolor='rgba(0, 0, 0 , 0)',
    title='Confusion Matrix',
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

accuracy = accuracy_score(y_test, y_pred)

accuracy_fig = go.Figure(
    data=[go.Bar(x=['Accuracy'], y=[accuracy])],
    layout=go.Layout(
        title='Model Accuracy',
        font=dict(
            family="JetBrainsMono, sans-serif",
            size=12,
            color="white"
        ),
        paper_bgcolor='rgba(0, 0, 0 , 0)',
        plot_bgcolor='rgba(0, 0, 0 , 0)',
    )
)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

classification_fig = go.Figure(
    data=[go.Table(
        header=dict(values=list(report_df.columns),
                    fill_color='rgba(32, 32, 32 , 1)',
                    align='left'),
        cells=dict(values=[report_df[k].tolist() for k in report_df.columns],
                   fill_color='rgba(0, 0, 0 , 0)',
                   align='left'))
          ],
    layout=go.Layout(
        title='Classification Report',
        font=dict(
            family="JetBrainsMono, sans-serif",
            size=12,
            color="white"
        ),
        paper_bgcolor='rgba(0, 0, 0 , 0)',
        plot_bgcolor='rgba(0, 0, 0 , 0)',
    )
)
y_test_roc = y_test.map({'No': 0, 'Yes': 1})
y_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_roc, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

roc_fig = go.Figure(
    data=[
        go.Scatter(x=fpr, y=tpr, mode='lines',
                   name='ROC curve (area = %0.2f)' % roc_auc),
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                   name='Random', line=dict(dash='dash')),
    ],
    layout=go.Layout(
        title='Receiver Operating Characteristic',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        paper_bgcolor='rgba(0, 0, 0 , 0)',
        plot_bgcolor='rgba(0, 0, 0 , 0)',
        font=dict(
            family="JetBrainsMono, sans-serif",
            size=10,
            color="white"
        ),
    )
)

layout = html.Div([
    html.Div(
        className="grid grid-cols-1 gap-4 lg:grid-cols-3 lg:gap-8",
        children=[
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="logreg-graph", figure=logreg_fig),
            ),
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="accuracy-graph", figure=accuracy_fig),
            ),
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="example-graph2", figure=classification_fig),
            ),
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="example-graph2", figure=roc_fig),
            ),
        ],
    ),
])
