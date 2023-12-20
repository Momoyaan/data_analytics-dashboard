import dash
from dash import Dash, dcc, html, Input, Output, callback
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd

dash.register_page(__name__, path='/regression')

telecom_cust = pd.read_csv("assets/dataset.csv")
telecom_cust.TotalCharges = pd.to_numeric(
    telecom_cust.TotalCharges, errors='coerce')
telecom_cust.isnull().sum()
telecom_cust.dropna(inplace=True)
df2 = telecom_cust.iloc[:, 1:]

df2['Churn'] = df2['Churn'].map({'No': 0, 'Yes': 1})
df_dummies = pd.get_dummies(df2)
y = df_dummies['Churn'].values
X = df_dummies.drop(columns=['Churn'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)

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
y_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
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
        autosize=True,
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

multiple_regression = LinearRegression()
multiple_regression.fit(X_train, y_train)
coefficients = multiple_regression.coef_
feature_names = X_train.columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
coef_fig = go.Figure(
    data=[go.Bar(x=coef_df['Feature'], y=coef_df['Coefficient'])],
    layout=go.Layout(
        title='Regression Coefficients',
        xaxis=dict(title='Feature'),
        yaxis=dict(title='Coefficient'),
        autosize=True,
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


layout = html.Div(
    className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3 lg:gap-8 mt-4",
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
                id="classification-graph", figure=classification_fig),
        ),
        html.Div(
            dcc.Graph(
                className="rounded-xl border border-neutral-800",
                id="roc-graph", figure=roc_fig),
        ),
        html.Div(
            dcc.Graph(
                className="rounded-xl border border-neutral-800",
                id="coef-graph", figure=coef_fig),
        ),
    ]
),
