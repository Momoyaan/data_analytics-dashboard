import dash
from dash import Dash, dcc, html, Input, Output, callback
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
dash.register_page(__name__, path='/')

telecom_cust_full = pd.read_csv("assets/dataset.csv")
telecom_cust = telecom_cust_full.sample(frac=0.1, random_state=42)
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

gender_counts = (telecom_cust['gender'].value_counts()
                 * 100.0 / len(telecom_cust))

# Create bar chart
colors = ['#4D3425', '#E4512B']
gender_bar = go.Figure(
    data=[go.Bar(
        x=gender_counts.index,
        y=gender_counts.values,
        text=[f"{val:.1f}%" for val in gender_counts.values],
        textposition='auto',
        marker_color=colors
    )],
    layout=go.Layout(
        title='Gender Distribution',
        xaxis=dict(title='Gender'),
        yaxis=dict(title='% Customers'),
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
# Calculate percentage of Senior Citizens
senior_counts = (
    telecom_cust['SeniorCitizen'].value_counts() * 100.0 / len(telecom_cust))

# Create pie chart
senior_pie = go.Figure(
    data=[go.Pie(
        labels=['No', 'Yes'],
        values=senior_counts.values,
        textinfo='label+percent',
        insidetextorientation='radial'
    )],
    layout=go.Layout(
        title='% of Senior Citizens',
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

# Calculate percentage of Dependents and Partners
df2 = pd.melt(telecom_cust, id_vars=['customerID'], value_vars=[
              'Dependents', 'Partner'])
df3 = df2.groupby(['variable', 'value']).count().unstack()
df3 = df3*100/len(telecom_cust)

# Create stacked bar chart
dep_part_bar = go.Figure(
    data=[
        go.Bar(
            name='No',
            x=df3.index,
            y=df3[('customerID', 'No')],
            text=[f"{val:.0f}%" for val in df3[('customerID', 'No')]],
            textposition='auto',
            marker_color='#eb6f92'
        ),
        go.Bar(
            name='Yes',
            x=df3.index,
            y=df3[('customerID', 'Yes')],
            text=[f"{val:.0f}%" for val in df3[('customerID', 'Yes')]],
            textposition='auto',
            marker_color='#f6c177'
        )
    ],
    layout=go.Layout(
        title='% Customers with dependents and partners',
        xaxis=dict(title=''),
        yaxis=dict(title='% Customers'),
        barmode='stack',
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

# Calculate percentage of Dependents and Partners based on whether they have a partner
partner_dependents = telecom_cust.groupby(
    ['Partner', 'Dependents']).size().unstack()
partner_dependents_percent = (
    partner_dependents.T * 100.0 / partner_dependents.T.sum()).T

# Create stacked bar chart
partner_dep_bar = go.Figure(
    data=[
        go.Bar(
            name='No',
            x=partner_dependents_percent.index,
            y=partner_dependents_percent['No'],
            text=[f"{val:.0f}%" for val in partner_dependents_percent['No']],
            textposition='auto',
            marker_color='#ebbcba'
        ),
        go.Bar(
            name='Yes',
            x=partner_dependents_percent.index,
            y=partner_dependents_percent['Yes'],
            text=[f"{val:.0f}%" for val in partner_dependents_percent['Yes']],
            textposition='auto',
            marker_color='#31748f'
        )
    ],
    layout=go.Layout(
        title='% Customers with/without dependents based on whether they have a partner',
        xaxis=dict(title='Partner'),
        yaxis=dict(title='% Customers'),
        barmode='stack',
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

services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Create subplot with 3 rows and 3 columns
services_fig = make_subplots(rows=3, cols=3)

# Loop through services and add bar charts to subplot
for i, item in enumerate(services):
    service_counts = telecom_cust[item].value_counts()
    row = i // 3 + 1
    col = i % 3 + 1
    services_fig.add_trace(
        go.Bar(
            x=service_counts.index,
            y=service_counts.values,
            name=item,
            text=[f"{val}" for val in service_counts.values],
            textposition='auto',
        ),
        row=row,
        col=col
    )

# Update layout
services_fig.update_layout(
    title_text="Service Usage",
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

# Create scatter plot
charges_scatter = go.Figure(
    data=[
        go.Scatter(
            x=telecom_cust['MonthlyCharges'],
            y=telecom_cust['TotalCharges'],
            mode='markers',
            marker=dict(
                # set color to an array/list of desired values
                color=telecom_cust['MonthlyCharges'],
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            )
        )
    ],
    layout=go.Layout(
        title='Total Charges vs Monthly Charges',
        xaxis=dict(title='Monthly Charges'),
        yaxis=dict(title='Total Charges'),
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

not_churn = telecom_cust.MonthlyCharges[(telecom_cust["Churn"] == 'No')]
churn = telecom_cust.MonthlyCharges[(telecom_cust["Churn"] == 'Yes')]

hist_data = [not_churn, churn]
group_labels = ['Not Churn', 'Churn']

dist_fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

dist_fig.update_layout(
    title_text='Distribution of monthly charges by churn',
    xaxis_title="Monthly Charges",
    yaxis_title="Density",
    paper_bgcolor='rgba(0, 0, 0 , 0)',
    plot_bgcolor='rgba(0, 0, 0 , 0)',
    font=dict(
        family="JetBrainsMono, sans-serif",
        size=10,
        color="white"
    ),
)

not_churn_total = telecom_cust.TotalCharges[(telecom_cust["Churn"] == 'No')]
churn_total = telecom_cust.TotalCharges[(telecom_cust["Churn"] == 'Yes')]

hist_data_total = [not_churn_total, churn_total]
group_labels_total = ['Not Churn', 'Churn']

fig_total = ff.create_distplot(
    hist_data_total, group_labels_total, bin_size=.2)

fig_total.update_layout(
    title_text='Distribution of total charges by churn',
    xaxis_title="Total Charges",
    yaxis_title="Density",
    paper_bgcolor='rgba(0, 0, 0 , 0)',
    plot_bgcolor='rgba(0, 0, 0 , 0)',
    font=dict(
        family="JetBrainsMono, sans-serif",
        size=10,
        color="white"
    ),
)


layout = html.Div([
    html.H1("A complete study of the Telco Customer Churn dataset is presented in this project. The purpose of this analysis is to identify the elements that contribute to customer churn in the telecoms industry. This study analyzes customer demographics, service usage patterns, and billing information in order to discover important indications of customer turnover. The research makes use of advanced data analytics and predictive modeling approaches. When it comes to improving service delivery and client retention strategies, the insights that were gathered provide significant recommendations for companies that provide telecommunications services. ", className="font-bold mb-8"),
    html.Div(
        className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4 lg:gap-8",
        children=[
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="gender-graph", figure=gender_bar),
            ),
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="senior-graph", figure=senior_pie),
            ),
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="dep-graph", figure=dep_part_bar),
            ),
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="partner-dep-graph", figure=partner_dep_bar),
            ),
        ],
    ),
    html.Div(
        className="grid grid-cols-1 gap-4 lg:gap-8 mt-4",
        children=[
            html.Div(
                dcc.Graph(
                  className="rounded-xl border border-neutral-800",
                  id="services-graph", figure=services_fig),
            ),
        ],
    ),
    html.Div(
        className="grid grid-cols-1 gap-4 lg:gap-8 mt-4",
        children=[
            html.Div(
                dcc.Graph(
                  className="rounded-xl border border-neutral-800",
                  id="charges-graph", figure=charges_scatter),
            ),
        ],
    ),
    html.Div(
        className="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:gap-8 mt-4",
        children=[
            html.Div(
                dcc.Graph(
                  className="rounded-xl border border-neutral-800",
                  id="dist-graph", figure=dist_fig),
            ),
            html.Div(
                dcc.Graph(
                    className="rounded-xl border border-neutral-800",
                    id="hist-graph", figure=fig_total),
            ),
        ],
    ),
])
