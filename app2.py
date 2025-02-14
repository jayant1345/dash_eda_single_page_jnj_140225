import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import base64
import io

# Initialize Dash App with Dark Mode and Suppressing Callback Exceptions
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Automated EDA Web App", className="text-center text-primary"), width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Upload(
            id="upload-data",
            children=html.Div(["📂 Drag & Drop or ", html.A("Select a File")]),
            style={
                "width": "100%", "height": "60px", "lineHeight": "60px",
                "borderWidth": "2px", "borderStyle": "dashed",
                "borderRadius": "10px", "textAlign": "center", "margin-bottom": "20px"
            },
            multiple=False
        ), width=12),
    ]),

    dbc.Row([
        dbc.Col(dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Dataset Overview", value="overview"),
            dcc.Tab(label="Correlation Heatmap", value="correlation"),
            dcc.Tab(label="Boxplots & Outliers", value="boxplots"),
            dcc.Tab(label="Missing Values", value="missing"),
            dcc.Tab(label="Duplicate Records", value="duplicates"),
        ]), width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(html.Div(id="tab-content"), width=12)
    ]),

    # Boxplot Dropdown & Graph (Always in Layout, Hidden Initially)
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="boxplot-column", placeholder="Select a Numeric Column", style={"display": "none"}), width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="boxplot-graph", style={"display": "none"}), width=12)
    ])
], fluid=True)


# Helper Function to Read Uploaded File
def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df


# Callback to Update Tab Content & Boxplot Dropdown
@app.callback(
    [Output("tab-content", "children"), Output("boxplot-column", "options"), Output("boxplot-column", "style")],
    [Input("tabs", "value"), Input("upload-data", "contents")]
)
def render_tab(selected_tab, contents):
    if contents is None:
        return html.Div("Please upload a CSV file to begin EDA.", className="alert alert-info text-center"), [], {"display": "none"}

    df = parse_contents(contents)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    column_options = [{"label": col, "value": col} for col in num_cols]

    if selected_tab == "overview":
        return html.Div([
            html.H3("Dataset Overview"),
            dash_table.DataTable(
                data=df.head().to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={"overflowX": "auto","color": "lightgray"},
                style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold","color": "lightgray"},
                page_size=5
            ),
            html.H5(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")
        ]), [], {"display": "none"}

    elif selected_tab == "correlation":
        corr_matrix = df[num_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        return dcc.Graph(figure=fig), [], {"display": "none"}

    elif selected_tab == "boxplots":
        return html.Div([html.H3("Boxplots for Outlier Detection")]), column_options, {"display": "block"}

    elif selected_tab == "missing":
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ["Feature", "Missing Values"]
        fig = px.bar(missing_data, x="Feature", y="Missing Values", color="Missing Values", title="Missing Values Count")
        return dcc.Graph(figure=fig), [], {"display": "none"}

    elif selected_tab == "duplicates":
        duplicate_count = df.duplicated().sum()
        return html.Div([
            html.H3(f"Total Duplicate Records: {duplicate_count}"),
            dash_table.DataTable(
                data=df[df.duplicated()].to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
                page_size=5
            ) if duplicate_count > 0 else html.P("No duplicate records found.", className="text-success")
        ]), [], {"display": "none"}

    return html.Div("Tab Not Found", className="alert alert-danger text-center"), [], {"display": "none"}


# Callback for Boxplots (Outlier Detection)
@app.callback(
    [Output("boxplot-graph", "figure"), Output("boxplot-graph", "style")],
    [Input("boxplot-column", "value"), Input("upload-data", "contents")]
)
def update_boxplot(column, contents):
    if contents is None or column is None:
        return {}, {"display": "none"}

    df = parse_contents(contents)
    
    # Outlier Detection using IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    fig = px.box(df, y=column, title=f"Boxplot for {column}", color_discrete_sequence=["red"])
    fig.add_trace(px.scatter(outliers, y=column).data[0])

    return fig, {"display": "block"}


# Run Server
if __name__ == "__main__":
    app.run_server(debug=True, port=8052)
