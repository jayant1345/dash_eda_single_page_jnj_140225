import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import base64
import io

# Initialize Dash App with Dark Theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# App Layout
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
        dbc.Col(dbc.Button("Toggle Dark Mode 🌙", id="dark-mode-toggle", color="secondary"), width=3),
        dbc.Col(dcc.Dropdown(id="column-filter", placeholder="Select a Column for Analysis"), width=6),
    ], className="mb-3"),

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
    ])
], fluid=True)


# Helper Function to Read Uploaded File
def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df


# Callbacks
@app.callback(
    [Output("tab-content", "children"), Output("column-filter", "options")],
    [Input("tabs", "value"), Input("upload-data", "contents")]
)
def render_tab(selected_tab, contents):
    if contents is None:
        return html.Div("Please upload a CSV file to begin EDA.", className="alert alert-info text-center"), []

    df = parse_contents(contents)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    column_options = [{"label": col, "value": col} for col in df.columns]

    if selected_tab == "overview":
        return html.Div([
            html.H3("Dataset Overview"),
            dash_table.DataTable(
                data=df.head().to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
                page_size=5
            ),
            html.H5(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}"),
            html.P(f"Numerical Columns: {len(num_cols)}"),
            html.P(f"Categorical Columns: {len(cat_cols)}")
        ]), column_options

    elif selected_tab == "correlation":
        corr_matrix = df[num_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        return dcc.Graph(figure=fig), column_options

    elif selected_tab == "boxplots":
        return html.Div([
            html.H3("Boxplots for Outlier Detection"),
            dcc.Dropdown(id="boxplot-column", options=column_options, placeholder="Select a Numeric Column"),
            dcc.Graph(id="boxplot-graph")
        ]), column_options

    elif selected_tab == "missing":
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ["Feature", "Missing Values"]
        fig = px.bar(missing_data, x="Feature", y="Missing Values", color="Missing Values", title="Missing Values Count")
        return dcc.Graph(figure=fig), column_options

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
        ]), column_options

    return html.Div("Tab Not Found", className="alert alert-danger text-center"), column_options


# Callback for Boxplots (Outlier Detection)
@app.callback(
    Output("boxplot-graph", "figure"),
    [Input("boxplot-column", "value"), Input("upload-data", "contents")]
)
def update_boxplot(column, contents):
    if contents is None or column is None:
        return {}

    df = parse_contents(contents)
    fig = px.box(df, y=column, title=f"Boxplot for {column}", color_discrete_sequence=["red"])
    return fig


# Dark Mode Toggle
@app.callback(
    Output("dark-mode-toggle", "color"),
    Input("dark-mode-toggle", "n_clicks"),
    prevent_initial_call=True
)
def toggle_dark_mode(n):
    return "dark" if n % 2 == 1 else "secondary"


# Run Server
if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
