import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import base64
import io

# Initialize Dash App with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Automated EDA Web App"

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Automated EDA Web App", className="text-center text-primary"), width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select a File")]),
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

    # Hidden storage for uploaded data
    dcc.Store(id="stored-data")
], fluid=True)


# Callback to process uploaded file
@app.callback(
    Output("stored-data", "data"),
    Input("upload-data", "contents")
)
def process_file(contents):
    if not contents:
        return None

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    return df.to_dict("records")


# Callback to render tabs
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value"), Input("stored-data", "data")]
)
def render_tab(selected_tab, data):
    if data is None:
        return html.Div("Please upload a CSV file to begin EDA.", className="alert alert-info text-center")

    df = pd.DataFrame(data)
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns

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
        ])

    elif selected_tab == "correlation":
        if len(num_cols) == 0:
            return html.Div("No numerical columns available for correlation.", className="alert alert-warning text-center")

        corr_matrix = df[num_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        return dcc.Graph(figure=fig)

    elif selected_tab == "boxplots":
        return html.Div([
            html.H3("Boxplots for Outlier Detection"),
            dcc.Dropdown(id="boxplot-column", options=[{"label": col, "value": col} for col in num_cols], value=num_cols[0]),
            dcc.Graph(id="boxplot-graph")
        ])

    elif selected_tab == "missing":
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ["Feature", "Missing Values"]
        fig = px.bar(missing_data, x="Feature", y="Missing Values", color="Missing Values", title="Missing Values Count")
        return dcc.Graph(figure=fig)

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
        ])

    return html.Div("Tab Not Found", className="alert alert-danger text-center")


# Callback for boxplot updates
@app.callback(
    Output("boxplot-graph", "figure"),
    Input("boxplot-column", "value"),
    State("stored-data", "data")
)
def update_boxplot(column, data):
    if data is None or column is None:
        return px.box(title="No data available")

    df = pd.DataFrame(data)
    fig = px.box(df, y=column, title=f"Boxplot of {column}")
    return fig


# Run Server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
