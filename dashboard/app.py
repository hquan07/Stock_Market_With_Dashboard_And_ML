import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from utils.paths import TRANSFORMED

# ================= Import Model Outputs =================
from Model.ml_models_1 import OUTPUT_GB_LOF, OUTPUT_CLUSTER, OUTPUT_ARIMA

# ================= Load Base Data =================
df = pd.read_csv(TRANSFORMED)
df['trade_date'] = pd.to_datetime(df['trade_date'])
tickers = df['ticker'].unique()

# Load ML results
df_ml, df_clusters, df_forecasts = None, None, None
if os.path.exists(OUTPUT_GB_LOF):
    df_ml = pd.read_csv(OUTPUT_GB_LOF)
    if 'trade_date' in df_ml.columns:
        df_ml['trade_date'] = pd.to_datetime(df_ml['trade_date'])

if os.path.exists(OUTPUT_CLUSTER):
    df_clusters = pd.read_csv(OUTPUT_CLUSTER)

if os.path.exists(OUTPUT_ARIMA):
    df_forecasts = pd.read_csv(OUTPUT_ARIMA)
    if 'date' in df_forecasts.columns:
        df_forecasts['date'] = pd.to_datetime(df_forecasts['date'])

# ================= Initialize App =================
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])
app.title = "Stock Dashboard"

GRAPH_TEMPLATE = go.layout.Template(
    layout=dict(
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font_color="white"
    )
)

# ================= Layout Components =================
header = html.Div([
    html.H1([
        html.I(className="fa-solid fa-chart-line me-2"),
        "Stock Market Dashboard"
    ], style={"fontFamily": "Segoe UI, Helvetica", "fontWeight": "300", "color": "#4DB8FF"}),
    html.P("Visualize financial data and machine learning insights",
           className="lead", style={"color": "#CCCCCC"})
], className="text-center my-4")

controls_card = dbc.Card([
    dbc.CardHeader(html.H4("Market Filters",
                           className="m-0 text-center",
                           style={"color": "#4DB8FF"})),
    dbc.CardBody([
        dbc.Label("Select Ticker(s)", html_for="ticker-dropdown"),
        dcc.Dropdown(
            id="ticker-dropdown",
            options=[{"label": t, "value": t} for t in tickers],
            value=[],
            multi=True,
            placeholder="Select tickers...",
        ),
        html.Br(),

        dbc.Label("Select Date Range", html_for="date-picker"),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=df['trade_date'].min(),
            end_date=df['trade_date'].max(),
            display_format='YYYY-MM-DD',
            style={"width": "100%"}
        ),
        html.Br(),

        dbc.Label("Top N Stocks by Volume", html_for="top-n"),
        dbc.Input(
            id='top-n',
            type='number',
            value=5,
            min=1,
            step=1,
        ),

        html.Hr(),
        html.Div([
            html.H6("Note:", style={"color": "#33CCCC"}),
            html.Ul([
                html.Li("Price Overview: Historical close price trends."),
                html.Li("Moving Averages: Short & long-term trend indicators."),
                html.Li("Correlation Matrix: Relationship between stock returns."),
                html.Li("Portfolio Allocation: Distribution of stock values."),
                html.Li("Gradient Boosting Prediction: Forecasted price movements."),
                html.Li("Anomaly Detection (LOF): Identify unusual price behaviors."),
                html.Li("Stock Clustering (GMM): Grouping stocks by return & risk."),
                html.Li("ARIMA/SARIMA Forecasting: Time-series predictions."),
            ], style={"fontSize": "12px", "color": "#CCCCCC", "paddingLeft": "20px"})
        ])
    ])
], className="h-100")

main_tabs = dbc.Tabs([
    dbc.Tab(dcc.Graph(id="line-chart"), label="Price Overview"),
    dbc.Tab(dcc.Graph(id="ma-chart"), label="Moving Averages"),
    dbc.Tab(dcc.Graph(id="correlation-chart"), label="Correlation Matrix"),
    dbc.Tab(dcc.Graph(id="portfolio-pie-chart"), label="Portfolio Allocation"),
])

ml_tabs = dbc.Tabs([
    dbc.Tab(
        dbc.Card(dbc.CardBody([
            dcc.Dropdown(id="gb-ticker", options=[{"label": t, "value": t} for t in tickers],
                         placeholder="Select a ticker"),
            dcc.Graph(id="gb-graph")
        ])), label="Gradient Boosting Prediction"
    ),
    dbc.Tab(
        dbc.Card(dbc.CardBody([
            dcc.Dropdown(id="lof-ticker", options=[{"label": t, "value": t} for t in tickers],
                         placeholder="Select a ticker"),
            dcc.Graph(id="lof-graph")
        ])), label="Anomaly Detection (LOF)"
    ),
    dbc.Tab(dbc.Card(dbc.CardBody([dcc.Graph(id="cluster-graph")])), label="Stock Clustering (GMM)"),
    dbc.Tab(
        dbc.Card(dbc.CardBody([
            dcc.Dropdown(id="forecast-ticker",
                         options=[{"label": t, "value": t} for t in df_forecasts['ticker'].unique()] if df_forecasts is not None else [],
                         placeholder="Select a ticker"),
            dcc.Graph(id="forecast-graph")
        ])), label="Forecasting (ARIMA/SARIMA)"
    ) if df_forecasts is not None else None,
])

app.layout = dbc.Container([
    html.Div([
        dcc.Markdown(
            """
            <style>
            .Select__control, .Select__menu, .Select__single-value, .Select__multi-value__label {
                color: black !important;
            }
            </style>
            """,
            dangerously_allow_html=True
        )
    ]),

    header,
    dbc.Row([
        dbc.Col(controls_card, md=3),
        dbc.Col([
            html.H3("Market Analysis",
                    className="mb-3 text-center",
                    style={"color": "#4DB8FF", "fontWeight": "500"}),
            dbc.Card(dbc.CardBody(main_tabs)),
            html.H3("Machine Learning Insights",
                    className="mt-4 mb-3 text-center",
                    style={"color": "#4DB8FF", "fontWeight": "500"}),
            dbc.Card(dbc.CardBody([ml_tabs]))
        ], md=9),
    ])
], fluid=True, className="dbc")

# ================= Callbacks =================
# Market Analysis
@app.callback(
    Output("line-chart", "figure"),
    Output("ma-chart", "figure"),
    Output("correlation-chart", "figure"),
    Output("portfolio-pie-chart", "figure"),
    Input("ticker-dropdown", "value"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
    Input("top-n", "value")
)
def update_charts(selected_tickers, start_date, end_date, top_n):
    if not selected_tickers:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Please select at least one ticker to show charts.",
                                 xref="paper", yref="paper", showarrow=False,
                                 font=dict(color="white", size=14))
        empty_fig.update_layout(template=GRAPH_TEMPLATE)
        return empty_fig, empty_fig, empty_fig, empty_fig

    df_sel = df[df['ticker'].isin(selected_tickers)].copy()
    df_sel = df_sel[(df_sel['trade_date'] >= pd.to_datetime(start_date)) & (df_sel['trade_date'] <= pd.to_datetime(end_date))]

    if df_sel.empty:
        msg_fig = go.Figure()
        msg_fig.add_annotation(text="No data for selected tickers / date range.",
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(color="white", size=14))
        msg_fig.update_layout(template=GRAPH_TEMPLATE)
        return msg_fig, msg_fig, msg_fig, msg_fig

    latest_date = df_sel['trade_date'].max()
    latest_volume = df_sel[df_sel['trade_date'] == latest_date].nlargest(top_n, 'volume')
    top_tickers = latest_volume['ticker'].tolist()

    unique_tickers = list(df_sel['ticker'].unique())
    palette = px.colors.qualitative.Plotly
    colors = (palette * ((len(unique_tickers) // len(palette)) + 1))[:len(unique_tickers)]
    color_map = dict(zip(unique_tickers, colors))

    # Line chart
    fig_line = go.Figure()
    for t in unique_tickers:
        df_t = df_sel[df_sel['ticker'] == t]
        fig_line.add_trace(go.Scatter(x=df_t['trade_date'], y=df_t['close'],
                                      mode="lines", name=t, line=dict(color=color_map[t])))
    fig_line.update_layout(title="Close Price Line Chart", template=GRAPH_TEMPLATE, hovermode="x unified")

    # Moving Average
    fig_ma = go.Figure()
    for t in unique_tickers:
        df_t = df_sel[df_sel['ticker'] == t].copy()
        df_t['MA20'] = df_t['close'].rolling(20).mean()
        df_t['MA50'] = df_t['close'].rolling(50).mean()
        color = color_map[t]
        fig_ma.add_trace(go.Scatter(x=df_t['trade_date'], y=df_t['MA20'], mode='lines',
                                    name=f"{t} MA20", line=dict(color=color)))
        fig_ma.add_trace(go.Scatter(x=df_t['trade_date'], y=df_t['MA50'], mode='lines',
                                    name=f"{t} MA50", line=dict(color=color, dash="dash")))
    fig_ma.update_layout(title="Moving Averages", template=GRAPH_TEMPLATE, hovermode="x unified")

    # Correlation Matrix
    df_pivot = df_sel.pivot(index='trade_date', columns='ticker', values='close')
    corr = df_pivot.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis',
                         title="Correlation Matrix")
    fig_corr.update_layout(template=GRAPH_TEMPLATE)

    # Portfolio Pie Chart
    latest = df_sel[df_sel['trade_date'] == latest_date]
    latest = latest[latest['ticker'].isin(top_tickers)]
    fig_pie = px.pie(
        latest,
        names='ticker',
        values='close',
        title="Portfolio Allocation (Top N by Volume)",
        color='ticker',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pie.update_traces(marker=dict(line=dict(color='#1E1E1E', width=2)))
    fig_pie.update_layout(template=GRAPH_TEMPLATE)

    return fig_line, fig_ma, fig_corr, fig_pie

# Machine Learning Callbacks
@app.callback(Output("gb-graph", "figure"), Input("gb-ticker", "value"))
def update_gb_graph(ticker):
    if df_ml is None or not ticker:
        fig = go.Figure()
        fig.add_annotation(text="No ML results available.", xref="paper", yref="paper", showarrow=False,
                           font=dict(color="white", size=14))
        fig.update_layout(template=GRAPH_TEMPLATE)
        return fig
    dff = df_ml[df_ml['ticker'] == ticker]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff['trade_date'], y=dff['close'],
                             mode="lines", name="Close", line=dict(color="cyan")))
    if 'gb_pred' in dff.columns:
        fig.add_trace(go.Scatter(x=dff['trade_date'], y=dff['gb_pred'],
                                 mode="lines", name="GB Prediction", line=dict(color="orange")))
    fig.update_layout(title=f"Gradient Boosting Prediction - {ticker}", template=GRAPH_TEMPLATE)
    return fig

@app.callback(Output("lof-graph", "figure"), Input("lof-ticker", "value"))
def update_lof_graph(ticker):
    if df_ml is None or not ticker:
        fig = go.Figure()
        fig.add_annotation(text="No anomaly results available.", xref="paper", yref="paper", showarrow=False,
                           font=dict(color="white", size=14))
        fig.update_layout(template=GRAPH_TEMPLATE)
        return fig
    dff = df_ml[df_ml['ticker'] == ticker]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff['trade_date'], y=dff['close'], mode="lines", name="Close"))
    if 'anomaly' in dff.columns:
        anomalies = dff[dff['anomaly'] == -1]
        fig.add_trace(go.Scatter(x=anomalies['trade_date'], y=anomalies['close'],
                                 mode="markers", name="Anomaly",
                                 marker=dict(color="red", size=10, symbol="x")))
    fig.update_layout(title=f"LOF Anomalies - {ticker}", template=GRAPH_TEMPLATE)
    return fig

@app.callback(Output("cluster-graph", "figure"), Input("ticker-dropdown", "value"))
def update_cluster_graph(_):
    if df_clusters is None:
        fig = go.Figure()
        fig.add_annotation(text="Cluster results not available.", xref="paper", yref="paper", showarrow=False,
                           font=dict(color="white", size=14))
        fig.update_layout(template=GRAPH_TEMPLATE)
        return fig
    fig = px.scatter(df_clusters, x="volatility", y="return",
                     size="volume", color="cluster", hover_name="ticker",
                     title="GMM Clustering of Stocks")
    fig.update_layout(template=GRAPH_TEMPLATE)
    return fig

@app.callback(Output("forecast-graph", "figure"), Input("forecast-ticker", "value"))
def update_forecast_graph(ticker):
    if df_forecasts is None or not ticker:
        fig = go.Figure()
        fig.add_annotation(text="No forecast data available.", xref="paper", yref="paper", showarrow=False,
                           font=dict(color="white", size=14))
        fig.update_layout(template=GRAPH_TEMPLATE)
        return fig
    dff = df_forecasts[df_forecasts['ticker'] == ticker]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff['date'], y=dff['actual'], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=dff['date'], y=dff['forecast'], mode="lines", name="Forecast"))
    fig.update_layout(title=f"ARIMA/SARIMA Forecast - {ticker}", template=GRAPH_TEMPLATE)
    return fig

# ================= Main =================
if __name__ == "__main__":
    app.run(debug=True)