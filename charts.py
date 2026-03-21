import plotly.express as px
import plotly.graph_objects as go

BG = "rgba(0,0,0,0)"
FONT = "#E5E7EB"
GRID = "rgba(148,163,184,0.12)"
PRIMARY = "#3B82F6"
PRIMARY_2 = "#60A5FA"
SUCCESS = "#10B981"
RISK = "#EF4444"
WARN = "#F59E0B"
MUTED = "#334155"
CARD = "#111827"


def _font():
    return dict(family="Inter, PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif", color=FONT)


def _base_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=20, family=_font()["family"], color=FONT)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=_font(),
        margin=dict(l=18, r=18, t=56, b=20),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#E5E7EB", bordercolor="rgba(255,255,255,0.08)"),
        showlegend=False,
        bargap=0.32,
        modebar_remove=["zoom", "pan", "select", "lasso2d", "autoScale", "toImage"],
    )
    try:
        fig.update_layout(barcornerradius=10)
    except Exception:
        pass
    return fig


def plot_priority_bar(df):
    d = df.copy().sort_values("priority_index", ascending=True).tail(8)
    top3 = set(d.sort_values("priority_index", ascending=False).head(3)["knowledge_point"])
    d["color"] = d["knowledge_point"].apply(lambda x: RISK if x in top3 else PRIMARY)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["priority_index"],
            y=d["knowledge_point"],
            orientation="h",
            marker=dict(color=d["color"], line=dict(color="rgba(255,255,255,0.12)", width=1)),
            text=[f"{v:.2f}" for v in d["priority_index"]],
            textposition="outside",
            hovertemplate="知识点：%{y}<br>优先级指数：%{x:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, title=None)
    fig.update_yaxes(showgrid=False, title=None)
    return _base_layout(fig, "知识点优先级排序（Top3 高亮）")


def plot_difficulty_accuracy(df):
    d = df.copy().sort_values("accuracy_rate", ascending=True)
    low3 = set(d.head(min(3, len(d)))["difficulty"])
    colors = [RISK if x in low3 else PRIMARY_2 for x in d["difficulty"]]
    fig = go.Figure(
        go.Bar(
            x=d["difficulty"],
            y=d["accuracy_rate"],
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.10)", width=1)),
            text=[f"{v:.1%}" for v in d["accuracy_rate"]],
            textposition="outside",
            hovertemplate="难度：%{x}<br>正确率：%{y:.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_xaxes(showgrid=False)
    return _base_layout(fig, "难度层级 vs 正确率")


def plot_trend_line(df):
    d = df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["label"],
            y=d["accuracy_rate"],
            mode="lines+markers+text",
            line=dict(color=SUCCESS, width=3, shape="spline"),
            marker=dict(size=10, color=SUCCESS, line=dict(color="#ffffff", width=1)),
            text=[f"{v:.0%}" for v in d["accuracy_rate"]],
            textposition="top center",
            hovertemplate="%{x}<br>正确率：%{y:.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor=GRID)
    fig.update_xaxes(showgrid=False)
    return _base_layout(fig, "作答阶段趋势（折线）")


def plot_risk_bar(df):
    d = df.copy().head(10).sort_values("risk_index", ascending=True)
    high3 = set(d.sort_values("risk_index", ascending=False).head(min(3, len(d)))["student_name"])
    d["color"] = d["student_name"].apply(lambda x: RISK if x in high3 else WARN)
    fig = go.Figure(
        go.Bar(
            x=d["risk_index"],
            y=d["student_name"],
            orientation="h",
            marker=dict(color=d["color"], line=dict(color="rgba(255,255,255,0.12)", width=1)),
            text=[f"{x:.1f}" for x in d["risk_index"]],
            textposition="outside",
            hovertemplate="学生：%{y}<br>风险指数：%{x:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _base_layout(fig, "学生风险指数排序（Top3 高亮）")


def plot_risk_donut(df):
    counts = df["risk_level"].value_counts().reindex(["高风险", "波动层", "稳定层"]).fillna(0)
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.62,
            marker=dict(colors=[RISK, WARN, SUCCESS], line=dict(color="#0F172A", width=2)),
            textinfo="label+percent",
            textfont=dict(color=FONT, size=13),
            hovertemplate="层级：%{label}<br>人数：%{value}<extra></extra>",
        )
    )
    fig.update_layout(
        annotations=[dict(text="学生分层", x=0.5, y=0.5, font=dict(size=18, color=FONT), showarrow=False)],
    )
    return _base_layout(fig, "风险学生分层（环形）")


def plot_teaching_gain(df):
    d = df.copy().sort_values("teaching_gain", ascending=True).tail(8)
    top3 = set(d.sort_values("teaching_gain", ascending=False).head(min(3, len(d)))["knowledge_point"])
    colors = [SUCCESS if x in top3 else PRIMARY for x in d["knowledge_point"]]
    fig = go.Figure(
        go.Bar(
            x=d["teaching_gain"],
            y=d["knowledge_point"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.12)", width=1)),
            text=[f"{x:.2f}" for x in d["teaching_gain"]],
            textposition="outside",
            hovertemplate="知识点：%{y}<br>教学收益预测：%{x:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _base_layout(fig, "教学收益预测（Top3 高亮）")
