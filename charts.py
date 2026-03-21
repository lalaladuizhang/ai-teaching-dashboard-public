import plotly.express as px
import plotly.graph_objects as go

PRIMARY = "#3B82F6"
SUCCESS = "#10B981"
RISK = "#EF4444"
MUTED = "#475569"
BG = "rgba(0,0,0,0)"
FONT = "#E5E7EB"


def _base_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=18, color=FONT)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FONT),
        margin=dict(l=20, r=20, t=55, b=20),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#E5E7EB"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(showgrid=False)
    return fig


def plot_priority_bar(df):
    d = df.copy().sort_values("priority_index", ascending=True)
    top3 = set(df.head(3)["knowledge_point"].tolist())
    d["color"] = d["knowledge_point"].apply(lambda x: RISK if x in top3 else PRIMARY)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["priority_index"],
            y=d["knowledge_point"],
            orientation="h",
            marker=dict(color=d["color"], line=dict(color="rgba(255,255,255,0.08)", width=1)),
            text=[f"{v:.2f}" for v in d["priority_index"]],
            textposition="outside",
            hovertemplate="知识点：%{y}<br>优先级指数：%{x:.2f}<extra></extra>",
        )
    )
    return _base_layout(fig, "知识点优先级排序（Top3 高亮）")


def plot_difficulty_accuracy(df):
    d = df.copy().sort_values("accuracy_rate", ascending=True)
    top3 = set(d.head(min(3, len(d)))["difficulty"].tolist())
    d["color"] = d["difficulty"].apply(lambda x: RISK if x in top3 else PRIMARY)
    fig = go.Figure(
        go.Bar(
            x=d["difficulty"],
            y=d["accuracy_rate"],
            marker=dict(color=d["color"], line=dict(color="rgba(255,255,255,0.08)", width=1)),
            text=[f"{v:.1%}" for v in d["accuracy_rate"]],
            textposition="outside",
            hovertemplate="难度：%{x}<br>正确率：%{y:.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return _base_layout(fig, "难度层级 vs 正确率（低正确率高亮）")


def plot_risk_bar(df):
    d = df.head(10).copy().sort_values("risk_index", ascending=True)
    top3 = set(df.head(3)["student_name"].tolist())
    d["color"] = d["student_name"].apply(lambda x: RISK if x in top3 else PRIMARY)
    fig = go.Figure(
        go.Bar(
            x=d["risk_index"],
            y=d["student_name"],
            orientation="h",
            marker=dict(color=d["color"]),
            text=[f"{v:.1f}" for v in d["risk_index"]],
            textposition="outside",
            hovertemplate="学生：%{y}<br>风险指数：%{x:.1f}<extra></extra>",
        )
    )
    return _base_layout(fig, "高风险学生排序（Top3 高亮）")


def plot_teaching_gain(df):
    d = df.head(8).copy().sort_values("teaching_gain", ascending=True)
    top3 = set(df.head(3)["knowledge_point"].tolist())
    d["color"] = d["knowledge_point"].apply(lambda x: SUCCESS if x in top3 else PRIMARY)
    fig = go.Figure(
        go.Bar(
            x=d["teaching_gain"],
            y=d["knowledge_point"],
            orientation="h",
            marker=dict(color=d["color"]),
            text=[f"{v:.2f}" for v in d["teaching_gain"]],
            textposition="outside",
            hovertemplate="知识点：%{y}<br>教学收益预测：%{x:.2f}<extra></extra>",
        )
    )
    return _base_layout(fig, "教学收益预测（Top3 高亮）")
