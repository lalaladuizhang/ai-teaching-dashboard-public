from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

BG = "rgba(0,0,0,0)"
TEXT = "#E5EEFf"
GRID = "rgba(148,163,184,0.18)"
PRIMARY = "#3B82F6"
SUCCESS = "#10B981"
DANGER = "#EF4444"
WARNING = "#F59E0B"
CARD = "#111827"


def base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, x=0.01, font=dict(size=20, color=TEXT)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, PingFang SC, Microsoft YaHei, sans-serif", size=13),
        margin=dict(l=24, r=24, t=60, b=30),
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor="#0B1220", font_color="#F8FAFC"),
    )



def _top3_colors(n: int, horizontal=False):
    cols = [PRIMARY] * n
    for i in range(min(3, n)):
        cols[i] = [DANGER, WARNING, SUCCESS][i]
    return cols



def fig_priority_bar(df: pd.DataFrame) -> go.Figure:
    data = df.head(8).sort_values("优先级指数", ascending=True)
    colors = _top3_colors(len(data))[::-1]
    fig = go.Figure(go.Bar(
        x=data["优先级指数"],
        y=data["knowledge_point"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.08)", width=1)),
        text=[f"{v:.2f}" for v in data["优先级指数"]],
        textposition="outside",
        hovertemplate="知识点：%{y}<br>优先级指数：%{x:.2f}<extra></extra>",
    ))
    fig.update_layout(**base_layout("知识点优先级排序（Top3高亮）"), bargap=0.34)
    fig.update_xaxes(title="优先级指数", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="")
    return fig



def fig_difficulty_bar(df: pd.DataFrame) -> go.Figure:
    order = ["基础题", "中档题", "提升题"]
    plot_df = df.copy()
    plot_df["difficulty_group"] = pd.Categorical(plot_df["difficulty_group"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("difficulty_group")
    colors = [SUCCESS if v >= 0.75 else WARNING if v >= 0.55 else DANGER for v in plot_df["正确率"]]
    fig = go.Figure(go.Bar(
        x=plot_df["difficulty_group"],
        y=plot_df["正确率"] * 100,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.08)", width=1.2)),
        text=[f"{v*100:.1f}%" for v in plot_df["正确率"]],
        textposition="outside",
        hovertemplate="难度：%{x}<br>正确率：%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**base_layout("难度层级 vs 正确率"), bargap=0.45)
    fig.update_yaxes(title="正确率(%)", range=[0, 100], showgrid=True, gridcolor=GRID)
    fig.update_xaxes(title="")
    return fig



def fig_risk_bar(df: pd.DataFrame) -> go.Figure:
    data = df.head(8).sort_values("风险指数", ascending=True)
    colors = [DANGER if x == "高风险" else WARNING if x == "波动层" else PRIMARY for x in data["分层"]]
    fig = go.Figure(go.Bar(
        x=data["风险指数"],
        y=data["student_name"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.08)", width=1)),
        text=[f"{v:.1f}" for v in data["风险指数"]],
        textposition="outside",
        hovertemplate="学生：%{y}<br>风险指数：%{x:.1f}<extra></extra>",
    ))
    fig.update_layout(**base_layout("风险学生识别"), bargap=0.34)
    fig.update_xaxes(title="风险指数", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="")
    return fig



def fig_gain_line(df: pd.DataFrame) -> go.Figure:
    data = df.head(8).sort_values("教学收益预测", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["knowledge_point"],
        y=data["教学收益预测"],
        mode="lines+markers+text",
        line=dict(color=PRIMARY, width=3),
        marker=dict(size=10, color=SUCCESS, line=dict(color="#ffffff22", width=1)),
        text=[f"{v:.2f}" for v in data["教学收益预测"]],
        textposition="top center",
        hovertemplate="知识点：%{x}<br>教学收益预测：%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**base_layout("教学收益预测"))
    fig.update_xaxes(title="", showgrid=False)
    fig.update_yaxes(title="收益值", showgrid=True, gridcolor=GRID)
    return fig



def fig_option_donut(df: pd.DataFrame, question_id: str) -> go.Figure:
    data = df[df["题目ID"] == question_id].copy()
    colors = [PRIMARY if x == "否" else SUCCESS for x in data["是否正确项"]]
    fig = go.Figure(go.Pie(
        labels=[f"{r['选项']}：{r['选项内容'][:10]}" for _, r in data.iterrows()],
        values=data["人数"],
        hole=0.58,
        marker=dict(colors=colors, line=dict(color="#0B1220", width=2)),
        textinfo="percent+label",
        sort=False,
    ))
    fig.update_layout(**base_layout(f"{question_id} 选项分布"))
    return fig



def fig_distractor_bar(df: pd.DataFrame, question_id: str) -> go.Figure:
    data = df[df["题目ID"] == question_id].sort_values("错误人数", ascending=True)
    fig = go.Figure(go.Bar(
        x=data["错误人数"],
        y=[f"{o}项" for o in data["错误选项"]],
        orientation="h",
        marker=dict(color=[DANGER, WARNING, PRIMARY][:len(data)] if len(data) <= 3 else [DANGER]*len(data)),
        text=[f"{v*100:.1f}%" for v in data["错误占比"]],
        textposition="outside",
        hovertemplate="错误选项：%{y}<br>人数：%{x}<extra></extra>",
    ))
    fig.update_layout(**base_layout(f"{question_id} 错误选项占比"), bargap=0.36)
    fig.update_xaxes(title="错误人数", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="")
    return fig
