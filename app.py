from pathlib import Path
import pandas as pd
import streamlit as st

from metrics import (
    build_action_suggestions,
    calc_class_metrics,
    calc_difficulty_accuracy,
    calc_item_discrimination,
    calc_knowledge_priority,
    calc_student_risk,
    calc_teaching_gain,
    calc_trend_by_order,
    clean_columns,
    normalize_input_dataframe,
)
from charts import (
    plot_difficulty_accuracy,
    plot_priority_bar,
    plot_risk_bar,
    plot_risk_donut,
    plot_teaching_gain,
    plot_trend_line,
)

st.set_page_config(page_title="AI精准教学分析平台", page_icon="📘", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
:root {
    --bg: #0F172A;
    --card: #111827;
    --card2: #0B1220;
    --primary: #3B82F6;
    --success: #10B981;
    --risk: #EF4444;
    --warn: #F59E0B;
    --text: #E5E7EB;
    --muted: #94A3B8;
    --border: rgba(148,163,184,0.14);
    --shadow: 0 16px 32px rgba(2,6,23,0.38);
}
html, body, [class*="css"] {font-family: Inter, PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif;}
.stApp {
    background:
      radial-gradient(circle at 20% 0%, rgba(59,130,246,0.18), transparent 26%),
      radial-gradient(circle at 100% 10%, rgba(16,185,129,0.10), transparent 20%),
      linear-gradient(180deg, #081120 0%, var(--bg) 100%);
    color: var(--text);
}
.block-container {max-width: 1480px; padding-top: 5.2rem; padding-bottom: 2rem;}
header[data-testid="stHeader"] {background: rgba(15,23,42,0.64); backdrop-filter: blur(8px);}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #111827 100%);
    border-right: 1px solid rgba(148,163,184,0.10);
}
[data-testid="stSidebar"] * {color: #E5E7EB !important;}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(148,163,184,0.28) !important;
    border-radius: 18px !important;
}
.page-hero {
    padding: 1.35rem 1.4rem 1.15rem;
    border: 1px solid var(--border);
    border-radius: 26px;
    background: linear-gradient(135deg, rgba(17,24,39,0.92), rgba(30,64,175,0.18));
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}
.page-title {
    font-size: 2.3rem; font-weight: 900; letter-spacing: -0.02em; line-height: 1.15;
    color: #F8FAFC; margin-bottom: .35rem;
}
.page-subtitle {font-size: 1rem; color: var(--muted); line-height: 1.7;}
.section-card {
    background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(10,15,30,0.98));
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.1rem 1.15rem 1.25rem;
    box-shadow: var(--shadow);
    transition: all .22s ease;
    margin-bottom: 1rem;
}
.section-card:hover, .metric-card:hover, .conclusion-card:hover, .action-card:hover, .table-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 22px 44px rgba(2,6,23,0.46);
    border-color: rgba(59,130,246,0.32);
}
.section-title {font-size: 1.42rem; font-weight: 800; color: #F8FAFC; margin-bottom: .25rem;}
.section-desc {font-size: .95rem; color: var(--muted); margin-bottom: .8rem; line-height: 1.7;}
.conclusion-card {
    background: linear-gradient(90deg, rgba(59,130,246,0.24), rgba(17,24,39,0.98));
    border: 1px solid rgba(59,130,246,0.22);
    border-left: 5px solid var(--primary);
    border-radius: 18px;
    padding: .95rem 1rem;
    margin-bottom: .95rem;
}
.conclusion-title {font-size: .82rem; color: #BFDBFE; margin-bottom: .35rem; font-weight: 700; letter-spacing: .04em;}
.conclusion-text {font-size: 1rem; color: #EFF6FF; line-height: 1.7;}
.metric-card {
    background: linear-gradient(145deg, rgba(17,24,39,0.98), rgba(31,41,55,0.92));
    border: 1px solid var(--border);
    border-radius: 22px;
    min-height: 140px;
    padding: 1rem 1.05rem;
    box-shadow: 0 12px 24px rgba(2,6,23,0.32);
    transition: all .22s ease;
}
.metric-card.hero {
    min-height: 205px;
    background: linear-gradient(145deg, rgba(17,24,39,1), rgba(59,130,246,0.18));
}
.metric-label {font-size: .95rem; color: var(--muted); margin-bottom: .5rem;}
.metric-value {font-size: 2.2rem; font-weight: 900; line-height: 1.1; color: #F8FAFC;}
.metric-card.hero .metric-value {font-size: 3.15rem;}
.metric-sub {font-size: .9rem; color: var(--muted); margin-top: .45rem; line-height: 1.6;}
.action-card {
    background: linear-gradient(180deg, rgba(17,24,39,.98), rgba(20,28,44,.98));
    border: 1px solid var(--border); border-radius: 18px; padding: 1rem 1rem .9rem;
    min-height: 180px; transition: all .22s ease;
}
.action-title {font-size: 1rem; font-weight: 800; margin-bottom: .55rem; color: #F8FAFC;}
.action-list {color: #E5E7EB; line-height: 1.8; font-size: .93rem; padding-left: 1rem;}
.table-card {
    background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(15,23,42,0.98));
    border: 1px solid var(--border); border-radius: 18px; padding: .7rem .8rem .2rem; box-shadow: 0 10px 20px rgba(2,6,23,0.25);
    transition: all .22s ease;
}
.table-title {font-size: .98rem; font-weight: 800; color: #F8FAFC; margin-bottom: .55rem;}
.table-wrap table {width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 14px;}
.table-wrap thead th {background: rgba(59,130,246,0.14); color: #DBEAFE; font-weight: 700; font-size: .88rem; text-align: left; padding: .72rem .72rem; border-bottom: 1px solid rgba(148,163,184,0.14);}
.table-wrap tbody td {background: rgba(255,255,255,0.02); color: #E5E7EB; font-size: .9rem; padding: .72rem .72rem; border-bottom: 1px solid rgba(148,163,184,0.10);}
.table-wrap tbody tr:nth-child(even) td {background: rgba(255,255,255,0.04);}
.table-wrap tbody tr:hover td {background: rgba(59,130,246,0.10);}
hr.section-split {border: none; border-top: 1px solid rgba(148,163,184,0.10); margin: 1rem 0 1rem;}
.stPlotlyChart {background: linear-gradient(180deg, rgba(15,23,42,0.35), rgba(15,23,42,0.15)); border: 1px solid rgba(148,163,184,0.10); border-radius: 18px; padding: .35rem;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "", hero: bool = False):
    cls = "metric-card hero" if hero else "metric-card"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def conclusion_card(text: str):
    st.markdown(
        f"""
        <div class="conclusion-card">
            <div class="conclusion-title">结论句</div>
            <div class="conclusion-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_card(title: str, items: list[str]):
    lis = "".join(f"<li>{x}</li>" for x in items)
    st.markdown(
        f"""
        <div class="action-card">
            <div class="action-title">{title}</div>
            <ul class="action-list">{lis}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def table_card(title: str, df: pd.DataFrame):
    html_table = df.to_html(index=False, escape=False)
    st.markdown(
        f"""
        <div class="table-card">
            <div class="table-title">{title}</div>
            <div class="table-wrap">{html_table}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        try:
            raw = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            raw = pd.read_csv(uploaded_file, encoding="gbk")
    else:
        raw = pd.read_excel(uploaded_file)
    raw.columns = clean_columns(raw.columns)
    return normalize_input_dataframe(raw)


st.markdown(
    """
    <div class="page-hero">
        <div class="page-title">AI精准教学分析平台</div>
        <div class="page-subtitle">面向教学汇报的决策型分析看板：先判断班级状态，再定位问题，再识别风险学生，最后形成可执行教学动作。新版重点修复标题遮挡、侧边栏风格割裂、图表粗糙和表格颜色冲突问题。</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 数据上传")
    uploaded = st.file_uploader("上传做题数据（CSV / XLSX）", type=["csv", "xlsx", "xls"])
    st.markdown("## 推荐字段")
    st.caption("学生姓名、题目ID、知识点、难度、是否正确、得分、满分、作答时间")
    st.markdown("## 页面逻辑")
    st.caption("1. 班级整体诊断\n2. 问题定位\n3. 学生分层\n4. 教学行动建议")

if not uploaded:
    st.info("请先上传数据文件。")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"数据读取失败：{e}")
    st.stop()

if df.empty:
    st.warning("当前数据为空，请检查文件内容。")
    st.stop()

class_metrics = calc_class_metrics(df)
priority_df = calc_knowledge_priority(df)
risk_df = calc_student_risk(df)
disc_df = calc_item_discrimination(df)
gain_df = calc_teaching_gain(priority_df)
difficulty_df = calc_difficulty_accuracy(df)
trend_df = calc_trend_by_order(df)
actions = build_action_suggestions(class_metrics, priority_df, risk_df, disc_df, gain_df)

# Layer 1
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">一、班级整体诊断</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">先判断本轮教学是需要整体补强，还是需要分层干预。核心指标不是越多越好，而是先看正确率和掌握指数是否处于安全区间。</div>', unsafe_allow_html=True)
conclusion_card(actions["class_conclusion"])
r1 = st.columns([1.2, 1.2, 1, 1, 1, 1])
with r1[0]:
    metric_card("班级正确率", f"{class_metrics['accuracy_rate']:.1%}", "反映整体答题稳定度", hero=True)
with r1[1]:
    metric_card("掌握指数", f"{class_metrics['mastery_index']:.1f}", "综合正确率、得分率、难度加权后的掌握度", hero=True)
with r1[2]:
    metric_card("学生人数", str(class_metrics["student_count"]), "参与分析学生数")
with r1[3]:
    metric_card("总作答数", str(class_metrics["attempt_count"]), "全部作答记录")
with r1[4]:
    metric_card("总错题数", str(class_metrics["wrong_count"]), "当前识别到的失分记录")
with r1[5]:
    metric_card("需关注学生", str(class_metrics["risk_student_count"]), "高风险层学生数量")
trend_c1, trend_c2 = st.columns([1.35, 1])
with trend_c1:
    st.plotly_chart(plot_trend_line(trend_df), use_container_width=True)
with trend_c2:
    donut_df = pd.DataFrame({"层级": ["高风险", "波动层", "稳定层"], "人数": [
        int((risk_df['risk_level'] == '高风险').sum()),
        int((risk_df['risk_level'] == '波动层').sum()),
        int((risk_df['risk_level'] == '稳定层').sum()),
    ]})
    st.plotly_chart(plot_risk_donut(risk_df), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Layer 2
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">二、问题定位</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">通过知识点优先级指数、难度层级正确率和题目区分度，判断“先讲什么”和“哪些题最值得讲”。</div>', unsafe_allow_html=True)
conclusion_card(actions["priority_conclusion"])
c21, c22 = st.columns([1.35, 1])
with c21:
    st.plotly_chart(plot_priority_bar(priority_df), use_container_width=True)
with c22:
    st.plotly_chart(plot_difficulty_accuracy(difficulty_df), use_container_width=True)
t21, t22 = st.columns([1.2, 1])
with t21:
    tmp = priority_df[["knowledge_point", "error_rate", "error_students", "question_weight", "priority_index"]].copy().head(8)
    tmp.columns = ["知识点", "错误率", "覆盖学生数", "题目权重", "优先级指数"]
    tmp["错误率"] = tmp["错误率"].map(lambda x: f"{x:.1%}")
    tmp["题目权重"] = tmp["题目权重"].map(lambda x: f"{x:.2f}")
    tmp["优先级指数"] = tmp["优先级指数"].map(lambda x: f"{x:.2f}")
    table_card("知识点优先级清单", tmp)
with t22:
    disc_show = disc_df[["question_id", "knowledge_point", "high_group_acc", "low_group_acc", "discrimination"]].copy().head(8)
    disc_show.columns = ["题目ID", "知识点", "高分组正确率", "低分组正确率", "区分度"]
    disc_show["高分组正确率"] = disc_show["高分组正确率"].map(lambda x: f"{x:.1%}")
    disc_show["低分组正确率"] = disc_show["低分组正确率"].map(lambda x: f"{x:.1%}")
    disc_show["区分度"] = disc_show["区分度"].map(lambda x: f"{x:.2f}")
    table_card("题目区分度清单", disc_show)
st.markdown('</div>', unsafe_allow_html=True)

# Layer 3
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">三、学生分层</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">不再只看“谁错得多”，而是综合连续错误和难题失分，识别真正需要优先干预的学生。</div>', unsafe_allow_html=True)
conclusion_card(actions["risk_conclusion"])
c31, c32 = st.columns([1.35, 1])
with c31:
    st.plotly_chart(plot_risk_bar(risk_df), use_container_width=True)
with c32:
    risk_show = risk_df[["student_name", "wrong_count", "longest_wrong_streak", "difficulty_wrong_weight", "risk_index", "risk_level"]].copy().head(8)
    risk_show.columns = ["学生", "错题数", "最长连续错", "难题失分权重", "风险指数", "层级"]
    risk_show["风险指数"] = risk_show["风险指数"].map(lambda x: f"{x:.1f}")
    risk_show["难题失分权重"] = risk_show["难题失分权重"].map(lambda x: f"{x:.1f}")
    table_card("风险学生清单", risk_show)
st.markdown('</div>', unsafe_allow_html=True)

# Layer 4
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">四、教学行动建议</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">让分析直接服务教学：给出优先讲评知识点、优先讲评题目和分层追踪方向。</div>', unsafe_allow_html=True)
conclusion_card(actions["action_conclusion"])
c41, c42 = st.columns([1.15, 1])
with c41:
    st.plotly_chart(plot_teaching_gain(gain_df), use_container_width=True)
with c42:
    gain_show = gain_df[["knowledge_point", "error_students", "avg_improvement_space", "teaching_gain"]].copy().head(8)
    gain_show.columns = ["知识点", "当前错误人数", "平均提升空间", "教学收益预测"]
    gain_show["平均提升空间"] = gain_show["平均提升空间"].map(lambda x: f"{x:.1%}")
    gain_show["教学收益预测"] = gain_show["教学收益预测"].map(lambda x: f"{x:.2f}")
    table_card("教学收益清单", gain_show)
a1, a2, a3 = st.columns(3)
with a1:
    action_card("班级共性讲评", actions["class_actions"])
with a2:
    action_card("风险学生补救", actions["risk_actions"])
with a3:
    action_card("二次训练与追踪", actions["action_actions"])
st.markdown('</div>', unsafe_allow_html=True)
