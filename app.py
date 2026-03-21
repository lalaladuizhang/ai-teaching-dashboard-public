import math
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
    clean_columns,
    normalize_input_dataframe,
)
from charts import (
    plot_difficulty_accuracy,
    plot_priority_bar,
    plot_risk_bar,
    plot_teaching_gain,
)

st.set_page_config(
    page_title="AI精准教学分析平台",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
    --bg: #0F172A;
    --card: #111827;
    --primary: #3B82F6;
    --success: #10B981;
    --risk: #EF4444;
    --text: #E5E7EB;
    --muted: #94A3B8;
    --border: rgba(148,163,184,0.18);
}
.stApp {
    background:
      radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 28%),
      radial-gradient(circle at top right, rgba(16,185,129,0.10), transparent 22%),
      linear-gradient(180deg, #0B1220 0%, var(--bg) 100%);
    color: var(--text);
}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1450px;}
.section-card {
    background: linear-gradient(180deg, rgba(17,24,39,0.98) 0%, rgba(15,23,42,0.96) 100%);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 12px 30px rgba(2,6,23,0.38);
    transition: all .22s ease;
}
.section-card:hover, .metric-card:hover, .conclusion-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 36px rgba(2,6,23,0.45);
    border-color: rgba(59,130,246,0.38);
}
.metric-card {
    background: linear-gradient(135deg, rgba(17,24,39,0.98), rgba(30,41,59,0.92));
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 24px rgba(2,6,23,0.35);
    transition: all .22s ease;
    min-height: 132px;
}
.metric-card.hero {
    min-height: 188px;
    background: linear-gradient(135deg, rgba(30,41,59,0.98), rgba(29,78,216,0.22));
}
.metric-label {font-size: 0.95rem; color: var(--muted); margin-bottom: .5rem;}
.metric-value {font-size: 2.1rem; font-weight: 800; line-height: 1.15;}
.metric-card.hero .metric-value {font-size: 3rem;}
.metric-sub {font-size: 0.88rem; color: var(--muted); margin-top: .35rem;}
.page-title {
    font-size: 2.15rem; font-weight: 800; margin-bottom: .25rem;
    background: linear-gradient(90deg, #F8FAFC, #93C5FD);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.page-subtitle {color: var(--muted); margin-bottom: 1rem;}
.section-title {font-size: 1.2rem; font-weight: 700; margin-bottom: .25rem; color: #F8FAFC;}
.section-desc {font-size: 0.92rem; color: var(--muted); margin-bottom: .75rem;}
.conclusion-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(17,24,39,0.98));
    border-left: 5px solid var(--primary);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin-bottom: .9rem;
    transition: all .22s ease;
}
.conclusion-title {font-size: 0.88rem; color: #BFDBFE; margin-bottom: .35rem; letter-spacing: .04em;}
.conclusion-text {font-size: 1rem; color: #EFF6FF; line-height: 1.65;}
.small-tag {
    display: inline-block; padding: .15rem .55rem; border-radius: 999px;
    background: rgba(59,130,246,0.16); color: #BFDBFE; font-size: 0.8rem;
    border: 1px solid rgba(59,130,246,0.25); margin-right: .35rem;
}
.action-card {
    background: linear-gradient(180deg, rgba(17,24,39,.98), rgba(31,41,55,.98));
    border: 1px solid var(--border); border-radius: 18px; padding: .95rem 1rem;
    min-height: 170px; transition: all .22s ease;
}
.action-card:hover {transform: translateY(-3px); border-color: rgba(16,185,129,0.35);}
.action-title {font-size: 1rem; font-weight: 700; margin-bottom: .55rem;}
.action-list {color: #E5E7EB; line-height: 1.75; font-size: .93rem;}
hr.section-split {border: none; border-top: 1px solid rgba(148,163,184,0.12); margin: 1rem 0 1.25rem;}
[data-testid="stDataFrame"] {border-radius: 18px; overflow: hidden; border: 1px solid var(--border);}
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


st.markdown('<div class="page-title">AI精准教学分析平台</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">面向教学汇报的决策型分析看板：先判断班级状态，再定位问题，再识别风险学生，最后输出可执行教学动作。</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 数据上传")
    uploaded = st.file_uploader("上传做题数据（CSV / XLSX）", type=["csv", "xlsx", "xls"])
    st.markdown("### 推荐字段")
    st.caption("学生姓名、题目ID、知识点、难度、是否正确、得分、满分、作答时间")
    st.markdown("### 页面逻辑")
    st.caption("1. 班级整体诊断\n2. 问题定位\n3. 学生分层\n4. 教学行动建议")

if not uploaded:
    st.info("请先上传数据文件。若已有模板，可直接使用 sample_template.csv 或改成 .xlsx 后上传。")
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
actions = build_action_suggestions(class_metrics, priority_df, risk_df, disc_df, gain_df)

# Layer 1
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">一、班级整体诊断</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">先给出班级整体健康度判断，帮助教师迅速确定本轮教学是“整体补强”还是“分层干预”。</div>', unsafe_allow_html=True)
conclusion_card(actions["class_conclusion"])
row1 = st.columns([1.2, 1.2, 1, 1, 1, 1])
with row1[0]:
    metric_card("班级正确率", f"{class_metrics['accuracy_rate']:.1%}", "核心判断指标：整体掌握水平", hero=True)
with row1[1]:
    metric_card("掌握指数", f"{class_metrics['mastery_index']:.1f}", "100分制综合掌握度", hero=True)
with row1[2]:
    metric_card("学生人数", f"{class_metrics['student_count']}", "参与分析学生数")
with row1[3]:
    metric_card("总作答数", f"{class_metrics['attempt_count']}", "全部题目作答记录")
with row1[4]:
    metric_card("总错题数", f"{class_metrics['wrong_count']}", "当前识别到的失分记录")
with row1[5]:
    metric_card("需关注学生", f"{class_metrics['risk_student_count']}", "高风险层学生数量")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr class='section-split'>", unsafe_allow_html=True)

# Layer 2
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">二、问题定位</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">核心不是展示所有知识点，而是通过“知识点优先级指数”给出教学先后顺序。</div>', unsafe_allow_html=True)
conclusion_card(actions["priority_conclusion"])
col21, col22 = st.columns([1.45, 1])
with col21:
    st.plotly_chart(plot_priority_bar(priority_df), use_container_width=True)
with col22:
    st.plotly_chart(plot_difficulty_accuracy(difficulty_df), use_container_width=True)
sub21, sub22 = st.columns([1.2, 1])
with sub21:
    show_priority = priority_df[["knowledge_point", "error_rate", "error_students", "question_weight", "priority_index"]].copy()
    show_priority.columns = ["知识点", "错误率", "覆盖学生数", "题目权重", "优先级指数"]
    st.dataframe(show_priority, use_container_width=True, hide_index=True)
with sub22:
    show_disc = disc_df[["question_id", "knowledge_point", "high_group_acc", "low_group_acc", "discrimination"]].copy()
    show_disc.columns = ["题目ID", "知识点", "高分组正确率", "低分组正确率", "区分度"]
    st.dataframe(show_disc.head(10), use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr class='section-split'>", unsafe_allow_html=True)

# Layer 3
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">三、学生分层</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">通过风险指数识别真正需要优先干预的学生，而不是仅按一次成绩高低排序。</div>', unsafe_allow_html=True)
conclusion_card(actions["risk_conclusion"])
col31, col32 = st.columns([1.1, 1])
with col31:
    st.plotly_chart(plot_risk_bar(risk_df), use_container_width=True)
with col32:
    risk_summary = risk_df.groupby("risk_level").agg(学生数=("student_name", "count"), 平均风险指数=("risk_index", "mean")).reset_index()
    st.dataframe(risk_summary, use_container_width=True, hide_index=True)

show_risk = risk_df[["student_name", "wrong_count", "continuous_error_penalty", "hard_error_weight", "risk_index", "risk_level"]].copy()
show_risk.columns = ["学生", "错题数", "连续错误惩罚", "难题错误权重", "风险指数", "风险层级"]
st.dataframe(show_risk, use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr class='section-split'>", unsafe_allow_html=True)

# Layer 4
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">四、教学行动建议</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">把数据翻译成可执行动作，直接支撑课堂讲评、课后过关和分层训练安排。</div>', unsafe_allow_html=True)
conclusion_card(actions["action_conclusion"])
col41, col42 = st.columns([1.1, 1])
with col41:
    st.plotly_chart(plot_teaching_gain(gain_df), use_container_width=True)
with col42:
    gain_show = gain_df[["knowledge_point", "wrong_students", "avg_improvement_space", "teaching_gain"]].copy()
    gain_show.columns = ["知识点", "当前错误人数", "平均提升空间", "教学收益预测"]
    st.dataframe(gain_show, use_container_width=True, hide_index=True)

a1, a2, a3 = st.columns(3)
with a1:
    action_card("班级层面", actions["class_actions"])
with a2:
    action_card("分层层面", actions["layered_actions"])
with a3:
    action_card("题目与训练层面", actions["item_actions"])
st.markdown('</div>', unsafe_allow_html=True)

st.download_button(
    "下载知识点优先级结果（CSV）",
    priority_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="knowledge_priority.csv",
    mime="text/csv",
)
