import io

import pandas as pd
import streamlit as st

from charts import (
    fig_difficulty_bar,
    fig_distractor_bar,
    fig_gain_line,
    fig_option_donut,
    fig_priority_bar,
    fig_risk_bar,
)
from metrics import (
    REQUIRED_COLS,
    build_conclusions,
    clean_df,
    compute_overview,
    distractor_analysis,
    item_discrimination,
    knowledge_priority,
    llm_prompt,
    question_option_overview,
    read_uploaded_file,
    sample_csv_bytes,
    sample_data,
    student_risk,
    teacher_actions,
    validate_df,
)

st.set_page_config(page_title="AI精准教学分析平台", page_icon="📘", layout="wide")


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0F172A 0%, #0B1220 100%);
            color: #E5E7EB;
        }
        .block-container {max-width: 1500px; padding-top: 5.6rem; padding-bottom: 2.5rem;}
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0F172A 100%);
            border-right: 1px solid rgba(148,163,184,0.16);
        }
        [data-testid="stSidebar"] * {color: #E5E7EB !important;}
        .hero {
            background: linear-gradient(135deg, rgba(30,41,59,0.96), rgba(15,23,42,0.98));
            border: 1px solid rgba(59,130,246,0.22);
            border-radius: 24px;
            padding: 26px 28px 22px 28px;
            box-shadow: 0 18px 60px rgba(2,6,23,.45);
            margin-bottom: 18px;
        }
        .hero h1 {margin:0; font-size: 34px; line-height:1.15; color:#F8FAFC;}
        .hero p {margin:10px 0 0 0; color:#CBD5E1; font-size:15px; line-height:1.7;}
        .section-card {
            background: linear-gradient(180deg, rgba(17,24,39,0.94), rgba(15,23,42,0.96));
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 22px;
            padding: 20px 22px;
            margin: 14px 0 18px 0;
            box-shadow: 0 10px 30px rgba(2,6,23,.26);
        }
        .section-title {font-size: 28px; font-weight: 800; color:#F8FAFC; margin-bottom:8px;}
        .section-sub {color:#94A3B8; font-size:14px; margin-bottom: 16px;}
        .conclusion {
            background: linear-gradient(90deg, rgba(29,78,216,.22), rgba(16,185,129,.10));
            border: 1px solid rgba(59,130,246,0.28);
            border-left: 6px solid #3B82F6;
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 18px;
        }
        .conclusion b {display:block; color:#BFDBFE; margin-bottom:6px;}
        .metric-big, .metric-small {
            background: linear-gradient(180deg, rgba(17,24,39,1), rgba(15,23,42,1));
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 22px;
            padding: 18px 18px;
            box-shadow: 0 10px 24px rgba(2,6,23,.24);
            transition: all .22s ease;
            height: 100%;
        }
        .metric-big:hover, .metric-small:hover, .section-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 36px rgba(2,6,23,.34);
            border-color: rgba(59,130,246,0.28);
        }
        .metric-big .label, .metric-small .label {color:#94A3B8; font-size:14px; margin-bottom:8px;}
        .metric-big .value {font-size:52px; font-weight:900; color:#F8FAFC; line-height:1;}
        .metric-small .value {font-size:32px; font-weight:850; color:#F8FAFC; line-height:1.1;}
        .metric-big .desc, .metric-small .desc {color:#CBD5E1; font-size:13px; margin-top:10px; line-height:1.6;}
        .action-item {
            background: rgba(30,41,59,0.62);
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }
        .stDataFrame, div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.14);
        }
        .note {color:#94A3B8; font-size:13px;}
        </style>
        """,
        unsafe_allow_html=True,
    )



def metric_card(label, value, desc, big=False):
    cls = "metric-big" if big else "metric-small"
    st.markdown(
        f"<div class='{cls}'><div class='label'>{label}</div><div class='value'>{value}</div><div class='desc'>{desc}</div></div>",
        unsafe_allow_html=True,
    )



def section_open(title, sub, conclusion):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div><div class='section-sub'>{sub}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='conclusion'><b>结论句</b>{conclusion}</div>", unsafe_allow_html=True)



def section_close():
    st.markdown("</div>", unsafe_allow_html=True)


inject_css()

st.markdown(
    """
    <div class="hero">
        <h1>AI精准教学分析平台</h1>
        <p>从“学生选了哪个选项”出发，完成班级诊断、知识点优先级排序、风险学生识别，以及错误选项级别的错因分析与教学建议。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 数据上传")
    uploaded = st.file_uploader("上传做题数据（CSV / XLSX）", type=["csv", "xlsx", "xls"])
    st.download_button("下载新版选择题模板", data=sample_csv_bytes(), file_name="sample_template_mcq.csv", mime="text/csv")
    st.markdown("### 推荐字段")
    st.caption("学生姓名、题目ID、知识点、难度、选项A-D、正确答案、学生选择、是否正确、得分、满分、作答时间")
    st.markdown("### 页面逻辑")
    st.caption("1. 班级整体诊断\n2. 问题定位\n3. 学生分层\n4. 错题分析与教学行动")

if uploaded is None:
    raw = sample_data()
    st.info("当前展示的是内置演示数据。上传真实选择题数据后会自动替换。")
else:
    raw = read_uploaded_file(uploaded)

raw = clean_df(raw)
ok, missing = validate_df(raw)
if not ok:
    st.error("缺少必填字段：" + "、".join([REQUIRED_COLS[m] for m in missing]))
    st.stop()

ov = compute_overview(raw)
kp = knowledge_priority(raw)
sr = student_risk(raw)
dis = item_discrimination(raw)
distractor = distractor_analysis(raw)
option_view = question_option_overview(raw)
con = build_conclusions(raw)
actions = teacher_actions(raw)
diff_df = raw.groupby("difficulty_group").agg(正确率=("wrong_flag", lambda s: 1 - s.mean())).reset_index()

section_open("一、班级整体诊断", "先判断班级整体是否需要共性讲评，再决定是否进入分层干预。", con["overall"])
row1 = st.columns([1.25, 1.25, 1, 1, 1, 1])
with row1[0]:
    metric_card("班级正确率", f"{ov['班级正确率']}%", "核心判断指标：整体掌握水平", big=True)
with row1[1]:
    metric_card("掌握指数", f"{ov['掌握指数']}", "100分制综合掌握度", big=True)
with row1[2]:
    metric_card("学生人数", ov["学生人数"], "参与分析学生数")
with row1[3]:
    metric_card("总作答数", ov["总作答数"], "全部题目作答记录")
with row1[4]:
    metric_card("总错题数", ov["总错题数"], "当前识别到的错误记录")
with row1[5]:
    metric_card("需关注学生", ov["需关注学生"], "高风险学生数量")
section_close()

section_open("二、问题定位", "核心不是看所有图，而是先确定‘先讲什么’与‘哪里最值得投入课堂时间’。", con["problem"])
col21, col22 = st.columns(2)
with col21:
    st.plotly_chart(fig_priority_bar(kp), use_container_width=True)
with col22:
    st.plotly_chart(fig_difficulty_bar(diff_df), use_container_width=True)
col23, col24 = st.columns([1.2, 0.8])
with col23:
    show_kp = kp[["knowledge_point", "错误率", "覆盖学生数", "题目权重", "优先级指数", "教学收益预测"]].copy().head(8)
    show_kp["错误率"] = (show_kp["错误率"] * 100).round(1).astype(str) + "%"
    show_kp["优先级指数"] = show_kp["优先级指数"].round(2)
    show_kp["教学收益预测"] = show_kp["教学收益预测"].round(2)
    st.dataframe(show_kp, use_container_width=True, hide_index=True)
with col24:
    dis_show = dis.copy().head(8)
    for col in ["高分组正确率", "低分组正确率", "区分度"]:
        dis_show[col] = (dis_show[col] * 100).round(1).astype(str) + "%"
    st.dataframe(dis_show, use_container_width=True, hide_index=True)
section_close()

section_open("三、学生分层", "学生分层不是只看错题数，而是综合连续错误和难题失分，优先抓真正会持续掉队的学生。", con["student"])
col31, col32 = st.columns([1.1, 0.9])
with col31:
    st.plotly_chart(fig_risk_bar(sr), use_container_width=True)
with col32:
    risk_show = sr[["student_name", "错题数", "连续错误次数", "难题错误权重", "风险指数", "分层", "正确率"]].copy().head(10)
    risk_show["难题错误权重"] = risk_show["难题错误权重"].round(1)
    risk_show["风险指数"] = risk_show["风险指数"].round(1)
    risk_show["正确率"] = risk_show["正确率"].astype(str) + "%"
    st.dataframe(risk_show, use_container_width=True, hide_index=True)
section_close()

section_open("四、错题分析与教学行动建议", "从‘错在哪个选项’走向‘为什么会选这个选项’，把错题诊断变成可执行讲评。", con["action"])
question_list = option_view["题目ID"].drop_duplicates().tolist()
selected_q = st.selectbox("选择题目查看选项与错误选项分析", question_list)
col41, col42 = st.columns(2)
with col41:
    st.plotly_chart(fig_option_donut(option_view, selected_q), use_container_width=True)
with col42:
    q_dis = distractor[distractor["题目ID"] == selected_q]
    if len(q_dis):
        st.plotly_chart(fig_distractor_bar(distractor, selected_q), use_container_width=True)
    else:
        st.success("该题当前没有错误选项，说明本题在本轮中没有形成明显误区。")

q_detail = raw[raw["question_id"] == selected_q].iloc[0]
st.markdown("#### 题目选项信息")
meta_df = pd.DataFrame([
    ["A", q_detail["option_a"], "是" if q_detail["correct_option"] == "A" else "否"],
    ["B", q_detail["option_b"], "是" if q_detail["correct_option"] == "B" else "否"],
    ["C", q_detail["option_c"], "是" if q_detail["correct_option"] == "C" else "否"],
    ["D", q_detail["option_d"], "是" if q_detail["correct_option"] == "D" else "否"],
], columns=["选项", "内容", "是否正确项"])
st.dataframe(meta_df, use_container_width=True, hide_index=True)

st.markdown("#### 错误选项分析")
if len(q_dis):
    show = q_dis[["错误选项", "错误选项内容", "错误人数", "错误占比", "常见误因", "分析建议"]].copy()
    show["错误占比"] = (show["错误占比"] * 100).round(1).astype(str) + "%"
    st.dataframe(show, use_container_width=True, hide_index=True)
else:
    st.info("当前题目未发现错误选项分析数据。")

st.markdown("#### 教学行动建议")
for i, action in enumerate(actions, 1):
    st.markdown(f"<div class='action-item'><b>建议{i}</b>：{action}</div>", unsafe_allow_html=True)

st.markdown("#### AI错题讲评 Prompt（可接大模型）")
st.code(llm_prompt(raw), language="text")
section_close()
