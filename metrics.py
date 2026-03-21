import numpy as np
import pandas as pd

COLUMN_ALIASES = {
    "student_name": ["学生姓名", "学生", "姓名", "student_name", "student"],
    "student_id": ["学生id", "学生编号", "学号", "student_id"],
    "question_id": ["题目id", "试题id", "question_id", "item_id"],
    "question_text": ["题目内容", "题干", "question_text", "question"],
    "knowledge_point": ["知识点", "知识点名称", "考点", "knowledge_point"],
    "difficulty": ["难度", "题目难度", "difficulty"],
    "is_correct": ["是否正确", "正确与否", "答对与否", "is_correct", "correct"],
    "score": ["得分", "score"],
    "full_score": ["满分", "full_score", "总分值"],
    "submit_time": ["作答时间", "提交时间", "时间", "submit_time", "answer_time"],
}


def clean_columns(columns):
    return [str(c).strip().replace("\n", "") for c in columns]


def _resolve_columns(df: pd.DataFrame):
    mapping = {}
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for std_col, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            key = a.strip().lower()
            if key in lower_map:
                mapping[lower_map[key]] = std_col
                break
    return mapping


def _coerce_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    true_vals = {"1", "true", "t", "yes", "y", "对", "正确", "答对"}
    false_vals = {"0", "false", "f", "no", "n", "错", "错误", "答错"}
    out = []
    for x in s:
        if x in true_vals:
            out.append(True)
        elif x in false_vals:
            out.append(False)
        else:
            try:
                out.append(float(x) > 0)
            except Exception:
                out.append(False)
    return pd.Series(out, index=series.index)


def difficulty_weight(x: str) -> float:
    text = str(x)
    if any(k in text for k in ["难", "困难", "hard", "高"]):
        return 1.5
    if any(k in text for k in ["中", "medium"]):
        return 1.2
    return 1.0


def normalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = _resolve_columns(df)
    df = df.rename(columns=mapping)

    required = ["student_name", "question_id", "knowledge_point", "is_correct"]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段：{', '.join(missing)}")

    if "difficulty" not in df.columns:
        df["difficulty"] = "中等"
    if "score" not in df.columns:
        df["score"] = np.where(_coerce_bool(df["is_correct"]), 1, 0)
    if "full_score" not in df.columns:
        df["full_score"] = 1
    if "submit_time" not in df.columns:
        df["submit_time"] = pd.RangeIndex(0, len(df))
    if "question_text" not in df.columns:
        df["question_text"] = df["question_id"].astype(str)

    df["student_name"] = df["student_name"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["knowledge_point"] = df["knowledge_point"].astype(str)
    df["difficulty"] = df["difficulty"].astype(str).replace({"nan": "中等"})
    df["is_correct"] = _coerce_bool(df["is_correct"])
    df["is_wrong"] = (~df["is_correct"]).astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["full_score"] = pd.to_numeric(df["full_score"], errors="coerce").replace(0, 1).fillna(1)
    df["score_rate"] = (df["score"] / df["full_score"]).clip(0, 1)
    df["difficulty_weight"] = df["difficulty"].apply(difficulty_weight)
    df["weighted_correct"] = np.where(df["is_correct"], df["difficulty_weight"], 0)
    df["weighted_total"] = df["difficulty_weight"]

    try:
        df["submit_time"] = pd.to_datetime(df["submit_time"], errors="coerce")
        if df["submit_time"].isna().all():
            df["submit_time"] = pd.RangeIndex(0, len(df))
    except Exception:
        df["submit_time"] = pd.RangeIndex(0, len(df))
    return df


def _longest_error_streak(group: pd.DataFrame) -> int:
    g = group.sort_values("submit_time")
    best = cur = 0
    for v in g["is_wrong"].tolist():
        if v == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def calc_class_metrics(df: pd.DataFrame) -> dict:
    accuracy = df["is_correct"].mean()
    score_rate = df["score_rate"].mean()
    weighted_accuracy = df["weighted_correct"].sum() / max(df["weighted_total"].sum(), 1)
    mastery_index = round((0.55 * accuracy + 0.25 * score_rate + 0.20 * weighted_accuracy) * 100, 1)
    risk_df = calc_student_risk(df)
    return {
        "accuracy_rate": float(accuracy),
        "score_rate": float(score_rate),
        "mastery_index": mastery_index,
        "student_count": int(df["student_name"].nunique()),
        "attempt_count": int(len(df)),
        "wrong_count": int(df["is_wrong"].sum()),
        "risk_student_count": int((risk_df["risk_level"] == "高风险").sum()),
        "avg_wrong_per_student": round(df["is_wrong"].sum() / max(df["student_name"].nunique(), 1), 2),
    }


def calc_knowledge_priority(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("knowledge_point")
        .agg(
            total_attempts=("question_id", "count"),
            wrong_attempts=("is_wrong", "sum"),
            error_students=("student_name", lambda x: x[df.loc[x.index, "is_wrong"] == 1].nunique()),
            avg_score_rate=("score_rate", "mean"),
            question_weight=("difficulty_weight", "mean"),
        )
        .reset_index()
    )
    grouped["error_rate"] = grouped["wrong_attempts"] / grouped["total_attempts"].replace(0, 1)
    grouped["priority_index"] = (
        grouped["error_rate"] * grouped["error_students"] * grouped["question_weight"]
    ).round(3)
    grouped["mastery_gap"] = (1 - grouped["avg_score_rate"]).round(3)
    grouped = grouped.sort_values(["priority_index", "error_students", "error_rate"], ascending=False).reset_index(drop=True)
    return grouped


def calc_student_risk(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["hard_wrong_weight"] = tmp["is_wrong"] * tmp["difficulty_weight"]

    streaks = tmp.groupby("student_name").apply(_longest_error_streak).reset_index(name="longest_wrong_streak")
    result = (
        tmp.groupby("student_name")
        .agg(
            wrong_count=("is_wrong", "sum"),
            difficulty_wrong_weight=("hard_wrong_weight", "sum"),
            accuracy_rate=("is_correct", "mean"),
            avg_score_rate=("score_rate", "mean"),
        )
        .reset_index()
        .merge(streaks, on="student_name", how="left")
    )
    result["continuous_error_penalty"] = result["longest_wrong_streak"] * 1.5
    result["risk_index"] = (
        result["wrong_count"] + result["continuous_error_penalty"] + result["difficulty_wrong_weight"]
    ).round(2)

    p50 = result["risk_index"].quantile(0.5)
    p80 = result["risk_index"].quantile(0.8)

    def _label(x):
        if x >= p80:
            return "高风险"
        if x >= p50:
            return "波动层"
        return "稳定层"

    result["risk_level"] = result["risk_index"].apply(_label)
    return result.sort_values("risk_index", ascending=False).reset_index(drop=True)


def calc_item_discrimination(df: pd.DataFrame) -> pd.DataFrame:
    stu = (
        df.groupby("student_name")
        .agg(total_score=("score", "sum"), total_full=("full_score", "sum"))
        .reset_index()
    )
    stu["score_rate"] = stu["total_score"] / stu["total_full"].replace(0, 1)
    stu = stu.sort_values("score_rate", ascending=False).reset_index(drop=True)
    n = max(1, int(len(stu) * 0.27))
    high_students = set(stu.head(n)["student_name"])
    low_students = set(stu.tail(n)["student_name"])

    high = (
        df[df["student_name"].isin(high_students)]
        .groupby("question_id")["is_correct"]
        .mean()
        .reset_index(name="high_group_acc")
    )
    low = (
        df[df["student_name"].isin(low_students)]
        .groupby("question_id")["is_correct"]
        .mean()
        .reset_index(name="low_group_acc")
    )
    meta = (
        df.groupby("question_id")
        .agg(knowledge_point=("knowledge_point", "first"), difficulty=("difficulty", "first"))
        .reset_index()
    )
    result = meta.merge(high, on="question_id", how="left").merge(low, on="question_id", how="left")
    result[["high_group_acc", "low_group_acc"]] = result[["high_group_acc", "low_group_acc"]].fillna(0)
    result["discrimination"] = (result["high_group_acc"] - result["low_group_acc"]).round(3)
    return result.sort_values("discrimination", ascending=False).reset_index(drop=True)


def calc_teaching_gain(priority_df: pd.DataFrame) -> pd.DataFrame:
    out = priority_df[["knowledge_point", "error_students", "error_rate", "priority_index"]].copy()
    out["avg_improvement_space"] = (1 - out["error_rate"]).round(3)
    out["teaching_gain"] = (out["error_students"] * out["avg_improvement_space"]).round(3)
    return out.sort_values("teaching_gain", ascending=False).reset_index(drop=True)


def calc_difficulty_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("difficulty")
        .agg(accuracy_rate=("is_correct", "mean"), attempts=("question_id", "count"), avg_score_rate=("score_rate", "mean"))
        .reset_index()
    )
    return out.sort_values("accuracy_rate", ascending=True).reset_index(drop=True)


def calc_trend_by_order(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy().sort_values("submit_time").reset_index(drop=True)
    if len(tmp) <= 1:
        tmp["batch"] = ["第1段"] * len(tmp)
    else:
        tmp["order"] = np.arange(len(tmp))
        tmp["batch"] = pd.qcut(tmp["order"], q=min(6, max(1, len(tmp))), duplicates="drop")
        tmp["batch"] = tmp["batch"].astype(str)
    out = tmp.groupby("batch").agg(accuracy_rate=("is_correct", "mean"), attempts=("question_id", "count")).reset_index()
    out["label"] = [f"阶段{i+1}" for i in range(len(out))]
    return out[["label", "accuracy_rate", "attempts"]]


def build_action_suggestions(class_metrics, priority_df, risk_df, disc_df, gain_df):
    top_kps = priority_df.head(3)["knowledge_point"].tolist()
    low_diff = priority_df.head(2)["knowledge_point"].tolist()
    top_items = disc_df.head(3)["question_id"].tolist()
    risk_names = risk_df.head(5)["student_name"].tolist()
    high_risk_count = int((risk_df["risk_level"] == "高风险").sum())
    top_gain = gain_df.head(3)["knowledge_point"].tolist()

    class_conclusion = (
        f"本班当前正确率为 {class_metrics['accuracy_rate']:.1%}，掌握指数为 {class_metrics['mastery_index']:.1f}。"
        f"从整体看，更适合先处理共性薄弱知识点，再对高风险学生进行定向补救。"
    )
    priority_conclusion = (
        f"当前最优先处理的知识点为 {', '.join(top_kps) if top_kps else '暂无'}，"
        f"这些知识点同时具备较高错误率、较广覆盖学生数和较强教学收益。"
    )
    risk_conclusion = (
        f"当前高风险学生共 {high_risk_count} 人，优先关注 {', '.join(risk_names) if risk_names else '暂无'}。"
        f"风险主要来自连续错误和难题失分累积。"
    )
    action_conclusion = (
        f"建议本轮教学先讲 {', '.join(top_gain) if top_gain else '暂无'}，再结合区分度较高的题目 {', '.join(top_items) if top_items else '暂无'} 做二次训练，"
        f"优先争取对高覆盖薄弱点形成集中修复。"
    )

    class_actions = [
        f"整班先讲：{', '.join(low_diff) if low_diff else '暂无'}。",
        "核心知识点讲评时，先讲典型错误路径，再讲标准解题框架。",
        "讲评后立刻安排1轮同类变式题，检验修复效果。",
    ]
    risk_actions = [
        f"优先盯防学生：{', '.join(risk_names) if risk_names else '暂无'}。",
        "对连续错误学生先做基础概念清理，再进阶到综合题。",
        "对难题持续失分学生安排分步过关，避免一次性整题压测。",
    ]
    action_actions = [
        f"优先讲评知识点：{', '.join(top_gain) if top_gain else '暂无'}。",
        f"推荐二次训练题：{', '.join(top_items) if top_items else '暂无'}。",
        "课后按高风险层、波动层、稳定层分别推送不同练习包。",
    ]
    return {
        "class_conclusion": class_conclusion,
        "priority_conclusion": priority_conclusion,
        "risk_conclusion": risk_conclusion,
        "action_conclusion": action_conclusion,
        "class_actions": class_actions,
        "risk_actions": risk_actions,
        "action_actions": action_actions,
    }
