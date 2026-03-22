import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS = {
    "student_name": "学生姓名",
    "student_id": "学生ID",
    "question_id": "题目ID",
    "question_text": "题目内容",
    "knowledge_point": "知识点",
    "difficulty": "难度",
    "is_correct": "是否正确",
    "score": "本题得分",
    "full_score": "本题满分",
    "submit_time": "作答时间",
    "option_a": "选项A",
    "option_b": "选项B",
    "option_c": "选项C",
    "option_d": "选项D",
    "correct_option": "正确答案",
    "student_option": "学生选择",
}

OPTIONAL_COLS = {
    "class_name": "班级",
    "exam_name": "考试/练习名称",
    "subject": "学科",
    "source": "来源APP",
}

DIFFICULTY_WEIGHT = {"基础题": 1.0, "中档题": 1.2, "提升题": 1.5}
OPTION_REASON_MAP = {
    "概念混淆": ["概念", "分类", "定义", "性质", "离子", "物质"],
    "审题不完整": ["正确的是", "不正确", "不能", "一定", "除外", "下列"],
    "条件迁移不足": ["实验", "装置", "步骤", "现象", "推断", "判断"],
    "单位/计算失误": ["质量", "体积", "浓度", "计算", "摩尔", "比例"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {v: k for k, v in REQUIRED_COLS.items()}
    rename.update({v: k for k, v in OPTIONAL_COLS.items()})
    return df.rename(columns={c: rename.get(c, c) for c in df.columns})



def read_uploaded_file(file) -> pd.DataFrame:
    if file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    for encoding in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except Exception:
            continue
    file.seek(0)
    return pd.read_csv(file)



def validate_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    return len(missing) == 0, missing



def parse_binary(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "对", "正确", "是"}:
        return 1
    if s in {"0", "false", "f", "no", "n", "错", "错误", "否"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return np.nan



def difficulty_bucket(value) -> str:
    try:
        v = float(value)
    except Exception:
        return "中档题"
    if v <= 2:
        return "基础题"
    if v <= 3:
        return "中档题"
    return "提升题"



def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df.copy())
    for c in [
        "student_name",
        "student_id",
        "question_id",
        "question_text",
        "knowledge_point",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "correct_option",
        "student_option",
    ]:
        df[c] = df[c].astype(str).replace("nan", "").str.strip()

    if "subject" not in df.columns:
        df["subject"] = "化学"
    if "class_name" not in df.columns:
        df["class_name"] = "默认班级"
    if "exam_name" not in df.columns:
        df["exam_name"] = "默认练习"
    if "source" not in df.columns:
        df["source"] = "未标注"

    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["full_score"] = pd.to_numeric(df["full_score"], errors="coerce")
    df["is_correct"] = df["is_correct"].apply(parse_binary)
    df["submit_time"] = pd.to_datetime(df["submit_time"], errors="coerce")
    df["wrong_flag"] = (df["is_correct"] == 0).astype(int)
    df["difficulty_group"] = df["difficulty"].apply(difficulty_bucket)
    df["difficulty_weight"] = df["difficulty_group"].map(DIFFICULTY_WEIGHT).fillna(1.2)
    df["accuracy_item"] = np.where(df["full_score"] > 0, df["score"] / df["full_score"], np.nan)
    return df



def compute_overview(df: pd.DataFrame) -> Dict[str, float]:
    students = df["student_id"].nunique()
    attempts = len(df)
    correct_rate = (1 - df["wrong_flag"].mean()) * 100 if attempts else 0
    wrongs = int(df["wrong_flag"].sum())
    student_wrong = df.groupby("student_id")["wrong_flag"].sum()
    risk_students = int((student_wrong >= max(2, np.percentile(student_wrong, 75) if len(student_wrong) else 2)).sum()) if len(student_wrong) else 0
    mastery = max(0, min(100, correct_rate * 0.65 + (100 - wrongs / max(students, 1) * 12) * 0.35))
    return {
        "班级正确率": round(correct_rate, 1),
        "掌握指数": round(mastery, 1),
        "学生人数": int(students),
        "总作答数": int(attempts),
        "总错题数": wrongs,
        "需关注学生": risk_students,
    }



def longest_wrong_streak(x: pd.Series) -> int:
    vals = x.fillna(0).astype(int).tolist()
    best = cur = 0
    for v in vals:
        if v == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best



def student_risk(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.sort_values(["student_id", "submit_time"]).copy()
    streak = temp.groupby(["student_id", "student_name"])["wrong_flag"].apply(longest_wrong_streak).reset_index(name="连续错误次数")
    base = temp.groupby(["student_id", "student_name"]).agg(
        错题数=("wrong_flag", "sum"),
        难题错误权重=("difficulty_weight", lambda s: float(s[temp.loc[s.index, 'wrong_flag'] == 1].sum()) if len(s) else 0),
        正确率=("wrong_flag", lambda s: round((1 - s.mean()) * 100, 1)),
        作答数=("question_id", "size"),
    ).reset_index()
    out = base.merge(streak, on=["student_id", "student_name"], how="left")
    out["风险指数"] = out["错题数"] + out["连续错误次数"] + out["难题错误权重"]
    q80 = out["风险指数"].quantile(0.8) if len(out) else 0
    q50 = out["风险指数"].quantile(0.5) if len(out) else 0
    out["分层"] = np.where(out["风险指数"] >= q80, "高风险", np.where(out["风险指数"] >= q50, "波动层", "稳定层"))
    return out.sort_values(["风险指数", "错题数"], ascending=False)



def knowledge_priority(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("knowledge_point").agg(
        总作答数=("question_id", "size"),
        错误数=("wrong_flag", "sum"),
        覆盖学生数=("student_id", lambda s: s[df.loc[s.index, "wrong_flag"] == 1].nunique()),
        题目权重=("difficulty_weight", "mean"),
    ).reset_index()
    grp["错误率"] = grp["错误数"] / grp["总作答数"].replace(0, np.nan)
    grp["优先级指数"] = grp["错误率"] * grp["覆盖学生数"] * grp["题目权重"]
    grp["教学收益预测"] = grp["覆盖学生数"] * (1 - (1 - grp["错误率"]))
    return grp.sort_values("优先级指数", ascending=False)



def item_discrimination(df: pd.DataFrame) -> pd.DataFrame:
    stu = df.groupby("student_id").agg(avg_score=("accuracy_item", "mean")).reset_index().sort_values("avg_score", ascending=False)
    if len(stu) < 4:
        use_n = max(1, len(stu) // 2)
    else:
        use_n = max(1, int(np.ceil(len(stu) * 0.27)))
    high_ids = set(stu.head(use_n)["student_id"])
    low_ids = set(stu.tail(use_n)["student_id"])
    rows = []
    for qid, sub in df.groupby("question_id"):
        high = sub[sub["student_id"].isin(high_ids)]
        low = sub[sub["student_id"].isin(low_ids)]
        high_acc = 1 - high["wrong_flag"].mean() if len(high) else np.nan
        low_acc = 1 - low["wrong_flag"].mean() if len(low) else np.nan
        rows.append({
            "题目ID": qid,
            "知识点": sub["knowledge_point"].iloc[0],
            "高分组正确率": round(float(high_acc or 0), 3),
            "低分组正确率": round(float(low_acc or 0), 3),
            "区分度": round(float((high_acc or 0) - (low_acc or 0)), 3),
        })
    return pd.DataFrame(rows).sort_values("区分度", ascending=False)



def option_text_map(row: pd.Series) -> Dict[str, str]:
    return {"A": row.get("option_a", ""), "B": row.get("option_b", ""), "C": row.get("option_c", ""), "D": row.get("option_d", "")}



def reason_from_question(question_text: str, distractor_text: str, correct_text: str) -> str:
    text = f"{question_text} {distractor_text} {correct_text}"
    for label, kws in OPTION_REASON_MAP.items():
        if any(k in text for k in kws):
            return label
    return "表层记忆替代了真正理解"



def distractor_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, sub in df.groupby("question_id"):
        meta = sub.iloc[0]
        opt_map = option_text_map(meta)
        total_wrong = int(sub["wrong_flag"].sum())
        if total_wrong == 0:
            continue
        wrong_sub = sub[sub["wrong_flag"] == 1]
        counts = wrong_sub["student_option"].value_counts()
        for opt, cnt in counts.items():
            if opt == meta["correct_option"]:
                continue
            rows.append({
                "题目ID": qid,
                "题目内容": meta["question_text"],
                "知识点": meta["knowledge_point"],
                "正确答案": meta["correct_option"],
                "错误选项": opt,
                "错误选项内容": opt_map.get(opt, ""),
                "错误人数": int(cnt),
                "错误占比": round(cnt / total_wrong, 3),
                "常见误因": reason_from_question(meta["question_text"], opt_map.get(opt, ""), opt_map.get(meta["correct_option"], "")),
                "分析建议": f"选择{opt}的学生多半将“{opt_map.get(opt, '')[:14]}”误当成正确结论，建议对比正确项与{opt}项的关键词差异，做1题同构辨析题。",
            })
    if not rows:
        return pd.DataFrame(columns=["题目ID", "错误选项", "错误占比", "常见误因", "分析建议"])
    return pd.DataFrame(rows).sort_values(["题目ID", "错误人数"], ascending=[True, False])



def question_option_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, sub in df.groupby("question_id"):
        meta = sub.iloc[0]
        opt_map = option_text_map(meta)
        vc = sub["student_option"].value_counts()
        total = len(sub)
        for opt in ["A", "B", "C", "D"]:
            rows.append({
                "题目ID": qid,
                "知识点": meta["knowledge_point"],
                "选项": opt,
                "选项内容": opt_map.get(opt, ""),
                "人数": int(vc.get(opt, 0)),
                "占比": round(vc.get(opt, 0) / total, 3) if total else 0,
                "是否正确项": "是" if opt == meta["correct_option"] else "否",
            })
    return pd.DataFrame(rows)



def build_conclusions(df: pd.DataFrame) -> Dict[str, str]:
    ov = compute_overview(df)
    kp = knowledge_priority(df)
    sr = student_risk(df)
    da = distractor_analysis(df)
    top_kp = kp.iloc[0]["knowledge_point"] if len(kp) else "暂无"
    top_risk = sr.iloc[0]["student_name"] if len(sr) else "暂无"
    top_dis = da.iloc[0]
    return {
        "overall": f"本班当前正确率为{ov['班级正确率']}%，掌握指数为{ov['掌握指数']}，建议采用“先处理高覆盖薄弱知识点，再追踪高风险学生”的教学策略。",
        "problem": f"当前最优先处理的知识点是{top_kp}，它同时具备较高错误率、较广学生覆盖和较高教学收益。",
        "student": f"当前需要优先关注的学生以{top_risk}为代表，高风险学生更集中地表现为连续错误和难题失分叠加。",
        "action": (
            f"建议先讲解{top_kp}的易错辨析题，再针对错误选项“{top_dis['错误选项']}”做对比讲评，"
            f"优先纠正“{top_dis['常见误因']}”这一类错误。" if len(da) else "建议先围绕高优先级知识点进行共性讲评，再安排课后分层订正。"
        ),
    }



def teacher_actions(df: pd.DataFrame) -> List[str]:
    kp = knowledge_priority(df).head(3)["knowledge_point"].tolist()
    da = distractor_analysis(df).head(3)
    texts = []
    if kp:
        texts.append(f"先讲 { '、'.join(kp) }，按‘概念澄清—典型辨析—变式巩固’三步完成共性讲评。")
    for _, r in da.iterrows():
        texts.append(f"针对题目{r['题目ID']}的错误选项{r['错误选项']}，重点说明{r['常见误因']}，并用正确项与错误项做关键词对照。")
    texts.append("对高风险学生安排1次小范围面批，优先处理连续错误链条，而不是只看单题对错。")
    return texts[:5]



def sample_data() -> pd.DataFrame:
    rows = [
        ["九年级2班", "阶段测验1", "小王", "S01", "Q001", "将下列物质按酸、碱、盐顺序排列，正确的是", "物质分类", 2, 0, 0, 2, "2026-03-20 08:00:00", "HCl、NaOH、Na2CO3", "NaOH、HCl、Na2CO3", "HCl、Na2CO3、NaOH", "Na2CO3、HCl、NaOH", "A", "B"],
        ["九年级2班", "阶段测验1", "小王玲", "S02", "Q001", "将下列物质按酸、碱、盐顺序排列，正确的是", "物质分类", 2, 1, 2, 2, "2026-03-20 08:01:00", "HCl、NaOH、Na2CO3", "NaOH、HCl、Na2CO3", "HCl、Na2CO3、NaOH", "Na2CO3、HCl、NaOH", "A", "A"],
        ["九年级2班", "阶段测验1", "李明", "S03", "Q001", "将下列物质按酸、碱、盐顺序排列，正确的是", "物质分类", 2, 0, 0, 2, "2026-03-20 08:02:00", "HCl、NaOH、Na2CO3", "NaOH、HCl、Na2CO3", "HCl、Na2CO3、NaOH", "Na2CO3、HCl、NaOH", "A", "C"],
        ["九年级2班", "阶段测验1", "小王", "S01", "Q002", "下列关于二氧化碳性质的说法正确的是", "二氧化碳性质", 4, 0, 0, 2, "2026-03-20 08:03:00", "能支持燃烧", "可使澄清石灰水变浑浊", "有刺激性气味", "难溶于水", "B", "D"],
        ["九年级2班", "阶段测验1", "小王玲", "S02", "Q002", "下列关于二氧化碳性质的说法正确的是", "二氧化碳性质", 4, 1, 2, 2, "2026-03-20 08:04:00", "能支持燃烧", "可使澄清石灰水变浑浊", "有刺激性气味", "难溶于水", "B", "B"],
        ["九年级2班", "阶段测验1", "李明", "S03", "Q002", "下列关于二氧化碳性质的说法正确的是", "二氧化碳性质", 4, 0, 0, 2, "2026-03-20 08:05:00", "能支持燃烧", "可使澄清石灰水变浑浊", "有刺激性气味", "难溶于水", "B", "A"],
        ["九年级2班", "阶段测验1", "小王", "S01", "Q003", "过滤操作中玻璃棒的作用是", "混合物分离", 3, 0, 0, 2, "2026-03-20 08:06:00", "搅拌加快溶解", "引流液体防止飞溅", "测量液体体积", "夹持试管", "B", "A"],
        ["九年级2班", "阶段测验1", "小王玲", "S02", "Q003", "过滤操作中玻璃棒的作用是", "混合物分离", 3, 1, 2, 2, "2026-03-20 08:07:00", "搅拌加快溶解", "引流液体防止飞溅", "测量液体体积", "夹持试管", "B", "B"],
        ["九年级2班", "阶段测验1", "李明", "S03", "Q003", "过滤操作中玻璃棒的作用是", "混合物分离", 3, 0, 0, 2, "2026-03-20 08:08:00", "搅拌加快溶解", "引流液体防止飞溅", "测量液体体积", "夹持试管", "B", "D"],
    ]
    cols = [
        "班级", "考试/练习名称", "学生姓名", "学生ID", "题目ID", "题目内容", "知识点", "难度", "是否正确", "本题得分", "本题满分", "作答时间",
        "选项A", "选项B", "选项C", "选项D", "正确答案", "学生选择"
    ]
    return pd.DataFrame(rows, columns=cols)



def sample_csv_bytes() -> bytes:
    return sample_data().to_csv(index=False).encode("utf-8-sig")



def llm_prompt(df: pd.DataFrame) -> str:
    sample = distractor_analysis(df).head(10).to_dict(orient="records")
    return (
        "你是一名中学教学分析助手。请基于以下错误选项分析数据，输出班级共性误区、每个错误选项对应的误因、以及可执行讲评建议：\n"
        + json.dumps(sample, ensure_ascii=False, indent=2)
    )
