#!/usr/bin/env python3
"""Build commercial-safe Chinese (zh-Hant & zh-Hans) lexicons in 8-column TSV format.

Sources (strictly permissive / commercial-safe):
- jieba-rs dict.txt (MIT)
- BYVoid/OpenCC dictionary tables (Apache-2.0)
- In-repo tables (src/doc/pos.rs, src/doc/token.rs, src/text.rs, src/language.rs)
- Statistical NPMI mining over CC0/PD corpora (scratch/corpora/zh/*.jsonl)

Outputs (atomic via temp files + os.replace):
- lexicon/data/zh_hant.tsv
- lexicon/data/zh_hans.tsv
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JIEBA_DICT = Path("/Users/chrischeng/faberlines/core/vendor/jieba-rs/src/data/dict.txt")
SCRATCH_DIR = Path(
    "/Users/chrischeng/.gemini/antigravity/brain/fcdf5ec7-a63d-4308-93f6-0436229a04ea/scratch"
)
OPENCC_DIR = SCRATCH_DIR / "lex_build_zh" / "opencc"
CORPORA_DIR = SCRATCH_DIR / "corpora" / "zh"
TMP_DIR = SCRATCH_DIR / "lex_build_zh"

NLP_CRATE_ROOT = REPO_ROOT / "crates" / "faberline-nlp"
OUT_HANT = NLP_CRATE_ROOT / "lexicon" / "data" / "zh_hant.tsv"
OUT_HANS = NLP_CRATE_ROOT / "lexicon" / "data" / "zh_hans.tsv"

VALID_UPOS = frozenset(
    {
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    }
)

ZH_NUMERALS = set("〇零一二三四五六七八九十百千萬万億亿兩两")

# Function/particle/coverb/pronoun chars that should not appear in NPMI-mined multi-char n-grams.
NPMI_STOP_CHARS = set(
    "的得地了著着過过嗎吗呢吧啊呀一不沒没都也就還还很太更最"
    "把被將将對对從从向往給给在有是這这那其你我他她它們们與与和跟或而"
)

# False coverb/particle/adverbial merges that must split into separate tokens.
EXCLUDED_WORDS = {
    "對此", "对此",
    "把門", "把门",
    "把它", "把他",
    "將他", "将他",
    "慢慢地",
    "實驗結果", "实验结果",
    "大家都",
    "還好", "还好",
    "提出了",
    "買了一", "买了一",
    "停了下來", "停了下来",
    "有一座",
    "天晚上",
}

# Curated person names (PROPN, sem=person) that should stay whole.
CURATED_PERSON_NAMES = {
    "阿芳", "阿杰", "小芸", "小林", "老王",
}

# Curated domain/classical/Taiwan terms that must have low enough cost to stay whole
# (e.g. 在下 cost < cost(在)+cost(下), while 還在下雨 still keeps 在 + 下雨).
CURATED_BOOST_WORDS = {
    "在下", "妾身", "老夫", "老身", "貧道", "贫道", "本座", "晚輩", "晚辈",
    "前輩", "前辈", "道友", "元嬰", "元婴", "筑基", "築基", "筑基丹", "築基丹",
    "結丹", "结丹", "化神", "煉虛", "炼虚", "大乘", "神識", "神识", "靈力", "灵力",
    "法寶", "法宝", "靈寶", "灵宝", "儲物袋", "储物袋", "丹藥", "丹药", "功法",
    "口訣", "口诀", "陣法", "阵法", "老者", "大漢", "大汉", "婦人", "妇人",
    "狼藉", "狼籍", "狼借",
    "中年", "中品", "中階", "中阶", "中計", "中计", "激射", "奧秘", "奥秘", "護具", "护具", "工坊",
    "這麼晚", "这么晚", "馬車", "马车", "隊列", "队列",
    "我們", "你們", "他們", "她們", "它們",
    "我们", "你们", "他们", "她们", "它们",
}

# Verbal-noun / open-class homographs that MUST stay ambiguous (NOUN|VERB)
NOUN_VERB_WORDS = {
    "研究", "发展", "發展", "开始", "開始", "工作", "学习", "學習", "训练", "訓練",
    "执行", "執行", "测量", "測量", "验证", "驗證", "记录", "記錄", "提升", "设计",
    "設計", "分析", "管理", "调查", "調查", "使用", "帮助", "幫助", "选择", "選擇",
    "变化", "變化", "影响", "影響", "支持", "保护", "保護", "建设", "建設", "生产",
    "生產", "生活", "希望", "准备", "準備", "计划", "計劃", "决定", "決定", "努力",
    "成功", "失败", "失敗", "需要", "要求", "建议", "建議", "解决", "解決", "证明",
    "證明", "表现", "表現", "解释", "解釋", "说明", "說明", "报告", "報告", "讨论",
    "討論", "思考", "感受", "体验", "體驗", "尝试", "嘗試", "练习", "練習", "比较",
    "比較", "联系", "聯繫", "交流", "合作", "竞争", "競爭", "交易", "投资", "投資",
    "消费", "消費", "服务", "服務", "教学", "教學", "治疗", "治療", "手术", "手術",
    "运动", "運動", "比赛", "比賽", "表演", "旅行", "旅游", "旅遊", "休息", "睡眠",
    "战争", "戰爭", "战斗", "戰鬥", "攻击", "攻擊", "防御", "防禦", "控制", "指挥",
    "指揮", "组织", "組織", "领导", "領導", "统治", "統治", "惩罚", "懲罰", "奖励",
    "獎勵", "承诺", "承諾", "拒绝", "拒絕", "同意", "反对", "反對", "怀疑", "懷疑",
    "相信", "梦想", "夢想", "记忆", "記憶", "回忆", "回憶", "想象", "想像", "创造",
    "創造", "发明", "發明", "发现", "發現", "观察", "觀察", "检查", "檢查", "测试",
    "測試", "实验", "實驗", "评估", "評估", "预测", "預測", "估计", "估計", "计算",
    "計算", "统计", "統計", "分类", "分類", "总结", "總結", "介绍", "介紹", "推荐",
    "推薦", "评价", "評價", "批评", "批評", "赞美", "讚美", "感谢", "感謝", "道歉",
    "庆祝", "慶祝", "纪念", "紀念", "出版", "编辑", "編輯", "翻译", "翻譯", "改革",
    "开放", "開放", "创新", "創新", "应用", "應用", "开发", "開發", "运行", "運行",
    "操作", "处理", "處理", "加工", "制造", "製造", "运输", "運輸", "销售", "銷售",
    "采购", "採購", "供应", "供應", "分配", "调整", "調整", "修改", "更新", "维护",
    "維護", "修理", "清理", "整理", "收集", "保存", "存储", "存儲", "传递", "傳遞",
    "表达", "表達", "沟通", "溝通", "协作", "協作", "协调", "協調", "监督", "監督",
    "审核", "審核", "审查", "審查", "批准", "授权", "授權", "禁止", "限制", "约束",
    "約束", "规范", "規範", "指导", "指導", "引导", "引導", "教育", "培养", "培養",
    "培训", "培訓", "考核", "考试", "考試", "毕业", "畢業", "就业", "就業", "创业",
    "創業", "退休", "休息", "娱乐", "娛樂", "游戏", "遊戲", "阅读", "閱讀", "写作",
    "寫作", "演讲", "演講", "汇报", "匯報", "采访", "採訪", "报道", "報道", "宣传",
    "宣傳", "广播", "廣播", "摄影", "攝影", "绘画", "繪畫", "雕刻", "建筑", "建築",
    "装修", "裝修", "种植", "種植", "养殖", "養殖", "捕捞", "捕撈", "开采", "開採",
    "勘探", "冶炼", "冶煉", "纺织", "紡織", "印刷", "包装", "包裝", "储存", "儲存",
    "排放", "污染", "治理", "绿化", "綠化", "节约", "節約", "浪费", "浪費", "损失",
    "損失", "盈利", "亏损", "虧損", "融资", "融資", "贷款", "貸款", "存款", "取款",
    "转账", "轉帳", "支付", "结算", "結算", "报销", "報銷", "审计", "審計", "纳税",
    "納稅", "保险", "保險", "理赔", "理賠", "诉讼", "訴訟", "仲裁", "调解", "調解",
    "判决", "判決", "辩护", "辯護", "起诉", "起訴", "举报", "舉報", "侦查", "偵查",
    "逮捕", "审判", "審判", "监禁", "監禁", "释放", "釋放", "逃跑", "追捕", "搜查",
    "巡逻", "巡邏", "警戒", "防卫", "防衛", "进攻", "進攻", "撤退", "投降", "占领",
    "佔領", "解放", "胜利", "勝利", "牺牲", "犧牲", "救援", "抢救", "搶救", "护理",
    "護理", "保健", "康复", "康復", "诊断", "診斷", "预防", "預防", "接种", "接種",
    "免疫", "感染", "传染", "傳染", "中毒", "过敏", "過敏", "发炎", "發炎", "出血",
    "恢复", "恢復", "成长", "成長", "发育", "發育", "衰老", "死亡", "埋葬", "祭祀",
    "祈祷", "祈禱", "祝福", "祝贺", "祝賀", "欢迎", "歡迎", "欢送", "歡送", "告别",
    "告別", "拜访", "拜訪", "接待", "陪伴", "照顾", "照顧", "抚养", "撫養", "赡养",
    "贍養", "结婚", "結婚", "离婚", "離婚", "恋爱", "戀愛", "约会", "約會", "分手",
    "聚会", "聚會", "宴会", "宴會", "游玩", "遊玩", "观光", "觀光", "探险", "探險",
    "攀登", "游泳", "滑冰", "滑雪", "跑步", "散步", "锻炼", "鍛煉", "竞赛", "競賽",
    "挑战", "挑戰", "突破", "进步", "進步", "退步", "提高", "降低", "增加", "减少",
    "減少", "扩大", "擴大", "缩小", "縮小", "加强", "加強", "削弱", "巩固", "鞏固",
    "完善", "改善", "恶化", "惡化", "稳定", "穩定", "波动", "波動", "增长", "增長",
    "下降", "上升", "跌落", "反弹", "反彈", "恢复", "恢復", "延续", "延續", "中断",
    "中斷", "停止", "暂停", "暫停", "结束", "結束", "完成", "实现", "實現", "达成",
    "達成", "获得", "獲得", "取得", "赢得", "贏得", "失去", "丧失", "喪失", "放弃",
    "放棄", "坚持", "堅持", "继续", "繼續", "维持", "維持", "保持", "保留", "取消",
    "撤销", "撤銷", "废除", "廢除", "建立", "成立", "设立", "設立", "创办", "創辦",
    "建设", "建設", "修建", "改造", "重建", "拆除", "破坏", "破壞", "毁坏", "毀壞",
    "修复", "修復", "补充", "補充", "替换", "替換", "更换", "更換", "转移", "轉移",
    "移动", "移動", "流动", "流動", "传播", "傳播", "扩散", "擴散", "集中", "分散",
    "集合", "聚集", "解散", "合并", "合併", "分离", "分離", "分裂", "统一", "統一",
    "独立", "獨立", "联合", "聯合", "结合", "結合", "连接", "連接", "联系", "聯繫",
    "接触", "接觸", "碰撞", "冲突", "衝突", "矛盾", "斗争", "鬥爭", "抗议", "抗議",
    "谈判", "談判", "协商", "協商", "妥协", "妥協", "让步", "讓步", "合作", "配合",
    "响应", "響應", "反馈", "反饋", "回应", "回應", "回答", "提问", "提問", "咨询",
    "諮詢", "商量", "讨论", "討論", "争论", "爭論", "辩论", "辯論", "吵架", "打架",
}

# Explicit cross-strait pairs (zh-Hant <-> zh-Hans)
CROSS_STRAIT_PAIRS = [
    ("軟體", "软件"),
    ("硬體", "硬件"),
    ("記憶體", "内存"),
    ("計程車", "出租车"),
    ("資訊", "信息"),
    ("網路", "网络"),
    ("網際網路", "互联网"),
    ("人工智慧", "人工智能"),
    ("機器學習", "机器学习"),
    ("智慧型手機", "智能手机"),
    ("積體電路", "集成电路"),
    ("資料庫", "数据库"),
    ("身分證", "身份证"),
    ("行動電源", "移动电源"),
    ("原始碼", "源代码"),
    ("作業系統", "操作系统"),
    ("隨身碟", "优盘"),
    ("印表機", "打印机"),
    ("滑鼠", "鼠标"),
    ("螢幕", "屏幕"),
    ("電腦", "电脑"),
    ("程式", "程序"),
    ("部落格", "博客"),
    ("伺服器", "服务器"),
    ("資料", "资料"),
    ("檔案", "文件"),
    ("光碟", "光盘"),
    ("雷射", "激光"),
    ("數位", "数字"),
    ("寬頻", "宽带"),
    ("頻寬", "带宽"),
    ("解析度", "分辨率"),
    ("預設", "默认"),
    ("載入", "加载"),
    ("執行", "执行"),
    ("設定", "设置"),
    ("連結", "链接"),
    ("超連結", "超链接"),
    ("瀏覽器", "浏览器"),
    ("搜尋引擎", "搜索引擎"),
    ("演算法", "算法"),
    ("物件導向", "面向对象"),
    ("介面", "界面"),
    ("相容", "兼容"),
    ("相容性", "兼容性"),
    ("同步", "同步"),
    ("非同步", "异步"),
    ("備份", "备份"),
    ("還原", "还原"),
    ("登錄", "登录"),
    ("註冊", "注册"),
    ("帳號", "账号"),
    ("密碼", "密码"),
    ("郵件", "邮件"),
    ("電子郵件", "电子邮件"),
    ("影片", "视频"),
    ("音訊", "音频"),
    ("視訊", "视频"),
    ("動畫", "动画"),
    ("捷運", "地铁"),
    ("公車", "公交车"),
    ("高鐵", "高铁"),
    ("腳踏車", "自行车"),
    ("機車", "摩托车"),
    ("便當", "盒饭"),
    ("鳳梨", "菠萝"),
    ("馬鈴薯", "土豆"),
    ("番茄", "西红柿"),
    ("高麗菜", "卷心菜"),
    ("花生", "花生"),
    ("起司", "奶酪"),
    ("優格", "酸奶"),
    ("巧克力", "巧克力"),
    ("冰淇淋", "冰淇淋"),
    ("咖啡", "咖啡"),
    ("臺灣", "台湾"),
    ("台灣", "台湾"),
    ("臺北", "台北"),
    ("台北", "台北"),
    ("臺中", "台中"),
    ("臺南", "台南"),
    ("高雄", "高雄"),
]


def is_clean_han(word: str) -> bool:
    """Return True iff word is non-empty and every char is a CJK Unified Ideograph."""
    if not word:
        return False
    for ch in word:
        cp = ord(ch)
        if not (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x20000 <= cp <= 0x2EBEF)
        ):
            return False
    return True


def load_opencc_dict(path: Path) -> dict[str, str]:
    """Load OpenCC tab-separated dictionary, taking the first target value."""
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                key = parts[0]
                val = parts[1].split()[0]
                if key and val:
                    mapping[key] = val
    return mapping


class OpenCCConverter:
    """Lightweight longest-match OpenCC converter using Apache-2.0 tables."""

    def __init__(self, opencc_dir: Path) -> None:
        self.st_phrases = load_opencc_dict(opencc_dir / "STPhrases.txt")
        self.st_chars = load_opencc_dict(opencc_dir / "STCharacters.txt")
        self.tw_phrases = load_opencc_dict(opencc_dir / "TWPhrases.txt")
        self.tw_variants = load_opencc_dict(opencc_dir / "TWVariants.txt")
        self.tw_var_phrases = load_opencc_dict(opencc_dir / "TWVariantsPhrases.txt")

        self.ts_phrases = load_opencc_dict(opencc_dir / "TSPhrases.txt")
        self.ts_chars = load_opencc_dict(opencc_dir / "TSCharacters.txt")
        self.tw_phrases_rev = load_opencc_dict(opencc_dir / "TWPhrasesRev.txt")

        # Build reverse TWVariants (for normalizing rare variant chars back if needed)
        self.hant_to_hans_override: dict[str, str] = {}
        self.hans_to_hant_override: dict[str, str] = {}
        for hant, hans in CROSS_STRAIT_PAIRS:
            if hant != hans:
                self.hant_to_hans_override[hant] = hans
                if hans not in self.hans_to_hant_override:
                    self.hans_to_hant_override[hans] = hant

        self.max_st_len = max((len(k) for k in self.st_phrases), default=1)
        self.max_tw_len = max((len(k) for k in self.tw_phrases), default=1)
        self.max_ts_len = max((len(k) for k in self.ts_phrases), default=1)
        self.max_tw_rev_len = max((len(k) for k in self.tw_phrases_rev), default=1)

    @staticmethod
    def _apply_longest(text: str, phrases: dict[str, str], chars: dict[str, str], max_len: int) -> str:
        out: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for L in range(min(max_len, n - i), 1, -1):
                sub = text[i : i + L]
                if sub in phrases:
                    out.append(phrases[sub])
                    i += L
                    matched = True
                    break
            if not matched:
                ch = text[i]
                out.append(chars.get(ch, ch))
                i += 1
        return "".join(out)

    def hans_to_hant(self, text: str) -> str:
        if text in self.hans_to_hant_override:
            return self.hans_to_hant_override[text]
        t = self._apply_longest(text, self.st_phrases, self.st_chars, self.max_st_len)
        t = self._apply_longest(t, self.tw_var_phrases, self.tw_variants, 8)
        t = self._apply_longest(t, self.tw_phrases, {}, self.max_tw_len)
        return t

    def hant_to_hans(self, text: str) -> str:
        if text in self.hant_to_hans_override:
            return self.hant_to_hans_override[text]
        t = self._apply_longest(text, self.tw_phrases_rev, {}, self.max_tw_rev_len)
        t = self._apply_longest(t, self.ts_phrases, self.ts_chars, self.max_ts_len)
        return t


def parse_pos_rs() -> tuple[dict[str, str], set[str]]:
    """Parse ZH_DECIDED and ZH_AMBIGUOUS from crates/faberline-nlp/src/doc/pos.rs."""
    pos_rs = NLP_CRATE_ROOT / "src" / "doc" / "pos.rs"
    src = pos_rs.read_text(encoding="utf-8")

    zh_decided: dict[str, str] = {}
    m_dec = re.search(r"const ZH_DECIDED:\s*&\[\(Upos,\s*&str\)\]\s*=\s*&\[(.*?)\n\];", src, re.S)
    if not m_dec:
        raise RuntimeError("Could not locate ZH_DECIDED in src/doc/pos.rs")
    for upos_str, words_str in re.findall(r'\(\s*Upos::(\w+)\s*,\s*"([^"]+)"\s*,?\s*\)', m_dec.group(1)):
        upos = upos_str.upper()
        for w in words_str.split():
            zh_decided[w] = upos

    zh_ambiguous: set[str] = set()
    m_amb = re.search(r"const ZH_AMBIGUOUS:\s*&\[&str\]\s*=\s*&\[(.*?)\n\];", src, re.S)
    if not m_amb:
        raise RuntimeError("Could not locate ZH_AMBIGUOUS in src/doc/pos.rs")
    for words_str in re.findall(r'"([^"]+)"', m_amb.group(1)):
        for w in words_str.split():
            zh_ambiguous.add(w)

    return zh_decided, zh_ambiguous


def load_in_repo_zh_words() -> list[str]:
    """Collect curated Chinese words from src/doc/token.rs, src/text.rs, src/language.rs."""
    words: list[str] = []
    for rel in ("src/doc/token.rs", "src/text.rs", "src/language.rs"):
        p = NLP_CRATE_ROOT / rel
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        for s in re.findall(r'"([^"\n]+)"', src):
            if is_clean_han(s) and 1 <= len(s) <= 8:
                words.append(s)
    return words


def load_gold_pos() -> dict[str, set[str]]:
    """Load all Chinese/mixed token -> set(UPOS) from pos_gold.json and gold_layers4_6_v1.jsonl."""
    gold: dict[str, set[str]] = defaultdict(set)

    pos_gold_path = NLP_CRATE_ROOT / "tests" / "data" / "pos_gold.json"
    with open(pos_gold_path, "r", encoding="utf-8") as f:
        for item in json.load(f):
            if item.get("lang") in ("zh", "mixed"):
                for tok, upos in item["tokens"]:
                    if is_clean_han(tok):
                        gold[tok].add(upos)

    layers_path = NLP_CRATE_ROOT / "tests" / "data" / "gold_layers4_6_v1.jsonl"
    with open(layers_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("lang") in ("zh", "mixed"):
                for tok, upos in zip(item["tokens"], item["upos"]):
                    if is_clean_han(tok):
                        gold[tok].add(upos)

    return gold


def mine_corpora_npmi(pattern: str, min_count: int = 6, min_pmi: float = 4.5) -> tuple[Counter, dict[str, float]]:
    """Scan CC0/PD corpora for Han unigram/n-gram counts and high-PMI 2-4 char words."""
    unigrams: Counter = Counter()
    ngrams: Counter = Counter()
    total_chars = 0

    han_run_re = re.compile(r"[\u4e00-\u9fff]+")
    for path_str in sorted(glob.glob(str(CORPORA_DIR / pattern))):
        with open(path_str, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line).get("text", "")
                for run in han_run_re.findall(text):
                    L = len(run)
                    total_chars += L
                    for ch in run:
                        unigrams[ch] += 1
                    for n in (2, 3, 4):
                        if L >= n:
                            for i in range(L - n + 1):
                                ngrams[run[i : i + n]] += 1

    mined: dict[str, float] = {}
    if total_chars == 0:
        return unigrams, mined

    for ng, cnt in ngrams.items():
        if cnt < min_count:
            continue
        if any(c in NPMI_STOP_CHARS for c in ng):
            continue
        # Compute minimum split PMI across all binary splits
        p_ng = cnt / total_chars
        min_split_pmi = float("inf")
        for split in range(1, len(ng)):
            left = ng[:split]
            right = ng[split:]
            c_left = unigrams[left] if len(left) == 1 else ngrams[left]
            c_right = unigrams[right] if len(right) == 1 else ngrams[right]
            if c_left == 0 or c_right == 0:
                min_split_pmi = -1.0
                break
            p_left = c_left / total_chars
            p_right = c_right / total_chars
            pmi = math.log2(p_ng / (p_left * p_right))
            if pmi < min_split_pmi:
                min_split_pmi = pmi
        if min_split_pmi >= min_pmi:
            mined[ng] = cnt * min_split_pmi

    return unigrams, mined


def map_ictclas(word: str, tag: str) -> tuple[str, str, str]:
    """Map ICTCLAS tag to (decided_upos, upos_mask, meta)."""
    meta = "-"
    if tag == "nr" or tag == "nrt" or tag == "nrfg":
        meta = "sem=person"
        return "PROPN", "PROPN", meta
    if tag == "ns":
        meta = "sem=location"
        return "PROPN", "PROPN", meta
    if tag == "nt":
        meta = "sem=organization"
        return "PROPN", "PROPN", meta
    if tag == "nz":
        return "PROPN", "PROPN", meta
    if tag == "t":
        meta = "sem=time"
        return "NOUN", "NOUN", meta
    if tag == "m":
        if all(c in ZH_NUMERALS for c in word):
            return "NUM", "NUM", "sem=quantity"
        # Multi-char 'm' in jieba often includes non-numeral quantifiers/adverbs (e.g. 一起, 整座)
        return "-", "DET|NUM", "sem=quantity"

    if word in NOUN_VERB_WORDS or tag == "vn":
        return "-", "NOUN|VERB", meta
    if tag in ("n", "s", "q", "ng"):
        return "NOUN", "NOUN", meta
    if tag == "f":
        return "-", "ADP|NOUN", meta
    if tag in ("v", "vg"):
        return "VERB", "VERB", meta
    if tag in ("a", "ag", "b", "z"):
        return "ADJ", "ADJ", meta
    if tag == "ad":
        return "-", "ADJ|ADV", meta
    if tag == "d":
        return "ADV", "ADV", meta
    if tag == "p":
        return "ADP", "ADP", meta
    if tag == "c":
        return "-", "CCONJ|SCONJ", meta
    if tag == "u":
        return "PART", "PART", meta
    if tag == "r":
        return "-", "DET|PRON", meta
    if tag in ("y", "e", "o"):
        return "-", "INTJ|PART", meta
    if tag in ("i", "l", "j"):
        return "-", "ADJ|NOUN|VERB", meta
    return "-", "NOUN|VERB", meta


def resolve_pos(
    word: str,
    ictclas_tag: str,
    zh_decided: dict[str, str],
    zh_ambiguous: set[str],
    gold_pos: dict[str, set[str]],
    conv: OpenCCConverter,
) -> tuple[str, str, str]:
    """Determine (decided_upos, upos_mask, meta) with 0 conflicts against pos.rs and gold."""
    decided, mask, meta = map_ictclas(word, ictclas_tag)

    # Check gold in both exact form and cross-script form
    gold_tags: set[str] = set()
    if word in gold_pos:
        gold_tags.update(gold_pos[word])
    w_hant = conv.hans_to_hant(word)
    w_hans = conv.hant_to_hans(word)
    if w_hant in gold_pos:
        gold_tags.update(gold_pos[w_hant])
    if w_hans in gold_pos:
        gold_tags.update(gold_pos[w_hans])

    # 1. pos.rs ZH_DECIDED takes absolute precedence
    if word in zh_decided:
        d = zh_decided[word]
        return d, d, meta

    # 2. pos.rs ZH_AMBIGUOUS must have decided_upos = "-"
    if word in zh_ambiguous:
        tags = set(mask.split("|")) if mask != "-" else set()
        tags.update(gold_tags)
        if not tags:
            tags = {"ADV", "PART", "VERB"}
        return "-", "|".join(sorted(tags)), meta

    # 3. Gold compatibility check
    if gold_tags:
        if len(gold_tags) > 1:
            # Multi-POS in gold -> MUST be "-"
            tags = set(mask.split("|")) | gold_tags
            return "-", "|".join(sorted(tags)), meta
        only_gold = next(iter(gold_tags))
        if decided != "-" and decided != only_gold:
            # Jieba tag disagreed with gold (e.g., jieba noisy 'nr' on common word)
            # Clear false semantic tag if PROPN was wrong
            if decided == "PROPN" and only_gold != "PROPN":
                meta = "-"
            return only_gold, only_gold, meta
        if decided == "-":
            tags = set(mask.split("|")) | gold_tags
            return "-", "|".join(sorted(tags)), meta

    return decided, mask, meta


def compute_cost(freq: float, total_freq: float) -> int:
    if freq <= 0 or total_freq <= 0:
        return 1800
    p = min(1.0, freq / total_freq)
    c = round(-math.log(p) * 100.0)
    return max(200, min(1800, c))


def write_tsv_atomic(
    out_path: Path,
    tmp_path: Path,
    variant: str,
    entries: dict[str, tuple[int, str, str, str, str]],
) -> int:
    """Sort by UTF-8 byte order of (surface, variant), write to tmp, and os.replace."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(
        entries.items(),
        key=lambda kv: (kv[0].encode("utf-8"), variant.encode("utf-8")),
    )

    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            f"# {variant} lexicon (8-col canonical TSV)\n"
            "# Sources: jieba-rs (MIT), OpenCC (Apache-2.0), in-repo tables, CC0/PD corpora\n"
            "# surface\tvariant\tcost\tdecided_upos\tupos_mask\tlemma\tcounterpart\tmeta\n"
        )
        for surface, (cost, dec_upos, upos_mask, lemma, counterpart, meta) in [
            (k, v) for k, v in sorted_items
        ]:
            f.write(
                f"{surface}\t{variant}\t{cost}\t{dec_upos}\t{upos_mask}\t{lemma}\t{counterpart}\t{meta}\n"
            )

    os.replace(tmp_path, out_path)
    return len(sorted_items)


def verify_tsv(
    path: Path,
    expected_variant: str,
    zh_decided: dict[str, str],
    zh_ambiguous: set[str],
    gold_pos: dict[str, set[str]],
) -> dict[str, object]:
    """Verify UTF-8 byte sort order, 8-column invariants, and 0 gold/pos.rs conflicts."""
    prev_key: tuple[bytes, bytes] | None = None
    seen: set[tuple[str, str]] = set()
    rows = 0
    one_char = 0
    multi_char = 0
    conflicts: list[str] = []
    entries_map: dict[str, tuple[str, str]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if raw.startswith("#"):
                continue
            line = raw.rstrip("\n")
            cols = line.split("\t")
            assert len(cols) == 8, f"{path}:{line_no}: expected 8 cols, got {len(cols)}"
            surf, var, cost_s, dec, mask, lemma, cp, meta = cols

            assert is_clean_han(surf), f"{path}:{line_no}: dirty surface {surf!r}"
            assert var == expected_variant, f"{path}:{line_no}: bad variant {var!r}"
            cost = int(cost_s)
            assert 200 <= cost <= 1800, f"{path}:{line_no}: cost out of range {cost}"
            assert dec == "-" or dec in VALID_UPOS, f"{path}:{line_no}: bad dec {dec!r}"
            mask_parts = mask.split("|")
            assert all(m in VALID_UPOS for m in mask_parts), f"{path}:{line_no}: bad mask {mask!r}"
            assert mask_parts == sorted(set(mask_parts)), f"{path}:{line_no}: unsorted mask {mask!r}"
            if dec != "-":
                assert mask == dec, f"{path}:{line_no}: dec={dec} != mask={mask}"
            assert lemma == "-", f"{path}:{line_no}: bad lemma {lemma!r}"
            if cp != "-":
                assert is_clean_han(cp) and cp != surf, f"{path}:{line_no}: bad counterpart {cp!r}"

            key = (surf.encode("utf-8"), var.encode("utf-8"))
            if prev_key is not None:
                assert key > prev_key, f"{path}:{line_no}: sort/dup violation at {surf!r}"
            prev_key = key
            assert (surf, var) not in seen
            seen.add((surf, var))

            if surf in zh_decided and dec != zh_decided[surf]:
                conflicts.append(f"ZH_DECIDED mismatch {surf}: {dec} vs {zh_decided[surf]}")
            if surf in zh_ambiguous and dec != "-":
                conflicts.append(f"ZH_AMBIGUOUS mismatch {surf}: {dec} != '-'")
            if surf in gold_pos:
                g = gold_pos[surf]
                if len(g) > 1 and dec != "-":
                    conflicts.append(f"Gold multi-POS mismatch {surf}: {dec} vs {g}")
                elif len(g) == 1 and dec != "-" and dec != next(iter(g)):
                    conflicts.append(f"Gold single-POS mismatch {surf}: {dec} vs {g}")
                if not (g & set(mask_parts)):
                    conflicts.append(f"Gold mask missing {surf}: {mask} vs {g}")

            if len(surf) == 1:
                one_char += 1
            else:
                multi_char += 1
            rows += 1
            entries_map[surf] = (dec, mask)

    for w in ("研究", "發展", "发展", "開始", "开始", "工作"):
        if w in entries_map:
            dec, mask = entries_map[w]
            assert dec == "-" and mask == "NOUN|VERB", f"{path}: {w} has {dec}, {mask}"

    assert not conflicts, f"Conflicts in {path}: {conflicts[:10]}"
    return {
        "rows": rows,
        "one_char": one_char,
        "multi_char": multi_char,
        "bytes": path.stat().st_size,
    }


def main() -> None:
    print("Loading OpenCC tables...")
    conv = OpenCCConverter(OPENCC_DIR)

    print("Parsing src/doc/pos.rs and gold datasets...")
    zh_decided, zh_ambiguous = parse_pos_rs()
    gold_pos = load_gold_pos()
    in_repo_words = load_in_repo_zh_words()
    print(f"  ZH_DECIDED={len(zh_decided)}, ZH_AMBIGUOUS={len(zh_ambiguous)}, gold_tokens={len(gold_pos)}")

    print("Mining CC0/PD zh corpora (NPMI + unigram counts)...")
    hans_uni, hans_mined = mine_corpora_npmi("zh_hans_*.jsonl", min_count=10, min_pmi=5.5)
    hant_uni, hant_mined = mine_corpora_npmi("zh_hant_*.jsonl", min_count=10, min_pmi=5.5)
    print(f"  zh-Hans mined={len(hans_mined)}, zh-Hant mined={len(hant_mined)}")

    print("Loading jieba-rs dict.txt...")
    jieba_entries: dict[str, tuple[int, str]] = {}
    with open(JIEBA_DICT, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            w = parts[0]
            freq = int(parts[1])
            tag = parts[2] if len(parts) >= 3 else "n"
            if not is_clean_han(w) or len(w) > 8:
                continue
            if w in jieba_entries:
                old_f, old_t = jieba_entries[w]
                if freq > old_f:
                    jieba_entries[w] = (freq, tag)
            else:
                jieba_entries[w] = (freq, tag)
    print(f"  Clean Han entries from jieba: {len(jieba_entries)}")

    # Mandatory words that MUST be included in both scripts
    mandatory_hans: set[str] = set()
    mandatory_hant: set[str] = set()

    for w in list(zh_decided) + list(zh_ambiguous) + list(gold_pos) + in_repo_words:
        if is_clean_han(w):
            mandatory_hans.add(conv.hant_to_hans(w))
            mandatory_hant.add(conv.hans_to_hant(w))
            # Keep original form too if it's valid in that script
            if w in gold_pos or w in zh_decided or w in zh_ambiguous:
                mandatory_hans.add(w)
                mandatory_hant.add(w)

    for hant, hans in CROSS_STRAIT_PAIRS:
        if is_clean_han(hant) and is_clean_han(hans):
            mandatory_hant.add(hant)
            mandatory_hans.add(hans)

    # Add OpenCC TWPhrases clean Han targets to mandatory_hant
    for k, v in conv.tw_phrases.items():
        if is_clean_han(v) and 2 <= len(v) <= 6:
            mandatory_hant.add(v)
        if is_clean_han(k) and 2 <= len(k) <= 6:
            mandatory_hans.add(conv.hant_to_hans(k))

    # Build scored candidate pools for zh-Hans and zh-Hant
    hans_pool: dict[str, tuple[float, str]] = {}
    hant_pool: dict[str, tuple[float, str]] = {}

    for w, (freq, tag) in jieba_entries.items():
        # Filter out 1-char jieba garbage
        if len(w) == 1:
            if freq < 200 and hans_uni[w] == 0 and w not in mandatory_hans:
                continue
        hans_w = conv.hant_to_hans(w)
        if not is_clean_han(hans_w):
            continue
        boost = hans_uni[hans_w] * 50.0 if len(hans_w) == 1 else 0.0
        score = float(freq) + boost
        if hans_w not in hans_pool or score > hans_pool[hans_w][0]:
            hans_pool[hans_w] = (score, tag)

        hant_w = conv.hans_to_hant(w)
        if is_clean_han(hant_w):
            if len(hant_w) == 1 and freq < 200 and hant_uni[hant_w] == 0 and hant_w not in mandatory_hant:
                pass
            else:
                h_boost = hant_uni[hant_w] * 50.0 if len(hant_w) == 1 else 0.0
                h_score = float(freq) + h_boost
                if hant_w not in hant_pool or h_score > hant_pool[hant_w][0]:
                    hant_pool[hant_w] = (h_score, tag)

    # Add NPMI-mined words from corpora
    for w, npmi_score in hans_mined.items():
        if is_clean_han(w) and w not in hans_pool:
            hans_pool[w] = (max(50.0, npmi_score * 5.0), "n")
    for w, npmi_score in hant_mined.items():
        if is_clean_han(w) and w not in hant_pool:
            hant_pool[w] = (max(50.0, npmi_score * 5.0), "n")

    # Ensure all mandatory words are in pools
    for w in mandatory_hans:
        if w not in hans_pool:
            tag = jieba_entries[w][1] if w in jieba_entries else "n"
            freq = float(jieba_entries[w][0]) if w in jieba_entries else 500.0
            hans_pool[w] = (freq, tag)
    for w in mandatory_hant:
        if w not in hant_pool:
            hans_eq = conv.hant_to_hans(w)
            tag = jieba_entries[hans_eq][1] if hans_eq in jieba_entries else "n"
            freq = float(jieba_entries[hans_eq][0]) if hans_eq in jieba_entries else 500.0
            hant_pool[w] = (freq, tag)

    # Boost curated domain/classical/Taiwan terms & plural pronouns so they stay whole,
    # and add curated person names.
    boost_set = set(CURATED_BOOST_WORDS)
    for hant, hans in CROSS_STRAIT_PAIRS:
        boost_set.add(hant)
        boost_set.add(hans)
    for w in boost_set:
        if is_clean_han(w):
            w_hans = conv.hant_to_hans(w)
            w_hant = conv.hans_to_hant(w)
            for target_w, pool in ((w, hans_pool), (w_hans, hans_pool), (w, hant_pool), (w_hant, hant_pool)):
                if is_clean_han(target_w):
                    old_sc, old_tag = pool.get(target_w, (0.0, "n"))
                    pool[target_w] = (max(old_sc, 12000.0), old_tag)
    for name in CURATED_PERSON_NAMES:
        for target_w, pool in ((name, hant_pool), (conv.hant_to_hans(name), hans_pool)):
            if is_clean_han(target_w):
                old_sc, _ = pool.get(target_w, (0.0, "nr"))
                pool[target_w] = (max(old_sc, 8000.0), "nr")

    # Remove excluded false merges from pools & mandatory sets.
    for w in EXCLUDED_WORDS:
        mandatory_hans.discard(w)
        mandatory_hant.discard(w)
        hans_pool.pop(w, None)
        hant_pool.pop(w, None)

    def select_top(
        pool: dict[str, tuple[float, str]],
        mandatory: set[str],
        target_multi: int,
    ) -> dict[str, tuple[float, str]]:
        selected: dict[str, tuple[float, str]] = {}
        multi_candidates: list[tuple[float, str, str]] = []
        for w, (sc, tag) in pool.items():
            if len(w) == 1 or w in mandatory:
                selected[w] = (sc, tag)
            else:
                multi_candidates.append((sc, w, tag))
        multi_candidates.sort(key=lambda x: (-x[0], x[1]))
        need = max(0, target_multi - sum(1 for w in selected if len(w) > 1))
        for sc, w, tag in multi_candidates[:need]:
            selected[w] = (sc, tag)
        return selected

    hans_sel = select_top(hans_pool, mandatory_hans, target_multi=66000)
    hant_sel = select_top(hant_pool, mandatory_hant, target_multi=62000)

    def build_rows(
        selected: dict[str, tuple[float, str]],
        is_hant: bool,
    ) -> dict[str, tuple[int, str, str, str, str, str]]:
        total_freq = sum(sc for sc, _ in selected.values())
        rows: dict[str, tuple[int, str, str, str, str, str]] = {}
        for w, (sc, tag) in selected.items():
            cost = compute_cost(sc, total_freq)
            dec, mask, meta = resolve_pos(w, tag, zh_decided, zh_ambiguous, gold_pos, conv)
            cp = conv.hant_to_hans(w) if is_hant else conv.hans_to_hant(w)
            if not is_clean_han(cp) or cp == w:
                cp = "-"
            rows[w] = (cost, dec, mask, "-", cp, meta)
        return rows

    hant_rows = build_rows(hant_sel, is_hant=True)
    hans_rows = build_rows(hans_sel, is_hant=False)

    print("Writing atomic TSVs...")
    write_tsv_atomic(OUT_HANT, TMP_DIR / "zh_hant.tsv.tmp", "zh-Hant", hant_rows)
    write_tsv_atomic(OUT_HANS, TMP_DIR / "zh_hans.tsv.tmp", "zh-Hans", hans_rows)

    print("Verifying invariants and gold compatibility...")
    stats_hant = verify_tsv(OUT_HANT, "zh-Hant", zh_decided, zh_ambiguous, gold_pos)
    stats_hans = verify_tsv(OUT_HANS, "zh-Hans", zh_decided, zh_ambiguous, gold_pos)
    print(f"  zh_hant.tsv: {stats_hant}")
    print(f"  zh_hans.tsv: {stats_hans}")


if __name__ == "__main__":
    main()
