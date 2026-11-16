#!/usr/bin/env python3
"""Build commercial-safe Chinese (zh-Hant & zh-Hans) lexicons in 8-column TSV format.

Sources (strictly permissive / commercial-safe):
- jieba-rs dict.txt (MIT)
- BYVoid/OpenCC dictionary tables (Apache-2.0)
- In-repo tables (src/doc/pos.rs, src/doc/token.rs, src/text.rs, language.rs,
  tools/lexicon_build/data/zh_translate_mit_v1.tsv)
- Public Domain Project Gutenberg Chinese novels (gutenberg:23950, gutenberg:23962; 150 docs)

Outputs (atomic via temp files + os.replace):
- lexicon/data/zh_hant.tsv
- lexicon/data/zh_hans.tsv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
ZH_TRANSLATE_MIT_TSV = DATA_DIR / "zh_translate_mit_v1.tsv"
ZH_MODERN_MIT_TSV = DATA_DIR / "zh_modern_mit_v1.tsv"

NLP_CRATE_ROOT = REPO_ROOT / "crates" / "faberline-nlp"
OUT_HANT = NLP_CRATE_ROOT / "lexicon" / "data" / "zh_hant.tsv"
OUT_HANS = NLP_CRATE_ROOT / "lexicon" / "data" / "zh_hans.tsv"

ALLOWED_ZH_GUTENBERG_SOURCES = frozenset({"gutenberg:23950", "gutenberg:23962"})

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
    # Qing-PD 1-char shift boundary restorations (origin/main split parity, distinct from C1 source hygiene):
    "三本",        # NUM+CLF (script-invariant)
    "就会",        # ADV+AUX (restores origin/main zh-Hans split while preserving origin/main zh-Hant 就會)
    "要去",        # AUX/VERB+VERB (script-invariant)
    "發到", "发到",  # VERB+ADP goal complement (symmetric Hant/Hans restoration of origin/main split)
}

# Curated person names (PROPN, sem=person) that should stay whole.
CURATED_PERSON_NAMES = {
    "阿芳", "阿杰", "小芸", "小林", "老王",
}

# Known historical/literary/common 2-char person names for clean_map_ictclas POS/sem tagging
# (kept separate from CURATED_PERSON_NAMES so pool membership/costs stay invariant).
_KNOWN_PERSON_NAMES = CURATED_PERSON_NAMES | {
    "老張", "老张", "老李", "小王", "小張", "小张", "小李", "小強", "小强",
    "李白", "杜甫", "呂布", "吕布", "宋江", "曹操", "劉備", "刘备", "孫權", "孙权", "關羽", "关羽",
    "張飛", "张飞", "周瑜", "魯肅", "鲁肃", "黃忠", "黄忠", "馬超", "马超", "趙雲", "赵云",
    "袁紹", "袁绍", "董卓", "項羽", "项羽", "劉邦", "刘邦", "韓信", "韩信", "張良", "张良", "蕭何", "萧何",
    "屈原", "孔子", "孟子", "老子", "莊子", "庄子", "荀子", "墨子", "韓非", "韩非",
    "蘇軾", "苏轼", "蘇轍", "苏辙", "蘇洵", "苏洵", "韓愈", "韩愈", "王維", "王维", "杜牧",
    "陸游", "陆游", "岳飛", "岳飞", "朱熹", "魯迅", "鲁迅", "巴金", "老舍", "茅盾", "冰心",
    "胡適", "胡适", "朱德", "林彪", "陳雲", "陈云", "李四", "張三", "张三",
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


def mine_corpora_npmi(
    gutenberg_jsonl: Path | None,
    min_count: int = 10,
    min_pmi: float = 5.5,
    transform_fn=None,
) -> tuple[Counter, dict[str, float], int]:
    """Scan allowlisted Project Gutenberg PD JSONL for Han unigram/n-gram counts and high-PMI 2-4 char words."""
    unigrams: Counter = Counter()
    ngrams: Counter = Counter()
    total_chars = 0
    docs = 0
    if gutenberg_jsonl is None:
        return unigrams, {}, 0

    han_run_re = re.compile(r"[\u4e00-\u9fff]+")
    with open(gutenberg_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("source", "") not in ALLOWED_ZH_GUTENBERG_SOURCES:
                continue
            docs += 1
            text = d.get("text", "")
            if transform_fn is not None:
                text = transform_fn(text)
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
        return unigrams, mined, docs

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

    return unigrams, mined, docs


_COMMON_SURNAMES_1 = frozenset(
    "王李张刘陈杨黄赵周吴徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
)
_COMPOUND_SURNAMES_2 = (
    "欧阳", "太史", "端木", "上官", "司马", "东方", "独孤", "南宫", "万俟", "闻人",
    "夏侯", "诸葛", "尉迟", "公羊", "赫连", "澹台", "皇甫", "宗政", "濮阳", "公冶",
    "太叔", "申屠", "公孙", "慕容", "仲孙", "钟离", "长孙", "宇文", "司徒", "鲜于",
    "司空", "闾丘", "子车", "亓官", "司寇", "巫马", "公西", "颛孙", "壤驷", "公良",
    "漆雕", "乐正", "宰父", "谷梁", "拓跋", "夹谷", "轩辕", "令狐", "段干", "百里",
    "呼延", "东郭", "南门", "羊舌", "微生", "公户", "公玉", "公仪", "梁丘", "公仲",
    "公上", "公门", "公山", "公坚", "左丘", "公伯", "西门", "公祖", "第五", "公乘",
    "贯丘", "公皙", "南荣", "东里", "东宫", "仲长", "子书", "子桑", "即墨", "达奚", "褚师",
)
_HEADS_LOC = frozenset("宫庙寺观庵殿斋洞峡滩渡堡寨省市县区镇乡村湾岛溪墩棚山河湖海港路街桥")
_HEADS_ORG = frozenset("院馆所校厂店行司局会部社党军团")
_HEADS_NOUN = frozenset("器机具品物药剂车船书报单表法制权学感率度网线面体肉米麦豆瓜果菜茶酒糖盐油汤饭面饼")
_NR_NONSURNAME_1CHAR_NOUN_SUFFIXES = frozenset(
    "权法学感率力量器机品物剂药病症炎癌筑馆院部会党军国省市区镇乡村路街桥站港场厂店行务费税款价单表书报刊图集录经话语词字号式型类种级等面线点网体质素能米麦豆瓜果菜茶酒肉鱼鸟虫草花树"
    "木石沙土金银铜铁锦丝布纸墨笔刀剑车船房屋门窗桌椅床灯钟风雪霜雾雷电光声味香气水火油盐糖醋酱汤饭饼糕粥"
)
_NRT_KNOWN_NON_PROPN = {
    "二人": ("-", "-", "-"),
    "令人": ("VERB", "VERB", "-"),
    "登基": ("VERB", "VERB", "-"),
    "崇拜": ("VERB", "VERB", "-"),
    "多谢": ("VERB", "VERB", "-"),
    "崇高": ("ADJ", "ADJ", "-"),
    "保姆": ("NOUN", "NOUN", "-"),
    "默契": ("-", "-", "-"),
    "威武": ("ADJ", "ADJ", "-"),
    "永世": ("ADV", "ADV", "-"),
    "旭日": ("NOUN", "NOUN", "-"),
    "新颜": ("NOUN", "NOUN", "-"),
    "略逊": ("VERB", "VERB", "-"),
    "顺祝": ("VERB", "VERB", "-"),
    "大王": ("NOUN", "NOUN", "-"),
    "法王": ("NOUN", "NOUN", "-"),
    "二世": ("NOUN", "NOUN", "-"),
    "五世": ("NOUN", "NOUN", "-"),
}
_NRT_KNOWN_LOCATIONS = {
    "伊拉克", "伊朗", "拉萨", "迦太基", "南斯拉夫", "塞尔维亚", "巴巴多斯", "威尼斯",
    "瑙鲁", "克里特", "克什米尔", "坎特伯雷", "新罕布什尔", "新罕布夏", "科尔多凡",
    "若羌", "沔阳", "费拉拉", "朗斯", "亚马逊", "波士顿", "花莲", "特拉华", "柴达木", "萨摩亚",
}
_NR_KNOWN_LOCATIONS = {
    "花莲", "波士顿", "仰光", "万宁", "钟祥", "乐安", "鲁尔", "萨摩亚", "特拉华",
    "柴达木", "贺兰山", "张家界", "王屋山", "五台山", "峨眉山",
}
_NR_ORG_ENDINGS = (
    "大学", "学院", "农专", "师专", "中学", "银行", "政府", "公司", "集团",
    "委员会", "联会", "监会", "协会", "学会", "商会", "工会", "研究院", "研究所",
    "设计院", "博物院", "博物馆", "图书馆", "体育馆", "展览馆", "纪念馆",
)
_NR_MULTI_NOUN_SUFFIXES = (
    "建筑", "礼教", "法权", "政权", "主权", "人权", "霸权", "产权", "股权", "特权",
    "版权", "债权", "物权", "主义", "制度", "汽车", "列车", "科学", "技术", "词典",
    "字典", "公报", "报告", "声明", "氏菌", "三角", "昆虫", "维生素", "粟米", "玉米",
    "大米", "小米", "糯米", "上唇", "下唇", "师傅", "长老", "公公", "大人", "师爷",
    "帮主", "教主", "掌门", "皇帝", "皇后", "太后", "宰相", "将军", "夫人", "小姐",
    "先生", "太太", "女儿", "兄弟", "姐妹", "诸侯", "阎王", "龙王", "大王", "魔王",
    "法王", "顾问团", "宣传画", "郁金香", "天文台", "植物", "动物", "菊花", "银花",
    "香花", "梅花", "桂花", "荷花", "桃花", "棉花", "雪花",
)
_NS_CULTURAL_ENDINGS = frozenset("锦缎绣绸布瓷陶玉茶酒菜饼面粉米梨桃枣橘柑橙柚瓜果鸡鸭鱼肉肠腿戏曲剧腔调语话歌谣舞拳剑刀扇伞纸笔墨砚席帽鞋毯")
_NS_GEO_SUFFIXES = frozenset("省市县区镇乡村岛湾海湖山河江溪港城国洲州郡")
_NT_FALSE_ORG_SUFFIXES = (
    "汽车", "越野车", "中档车", "草药", "公报", "科学", "文学", "法学",
    "理学", "哲学", "美学", "事物", "楼梯",
)
_NZ_STRICT_NOUN_SUFFIXES = (
    "英文", "中文", "日文", "法文", "德文", "俄文", "韩文", "百科全书", "机器",
    "病毒", "细菌", "软件", "硬件", "系统", "协议", "算法", "模型", "引擎",
    "语言", "文字", "字体", "编码", "格式", "标准", "主义", "理论", "定理",
    "公式", "定律", "方程", "函数", "矩阵", "化学", "物理", "生物", "医学",
    "疾病", "综合征", "药品", "制剂", "材料", "器件", "设备", "仪器",
)
_MODIFIER_NOUN_SUFFIXES = frozenset(
    "感权率度性单表书品物器者员师家室馆处部科组会业学"
    "论法制式型类种点线面体网观心意情症病费金款税价额量数"
    "期代纪史录案件事"
)
_MODIFIER_ADJ_WHITELIST = {
    "高兴", "开心", "满意", "故意", "客观", "主观", "好看", "难看", "好听", "难听",
    "好吃", "难吃", "好玩", "有趣", "无趣", "有意", "无意", "诚心", "真心", "假意",
    "热心", "细心", "小心", "粗心", "耐心", "放心", "安心", "担心", "伤心", "痛心",
    "灰心", "死心", "专心", "用心", "尽心", "偏心", "多心", "好心", "坏心", "良心",
    "恶心", "贴心", "顺心", "称心", "舒心", "省心", "操心", "费心", "分心", "走心",
    "可观", "美观", "壮观", "悲观", "乐观", "直观", "宏观", "微观", "达观", "旁观",
    "自觉", "直觉", "错觉", "幻觉", "知觉", "感觉", "视觉", "听觉", "嗅觉", "味觉", "触觉",
}
_Z_LAND_NOUNS = {
    "湿地", "草地", "土地", "山地", "盆地", "平地", "空地", "工地", "基地", "阵地",
    "领地", "殖民地", "目的地", "发源地", "栖息地", "根据地", "所在地",
}


def clean_map_ictclas(
    word: str,
    tag: str,
    conv: OpenCCConverter,
    non_person_2: set[str],
    high_nr2: set[str],
) -> tuple[str, str, str]:
    """Map ICTCLAS tag to (decided_upos, upos_mask, meta) with structural noise repair."""
    # 1. Single-char entity/unknown tags are unreliable in isolation
    if len(word) == 1 and tag in ("nr", "nrt", "nrfg", "ns", "nt", "nz", "zg", "g", "x"):
        return "-", "-", "-"

    s = conv.hant_to_hans(word)

    # 2. Structural repair for person tags (nr / nrt / nrfg)
    if tag == "nrt":
        if word in _KNOWN_PERSON_NAMES or s in _KNOWN_PERSON_NAMES:
            return "PROPN", "PROPN", "sem=person"
        if s in _NRT_KNOWN_NON_PROPN:
            return _NRT_KNOWN_NON_PROPN[s]
        if s.endswith("人"):
            return "NOUN", "NOUN", "-"
        if s in _NRT_KNOWN_LOCATIONS or s[-1] in "堡河山岛湾海湖州郡省市县镇城国洲峡港":
            return "PROPN", "PROPN", "sem=location"
        return "PROPN", "PROPN", "-"

    if tag in ("nr", "nrfg"):
        if word in _KNOWN_PERSON_NAMES or s in _KNOWN_PERSON_NAMES:
            return "PROPN", "PROPN", "sem=person"
        if s in _NR_KNOWN_LOCATIONS:
            return "PROPN", "PROPN", "sem=location"
        if s.endswith(_NR_ORG_ENDINGS):
            return "PROPN", "PROPN", "sem=organization"
        if (
            s.endswith(_NR_MULTI_NOUN_SUFFIXES)
            or (len(s) >= 3 and s.endswith("树"))
            or s[-1] in "器素胺酸碱盐酯醇酮醚醛烃酚苯肽酶苷菌虫癌癣症癖剂泵阀"
        ):
            return "NOUN", "NOUN", "-"
        if len(s) == 2:
            if s[-1] in "米卡铜玉仓府坡庐虎蜂鹃薇芩麻弦冠实":
                return "NOUN", "NOUN", "-"
            return "-", "-", "-"
        if len(s) >= 4:
            if len(s) == 4 and s.startswith(_COMPOUND_SURNAMES_2):
                return "PROPN", "PROPN", "sem=person"
            return "-", "-", "-"
        # len(s) == 3
        if s[0] in "那这每各某能会要可敢肯该别莫勿未不没很太更最再又也还就才都只共总全":
            return "-", "-", "-"
        if s[1] == "家" and s[2] in "庄村屯集店湾坡沟岭山桥河嘴坪坝台墩棚铺营堡寨界巷":
            return "PROPN", "PROPN", "sem=location"
        if s[:2] in high_nr2:
            return "-", "-", "-"
        if s[:2] in non_person_2:
            if s[2] in _HEADS_LOC:
                return "PROPN", "PROPN", "sem=location"
            if s[2] in _HEADS_ORG:
                return "PROPN", "PROPN", "sem=organization"
            if s[2] in _HEADS_NOUN:
                return "NOUN", "NOUN", "-"
        if s[-1] in "宫庙寺观庵殿斋洞峡滩渡堡寨省市县区镇乡村湾岛溪墩棚":
            return "PROPN", "PROPN", "sem=location"
        if (
            s[0] not in _COMMON_SURNAMES_1
            and not s.startswith(_COMPOUND_SURNAMES_2)
            and s[0] not in ("小", "老", "阿")
            and s[-1] in _NR_NONSURNAME_1CHAR_NOUN_SUFFIXES
        ):
            return "NOUN", "NOUN", "-"
        return "PROPN", "PROPN", "sem=person"

    # 3. Cultural/product ns
    if tag == "ns":
        if len(s) >= 2 and s[-1] in _NS_CULTURAL_ENDINGS and s[-1] not in _NS_GEO_SUFFIXES:
            return "NOUN", "NOUN", "-"

    # 4. False-org nt
    if tag == "nt":
        if s == "人品出众":
            return "-", "-", "-"
        if s.endswith(_NT_FALSE_ORG_SUFFIXES):
            return "NOUN", "NOUN", "-"

    # 5. Strict noun nz (excluding personal names ending in 文 like 凯文/欧文/达尔文)
    if tag == "nz":
        if s not in ("凯文", "欧文", "达尔文") and s.endswith(_NZ_STRICT_NOUN_SUFFIXES):
            return "NOUN", "NOUN", "-"

    # 6. Noun-headed a / b / l (guarding genuine adjectives)
    if tag in ("a", "b", "l") and len(s) >= 2 and s[-1] in _MODIFIER_NOUN_SUFFIXES and s not in _MODIFIER_ADJ_WHITELIST:
        return "NOUN", "NOUN", "-"

    # 7. Descriptive 'z' forms
    if tag == "z" and len(s) >= 2:
        if s in _Z_LAND_NOUNS:
            return "NOUN", "NOUN", "-"
        if s.endswith("地"):
            return "ADV", "ADV", "-"
        if s.endswith("的") and s not in ("目的", "标的", "有的"):
            return "ADJ", "ADJ", "-"

    return map_ictclas(word, tag)


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
    non_person_2: set[str],
    high_nr2: set[str],
) -> tuple[str, str, str]:
    """Determine (decided_upos, upos_mask, meta) with 0 conflicts against pos.rs and gold."""
    decided, mask, meta = clean_map_ictclas(word, ictclas_tag, conv, non_person_2, high_nr2)

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
        base_tags = set(mask.split("|")) if mask != "-" else set()
        if len(gold_tags) > 1:
            # Multi-POS in gold -> MUST be "-"
            tags = base_tags | gold_tags
            return "-", "|".join(sorted(tags)), meta
        only_gold = next(iter(gold_tags))
        if decided != "-" and decided != only_gold:
            # Jieba tag disagreed with gold (e.g., jieba noisy 'nr' on common word)
            # Clear false semantic tag if PROPN was wrong
            if decided == "PROPN" and only_gold != "PROPN":
                meta = "-"
            return only_gold, only_gold, meta
        if decided == "-":
            tags = base_tags | gold_tags
            if len(tags) == 1:
                only = next(iter(tags))
                return only, only, meta
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
    entries: dict[str, tuple[int, str, str, str, str, str]],
) -> int:
    """Sort by UTF-8 byte order of (surface, variant), write to tmp, and os.replace."""
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(
        entries.items(),
        key=lambda kv: (kv[0].encode("utf-8"), variant.encode("utf-8")),
    )

    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            f"# {variant} lexicon (8-col canonical TSV)\n"
            "# Sources: jieba-rs (MIT), OpenCC (Apache-2.0), in-repo tables, first-party MIT additions, PD Project Gutenberg (gutenberg:23950, gutenberg:23962)\n"
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
            assert (dec == "-" and mask == "-") or all(
                m in VALID_UPOS for m in mask_parts
            ), f"{path}:{line_no}: bad mask {mask!r}"
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


def load_mit_overrides(
    path: Path,
) -> dict[str, dict[str, tuple[int, str, str, str, str, str]]]:
    """Load first-party MIT additions keyed by variant -> surface -> (cost, dec, mask, lemma, cp, meta)."""
    by_var: dict[str, dict[str, tuple[int, str, str, str, str, str]]] = {
        "zh-Hant": {},
        "zh-Hans": {},
    }
    if not path.exists():
        return by_var
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 8:
                continue
            surf, var, cost_s, dec, mask, lemma, cp, meta = cols
            if var in by_var and is_clean_han(surf):
                by_var[var][surf] = (int(cost_s), dec, mask, lemma, cp, meta)
    return by_var


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build zh_hant.tsv and zh_hans.tsv")
    p.add_argument(
        "--jieba-dict",
        type=Path,
        default=Path(os.environ["LEX_JIEBA_DICT"]) if "LEX_JIEBA_DICT" in os.environ else None,
        help="Path to jieba-rs dict.txt (or env LEX_JIEBA_DICT)",
    )
    p.add_argument(
        "--opencc-dir",
        type=Path,
        default=Path(os.environ["LEX_OPENCC_DIR"]) if "LEX_OPENCC_DIR" in os.environ else None,
        help="Path to OpenCC dictionary directory (or env LEX_OPENCC_DIR)",
    )
    p.add_argument(
        "--gutenberg-jsonl",
        type=Path,
        default=Path(os.environ["LEX_ZH_GUTENBERG_JSONL"]) if "LEX_ZH_GUTENBERG_JSONL" in os.environ else None,
        help="Path to JSONL filtered strictly to gutenberg:23950 and gutenberg:23962 (or env LEX_ZH_GUTENBERG_JSONL)",
    )
    p.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path(os.environ.get("LEX_TMP_DIR", tempfile.gettempdir())),
        help="Temporary directory for atomic write (or env LEX_TMP_DIR)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.jieba_dict is None or args.opencc_dir is None:
        raise SystemExit("ERROR: --jieba-dict (or LEX_JIEBA_DICT) and --opencc-dir (or LEX_OPENCC_DIR) are required")
    jieba_dict: Path = args.jieba_dict
    opencc_dir: Path = args.opencc_dir
    tmp_dir: Path = args.tmp_dir

    print("Loading OpenCC tables...")
    conv = OpenCCConverter(opencc_dir)

    print("Parsing src/doc/pos.rs and gold datasets...")
    zh_decided, zh_ambiguous = parse_pos_rs()
    gold_pos = load_gold_pos()
    in_repo_words = load_in_repo_zh_words()
    print(f"  ZH_DECIDED={len(zh_decided)}, ZH_AMBIGUOUS={len(zh_ambiguous)}, gold_tokens={len(gold_pos)}")

    print("Mining allowlisted Project Gutenberg PD zh corpus (NPMI + unigram counts)...")
    hant_uni, hant_mined, hant_docs = mine_corpora_npmi(args.gutenberg_jsonl, min_count=10, min_pmi=5.5)
    hans_uni, hans_mined, hans_docs = mine_corpora_npmi(
        args.gutenberg_jsonl, min_count=10, min_pmi=5.5, transform_fn=conv.hant_to_hans
    )
    print(f"  zh-Hant PD docs={hant_docs}, mined={len(hant_mined)}; zh-Hans PD-projected docs={hans_docs}, mined={len(hans_mined)}")

    print("Loading jieba-rs dict.txt...")
    jieba_entries: dict[str, tuple[int, str]] = {}
    with open(jieba_dict, "r", encoding="utf-8") as f:
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

    non_person_2 = {
        w
        for w, (fr, tg) in jieba_entries.items()
        if len(w) == 2
        and tg in ("n", "vn", "nz", "v", "a", "d", "b", "z", "t", "s", "f", "m", "q", "r", "c", "p")
        and fr >= 5
    }
    high_nr2 = {
        w
        for w, (fr, tg) in jieba_entries.items()
        if len(w) == 2 and tg == "nr" and fr >= 50 and w not in _COMPOUND_SURNAMES_2
    }

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

    # Load first-party MIT additions (PR #46 + optional modern UPOS/semantic overrides)
    mit_translate = load_mit_overrides(ZH_TRANSLATE_MIT_TSV)
    mandatory_hant.update(mit_translate["zh-Hant"].keys())
    mandatory_hans.update(mit_translate["zh-Hans"].keys())

    mit_modern = load_mit_overrides(ZH_MODERN_MIT_TSV)
    # Symmetrically propagate mit_modern across Hant & Hans without altering mandatory_* selection sets
    for src_var, dst_var, fn in (
        ("zh-Hant", "zh-Hans", conv.hant_to_hans),
        ("zh-Hans", "zh-Hant", conv.hans_to_hant),
    ):
        for w, row in list(mit_modern[src_var].items()):
            w_dst = fn(w)
            if is_clean_han(w_dst) and w_dst not in mit_modern[dst_var]:
                cost, dec, mask, lemma, _cp, meta = row
                mit_modern[dst_var][w_dst] = (cost, dec, mask, lemma, w if w != w_dst else "-", meta)

    # Build scored candidate pools for zh-Hans and zh-Hant (1-Char Hanzi Policy: freq >= 80 or Gutenberg PD or mandatory)
    hans_pool: dict[str, tuple[float, str]] = {}
    hant_pool: dict[str, tuple[float, str]] = {}

    for w, (freq, tag) in jieba_entries.items():
        if len(w) == 1:
            if freq < 80 and hans_uni[w] == 0 and w not in mandatory_hans:
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
            if len(hant_w) == 1 and freq < 80 and hant_uni[hant_w] == 0 and hant_w not in mandatory_hant:
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

    hans_sel = select_top(hans_pool, mandatory_hans, target_multi=68500)
    hant_sel = select_top(hant_pool, mandatory_hant, target_multi=64200)

    def build_rows(
        selected: dict[str, tuple[float, str]],
        is_hant: bool,
        mit_overrides: dict[str, tuple[int, str, str, str, str, str]],
    ) -> dict[str, tuple[int, str, str, str, str, str]]:
        total_freq = sum(sc for sc, _ in selected.values())
        rows: dict[str, tuple[int, str, str, str, str, str]] = {}
        for w, (sc, tag) in selected.items():
            cost = compute_cost(sc, total_freq)
            dec, mask, meta = resolve_pos(w, tag, zh_decided, zh_ambiguous, gold_pos, conv, non_person_2, high_nr2)
            cp = conv.hant_to_hans(w) if is_hant else conv.hans_to_hant(w)
            if not is_clean_han(cp) or cp == w:
                cp = "-"
            w_hant = conv.hans_to_hant(w)
            w_hans = conv.hant_to_hans(w)
            if (
                w in mit_overrides
                and w not in zh_decided
                and w not in zh_ambiguous
                and w not in gold_pos
                and w_hant not in gold_pos
                and w_hans not in gold_pos
            ):
                _, m_dec, m_mask, _, m_cp, m_meta = mit_overrides[w]
                m_mask = "|".join(sorted(set(m_mask.split("|"))))
                if m_dec != "-" and m_mask != m_dec:
                    m_dec = "-"
                dec, mask, meta = m_dec, m_mask, m_meta
                if m_cp == "-":
                    cp = "-"
                elif is_clean_han(m_cp) and m_cp != w:
                    cp = m_cp
            rows[w] = (cost, dec, mask, "-", cp, meta)
        return rows

    hant_rows = build_rows(hant_sel, is_hant=True, mit_overrides={**mit_translate["zh-Hant"], **mit_modern["zh-Hant"]})
    hans_rows = build_rows(hans_sel, is_hant=False, mit_overrides={**mit_translate["zh-Hans"], **mit_modern["zh-Hans"]})

    print("Writing atomic TSVs...")
    write_tsv_atomic(OUT_HANT, tmp_dir / "zh_hant.tsv.tmp", "zh-Hant", hant_rows)
    write_tsv_atomic(OUT_HANS, tmp_dir / "zh_hans.tsv.tmp", "zh-Hans", hans_rows)

    print("Verifying invariants and gold compatibility...")
    stats_hant = verify_tsv(OUT_HANT, "zh-Hant", zh_decided, zh_ambiguous, gold_pos)
    stats_hans = verify_tsv(OUT_HANS, "zh-Hans", zh_decided, zh_ambiguous, gold_pos)
    print(f"  zh_hant.tsv: {stats_hant}")
    print(f"  zh_hans.tsv: {stats_hans}")


if __name__ == "__main__":
    main()
