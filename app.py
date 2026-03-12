import streamlit as st
import os
import base64
import re
from zhipuai import ZhipuAI
from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- 样式设置（按钮配色+绿色成功卡片） ----------
st.set_page_config(
    page_title="中医智能小助手",
    page_icon="🌿",
    layout="wide"
)
st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF !important; }
        .title {
            text-align: center;
            color: #3A5F0B;
            font-family: 'KaiTi', 'SimSun', serif;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #8B4513;
            color: #8B4513;
            background-color: #FFFFFF;
        }
        .stButton>button:hover {
            border-color: #3A5F0B;
            color: #3A5F0B;
        }
        .stButton>button[data-baseweb="button"] {
            background-color: #3A5F0B;
            color: #FFFFFF;
            border: none;
        }
        .stButton>button[data-baseweb="button"]:hover {
            background-color: #556B2F;
            color: #FFFFFF;
        }
        .stButton>button[kind="secondary"] {
            background-color: #A0522D;
            color: white;
            border: none;
        }
        .stAlert { border-radius: 8px; }
        .continue-card {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #FFF8DC;
            border: 1px solid #D2B48C;
            margin: 1rem 0;
        }
        .success-card {
            background-color: #e6f4ea;
            border: 2px solid #4caf50;
            border-radius: 8px;
            padding: 1em;
            margin-bottom: 1em;
            color: #2e7d32;
            font-size: 1.1em;
        }
        .diagnosis-time {
            color: #696969;
            font-size: 0.8em;
            text-align: right;
        }
        .st-expander, .st-expander header {
            background-color: #FAF0E6;
            border-radius: 8px;
        }
        .modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 650px;
            background: #f2e7ff;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }
        div[data-testid="stExpander"] .stButton>button {
            border: 1px solid #3A5F0B !important;
            color: #3A5F0B !important;
            background-color: #F0FFF0 !important;
            border-radius: 8px;
        }
        div[data-testid="stExpander"] .stButton>button:hover {
            background-color: #E0EEE0 !important;
            color: #1C2F0C !important;
        }
        div[data-testid="stExpander"] .stButton>button[data-baseweb="button"] {
            background-color: #3A5F0B !important;
            color: #FFFFFF !important;
            border: none !important;
        }
        .risk-warning {
            background-color: #FFF3CD;
            padding: 10px;
            border-left: 4px solid #FF9800;
            margin-bottom: 20px;
        }
        .doctor-avatar-box {
            display: flex;
            align-items: flex-start;
            gap: 15px;
            margin-bottom: 10px;
        }
        .doctor-avatar-img {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #3A5F0B;
            background: #FFF;
            flex-shrink: 0;
        }
        .doctor-avatar-content {
            flex: 1;
            min-width: 0;
            font-size: 1rem;
            line-height: 1.7;
        }
        .info-card {
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
            border: 1px solid #dee2e6;
        }
    </style>
    """, unsafe_allow_html=True
)

# ----------- session_state初始化 -----------
if "show_constitution_test" not in st.session_state:
    st.session_state.show_constitution_test = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = set()
if "user_gender" not in st.session_state:
    st.session_state.user_gender = None
if "user_age" not in st.session_state:
    st.session_state.user_age = None
if "info_collected" not in st.session_state:
    st.session_state.info_collected = False

# ----------- 工具函数（图片和内容格式化） -----------
def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def format_ai_content(content):
    # 修复错误的HTML标签
    # 1. 修复不正确的加粗标签
    content = re.sub(r'b>([^<]+?)/b>', r'<b>\1</b>', content)
    # 2. 修复不正确的换行标签
    content = re.sub(r'br>', r'<br>', content)
    
    # 处理Markdown格式的加粗
    content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
    
    # 处理数字序号和中文序号前加换行
    content = re.sub(r'(\d+\.)', r'<br>\1', content)
    content = re.sub(r'（[一二三四五六七八九十]）', r'<br>\g<0>', content)
    
    # 处理主标题
    content = re.sub(r'([一二三四五六七八九十])、([^\n<]+)', r'<br>\1、\2', content)
    
    # 列表点转HTML
    content = re.sub(r'•\s*(.+)', r'<ul><li>\1</li></ul>', content)
    
    # 处理独立的<，但不破坏HTML标签
    content = re.sub(r'(?<!<)(<)(?![a-z/])', '', content)
    
    # 去掉多余空行
    content = content.replace("\n", "")
    
    return content

def format_ai_content_no_bold(content):
    # 修复错误的HTML标签
    # 1. 修复不正确的加粗标签（但不添加新的加粗）
    content = re.sub(r'b>([^<]+?)/b>', r'\1', content)
    # 2. 修复不正确的换行标签
    content = re.sub(r'br>', r'<br>', content)
    
    # 处理Markdown格式的加粗（不添加HTML加粗）
    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
    
    # 处理数字序号和中文序号前加换行
    content = re.sub(r'(\d+\.)', r'<br>\1', content)
    content = re.sub(r'（[一二三四五六七八九十]）', r'<br>\g<0>', content)
    
    # 处理主标题
    content = re.sub(r'([一二三四五六七八九十])、([^\n<]+)', r'<br>\1、\2', content)
    
    # 列表点转HTML
    content = re.sub(r'•\s*(.+)', r'<ul><li>\1</li></ul>', content)
    
    # 处理独立的<，但不破坏HTML标签
    content = re.sub(r'(?<!<)(<)(?![a-z/])', '', content)
    
    # 去掉多余空行
    content = content.replace("\n", "")
    
    return content

def collect_user_info():
    if not st.session_state.info_collected:
        st.markdown("""
        <div class="info-card">
        <h3 style="color:#3A5F0B">👤 欢迎使用中医智能小助手</h3>
        <p>为了提供更符合您体质特点的中医分析与建议，请提供以下基本信息：</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("性别", ["男", "女"], index=0)
        with col2:
            age = st.number_input("年龄", min_value=1, max_value=120, value=30)
        
        if st.button("确认并继续", type="primary"):
            st.session_state.user_gender = gender
            st.session_state.user_age = age
            st.session_state.info_collected = True
            st.rerun()
        return False
    return True

# ---------------- 体质测试 ----------------
CONSTITUTION_QUESTIONS = [
    {"q": "1. 您手脚发凉的情况多吗？", "options": ["没有", "很少", "有时", "经常", "总是"], "type": "阳虚质"},
    {"q": "2. 您感到精力不济，容易疲乏吗？", "options": ["没有", "很少", "有时", "经常", "总是"], "type": "气虚质"},
    {"q": "3. 您皮肤或口唇感觉干燥吗？", "options": ["没有", "很少", "有时", "经常", "总是"], "type": "阴虚质"},
    {"q": "4. 您感觉身体沉重，或腹部肥满松软吗？", "options": ["没有", "很少", "有时", "经常", "总是"], "type": "痰湿质"},
    {"q": "5. 您面部或鼻部是否总是油光发亮，易生粉刺？", "options": ["没有", "很少", "有时", "经常", "总是"], "type": "湿热质"},
    {"q": "6. 您的皮肤在抓挠后是否容易出现紫色瘀斑？", "options": ["没有", "很少", "有时", "经常", "总是"], "type": "血瘀质"},
    {"q": "7. 您是否经常感到情绪抑郁、紧张焦虑？", "options": ["没有", "很少", "有时", "经常", "总是"], "type": "气郁质"},
    {"q": "8. 您是否精力充沛、面色红润、适应能力强？", "options": ["是的", "大部分是", "有时是", "很少是", "不是"], "type": "平和质"}
]
CONSTITUTION_DESCRIPTIONS = {
    "平和质": "恭喜您！这是最健康的体质。形体匀称健壮，面色红润，精力充沛，适应能力强。请继续保持良好的生活习惯。",
    "气虚质": "表现为元气不足，易疲乏，声音低弱，易出汗，易感冒。建议多食用补气健脾的食物，如山药、黄芪、大枣，并进行适度、缓和的锻炼。",
    "阳虚质": "即\"火力不足\"，表现为畏寒怕冷，手脚冰凉，精神不振，大便稀溏。建议多吃温补肾阳的食物如羊肉、韭菜，并注意保暖，多晒太阳。",
    "阴虚质": "体内津液精血等阴液亏少，表现为手足心热，口燥咽干，鼻微干，喜冷饮，大便干燥。建议多吃滋阴润燥的食物，如银耳、百合、梨，避免熬夜和辛辣食物。",
    "痰湿质": "体内水湿停聚，表现为体形肥胖，腹部肥满，口黏苔腻，身体困重。建议饮食清淡，多吃健脾祛湿的食物如薏米、赤小豆，并增加运动量。",
    "湿热质": "湿与热并存，表现为面垢油光，易生痤疮，口苦口干，大便黏滞。建议饮食清淡，多吃清热利湿的食物如绿豆、冬瓜、苦瓜，忌辛辣油腻。",
    "血瘀质": "血液运行不畅，表现为面色晦暗，皮肤粗糙，易出现瘀斑，口唇暗淡。建议多进行可促进血液循环的运动，并可适量食用活血化瘀的食物如山楂、黑木耳。",
    "气郁质": "气的运行不畅，表现为神情抑郁，情感脆弱，烦闷不乐，易失眠。建议多参加社交活动，听轻松音乐，多食用能行气解郁的食物如佛手、玫瑰花茶。",
}
def judge_constitution(answers):
    scores = {"平和质": 0, "气虚质": 0, "阳虚质": 0, "阴虚质": 0, "痰湿质": 0, "湿热质": 0, "血瘀质": 0, "气郁质": 0}
    option_map = {"没有": 1, "很少": 2, "有时": 3, "经常": 4, "总是": 5}
    peaceful_reverse_map = {"是的": 1, "大部分是": 2, "有时是": 3, "很少是": 4, "不是": 5}
    scores["平和质"] = peaceful_reverse_map[answers[7]]
    for i in range(7):
        q_type = CONSTITUTION_QUESTIONS[i]["type"]
        scores[q_type] = option_map[answers[i]]
    non_peaceful_scores = {k: v for k, v in scores.items() if k != "平和质"}
    max_score_type = max(non_peaceful_scores, key=non_peaceful_scores.get)
    max_score = non_peaceful_scores[max_score_type]
    if scores["平和质"] <= 2 and all(score < 3 for score in non_peaceful_scores.values()):
        return "平和质", CONSTITUTION_DESCRIPTIONS["平和质"]
    if max_score >= 3:
        return max_score_type, CONSTITUTION_DESCRIPTIONS[max_score_type]
    else:
        return "混合或不明显体质", "您的体质倾向不太明显，建议结合具体症状进行综合判断，并保持健康的生活方式。"

# ----------- 知识库 -----------
@st.cache_resource
def load_knowledge_base():
    try:
        persist_dir = "./chroma_db"
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if os.path.exists(persist_dir):
            return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        st.info("首次运行或知识库更新，正在构建向量数据库...")
        loader = TextLoader("knowledge/knowledge.txt", encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)
        vectorstore.persist()
        st.success("知识库构建完成并已持久化！")
        return vectorstore
    except Exception as e:
        st.error(f"加载知识库失败：{str(e)}")
        return None

if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_knowledge_base()

try:
    client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])
except KeyError:
    st.error("❌ 请在Streamlit的Secrets中配置ZHIPUAI_API_KEY。")
    st.stop()

def clean_model_output(text):
    if text:
        return text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    return text

def call_zhipu_llm(user_query, history, more_advice=False):
    related_knowledge = ""
    if st.session_state.vectorstore:
        search_k = 8 if more_advice else 4
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": search_k})
        retrieved_docs = retriever.get_relevant_documents(user_query)
        related_knowledge = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # 获取性别和年龄
    gender = st.session_state.user_gender or "未知"
    age = st.session_state.user_age or "未知"
    
    # 年龄段判断
    age_category = "未知"
    if isinstance(age, int):
        if age <= 14:
            age_category = "少年期"
        elif age <= 35:
            age_category = "青年期"
        elif age <= 55:
            age_category = "壮年期"
        elif age <= 70:
            age_category = "中年期"
        else:
            age_category = "老年期"
    
    user_info = f"用户信息：性别 {gender}，年龄 {age}（{age_category}）。"
    
    if not history:
        system_prompt = f"""{user_info}
作为一名资深的中医专家，你的首要任务是进行严谨的"问诊"。用户刚刚提供了初步症状，你的唯一目标是提出2-3个关键的追问问题，以获取更全面的信息。请遵循以下规则：
1.  禁止诊断：在这一轮对话中，绝对不允许给出任何形式的证型判断或养生建议。
2.  聚焦关键问题: 你的问题必须围绕以下核心方面展开：
    - 过往病史及身体异常指标: 例如："之前有没有严重病史或身体哪些指标不正常如血压血糖？"
    - 症状持续时间: 例如："这种情况持续多久了？"
    - 具体表现与诱因: 例如："咳嗽是干咳还是有痰？什么情况下会加重？"
    - 伴随症状: 根据初步症状，推断并询问可能被忽略的其他相关症状。例如，如果用户说"头痛"，你可以问"是否伴有恶心、畏光或鼻塞？"
3.  引用知识: 你可以参考以下检索到的资料来构思更专业的问题。
    --- 检索到的资料 ---
    {related_knowledge}
    --- 资料结束 ---
4.  结尾引导: 在提出问题后，以一句话引导用户回答，例如："请您补充这些信息，以便我能更准确地为您分析。"
你的回答必须直接以问题开始，简洁明了。"""
    elif more_advice:
        system_prompt = f"""{user_info}
作为一名资深的中医专家，请严格依据以下从本地知识库检索到的资料，为用户提供专业的调理建议。
--- 检索到的资料 ---
{related_knowledge}
--- 资料结束 ---
要求：
1. 内容来源: 你的回答必须完全基于上述"检索到的资料"。
2. 输出结构: 分"一、膏方建议"、"二、茶饮建议"、"三、药膳建议"、"四、理疗建议"四个部分清晰作答。
3. 专业性: 语言专业、严谨，给出建议时可简要说明其适应证。
4. 性别针对性: 根据用户性别（{gender}）结合中医阴阳理论，给出更加针对性的建议。例如，男性属阳，女性属阴，调理方法有所不同。
5. 年龄特异性: 根据用户年龄段（{age_category}）结合中医盛衰理论，考虑不同年龄段的生理特点。例如，壮年气血充沛，老年气血渐衰，调理方法应有所区别。
6. 补充原则: 如果资料不全，无法覆盖所有四个方面，请仅就资料中有的部分作答，并明确指出"关于XX方面的建议，资料中暂未提及"。绝不允许自行编撰。
7. 输出时禁止缩进和多余空格，分层结构请用正常的数字序号和小黑点（•），或直接输出HTML ul/li列表结构，禁止markdown缩进。"""
    else:
        system_prompt = f"""{user_info}
作为一名资深的中医专家，你的任务是基于用户描述的症状及补充信息，结合本地知识库的资料，进行严谨的辨证分析。
--- 检索到的资料 ---
{related_knowledge}
--- 资料结束 ---
请遵循以下规则进行回复：
1. 辨证分析:
   - 优先引用: 必须优先结合并引用"检索到的资料"进行分析。
   - 补充诊断: 若资料不足以支撑诊断，你可以结合自身庞大的中医知识库进行补充和推断，但需明确告知用户"根据资料并结合我的知识判断..."。
   - 性别相关分析: 结合中医阴阳理论，考虑用户性别（{gender}）在辨证中的影响。
   - 年龄相关分析: 结合中医盛衰理论，考虑用户年龄段（{age_category}）在辨证中的影响。
2. 养生建议:
   - 给出3-5条具体、可操作的非药物建议（如饮食、起居、运动、情绪调理）。
   - 针对性别给出更具针对性的建议，例如：男性阳气更盛，女性阴血更丰等特点。
   - 针对年龄段给出更精准的建议，例如：青壮年气血旺盛，中老年气血渐衰等特点。
3. 格式要求:
   - 回复必须分为"一、辨证分析"和"二、养生建议"两部分。
   - 语言专业、沉稳、易于理解。"""
    
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_query}]
    try:
        response = client.chat.completions.create(model="GLM-4.5V", messages=messages, temperature=0.2)
        cleaned_content = clean_model_output(response.choices[0].message.content)
        return cleaned_content
    except Exception as e:
        return f"❌ API调用失败：{str(e)}"

doctor_avatar_b64 = get_base64_image("images/doctor_avatar.png")
tcm_logo_b64 = get_base64_image("images/tcm_logo.png")

if st.session_state.show_constitution_test:
    st.header("🧬 中医体质自测")
    st.markdown('<div class="risk-warning"><strong>⚠️ 风险提示：</strong>本产品仅为AI技术演示，内容仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。</div>', unsafe_allow_html=True)
    st.caption("根据您近期的身体感受，选择最符合的选项。")
    answers = []
    for i, item in enumerate(CONSTITUTION_QUESTIONS):
        st.write(f"**{item['q']}**")
        answer = st.radio(
            label=f"问题{i+1}",
            options=item['options'],
            key=f"test_q_{i}",
            horizontal=True,
            label_visibility="collapsed"
        )
        answers.append(answer)
    if st.button("查看我的体质结果", type="primary"):
        constitution_type, description = judge_constitution(answers)
        st.success(f"**您的体质类型是：{constitution_type}**")
        st.info(description)
    st.markdown("---")
    if st.button("关闭测试"):
        st.session_state.show_constitution_test = False
        st.rerun()
else:
    # 显示页面标题和风险提示
    col_logo, col_main_title, col_main_popup = st.columns([1,5,1])
    with col_logo:
        st.markdown(f'<img src="data:image/png;base64,{tcm_logo_b64}" width="144" />', unsafe_allow_html=True)
    with col_main_title:
        st.markdown('<h1 class="title">🌿 中医智能小助手</h1>', unsafe_allow_html=True)
        st.markdown('<div class="risk-warning"><strong>⚠️ 风险提示：</strong>本产品仅为AI技术演示，内容仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。</div>', unsafe_allow_html=True)
    with col_main_popup:
        if st.button("🧬 体质测试", use_container_width=True):
            st.session_state.show_constitution_test = True
            st.rerun()
    
    # 收集用户信息
    can_proceed = collect_user_info()
    
    # 只有收集完信息才显示主界面
    if can_proceed:
        st.subheader("💡 常见症状参考（点击选择）")
        # 个人信息设置区域
        with st.expander("⚙️ 个人信息设置"):
            st.write(f"当前信息：{st.session_state.user_gender}，{st.session_state.user_age}岁")
            if st.button("修改个人信息"):
                st.session_state.info_collected = False
                st.rerun()
                
        SYMPTOM_KEYWORDS = {
            "头部": ["头痛", "头晕", "偏头痛", "头重", "头胀"], "呼吸": ["咳嗽", "痰多", "咽痛", "流涕", "鼻塞", "打喷嚏", "呼吸急促"],
            "消化": ["腹痛", "腹胀", "消化不良", "食欲不振", "恶心", "呕吐"], "睡眠": ["失眠", "多梦", "早醒", "嗜睡", "睡眠质量差"],
            "情绪": ["焦虑", "抑郁", "烦躁", "易怒", "心神不宁", "心慌", "心悸"], "其他": ["疲劳", "乏力", "手脚冰凉", "出汗异常", "浮肿", "腰酸背痛"]
        }
        for category, symptoms in SYMPTOM_KEYWORDS.items():
            with st.expander(f"📌 {category}相关症状"):
                cols = st.columns(5)
                for i, symptom in enumerate(symptoms):
                    with cols[i % 5]:
                        if symptom in st.session_state.selected_symptoms:
                            if st.button(f"✅ {symptom}", key=f"btn_{symptom}", type="primary"):
                                st.session_state.selected_symptoms.remove(symptom); st.rerun()
                        else:
                            if st.button(f"➕ {symptom}", key=f"btn_{symptom}"):
                                st.session_state.selected_symptoms.add(symptom); st.rerun()
        if st.session_state.selected_symptoms:
            st.markdown("##### 🔍 已选症状：")
            st.info("、".join(st.session_state.selected_symptoms))
            if st.button("❌ 清空已选症状"):
                st.session_state.selected_symptoms = set(); st.rerun()
        with st.form("input_form", clear_on_submit=True):
            user_input = st.text_area("🌱 补充描述或直接提问：", placeholder="请在此描述您的症状，或回答下方助手提出的问题...所有回答都在该输入框进行", height=120)
            col1, col2 = st.columns(2)
            with col1:
                submit_btn = st.form_submit_button("提交信息", type="primary", use_container_width=True)
            with col2:
                clear_btn = st.form_submit_button("清空记录", type="secondary", use_container_width=True)
        if clear_btn:
            st.session_state.chat_history = []; st.session_state.selected_symptoms = set(); st.success("✨ 已清空所有记录"); st.rerun()
        if submit_btn:
            symptoms_text = "、".join(st.session_state.selected_symptoms)
            combined_input = f"{symptoms_text}；{user_input.strip()}" if symptoms_text and user_input.strip() else (symptoms_text or user_input.strip())
            if combined_input:
                with st.spinner("🌿 AI专家正在分析..."):
                    ai_response = call_zhipu_llm(combined_input, st.session_state.chat_history)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.chat_history.append({"role": "user", "content": combined_input, "timestamp": timestamp})
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.session_state.selected_symptoms = set()
                st.rerun()
        if st.session_state.chat_history:
            st.divider()
            st.subheader("📝 问诊记录")
            for i in range(len(st.session_state.chat_history) - 2, -1, -2):
                user_msg = st.session_state.chat_history[i]
                ai_msg = st.session_state.chat_history[i + 1]
                st.markdown(f'<p class="diagnosis-time">问诊时间：{user_msg["timestamp"]}</p>', unsafe_allow_html=True)
                st.info(f"👤 您的描述：\n> {user_msg['content']}")
                content = ai_msg['content']
                if "一、辨证分析" in content and "二、养生建议" in content:
                    parts = content.split("二、养生建议")
                    clean_analysis = parts[0].replace('一、辨证分析', '').strip()
                    clean_suggestions = parts[1].strip()
                    st.markdown(
                        f"""
                        <div class="doctor-avatar-box">
                            <img src="data:image/png;base64,{doctor_avatar_b64}" class="doctor-avatar-img" alt="AI医生头像"/>
                            <div class="doctor-avatar-content">
                                <span style="color:#3A5F0B;font-weight:bold;">🌿 中医辨证</span>
                                {format_ai_content(clean_analysis)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.markdown(
                        f"""
                        <div class="doctor-avatar-box">
                            <img src="data:image/png;base64,{doctor_avatar_b64}" class="doctor-avatar-img" alt="AI医生头像"/>
                            <div class="doctor-avatar-content">
                                <span style="color:#A0522D;font-weight:bold;">🍵 养生建议</span>
                                {format_ai_content(clean_suggestions)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.markdown("""<div class="continue-card">💡 <b>需要更详细的调理方案？</b><br>点击下方按钮，获取膏方、茶饮、药膳等专业建议。</div>""", unsafe_allow_html=True)
                    if st.button("获取更多中医建议", key=f"more_{i}"):
                        with st.spinner("正在检索更多方案..."):
                            more_advice = call_zhipu_llm(user_msg['content'], st.session_state.chat_history, more_advice=True)
                        st.markdown(
                            f"""<div class="success-card">🌟 专业调理方案：<br>{format_ai_content_no_bold(more_advice)}</div>""",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        f"""
                        <div class="doctor-avatar-box">
                            <img src="data:image/png;base64,{doctor_avatar_b64}" class="doctor-avatar-img" alt="AI医生头像"/>
                            <div class="doctor-avatar-content">
                                <span style="color:#3A5F0B;font-weight:bold;">🤖 AI专家追问</span>
                                {format_ai_content(content)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                st.divider()
        with st.expander("💡 使用说明"):
            st.markdown("""
            - **体质测试**: 点击右上角"体质测试"按钮，在弹窗中完成问卷，了解您的基本体质。
            - **个人信息**: 您可以在"个人信息设置"中查看或修改您的性别和年龄信息，以获取更精准的建议。
            - **症状问诊**: 在主页面选择或输入您的症状，点击"提交信息"进行智能辨证。AI专家会先进行追问，请您在下方输入框回答后再次提交。
            - **深入调理**: 在获取初步建议后，可点击"获取更多中医建议"得到更详细的方案。
            - **清空记录**: 使用"清空记录"可开始一次全新的问诊。
            """)
