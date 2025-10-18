import streamlit as st
import os
from zhipuai import ZhipuAI
from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------
# 1. 页面基础设置
# --------------------------
st.set_page_config(
    page_title="中医智能小助手",
    page_icon="🌿",
    layout="wide"
)

# ========== 全新配色方案与样式 ==========
st.markdown("""
    <style>
        /* 全局背景和字体 - 已修改为纯白色 */
        .stApp {
            background-color: #FFFFFF !important; /* 纯白色背景 */
        }
        /* 主标题样式 */
        .title {
            text-align: center;
            color: #3A5F0B; /* 深竹青色 */
            font-family: 'KaiTi', 'SimSun', serif; /* 楷体或宋体，增加古典感 */
        }
        /* Streamlit原生按钮样式覆盖 */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #8B4513; /* 鞍褐色边框 */
            color: #8B4513; /* 鞍褐色文字 */
            background-color: #FFFFFF;
        }
        .stButton>button:hover {
            border-color: #3A5F0B;
            color: #3A5F0B;
        }
        /* 主操作按钮（例如"获取分析"） */
        .stButton>button[data-baseweb="button"] {
            background-color: #3A5F0B; /* 深竹青色 */
            color: #FFFFFF;
            border: none;
        }
        .stButton>button[data-baseweb="button"]:hover {
            background-color: #556B2F; /* 暗橄榄绿 */
            color: #FFFFFF;
        }
        /* 次要按钮（例如"清空记录"） */
        .stButton>button[kind="secondary"] {
            background-color: #A0522D; /* 赭色 */
            color: white;
            border: none;
        }
        /* 信息提示框样式 */
        .stAlert {
            border-radius: 8px;
        }
        /* 更多建议卡片 */
        .continue-card {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #FFF8DC; /* 玉米色 */
            border: 1px solid #D2B48C; /* 鞣革色边框 */
            margin: 1rem 0;
        }
        /* 诊断时间戳 */
        .diagnosis-time {
            color: #696969;
            font-size: 0.8em;
            text-align: right;
        }
        /* Expander 展开组件样式 */
        .st-expander, .st-expander header {
            background-color: #FAF0E6; /* 亚麻色 */
            border-radius: 8px;
        }
        /* 体质测试中央弹窗容器样式 */
        .modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 650px;
            background: #f2e7ff; /* 淡紫色背景 */
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            z-index: 1000; /* 置于顶层 */
        }
        
        /* 症状按钮样式 - 改为竹叶青色 */
        div[data-testid="stExpander"] .stButton>button {
            border: 1px solid #3A5F0B !important; /* 深竹青色边框 */
            color: #3A5F0B !important; /* 深竹青色文字 */
            background-color: #F0FFF0 !important; /* 蜜瓜绿背景 */
            border-radius: 8px;
        }
        div[data-testid="stExpander"] .stButton>button:hover {
            background-color: #E0EEE0 !important; /* 悬停时稍暗一点 */
            color: #1C2F0C !important;
        }
        
        /* 已选中的症状按钮 - 竹叶青色填充 */
        div[data-testid="stExpander"] .stButton>button[data-baseweb="button"] {
            background-color: #3A5F0B !important; /* 深竹青色背景 */
            color: #FFFFFF !important; /* 白色文字 */
            border: none !important;
        }
        
        /* 风险提示样式 */
        .risk-warning {
            background-color: #FFF3CD;
            padding: 10px; 
            border-left: 4px solid #FF9800;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# 2. 中医体质测试模块 (硬编码)
# --------------------------
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

# --------------------------
# 3. 知识库加载与核心初始化
# --------------------------
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

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_knowledge_base()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = set()
if "show_constitution_test" not in st.session_state:
    st.session_state.show_constitution_test = False

try:
    # 首先尝试从 Streamlit secrets 获取
    api_key = st.secrets["ZHIPUAI_API_KEY"]
except (KeyError, AttributeError):
    # 如果失败，再尝试从环境变量获取
    try:
        api_key = os.environ["ZHIPUAI_API_KEY"]
    except KeyError:
        st.error("❌ 未找到 API 密钥。请在 Streamlit 的 Secrets 或环境变量中配置 ZHIPUAI_API_KEY。")
        st.stop()

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=api_key)

# --------------------------
# 4. 调用模型函数 (全新多轮对话逻辑)
# --------------------------
def clean_model_output(text):
    """清除模型输出中的特殊标记"""
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

    # 根据对话历史判断使用哪个Prompt
    if not history: # 这是第一轮对话，强制追问
        system_prompt = f"""作为一名资深的中医专家，你的首要任务是进行严谨的"问诊"。用户刚刚提供了初步症状，你的唯一目标是提出2-3个关键的追问问题，以获取更全面的信息。请遵循以下规则：
1.  **禁止诊断**：在这一轮对话中，绝对不允许给出任何形式的证型判断或养生建议。
2.  **聚焦关键问题**: 你的问题必须围绕以下核心方面展开：
    - **症状持续时间**: 例如："这种情况持续多久了？"
    - **具体表现与诱因**: 例如："咳嗽是干咳还是有痰？什么情况下会加重？"
    - **伴随症状**: 根据初步症状，推断并询问可能被忽略的其他相关症状。例如，如果用户说"头痛"，你可以问"是否伴有恶心、畏光或鼻塞？"
3.  **引用知识**: 你可以参考以下检索到的资料来构思更专业的问题。
    --- 检索到的资料 ---
    {related_knowledge}
    --- 资料结束 ---
4.  **结尾引导**: 在提出问题后，以一句话引导用户回答，例如："请您补充这些信息，以便我能更准确地为您分析。"
你的回答必须直接以问题开始，简洁明了。"""
    elif more_advice: # 获取更多建议模式
        system_prompt = f"""作为一名资深的中医专家，请严格依据以下从本地知识库检索到的资料，为用户提供专业的调理建议。
--- 检索到的资料 ---
{related_knowledge}
--- 资料结束 ---
要求：
1. **内容来源**: 你的回答必须完全基于上述"检索到的资料"。
2. **输出结构**: 分"一、膏方建议"、"二、茶饮建议"、"三、药膳建议"、"四、理疗建议"四个部分清晰作答。
3. **专业性**: 语言专业、严谨，给出建议时可简要说明其适应证。
4. **补充原则**: 如果资料不全，无法覆盖所有四个方面，请仅就资料中有的部分作答，并明确指出"关于XX方面的建议，资料中暂未提及"。绝不允许自行编撰。"""
    else: # 这是第二轮或之后的对话，可以进行诊断
        system_prompt = f"""作为一名资深的中医专家，你的任务是基于用户描述的症状及补充信息，结合本地知识库的资料，进行严谨的辨证分析。
--- 检索到的资料 ---
{related_knowledge}
--- 资料结束 ---
请遵循以下规则进行回复：
1. **辨证分析**:
   - **优先引用**: 必须优先结合并引用"检索到的资料"进行分析。
   - **补充诊断**: 若资料不足以支撑诊断，你可以结合自身庞大的中医知识库进行补充和推断，但需明确告知用户"根据资料并结合我的知识判断..."。
2. **养生建议**:
   - 给出3-5条具体、可操作的非药物建议（如饮食、起居、运动、情绪调理）。
3. **格式要求**:
   - 回复必须分为"一、辨证分析"和"二、养生建议"两部分。
   - 语言专业、沉稳、易于理解。"""
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_query}]
    try:
        # 使用GLM-4.5V模型
        response = client.chat.completions.create(model="GLM-4.5V", messages=messages, temperature=0.2)
        # 清理输出中的特殊标记
        cleaned_content = clean_model_output(response.choices[0].message.content)
        return cleaned_content
    except Exception as e:
        return f"❌ API调用失败：{str(e)}"

# --------------------------
# 5. 主页面与弹窗视图切换
# --------------------------
# 注意：将分支逻辑放在最外层，明确区分两个视图，避免中间的白框问题

if st.session_state.show_constitution_test:
    # ========== 体质测试视图 ==========
    st.header("🧬 中医体质自测")
    
    # 添加风险提示
    st.markdown('<div class="risk-warning"><strong>⚠️ 风险提示：</strong>本产品仅为AI技术演示，内容仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。</div>', unsafe_allow_html=True)
    
    st.caption("根据您近期的身体感受，选择最符合的选项。")
    answers = []
    for i, item in enumerate(CONSTITUTION_QUESTIONS):
        st.write(f"**{item['q']}**")
        # 使用唯一的key和label_visibility="collapsed"避免产生多余标签
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
    # ========== 主应用视图 ==========
    col_main_title, col_main_popup = st.columns([5,1])
    with col_main_title:
        st.markdown('<h1 class="title">🌿 中医智能小助手</h1>', unsafe_allow_html=True)
        
        # 添加醒目的风险提示
        st.markdown('<div class="risk-warning"><strong>⚠️ 风险提示：</strong>本产品仅为AI技术演示，内容仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。</div>', unsafe_allow_html=True)
        
    with col_main_popup:
        if st.button("🧬 体质测试", use_container_width=True):
            st.session_state.show_constitution_test = True
            st.rerun()

    st.subheader("💡 常见症状参考（点击选择）")
    SYMPTOM_KEYWORDS = {
        "头部": ["头痛", "头晕", "偏头痛", "头重", "头胀"], "呼吸": ["咳嗽", "咽痛", "流涕", "鼻塞", "打喷嚏", "呼吸急促"],
        "消化": ["腹痛", "腹胀", "消化不良", "食欲不振", "恶心", "呕吐"], "睡眠": ["失眠", "多梦", "早醒", "嗜睡", "睡眠质量差"],
        "情绪": ["焦虑", "抑郁", "烦躁", "易怒", "心神不宁"], "其他": ["疲劳", "乏力", "手脚冰凉", "出汗异常", "浮肿"]
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
        user_input = st.text_area("🌱 补充描述或直接提问：", placeholder="请在此描述您的症状，或回答上方助手提出的问题...", height=120)
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
            st.info(f"👤 **您的描述**：\n> {user_msg['content']}")
            content = ai_msg['content']
            if "一、辨证分析" in content and "二、养生建议" in content:
                parts = content.split("二、养生建议")
                # 清理辨证分析部分的特殊标记
                clean_analysis = parts[0].replace('一、辨证分析', '').strip()
                st.info(f"🌿 **中医辨证**\n{clean_analysis}")
                
                # 清理养生建议部分的特殊标记
                clean_suggestions = parts[1].strip()
                st.success(f"🍵 **养生建议**\n{clean_suggestions}")
                
                st.markdown("""<div class="continue-card">💡 <b>需要更详细的调理方案？</b><br>点击下方按钮，获取膏方、茶饮、药膳等专业建议。</div>""", unsafe_allow_html=True)
                if st.button("获取更多中医建议", key=f"more_{i}"):
                    with st.spinner("正在检索更多方案..."):
                        more_advice = call_zhipu_llm(user_msg['content'], st.session_state.chat_history, more_advice=True)
                    st.success(f"🌟 **专业调理方案**：\n\n{more_advice}")
            else: # AI在追问
                st.info(f"🤖 **AI专家追问**：\n> {content}")
            st.divider()

    with st.expander("💡 使用说明"):
        st.markdown("""
        - **体质测试**: 点击右上角"体质测试"按钮，在弹窗中完成问卷，了解您的基本体质。
        - **症状问诊**: 在主页面选择或输入您的症状，点击"提交信息"进行智能辨证。AI专家会先进行追问，请您在下方输入框回答后再次提交。
        - **深入调理**: 在获取初步建议后，可点击"获取更多中医建议"得到更详细的方案。
        - **清空记录**: 使用"清空记录"可开始一次全新的问诊。
        """)
