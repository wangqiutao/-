"""羽毛球教练智能体 —— Streamlit + 智谱 GLM 免费大模型"""

import os
import json

import requests
import streamlit as st

# ---------------- 页面配置 ----------------
st.set_page_config(
    page_title="羽毛球教练 · 智能问答",
    page_icon="🏸",
    layout="centered",
)

# ---------------- 常量 ----------------
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

MODEL_OPTIONS = {
    "GLM-4-Flash（免费·推荐）": "glm-4-flash",
    "GLM-4.7-Flash（免费·超长上下文）": "glm-4.7-flash",
}

SYSTEM_PROMPT = """你是一位专业、耐心、经验丰富的羽毛球教练。你擅长把复杂的羽毛球技术讲得简单易懂，帮助各个水平的球友进步。

【你的专业范围】
1. 技术与动作：握拍（正手/反手）、发球（正手/反手发球、发小球、发高远球）、高远球、杀球、吊球、网前搓球/勾球/挑球、平抽挡、步法（启动、移动、制动、回位）等动作要领、发力技巧、常见错误与纠正方法。
2. 规则与裁判：单双打规则、计分制度、发球规则、违例判定、界内界外、双打轮转、常见争议与判罚依据。
3. 训练计划与体能：针对零基础/初学/进阶/高手的训练安排、专项体能（力量、速度、灵敏、耐力）、热身与放松、日常练习方法。
4. 战术与打法：单打/双打战术、球路变化、落点控制、针对不同对手（进攻型/防守型/拉吊型）的策略、比赛心理调整。

【回答要求】
- 默认用中文回答；语言清晰、专业、通俗易懂。
- 根据用户水平和具体场景调整讲解深度；用户未说明水平时，可用通俗方式先讲基础，或简短询问。
- 讲解技术动作时：先分步骤讲动作要领，再指出常见错误，最后给出练习/纠正建议。
- 回答要具体、可操作，避免空泛套话；必要时用"第一步/第二步"等结构便于执行。
- 涉及运动损伤或身体不适时，只给通用防护建议，并明确提醒"如持续疼痛或受伤应及时就医"，不做医疗诊断或治疗。
- 语气积极、鼓励，像一位认真负责的教练在指导学员。

【边界】
- 只回答羽毛球运动相关的问题；遇到无关问题，礼貌说明并引导回羽毛球话题。
- 不提供医疗诊断、不承诺受伤恢复效果。"""

WELCOME_MSG = (
    "你好！我是你的羽毛球教练 🏸\n\n"
    "无论是技术动作、比赛规则、训练计划还是战术打法，都可以随时问我。\n\n"
    "先告诉我你的水平（零基础 / 初学者 / 进阶 / 高手）和想提高的方面，"
    "我会给你更有针对性的指导。"
)


# ---------------- 工具函数 ----------------
def get_api_key() -> str:
    """按优先级读取 API Key：Secrets → 环境变量 → 侧边栏输入。"""
    key = None
    try:
        key = st.secrets.get("ZHIPU_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("ZHIPU_API_KEY")
    return key


def stream_chat(messages, api_key: str, model: str):
    """调用智谱 GLM 接口，流式返回回答内容。失败时抛出异常。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    try:
        resp = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(10, 120),
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"网络请求失败：{e}")

    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise RuntimeError(f"API 调用失败（{resp.status_code}）：{msg}")

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8", errors="ignore")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            yield content


# ---------------- 会话状态 ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MSG}]


# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.title("🏸 羽毛球教练")
    st.caption("技术 · 规则 · 训练 · 战术")

    api_key = get_api_key()
    if api_key:
        st.success("✅ 已检测到 API Key")
    else:
        api_key = st.text_input(
            "智谱 API Key",
            type="password",
            help="在 open.bigmodel.cn 注册后，控制台→API 密钥 中创建。仅本次会话使用，不会保存。",
        )

    model_label = st.selectbox("模型", list(MODEL_OPTIONS.keys()))
    model = MODEL_OPTIONS[model_label]

    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MSG}]
        st.rerun()

    st.divider()
    st.caption(
        "如何获取 API Key：\n"
        "1. 打开 open.bigmodel.cn 注册登录\n"
        "2. 进入「控制台 → API 密钥」\n"
        "3. 新建并复制 Key（免费，无需充值）"
    )


# ---------------- 主界面 ----------------
st.title("🏸 羽毛球教练")
st.caption("专业羽毛球问答助手 · 基于智谱 GLM 免费大模型")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("向教练提问，例如：高远球怎么发力？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        st.warning("请先在左侧侧边栏填入智谱 API Key，再开始对话。")
    else:
        with st.chat_message("assistant"):
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                full_response = st.write_stream(
                    stream_chat(messages, api_key, model)
                )
            except Exception as e:
                full_response = None
                st.error(f"出错了：{e}")

        if full_response is not None:
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
