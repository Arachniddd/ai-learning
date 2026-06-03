from app.models.chunk import Chunk


DEFAULT_SYSTEM_PROMPT = "你是一个可靠的 AI 助手。"
JSON_ONLY_SYSTEM_PROMPT = "你只返回合法 JSON，不要输出 Markdown，不要输出解释。"
RAG_ANSWER_SYSTEM_PROMPT = "你是一个严格基于资料回答问题的学习助手。"
SUMMARY_SYSTEM_PROMPT = "你只返回总结正文，不要输出 Markdown，不要输出解释。"


def build_query_rewrite_prompt(question: str) -> str:
    return f"""
你是一个 RAG 检索查询改写器。

你的任务：
把用户问题改写成更适合向量数据库检索的查询。

要求：
1. 保留用户原始问题的真实意图。
2. 补充必要的专业关键词。
3. 不要改变问题主题。
4. 不要添加没有根据的新实体。
5. 不要回答问题。
6. 只输出改写后的查询，不要输出解释。

用户原始问题：
{question}

改写后的检索查询：
""".strip()


def build_rerank_prompt(question: str, chunks: list[Chunk]) -> str:
    candidates_text = ""

    for idx, chunk in enumerate(chunks):
        candidates_text += f"""
候选 {idx}:
ID: {chunk.id}
来源: {chunk.source}
章节: {chunk.section or ""}
向量检索分数: {getattr(chunk, "score", "")}
内容:
{chunk.content[:800]}
""".strip() + "\n\n"

    return f"""
你是一个 RAG 检索结果重排器。

用户问题：
{question}

下面是向量数据库召回的候选片段。请判断每个片段对回答用户问题的帮助程度。

评分规则：
0 = 完全无关
3 = 只有关键词相似，但不能用于回答
5 = 有一定相关性，但信息不完整
8 = 比较相关，可以辅助回答
10 = 非常相关，可以直接用于回答

输出要求：
1. 只输出 JSON。
2. 不要输出解释性文字。
3. candidate_index 必须对应候选编号。
4. score 必须是 0 到 10 的整数。
5. reason 用一句话说明原因。

输出格式：
[
  {{"candidate_index": 0, "score": 8, "reason": "该片段解释了系统调用进入内核态的过程"}},
  {{"candidate_index": 1, "score": 2, "reason": "该片段主题不相关"}}
]

候选片段：
{candidates_text}
""".strip()


def build_rag_answer_prompt(question: str, chunks: list[Chunk]) -> str:
    context_text = ""

    for idx, chunk in enumerate(chunks):
        context_text += f"""
[资料 {idx}]
chunk_id: {chunk.id}
source: {chunk.source}
section: {chunk.section or ""}
score: {getattr(chunk, "score", "")}
rerank_score: {getattr(chunk, "rerank_score", "")}

content:
{chunk.content}
""".strip() + "\n\n"

    return f"""
你是一个严格基于资料回答问题的学习助手。

回答规则：
1. 只能使用下面给出的资料回答。
2. 如果资料不足，必须明确说“资料不足，无法确定”，不要编造。
3. 回答要适合计算机专业学生理解。
4. 关键结论后面要标注来源，例如：[资料来源0]、[资料来源1]。这里的来源从每个chunk中的source取。
5. 如果多个资料都支持同一个结论，可以同时引用多个资料。
6. 不要引用没有实际使用的资料。
7. 最后用“使用资料：”列出你使用的资料编号。

用户问题：
{question}

可用资料：
{context_text}

请基于资料回答：
""".strip()


def build_summarize_prompt(text: str) -> str:
    return f"""
你是一个学习笔记总结助手。

请总结下面的内容。

要求：
1. 保留核心概念。
2. 用清晰的条理组织。
3. 不要编造原文没有的信息。
4. 如果内容是计算机课程笔记，要保留关键术语。
5. 输出中文。

待总结内容：
{text}

总结：
""".strip()


def build_generate_quiz_prompt(
    topic: str,
    chunks: list[Chunk],
    num_questions: int,
) -> str:
    context_text = ""

    for idx, chunk in enumerate(chunks):
        context_text += f"""
[资料 {idx}]
source: {chunk.source}
section: {chunk.section or ""}
content:
{chunk.content}
""".strip() + "\n\n"

    return f"""
你是一个计算机课程助教。

请基于资料围绕主题「{topic}」生成 {num_questions} 道复习题。

要求：
1. 题目必须基于资料，不要编造资料外内容。
2. 题型可以包括选择题、简答题、判断题。
3. 每道题都要给出答案和解析。
4. 难度适合计算机专业本科生。
5. 输出中文。

资料：
{context_text}

请生成题目：
""".strip()
