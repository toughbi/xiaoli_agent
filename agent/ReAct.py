import re
from llm import XiaoliAgentsLLM
from tools import ToolExecutor, bocha_web_search_tool, search

# (此处省略 REACT_PROMPT_TEMPLATE 的定义)
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]`：调用一个可用工具。
- `Finish[最终答案]`：当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `Finish[最终答案]` 来输出最终答案。


现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: XiaoliAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run_stream(self, question: str):
        """
        流式执行Agent，逐步yield内容
        """
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            step_info = f"\n--- 第 {current_step} 步 ---\n"
            print(step_info)
            yield {'type': 'info', 'content': step_info}

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)

            messages = [{"role": "user", "content": prompt}]
            
            collected_response = []
            for chunk in self.llm_client.think_stream(messages=messages):
                if chunk:
                    collected_response.append(chunk)
                    yield {'type': 'content', 'content': chunk}
            
            response_text = "".join(collected_response)
            if not response_text:
                error_msg = "错误：LLM未能返回有效响应。"
                print(error_msg)
                yield {'type': 'error', 'error': error_msg}
                break

            thought, action = self._parse_output(response_text)
            if thought:
                thought_info = f"🤔 思考: {thought}\n"
                print(thought_info)
                yield {'type': 'info', 'content': thought_info}
            if not action:
                warn_msg = "警告：未能解析出有效的Action，流程终止。"
                print(warn_msg)
                yield {'type': 'error', 'error': warn_msg}
                break
            
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                result_info = f"🎉 最终答案: {final_answer}"
                print(result_info)
                yield {'type': 'final', 'content': final_answer}
                yield {'type': 'done', 'done': True}
                return
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                self.history.append("Observation: 无效的Action格式，请检查。")
                continue

            action_info = f"🎬 行动: {tool_name}[{tool_input}]\n"
            print(action_info)
            yield {'type': 'info', 'content': action_info}
            
            tool_function = self.tool_executor.getTool(tool_name)
            observation = tool_function(tool_input) if tool_function else f"错误：未找到名为 '{tool_name}' 的工具。"
            
            obs_info = f"👀 观察: {observation}\n"
            print(obs_info)
            yield {'type': 'info', 'content': obs_info}
            
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        yield {'type': 'error', 'error': "已达到最大步数，流程终止。"}
        yield {'type': 'done', 'done': True}

    def _parse_output(self, text: str):
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""

if __name__ == '__main__':
    llm = XiaoliAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, bocha_web_search_tool)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)
