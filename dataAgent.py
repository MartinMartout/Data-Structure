import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
import io
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')



from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

load_dotenv()

async def process_chunk(chunk, start_idx, total_records, model_client, termination_condition):
    """
    處理單一批次資料：
      - 解析該批次的寶可夢數據
      - 提供戰略建議，如屬性克制、戰鬥策略等
      - 讓 MultimodalWebSurfer 搜尋最新的寶可夢對戰資訊，整合進建議
    """
    chunk_data = chunk.to_dict(orient='records')
    prompt = (
        f"目前正在處理第 {start_idx} 至 {start_idx + len(chunk) - 1} 筆資料（共 {total_records} 筆）。\n"
        f"以下為該批次的寶可夢數據:\n{chunk_data}\n\n"
        "請根據以上數據進行分析，並提供寶可夢對戰策略建議。"
        "請特別注意：\n"
        "  1. 分析寶可夢的屬性、總能力值，提供戰鬥優勢建議；\n"
        "  2. 利用 MultimodalWebSurfer 搜尋最新的寶可夢對戰建議，包括技能搭配、隊伍配置等，並整合進回覆；\n"
        "  3. 最後請提供具體的戰術建議，包括適合的對戰方式和可能的最佳搭配組合。"
    )
    
    local_data_agent = AssistantAgent("data_agent", model_client)
    local_web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    local_assistant = AssistantAgent("assistant", model_client)
    local_user_proxy = UserProxyAgent("user_proxy")
    local_team = RoundRobinGroupChat(
        [local_data_agent, local_web_surfer, local_assistant, local_user_proxy],
        termination_condition=termination_condition
    )
    
    messages = []
    async for event in local_team.run_stream(task=prompt):
        if isinstance(event, TextMessage):
            print(f"[{event.source}] => {event.content}\n")
            messages.append({
                "batch_start": start_idx,
                "batch_end": start_idx + len(chunk) - 1,
                "source": event.source,
                "content": event.content,
                "type": event.type,
                "prompt_tokens": event.models_usage.prompt_tokens if event.models_usage else None,
                "completion_tokens": event.models_usage.completion_tokens if event.models_usage else None
            })
    return messages

async def main():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("請檢查 .env 檔案中的 GEMINI_API_KEY。")
        return

    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=gemini_api_key,
    )
    
    termination_condition = TextMentionTermination("exit")
    
    csv_file_path = r"C:\Users\User\Downloads\autogen_project-main\pokemon.csv"

    chunk_size = 1000
    chunks = list(pd.read_csv(csv_file_path, chunksize=chunk_size))
    total_records = sum(chunk.shape[0] for chunk in chunks)
    
    tasks = list(map(
        lambda idx_chunk: process_chunk(
            idx_chunk[1],
            idx_chunk[0] * chunk_size,
            total_records,
            model_client,
            termination_condition
        ),
        enumerate(chunks)
    ))
    
    results = await asyncio.gather(*tasks)
    all_messages = [msg for batch in results for msg in batch]
    
    df_log = pd.DataFrame(all_messages)
    output_file = "pokemon_analysis_log.csv"
    df_log.to_csv(output_file, index=False, encoding="utf-8")
    print(f"已將所有寶可夢對戰分析輸出為 {output_file}")

if __name__ == '__main__':
    asyncio.run(main())