#!/usr/bin/env python3
"""
测试智能体模式 - 让大模型主动调用Deeplab工具
"""
import json
from src.agent.default_server import create_default_server


def test_agent_mode():
    print("=== 测试智能体模式 - LLM主动调用工具 ===\n")
    
    # 创建服务器实例
    server = create_default_server()
    
    # 构建聊天消息，让LLM分析滑坡图像
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image_path": "data/landslide_image.png"
                },
                {
                    "type": "text", 
                    "text": "请分析这张图片，检测是否存在滑坡。如有必要，请使用适当的工具进行详细分析。"
                }
            ]
        }
    ]
    
    print("发送请求到智能体...")
    print(f"消息内容: {json.dumps(messages, ensure_ascii=False, indent=2)}\n")
    
    # 调用chat方法，让LLM自主决定是否调用工具
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "chat",
        "params": {
            "messages": messages,
            "max_turns": 6
        }
    }
    
    result = server.handle(request)
    
    print("=== 智能体响应 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 分析结果
    if "result" in result:
        chat_result = result["result"]
        if "history" in chat_result:
            print("\n=== 对话历史分析 ===")
            for i, msg in enumerate(chat_result["history"]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                
                print(f"步骤 {i+1} - 角色: {role}")
                if content:
                    print(f"  内容: {content}")
                if tool_calls:
                    print(f"  工具调用: {tool_calls}")
                print()


if __name__ == "__main__":
    test_agent_mode()