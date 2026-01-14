from openai import OpenAI
import os
OPENROUTER_API_KEY = "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"  # 替换为真实的OpenRouter API Key

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter固定的API地址
    api_key=OPENROUTER_API_KEY                # 核心：OpenRouter的API Key
)
def get_ai_suggestion(sensor_data):
    """
    获取AI生成的粮仓管理建议
    Args:
        sensor_data: 传感器数据，格式为：
            {
                "sensor1": {"temperature": float, "humidity": float},
                "sensor2": {"temperature": float, "humidity": float},
                "sensor3": {"temperature": float, "humidity": float}
            }
    
    Returns:
        str: AI生成的专业建议
    """
    try:
        # 验证传感器数据格式
        if not isinstance(sensor_data, dict):
            raise ValueError("传感器数据必须是字典格式")


        # 准备用户消息内容
        user_content = f"""现在的1号传感器的温湿度为{sensor_data['sensor1']['temperature']}°C，{sensor_data['sensor1']['humidity']}%；
现在的2号传感器的温湿度为{sensor_data['sensor2']['temperature']}°C，{sensor_data['sensor2']['humidity']}%；
现在的3号传感器的温湿度为{sensor_data['sensor3']['temperature']}°C，{sensor_data['sensor3']['humidity']}%"""
        
        # 创建对话完成请求
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=[
                {
                    "role": "system",
                    "content": """你是一位拥有15年一线粮食仓储管理经验的资深农业专家，擅长玉米、小麦等主要粮食作物的仓储技术和管理。
                                请你根据2000平玉米粮仓的3个传感器的温度和湿度数据，提供专业、准确、易懂的管理建议。
                                建议内容应包括：
                                1. 当前粮仓环境状态评估
                                2. 针对性的管理措施建议
                                3. 潜在风险提示（如果有）
                                
                                要求：
                                - 语言简洁明了，专业准确
                                - 建议具有可操作性
                                - 格式清晰，易于阅读
                                - 避免使用过于专业的术语，确保一线工作人员能理解
                                - 总字数控制在50-100字左右
                                - 字数不要写出来"""
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            extra_headers={},
            extra_body={},
            temperature=0.5,  # 降低随机性，确保建议更可靠
            max_tokens=1500     # 限制生成的最大token数
        )
        # 返回模型响应结果
        return completion.choices[0].message.content
    
    # 捕获认证错误（最常见，重点处理）
    except OpenAI.AuthenticationError as e:
        return f"认证失败：请检查API Key是否正确或已过期"
    
    # 捕获其他API错误（如模型不存在、请求超限等）
    except OpenAI.APIError as e:
        return f"API调用错误：{str(e)}"
    
    # 捕获参数错误
    except ValueError as e:
        return f"参数错误：{str(e)}"
    
    # 捕获网络/连接错误
    except Exception as e:
        return f"获取建议失败：{str(e)}"

# 测试代码（仅在直接运行时执行）
if __name__ == "__main__":
    # 测试数据
    test_data = {
        "sensor1": {"temperature": 18.5, "humidity": 65},
        "sensor2": {"temperature": 19.2, "humidity": 68},
        "sensor3": {"temperature": 17.8, "humidity": 63}
    }
    # 获取建议
    suggestion = get_ai_suggestion(test_data)
    print("AI建议：\n", suggestion)