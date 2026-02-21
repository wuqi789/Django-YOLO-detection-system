from openai import OpenAI
from openai import AuthenticationError, APIError
import os

OPENROUTER_API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def get_ai_suggestion(sensor_data):
    """
    获取AI生成的粮仓管理建议
    """
    try:
        if not isinstance(sensor_data, dict):
            raise ValueError("传感器数据必须是字典格式")

        user_content = f"""现在的1号传感器的温湿度为{sensor_data['sensor1']['temperature']}°C，{sensor_data['sensor1']['humidity']}%；
现在的2号传感器的温湿度为{sensor_data['sensor2']['temperature']}°C，{sensor_data['sensor2']['humidity']}%；
现在的3号传感器的温湿度为{sensor_data['sensor3']['temperature']}°C，{sensor_data['sensor3']['humidity']}%"""

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
            temperature=0.5,
            max_tokens=1800
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"获取建议失败：{str(e)}"


if __name__ == "__main__":
    test_data = {
        "sensor1": {"temperature": 18.5, "humidity": 65},
        "sensor2": {"temperature": 19.2, "humidity": 68},
        "sensor3": {"temperature": 17.8, "humidity": 63}
    }
    suggestion = get_ai_suggestion(test_data)
    print("AI建议：\n", suggestion)
