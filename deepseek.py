from openai import OpenAI
from openai import AuthenticationError, APIError
import os
import json
import re

OPENROUTER_API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def get_ai_suggestion(sensor_data):
    """
    获取AI生成的粮仓管理建议，包含安全风险等级
    
    返回格式：
    {
        "suggestion": "AI建议内容",
        "risk_level": 0-2 (0:无风险, 1:一般风险, 2:高风险),
        "risk_description": "风险描述"
    }
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

【重要】你必须严格按照以下JSON格式返回结果，不要包含任何其他文字：
{
    "risk_level": 0或1或2,
    "risk_description": "风险描述（简短说明为什么是这个风险等级）",
    "suggestion": "管理建议内容"
}

【风险等级判定标准】：
- risk_level = 0（无风险）：所有传感器温度在15-22°C之间，湿度在50-65%之间，环境条件理想
- risk_level = 1（一般风险）：温度在22-25°C或湿度在65-75%，存在一定风险需要关注
- risk_level = 2（高风险）：温度>25°C或湿度>75%，存在较大风险需要立即处理

【建议内容要求】：
1. 当前粮仓环境状态评估
2. 针对性的管理措施建议
3. 潜在风险提示（如果有）

要求：
- 语言简洁明了，专业准确
- 建议具有可操作性
- 格式清晰，易于阅读
- 避免使用过于专业的术语，确保一线工作人员能理解
- 总字数控制在50-100字左右
- 必须严格按照JSON格式返回，不要包含markdown代码块标记"""
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
        
        response_content = completion.choices[0].message.content
        print("AI原始响应:", response_content)
        
        # 尝试解析JSON响应
        try:
            # 移除可能的markdown代码块标记
            cleaned_content = response_content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            
            # 尝试提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', cleaned_content)
            if json_match:
                cleaned_content = json_match.group(0)
            
            result = json.loads(cleaned_content)
            
            # 验证返回的数据格式
            if 'risk_level' not in result:
                result['risk_level'] = 1
            if 'risk_description' not in result:
                result['risk_description'] = '风险等级已评估'
            if 'suggestion' not in result:
                result['suggestion'] = response_content
            
            # 确保risk_level在0-2范围内
            result['risk_level'] = max(0, min(2, int(result['risk_level'])))
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            # 如果JSON解析失败，尝试从文本中提取风险等级
            risk_level = 1  # 默认一般风险
            if '无风险' in response_content or '风险较低' in response_content or '理想' in response_content:
                risk_level = 0
            elif '高风险' in response_content or '危险' in response_content or '立即' in response_content:
                risk_level = 2
            
            return {
                "suggestion": response_content,
                "risk_level": risk_level,
                "risk_description": "根据AI分析自动判定"
            }

    except Exception as e:
        return {
            "suggestion": f"获取建议失败：{str(e)}",
            "risk_level": 1,
            "risk_description": "分析失败，请检查系统"
        }


if __name__ == "__main__":
    test_data = {
        "sensor1": {"temperature": 18.5, "humidity": 65},
        "sensor2": {"temperature": 19.2, "humidity": 68},
        "sensor3": {"temperature": 17.8, "humidity": 63}
    }
    result = get_ai_suggestion(test_data)
    print("AI分析结果：\n", result)
