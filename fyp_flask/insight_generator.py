import re
import aiohttp
from config import Config
from typing import List, Dict

def clean_insight(insight: str) -> str:
    """Simplify formatting for display and remove emojis"""
    # Remove emojis and special characters
    insight = re.sub(r'[^\x00-\x7F]+', '', insight)
    
    # Replace bullet points with dashes
    insight = insight.replace('•', '-')
    
    # Remove markdown formatting
    insight = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', insight)
    insight = re.sub(r'#{1,6}\s*', '', insight)
    insight = re.sub(r'-\s+', '- ', insight)
    insight = re.sub(r'```.*?```', '', insight, flags=re.DOTALL)
    return insight.replace('**', '').replace('__', '').strip()

def build_prompt(text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant with computer science expertise."
        },
        {
            "role": "user",
            "content": f"""
From this meeting transcript:
1. List unique computer science terms with one-sentence explanations
2. Provide a concise overall takeaway
3. Add a 1-2 sentence summary

Use PLAIN TEXT only. No markdown or special formatting.

Transcript:
\"\"\"{text}\"\"\"
"""
        }
    ]

async def get_insight(text: str) -> str:
    """Get insights from DeepSeek API"""
    headers = {"Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}"}
    payload = {
        "model": "deepseek-chat",
        "messages": build_prompt(text),
        "temperature": Config.TEMPERATURE,
        "max_tokens": Config.MAX_TOKENS
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            Config.DEEPSEEK_API_URL, 
            json=payload, 
            headers=headers
        ) as response:
            data = await response.json()
            return clean_insight(data['choices'][0]['message']['content'])