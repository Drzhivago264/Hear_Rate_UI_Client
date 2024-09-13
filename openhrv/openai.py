
import openai
from decouple import config


def send_request_openai(
        average_heart_rate: int,
        model: str = "gpt-4o-mini"
) -> str:
    try:
        client = openai.OpenAI(api_key=config(
            "GPT_KEY"), timeout=10, max_retries=2)
        raw_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful medical assistant."},
                {"role": "user", "content": f"My heart rate is {average_heart_rate}, can you provide detail interpretation and suggestion for me"},
            ],
            stream=True
        )
        return raw_response

    except openai.APIConnectionError as e:
        return f"Failed to connect to OpenAI API: {e}"

    except openai.RateLimitError as e:
        return f"OpenAI API request exceeded rate limit: {e}",

    except openai.APIError as e:
        return f"OpenAI API returned an API Error: {e}",
