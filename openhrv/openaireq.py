import openhrv.openaireq as openaireq
from decouple import config
from PySide6.QtCore import QObject

class OpenAIView(QObject):
    def __init__(self):
        super().__init__()

    def send_chat_request_openai(self,
                                 stream: bool,
                                 session_history: list,
                                 model: str,
                                 clean_response: str,
                                 max_tokens: int | None,
                                 frequency_penalty: float,
                                 temperature: float,
                                 top_p: float,
                                 presence_penalty: float) -> str:

        try:
            client = openaireq.OpenAI(api_key=config(
                "GPT_KEY"), timeout=10, max_retries=2)
            raw_response = client.chat.completions.create(model=model,
                                                          messages=session_history,
                                                          stream=stream,
                                                          max_tokens=max_tokens,
                                                          temperature=temperature,
                                                          top_p=top_p,
                                                          frequency_penalty=frequency_penalty,
                                                          presence_penalty=presence_penalty
                                                          )
            for chunk in raw_response:
                if chunk:
                    data = chunk.choices[0].delta.content
                    if data != None:
                        clean_response += data
                        print(data)
            return clean_response

        except openaireq.APIConnectionError as e:
            return f"Failed to connect to OpenAI API: {e}"

        except openaireq.RateLimitError as e:
            return f"OpenAI API request exceeded rate limit: {e}",

        except openaireq.APIError as e:
            return f"OpenAI API returned an API Error: {e}",
