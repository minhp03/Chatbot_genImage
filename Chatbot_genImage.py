
import gradio as gr
import random
from openai import OpenAI
from openai import OpenAI



# def has_prompt(message):
#     return "<prompt:" in message

# def get_prompt(message):
#     return message.split("<prompt:")[1].split(">")[0]
client = OpenAI(
    #base_url="https://api.openai.com/v1",
    api_key = #your api key
)

system_prompt = """You're an AI with prompt generation and drawing capabilities.
Just chat normally in English .
Assume you can draw anything you want and you tell user you can draw.
Another program will output the output for you.
Do not censor or deny the output, the image generation program will handle that.

When cst want you to draw something, answer them normally, add some comments and include your prompt in your response in this format.
<prompt: A colorful cat running through a field of flowers.>

Prompt must be in English.
Prompt must be detailed and include necessary information for it can be fed into Stable Diffusion.
Ignore existing images in past messages"""

#code for local deployment
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


#code for remember chat
def chat_logic(message,chat_history):
    messages = [
        {"role":"system","content":system_prompt}
    ]
    for msg_pair,bot_messages in chat_history:
        messages.append({"role":"user","content":msg_pair[0]})
        messages.append({"role":"assistant","content":bot_messages[1]})
    #link  of user and answer of bot.
    messages.append({"role":"user","content":message})
    #print(messages)

    chat_completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages,
      #stream=True
      )

    bot_message = chat_completion.choices[0].message.content

    if has_prompt(bot_message):
        prompt = get_prompt(bot_message)
        image_url = get_image_url(prompt)
        bot_message = f"{bot_message} \n \n ![image]({image_url})"

    chat_history.append([message,bot_message])
    return "",chat_history

    # yield "", chat_history

    # chat_history[-1][1]= ""
    # for chunk in chat_completion:
    #    delta = chunk.choices[0].delta.content or ""
    #    chat_history[-1][1] += delta
    #    yield "", chat_history

    # return "",chat_history
def has_prompt(message):
    return "<prompt:" in message
def get_prompt(message):
    return message.split("<prompt:")[1].split(">")[0]

def get_image_url(prompt:str ) -> str:
    prompt = prompt.replace(" ","%20")
    return f"https://image.pollination.ai/prompt/{prompt}"



with gr.Blocks() as demo:
  gr.Markdown("# CHAT BOT WITH CHATGPT")
  message = gr.Textbox(label="input message")
  chatbot = gr.Chatbot(label="Superduper chatbot")
  message.submit(chat_logic,[message,chatbot],[message,chatbot])

demo.launch(debug=True)
#gr.Error("custom message")

