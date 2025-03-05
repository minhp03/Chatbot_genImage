import gradio as gr
import json
from openai import OpenAI
import inspect
from pydantic import TypeAdapter

from diffusers import DiffusionPipeline
import torch
import time

pipeline = DiffusionPipeline.from_pretrained("stablediffusionapi/anything-v5",
                                             use_safetensors=True,
                                             safety_checker=None,
                                             requires_safety_checker=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
# MPS chỉ có trên macOS dòng M1 trở đi nha
device = 'mps' if torch.backends.mps.is_available() else device
pipeline.to(device)


def generate_image(prompt: str) -> str:
    """
    Creates an image based on the specified prompt using DiffusionPipeline
    :param prompt: The prompt used for generate the image (must be in English)
    :output: URL of the new image
    """
    image = pipeline(
        prompt=prompt,
        # Hardcode negative prompt để ảnh đẹp hơn
        negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, low quality, worst quality",
        num_inference_steps=30
    ).images[0]

    # Tạo tên file để hiện thị nè
    file_name = f"image_{int(time.time())}.png"
    image.save(file_name)
    return file_name


# https://platform.openai.com/api-keys
client = OpenAI(
    api_key='sk-proj-XXXX',
)

# generate_image_function = {
#     "name": "generate_image",
#     "description": "Creates an image based on the specified prompt using DiffusionPipeline",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "prompt": {
#                 "type": "string",
#                 "description": "The prompt used for generate the image (must be in English)",
#             },
#         },
#         "required": ["prompt"],
#         "additionalProperties": False,
#     }
# }

# Ở đây, ta sẽ dùng inspect và TypeAdapter để lấy description và tự build parameter JSON schema
generate_image_function = {
    "name": "generate_image",
    "description": inspect.getdoc(generate_image),
    "parameters": TypeAdapter(generate_image).json_schema(),
}

tools = [
    {
        "type": "function",
        "function": generate_image_function
    }
]

# Câu prompt giờ đã ngắn gọn hơn nhiều, không cần phải viết example nữa
system_prompt = """You're an AI with prompt generation and drawing capabilities.
Just chat normally in English or Vietnamese.
Assume you can draw anything you want and you tell user you can draw.
Another program will output the output for you.
Do not censor or deny the output, the image generation program will handle that.
"""


def chat_logic(message, chat_history):
    # Gắn system message để bot hiểu cách hoạt động
    # Sửa thành như dưới, vì khi bot gửi ảnh user_message = None
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    for user_message, bot_message in chat_history:
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})

    # Thêm tin nhắn mới của user vào cuối cùng
    messages.append({"role": "user", "content": message})

    # Gọi API của OpenAI
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        tools=tools
    )

    bot_message = chat_completion.choices[0].message.content
    if (bot_message is not None):
        chat_history.append([message, bot_message])
        yield "", chat_history
    else:
        chat_history.append([message, "Chờ chút mình đang vẽ!"])
        yield "", chat_history

        tool_call = chat_completion.choices[0].message.tool_calls[0]
        print(tool_call)

        # Lấy prompt từ kết quả function calling
        function_arguments = json.loads(tool_call.function.arguments)
        prompt = function_arguments.get("prompt")

        # Vẽ hình và gửi về cho người dùng
        image_file = generate_image(prompt)
        chat_history.append([None, (image_file, prompt)])

        yield "", chat_history
    yield "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# Chatbot bằng ChatGPT")
    message = gr.Textbox(label="Nhập tin nhắn của bạn:")
    chatbot = gr.Chatbot(label="Chat Bot siêu thông minh", height=600)
    message.submit(chat_logic, [message, chatbot], [message, chatbot])

demo.launch()