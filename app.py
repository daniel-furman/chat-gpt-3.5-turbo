import time
import logging
import gradio as gr

from src.llm_boilers import llm_boiler


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.warning("READY. App started...")


class Chat:
    default_system_prompt = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."
    system_format = "<|im_start|>system\n{}<|im_end|>\n"

    def __init__(
        self, system: str = None, user: str = None, assistant: str = None
    ) -> None:
        if system is not None:
            self.set_system_prompt(system)
        else:
            self.reset_system_prompt()
        self.user = user if user else "<|im_start|>user\n{}<|im_end|>\n"
        self.assistant = (
            assistant if assistant else "<|im_start|>assistant\n{}<|im_end|>\n"
        )
        self.response_prefix = self.assistant.split("{}")[0]

    def set_system_prompt(self, system_prompt):
        # self.system = self.system_format.format(system_prompt)
        return system_prompt

    def reset_system_prompt(self):
        return self.set_system_prompt(self.default_system_prompt)

    def history_as_formatted_str(self, system, history) -> str:
        system = self.system_format.format(system)
        text = system + "".join(
            [
                "\n".join(
                    [
                        self.user.format(item[0]),
                        self.assistant.format(item[1]),
                    ]
                )
                for item in history[:-1]
            ]
        )
        text += self.user.format(history[-1][0])
        text += self.response_prefix
        # stopgap solution to too long sequences
        if len(text) > 4500:
            # delete from the middle between <|im_start|> and <|im_end|>
            # find the middle ones, then expand out
            start = text.find("<|im_start|>", 139)
            end = text.find("<|im_end|>", 139)
            while end < len(text) and len(text) > 4500:
                end = text.find("<|im_end|>", end + 1)
                text = text[:start] + text[end + 1 :]
        if len(text) > 4500:
            # the nice way didn't work, just truncate
            # deleting the beginning
            text = text[-4500:]

        return text

    def clear_history(self, history):
        return []

    def turn(self, user_input: str):
        self.user_turn(user_input)
        return self.bot_turn()

    def user_turn(self, user_input: str, history):
        history.append([user_input, ""])
        return user_input, history

    def bot_turn(self, system, history, openai_key):
        conversation = self.history_as_formatted_str(system, history)
        assistant_response = call_inf_server(conversation, openai_key)
        # history[-1][-1] = assistant_response
        # return history
        history[-1][1] = ""
        for chunk in assistant_response:
            try:
                decoded_output = chunk["choices"][0]["delta"]["content"]
                history[-1][1] += decoded_output
                yield history
            except KeyError:
                pass


def call_inf_server(prompt, openai_key):
    model_id = "gpt-3.5-turbo"  # "gpt-3.5-turbo-16k",
    model = llm_boiler(model_id, openai_key)
    logging.warning(f'Inf via "{model_id}"" for prompt "{prompt}"')

    try:
        # run text generation
        response = model.run(prompt, temperature=1.0)
        logging.warning(f"Result of text generation: {response}")
        return response

    except Exception as e:
        # assume it is our error
        # just wait and try one more time
        print(e)
        time.sleep(2)
        response = model.run(prompt, temperature=1.0)
        logging.warning(f"Result of text generation: {response}")
        return response


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    gr.Markdown(
        """<h1><center>Chat with gpt-3.5-turbo</center></h1>

        This is a lightweight demo of gpt-3.5-turbo conversation completion. It was designed as a template for in-context learning applications to be built on top of.
"""
    )
    conversation = Chat()
    with gr.Row():
        with gr.Column():
            # to do: change to openaikey input for public release
            openai_key = gr.Textbox(
                label="OpenAI Key",
                value="",
                type="password",
                placeholder="sk..",
                info="You have to provide your own OpenAI API key.",
            )
    chatbot = gr.Chatbot().style(height=400)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    with gr.Row():
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column(scale=2):
                    system = gr.Textbox(
                        label="System Prompt",
                        value=Chat.default_system_prompt,
                        show_label=False,
                    ).style(container=False)
                with gr.Column():
                    with gr.Row():
                        change = gr.Button("Change System Prompt")
                        reset = gr.Button("Reset System Prompt")
    with gr.Row():
        gr.Markdown(
            "Disclaimer: The gpt-3.5-turbo model can produce factually incorrect output, and should not be solely relied on to produce "
            "factually accurate information. The gpt-3.5-turbo model was trained on various public datasets; while great efforts "
            "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
            "biased, or otherwise offensive outputs.",
            elem_classes=["disclaimer"],
        )

    submit_event = msg.submit(
        fn=conversation.user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=conversation.bot_turn,
        inputs=[system, chatbot, openai_key],
        outputs=[chatbot],
        queue=True,
    )
    submit_click_event = submit.click(
        fn=conversation.user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=conversation.bot_turn,
        inputs=[system, chatbot, openai_key],
        outputs=[chatbot],
        queue=True,
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False).then(
        fn=conversation.clear_history,
        inputs=[chatbot],
        outputs=[chatbot],
        queue=False,
    )
    change.click(
        fn=conversation.set_system_prompt,
        inputs=[system],
        outputs=[system],
        queue=False,
    )
    reset.click(
        fn=conversation.reset_system_prompt,
        inputs=[],
        outputs=[system],
        queue=False,
    )


demo.queue(max_size=36, concurrency_count=14).launch(debug=True)
