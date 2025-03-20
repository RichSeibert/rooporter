import subprocess
import logging

class PromptInfo:
    article_summary = "article_summary"
    video_prompt = "video_prompt"
    make_title = "make_title"
    def __init__(self, prompt_type, prompt_data):
        self.prompts = prompt_data
        if prompt_type == self.article_summary:
            self.system_prompt = "Summarize the input article into 1 sentence. You cannot write anything except for the article summary. Do not write something like 'Here is a summary of the article:', you can only write the summary."
            self.prompt_type = self.article_summary
        elif prompt_type == self.video_prompt:
            self.system_prompt = "Write a short, simple, descriptive, and funny 2 sentence scene of following article. Only describe the visuals of the scene. Do not write anything except for the prompt. Do not include the time duration of the video. Here is an example prompt: 'A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.'"
            self.prompt_type = self.video_prompt
        elif prompt_type == self.make_title:
            self.system_prompt = "Summarize the input into a 5-10 word title for a youtube video. It must be 70 characters or less."
            self.prompt_type = self.make_title

def generate_text(prompt_info, settings):
    logging.info(f"Generating text: {prompt_info.prompt_type}")
    llamaCpp_file_path = "llama.cpp/build/bin/llama-cli"
    model_file_path = "models/" + settings["model_file_name"]
    cpu_threads = str(settings["cpu_threads"])
    gpu_layers = str(settings["llama_cpp_gpu_layers"])
    context_len = "10000"

    outputs = []

    full_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    # TODO use 'popen' instead of 'run' for parallization
    for prompt in prompt_info.prompts:
        complete_prompt = full_prompt.replace("{system_prompt}", prompt_info.system_prompt)
        complete_prompt = complete_prompt.replace("{prompt}", prompt)
        try:
            result = subprocess.run(
                [llamaCpp_file_path,
                 "-m", model_file_path,
                 "-t", cpu_threads,
                 "-ngl", gpu_layers,
                 "--temp", "0.9",
                 "-c", context_len,
                 "-p", complete_prompt,
                 "-no-cnv"],
                capture_output=True,
                text=True
            )
            logging.debug(f"llama output: {result}")
            llm_output = result.stdout.strip()
            start_string = "assistant\n\n"
            start_idx = llm_output.find(start_string) + len(start_string)
            extra_ending = " [end of text]"
            llm_output_stripped = llm_output[start_idx:-len(extra_ending)]
            outputs.append(llm_output_stripped)
        except Exception as e:
            logging.error(f"Error running llama.cpp: {e}")
            outputs.append("")
            continue
    return outputs

