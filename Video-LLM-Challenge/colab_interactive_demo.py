"""Interactive temporal retrieval with Mobile-VideoGPT.
Uses the official inference API: github.com/Amshaker/Mobile-VideoGPT.
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path
import gradio as gr
import torch
import mobilevideogpt
from mobilevideogpt.utils import preprocess_input
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'Amshaker/Mobile-VideoGPT-1.5B'
model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, config=AutoConfig.from_pretrained(MODEL_NAME),
            torch_dtype=torch.float16,
        ).cuda().eval()
    return model, tokenizer


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def make_clips(video, folder):
    metadata = json.loads(command([
        'ffprobe', '-v', 'error', '-show_format', '-of', 'json', video
    ]))
    duration = float(metadata['format']['duration'])
    if not 0 < duration <= 60:
        raise gr.Error('Please use a video no longer than 60 seconds.')
    clips = []
    start = 0.0
    while start < duration - 1e-6:
        end = min(start + 3, duration)
        clip = folder / f'clip_{len(clips):02d}.mp4'
        preview = folder / f'frame_{len(clips):02d}.jpg'
        command(['ffmpeg', '-y', '-v', 'error', '-ss', str(start),
                 '-i', video, '-t', str(end-start), '-an', '-c:v', 'libx264',
                 '-pix_fmt', 'yuv420p', str(clip)])
        command(['ffmpeg', '-y', '-v', 'error', '-ss', str((end-start)/2),
                 '-i', str(clip), '-frames:v', '1', '-update', '1', str(preview)])
        clips.append((start, end, clip, preview))
        start = end
    return clips


def prompt_for(question):
    # Handle the object's visibility directly for this common question.
    visible = re.fullmatch(r'when\s+is\s+(.+?)\s+visible\??', question.strip(), re.I)
    if visible:
        task = f'Is {visible.group(1)} visibly present in this video clip?'
    else:
        task = (f'The user asks: {question}\n'
                'Does this clip visibly show the object or event needed to answer that question?')
    return (task + ' Start with YES, NO, or UNCERTAIN, then briefly describe '
            'what you actually see. Do not infer content outside this clip.')


def find_moments(video, question, progress=gr.Progress()):
    if not video or not question.strip():
        raise gr.Error('Upload a video and enter a question first.')
    folder = Path(tempfile.mkdtemp(prefix='video_demo_'))
    progress(0, desc='Preparing video clips')
    clips = make_clips(str(video), folder)
    progress(0.1, desc='Loading Mobile-VideoGPT')
    video_model, video_tokenizer = load_model()
    rows, gallery, matches = [], [], []
    prompt = prompt_for(question)
    for i, (start, end, clip, preview) in enumerate(clips):
        progress(0.1 + 0.85*i/len(clips), desc=f'Checking clip {i+1} of {len(clips)}')
        ids, frames, context, stop = preprocess_input(
            video_model, video_tokenizer, str(clip), prompt
        )
        with torch.inference_mode():
            output = video_model.generate(
                ids, images=torch.stack(frames).half().cuda(),
                context_images=torch.stack(context).half().cuda(),
                do_sample=False, num_beams=1, max_new_tokens=96, use_cache=True,
            )
        if output.shape[1] >= ids.shape[1] and torch.equal(output[0, :ids.shape[1]], ids[0]):
            output = output[:, ids.shape[1]:]
        answer = video_tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        if stop and answer.endswith(stop):
            answer = answer[:-len(stop)].strip()
        decision_match = re.match(r'^(YES|NO|UNCERTAIN)\b', answer, re.I)
        decision = decision_match.group(1).upper() if decision_match else 'UNCERTAIN'
        rows.append([round(start, 2), round(end, 2), decision, answer])
        if decision == 'YES':
            matches.append((start, end))
            gallery.append((str(preview), f'{start:.2f}–{end:.2f} seconds'))
    if matches:
        intervals = ', '.join(f'{start:.2f}–{end:.2f} s' for start, end in matches)
        response = f'**Model-selected candidate intervals:** {intervals}'
    else:
        response = 'The model did not select any matching clips. Check the raw answers below.'
    response += ('\n\nTimes are approximate 3-second clip boundaries. '
                 'Predictions may contain false positives. Frames are midpoint previews; '
                 'they are not object bounding boxes.')
    result_file = folder / 'interactive_results.json'
    result_file.write_text(json.dumps({
        'model': MODEL_NAME, 'question': question, 'prompt': prompt,
        'predictions': [dict(zip(['start', 'end', 'decision', 'answer'], row)) for row in rows],
    }, indent=2))
    return response, gallery, rows, str(result_file)


with gr.Blocks(title='Temporal Video Grounding Assistant') as demo:
    gr.Markdown('# Temporal Video Grounding Assistant\n'
                'Cherisha Killari Giribabu · Panther ID 002901729\n\n'
                'Upload a short video and ask when an object or action is visible.')
    video = gr.Video(label='Input video', sources=['upload'])
    question = gr.Textbox(label='Your question', value='When is the water bottle visible?')
    button = gr.Button('Find moments', variant='primary')
    response = gr.Markdown()
    gallery = gr.Gallery(label='Preview frames from selected clips', columns=3)
    table = gr.Dataframe(headers=['Start (s)', 'End (s)', 'Decision', 'Raw model answer'],
                         interactive=False, label='Actual model output')
    result_file = gr.File(label='Download this test result')
    button.click(find_moments, inputs=[video, question], outputs=[response, gallery, table, result_file])

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(share=True, server_name='0.0.0.0', show_error=True)
