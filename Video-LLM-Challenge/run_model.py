"""GPU adapter based on the official Mobile-VideoGPT inference example.
Run from an environment with the official Mobile-VideoGPT repository installed.
Inference has not been executed in the authoring environment.
"""
import argparse
import json
import re
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--event', required=True)
    parser.add_argument('--model', default='Amshaker/Mobile-VideoGPT-1.5B')
    parser.add_argument('--out', default='results.json')
    args = parser.parse_args()
    import torch
    import mobilevideogpt  # registers the custom model with Transformers
    from mobilevideogpt.utils import preprocess_input
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('Mobile-VideoGPT requires a CUDA GPU in this adapter')
    folder = Path(args.prepared)
    manifest = json.loads((folder / 'manifest.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,
            config=AutoConfig.from_pretrained(args.model), torch_dtype=torch.float16).cuda().eval()
    predictions = []
    prompt = (f'Is this event visible in this video clip: {args.event}? '
              'Start your answer with YES, NO, or UNCERTAIN, followed by a short description of the visual evidence. '
              'Only answer YES if the action itself occurs; merely seeing the object is insufficient.')
    for row in manifest['windows']:
        started = time.perf_counter()
        ids, frames, context, stop = preprocess_input(model, tokenizer, str(folder / row['clip']), prompt)
        with torch.inference_mode():
            output = model.generate(ids, images=torch.stack(frames, dim=0).half().cuda(),
                context_images=torch.stack(context, dim=0).half().cuda(), do_sample=False,
                num_beams=1, max_new_tokens=128, use_cache=True)
        # Some custom models return only new tokens; others include the prompt.
        if output.shape[1] >= ids.shape[1] and torch.equal(output[0, :ids.shape[1]], ids[0]):
            output = output[:, ids.shape[1]:]
        answer = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        if stop and answer.endswith(stop):
            answer = answer[:-len(stop)].strip()
        match = re.match(r'^(YES|NO|UNCERTAIN)\b', answer, re.I)
        decision = match.group(1).upper() if match else 'UNCERTAIN'
        predictions.append({'index': row['index'], 'start': row['start'], 'end': row['end'],
            'decision': decision, 'answer': answer, 'seconds': round(time.perf_counter()-started, 3)})
        print(predictions[-1], flush=True)
    result = {**manifest, 'model': args.model, 'event': args.event, 'predictions': predictions,
              'timestamp_method': 'external window boundaries; not model-predicted exact times'}
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
