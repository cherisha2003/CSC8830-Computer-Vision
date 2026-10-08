"""Video windows and evaluation; no model or hand-written answers here."""
import json
import subprocess
from pathlib import Path


def run(args):
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def duration(path):
    data = json.loads(run(['ffprobe', '-v', 'error', '-show_format', '-of', 'json', str(path)]))
    seconds = float(data['format']['duration'])
    if seconds <= 0:
        raise ValueError('Video must have a positive duration')
    return seconds


def prepare(path, output, window=3.0):
    if window <= 0:
        raise ValueError('Window length must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    length = duration(path)
    if length > 60:
        raise ValueError('Use a video no longer than 60 seconds for this experiment')
    rows, start = [], 0.0
    while start < length - 1e-6:
        end = min(length, start + window)
        name = f'window_{len(rows):03d}'
        clip = output / f'{name}.mp4'
        frame = output / f'{name}.jpg'
        run(['ffmpeg', '-y', '-v', 'error', '-ss', str(start), '-i', str(path),
             '-t', str(end-start), '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(clip)])
        run(['ffmpeg', '-y', '-v', 'error', '-ss', str((end-start)/2), '-i', str(clip),
             '-frames:v', '1', '-update', '1', str(frame)])
        rows.append({'index': len(rows), 'start': start, 'end': end,
                     'clip': clip.name, 'frame': frame.name})
        start = end
    manifest = {'source': Path(path).name, 'duration': length, 'window_seconds': window, 'windows': rows}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def overlap(a, b):
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--out', default='prepared')
    parser.add_argument('--window', type=float, default=3)
    args = parser.parse_args()
    print(json.dumps(prepare(args.video, args.out, args.window), indent=2))
