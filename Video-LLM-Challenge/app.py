"""Prepare a video locally and review real Mobile-VideoGPT results."""
import hashlib
import json
import tempfile
from pathlib import Path
import streamlit as st
from core import prepare, overlap

st.set_page_config(page_title='Temporal Video Grounding Assistant', layout='wide')
st.title('Temporal Video Grounding Assistant')
st.caption('Cherisha Killari Giribabu · Panther ID 002901729 · CSC 8830')
st.info('This app prepares and displays video windows. Model inference runs separately on a CUDA GPU using run_model.py. No automatic answer is generated here.')
video = st.file_uploader('Upload a short MP4 or MOV (10–30 seconds works well)', type=['mp4', 'mov'])
window = st.select_slider('Window duration in seconds', options=[1, 2, 3, 5], value=3)
event = st.text_input('Event to locate', 'The person picks up the bottle')
if video:
    content = video.getvalue()
    key = hashlib.sha256(content + str(window).encode()).hexdigest()
    if st.button('Prepare video'):
        try:
            directory = Path(tempfile.mkdtemp(prefix='video_grounding_'))
            path = directory / ('input' + Path(video.name).suffix.lower())
            path.write_bytes(content)
            with st.spinner('Preparing clips and timestamped evidence…'):
                manifest = prepare(path, directory / 'prepared', window)
            st.session_state['prepared'] = (key, str(directory), manifest)
        except Exception as error:
            st.error(f'Could not process video: {error}')
    state = st.session_state.get('prepared')
    if state and state[0] == key:
        _, directory, manifest = state
        prepared = Path(directory) / 'prepared'
        st.video(content)
        st.write(f"{manifest['duration']:.2f} seconds → {len(manifest['windows'])} windows")
        columns = st.columns(3)
        for row in manifest['windows']:
            columns[row['index'] % 3].image(str(prepared / row['frame']), caption=f"{row['start']:.2f}–{row['end']:.2f} s")
        import io, zipfile
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(prepared.iterdir()):
                archive.write(file, file.name)
        st.download_button('Download prepared clips for GPU inference', buffer.getvalue(), 'prepared.zip')
        st.code(f'python run_model.py --prepared prepared --event {json.dumps(event)} --out results.json', language='bash')
        results = st.file_uploader('Import results.json after running the model', type=['json'])
        if results:
            try:
                data = json.load(results)
                if data['source'] != manifest['source'] or data['windows'] != manifest['windows']:
                    raise ValueError('Results do not match this prepared video')
                st.write('Model:', data['model'], 'Event:', data['event'])
                st.dataframe(data['predictions'], use_container_width=True)
                matches = [r for r in data['predictions'] if r['decision'] == 'YES']
                if matches:
                    st.success('Matching intervals: ' + ', '.join(f"{r['start']:.2f}–{r['end']:.2f} s" for r in matches))
                    for row in matches:
                        st.image(str(prepared / manifest['windows'][row['index']]['frame']), caption=f"Evidence from {row['start']:.2f}–{row['end']:.2f} s")
                else:
                    st.warning('No window received an explicit YES; inspect raw responses before drawing a conclusion.')
                st.caption('Times come from clip boundaries. They are approximate locations, not exact action onset/offset predictions.')
                st.subheader('Compare with your manually recorded event interval')
                left, right = st.columns(2)
                start = left.number_input('Observed start (seconds)', 0.0, manifest['duration'], 0.0)
                end = right.number_input('Observed end (seconds)', 0.0, manifest['duration'], manifest['duration'])
                if end > start and matches:
                    st.write('Best matching-window temporal IoU:', round(max(overlap((r['start'], r['end']), (start, end)) for r in matches), 3))
            except (KeyError, ValueError, TypeError) as error:
                st.error(f'Invalid results file: {error}')
