# Temporal Video Grounding Assistant

Cherisha Killari Giribabu  
Panther ID: 002901729  
CSC 8830 – Computer Vision

## Problem statement
How can a video language model identify when an object or action appears in a video using a natural language question?

## Approach
The prototype divides a video into approximately three-second clips. MobileVideoGPT checks each clip against the question. Selected clips are displayed as candidate timestamps with midpoint preview frames.

## Completed prototype
The model ran successfully on a Google Colab GPU using my own video.

Three action queries were tested:
- The person removes the lid from the bottle.
- The person opens the book.
- The person drinks from the bottle (absent-action control).

The interactive Gradio demo was also run with:
“When is the water bottle visible?”

It returned candidate timestamps, preview frames, a table of model responses, and downloadable JSON results.

## Files
- Temporal_Video_Grounding_Assistant.ipynb: Colab setup, action tests, and demo launcher.
- colab_interactive_demo.py: Interactive Gradio demo.
- run_model.py: Model inference for prepared clips.
- app.py: Local Streamlit app for video preparation and reviewing imported results.
- core.py: Video preparation and timestamp utilities.
- requirements.txt: Dependencies for the local Streamlit app.
- TEST_STATUS.json: Completed checks and limitations.

## Run the interactive demo
1. Open Temporal_Video_Grounding_Assistant.ipynb in Google Colab.
2. Select a GPU runtime.
3. Run the cells in order.
4. Upload Video_Test_Clips.zip when prompted for the action tests.
5. Upload colab_interactive_demo.py when prompted.
6. Open the printed Gradio link, upload your video, and enter a question.

Keep the Colab runtime running while using the demo.

## Run the local Streamlit app
Install FFmpeg, then run:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

## Limitations
Predictions can contain false positives. In the bottle demo, one selected preview showed a notebook. Timestamps follow clip boundaries and are approximate.

This prototype returns candidate moments. It does not provide object bounding boxes or automatically create a compressed highlight video. No benchmark accuracy is claimed.

## References
Selected keynote: Fahad Khan, Keynote 6
https://www.youtube.com/watch?v=qF9fsNVdTyI

MobileVideoGPT:
https://github.com/Amshaker/Mobile-VideoGPT

VideoMamba:
https://github.com/OpenGVLab/VideoMamba