"""Explanations used by app.py. Start the app with `python -m streamlit run app.py`."""
import streamlit as st


def explain_method(kind):
    if kind == "rgb":
        st.markdown("""
**How the RGB result is produced**

1. **Mark the target.** A rectangle encloses the person. Green points mark known foreground, and red points mark background. Everything outside the rectangle is also background.
2. **Smooth the image.** A small Gaussian blur reduces noise that could create tiny unwanted regions.
3. **Run watershed.** Foreground and background regions spread from the markers according to local color differences. Strong changes can form dividing boundaries.
4. **Build the mask.** Pixels assigned to the foreground become white; other pixels become black. Optional closing joins small gaps. The saved example uses a 1-pixel kernel, so closing has no effect there.
5. **Trace the outline.** OpenCV finds the outer contours of the mask and draws them in red.

**Why parts can be missed:** A hand, shirt, or shoe may have a different color from the torso. More foreground points can help, but weak contrast and clutter still cause errors. The points are supplied manually; this is prompted segmentation, not automatic human recognition.
""")
    else:
        st.markdown("""
**How the thermal result is produced**

1. **Read intensity.** Brighter pixels represent stronger displayed infrared intensity in this example; these values are not temperatures in degrees.
2. **Smooth the frame.** A Gaussian blur reduces small intensity fluctuations.
3. **Choose a threshold.** Otsu's method examines intensities inside the target box and selects the split that best separates two intensity groups.
4. **Clean the mask.** Opening removes small specks, and closing fills narrow gaps. The component under the box center is retained; if no suitable component is there, the largest remaining one is used.
5. **Trace the outline.** The selected region becomes the white mask, with its boundary drawn in red.

**Why this example works relatively well:** The person has strong contrast against the background. Warm objects, dark clothing, and weak contrast can still cause missing areas or extra foreground. The box selects the target; thresholding itself does not know what a person is.
""")


def explain_scores():
    st.markdown("""**Reading the comparison**

White pixels belong to the selected person. The red contour is the classical result; the green contour is SAM2. Where the contours overlap, the methods agree. Where they separate, inspect what was added or missed.

SAM2 is a pretrained deep-learning model used only for the requested comparison. Its included masks were generated separately. The website displays those saved masks; it does not run SAM2 every time the page opens.
""")
    st.latex(r"\operatorname{IoU}=\frac{|A\cap S|}{|A\cup S|},\qquad \operatorname{Dice}=\frac{2|A\cap S|}{|A|+|S|}")
    st.caption("A is the classical mask and S is the SAM2 mask. Scores range from 0 to 1; 1 means identical masks. SAM2 is a reference, not independently verified ground truth.")
