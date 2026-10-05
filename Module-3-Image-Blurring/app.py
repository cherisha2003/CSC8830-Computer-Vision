import streamlit as st
import cv2
import numpy as np
from PIL import Image


def spatial_blur(img, kernel):
    result = cv2.filter2D(
        img.astype(np.float64),
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT
    )
    return result


def fourier_blur(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape

    new_h = h + kh - 1
    new_w = w + kw - 1

    img_fft = np.fft.fft2(img, s=(new_h, new_w))
    kernel_fft = np.fft.fft2(kernel, s=(new_h, new_w))

    result_fft = img_fft * kernel_fft

    result = np.fft.ifft2(result_fft)
    result = np.real(result)

    start_h = kh // 2
    start_w = kw // 2

    result = result[
        start_h:start_h + h,
        start_w:start_w + w
    ]

    return result


st.title("Image Blurring using Spatial and Fourier Filtering")

st.write(
    "This program compares image blurring in the spatial domain "
    "and the Fourier domain."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

kernel_size = st.slider(
    "Filter size",
    3,
    15,
    5,
    step=2
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.float64
    )

    kernel = kernel / (kernel_size * kernel_size)

    spatial_result = spatial_blur(gray, kernel)

    fourier_result = fourier_blur(
        gray.astype(np.float64),
        kernel
    )

    difference = np.abs(
        spatial_result - fourier_result
    )

    mse = np.mean(
        (spatial_result - fourier_result) ** 2
    )

    max_diff = np.max(difference)

    st.subheader("Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Original")
        st.image(
            gray,
            width="stretch"
        )

    with col2:
        st.write("Spatial Domain")
        st.image(
            np.clip(
                spatial_result,
                0,
                255
            ).astype(np.uint8),
            width="stretch"
        )

    with col3:
        st.write("Fourier Domain")
        st.image(
            np.clip(
                fourier_result,
                0,
                255
            ).astype(np.uint8),
            width="stretch"
        )

    st.subheader("Difference Image")

    st.image(
        np.clip(
            difference,
            0,
            255
        ).astype(np.uint8),
        width="stretch"
    )

    st.subheader("Comparison")

    st.write("Mean Squared Error:", mse)
    st.write("Maximum Pixel Difference:", max_diff)

    st.subheader("Filter Used")

    st.write(kernel)

    st.subheader("Convolution Theorem")

    st.write("Spatial domain:")

    st.latex(
        r"g(x,y) = f(x,y) * h(x,y)"
    )

    st.write("Fourier domain:")

    st.latex(
        r"G(u,v) = F(u,v)H(u,v)"
    )

    st.write(
        "If both outputs are nearly identical, it shows that "
        "convolution in the spatial domain is equivalent to "
        "multiplication in the Fourier domain."
    )