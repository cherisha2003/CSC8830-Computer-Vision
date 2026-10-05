# CSC 8830 - Module 3

## Image Blurring using Spatial and Fourier Filtering

This project implements image blurring using an averaging filter.

The same filter is applied in two different ways:

- Spatial domain convolution
- Fourier domain multiplication

The goal is to show that convolution in the spatial domain produces the same result as multiplication in the Fourier domain.

## How it works

An image is uploaded through the Streamlit application and converted to grayscale.

The user can select different filter sizes such as 3x3, 5x5, 7x7, and 15x15.

The program then:

1. Applies the averaging filter using spatial convolution.
2. Applies the same filter using the Fourier transform.
3. Displays both results.
4. Calculates the difference between the two images.
5. Calculates the Mean Squared Error and maximum pixel difference.

The difference between the two results should be very close to zero.

## Convolution Theorem

In the spatial domain:

g(x,y) = f(x,y) * h(x,y)

In the Fourier domain:

G(u,v) = F(u,v)H(u,v)

This shows that convolution in the spatial domain is equivalent to multiplication in the Fourier domain.

## Run the program

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Files

- `app.py` - main Python and Streamlit application
- `requirements.txt` - Python packages required to run the project
- `README.md` - project description and instructions

## Web Application

The deployed application can be accessed here:

https://csc8830-module3-image-blurring-jshaur3ekyrebw9tuvvfhb.streamlit.app/

## GitHub Repository

https://github.com/cherisha2003/CSC8830-Module3-Image-Blurring
