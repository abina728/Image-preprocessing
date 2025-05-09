import cv2
import numpy as np
import streamlit as st

def resize_image(image, width, height):
    """
    Resizes an image to the specified width and height using OpenCV.

    Args:
        image (numpy.ndarray): The input image.
        width (int): The desired width.
        height (int): The desired height.

    Returns:
        numpy.ndarray: The resized image.
    """
    try:
        resized_image = cv2.resize(image, (width, height))
        return resized_image
    except Exception as e:
        st.error(f"Error resizing image: {e}")
        return None

def apply_grayscale(image):
    """
    Converts an image to grayscale using OpenCV.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    except Exception as e:
        st.error(f"Error converting to grayscale: {e}")
        return None

def apply_gaussian_blur(image, kernel_size=(5, 5), sigmaX=0):
    """
    Applies Gaussian blur to an image using OpenCV.

    Args:
        image (numpy.ndarray): The input image.
        kernel_size (tuple): The size of the Gaussian kernel.
        sigmaX (int): The standard deviation of the Gaussian distribution in the X-direction.

    Returns:
        numpy.ndarray: The blurred image.
    """
    try:
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX)
        return blurred_image
    except Exception as e:
        st.error(f"Error applying Gaussian blur: {e}")
        return None

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image using OpenCV.

    Args:
        image (numpy.ndarray): The input image (grayscale).
        clip_limit (float): The clipping limit for contrast enhancement.
        tile_grid_size (tuple): The size of the grid for histogram equalization.

    Returns:
        numpy.ndarray: The CLAHE enhanced image.
    """
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        clahe_image = clahe.apply(image)
        return clahe_image
    except Exception as e:
        st.error(f"Error applying CLAHE: {e}")
        return None

def apply_unsharp_masking(image, kernel_size=(5, 5), weight=1.0, sigma=0.0):
    """
    Applies unsharp masking to sharpen an image using OpenCV.

    Args:
        image (numpy.ndarray): The input image.
        kernel_size (tuple): The size of the Gaussian kernel for blurring.
        weight (float):  A weight controlling the amount of sharpening.
        sigma (float):  Standard deviation of the Gaussian kernel. If zero, it is computed
                        from the kernel size.
    Returns:
        numpy.ndarray: The sharpened image.
    """
    try:
        blurred = cv2.GaussianBlur(image.copy(), kernel_size, sigma)
        sharpened = float(weight + 1) * image - float(weight) * blurred
        sharpened = np.maximum(sharpened, 0).astype(np.uint8)  # Ensure pixel values are >= 0
        sharpened = np.minimum(sharpened, 255).astype(np.uint8)
        return sharpened
    except Exception as e:
        st.error(f"Error applying unsharp masking: {e}")
        return None

def preprocess_image(image, target_width, target_height, use_clahe=True, use_unsharp=False):
    """
    Preprocesses a damaged car image.

    Args:
        image (numpy.ndarray): The input image.
        target_width (int): The desired width of the preprocessed image.
        target_height (int): The desired height of the preprocessed image.
        use_clahe (bool): Flag to apply CLAHE.
        use_unsharp (bool): Flag to apply Unsharp Masking

    Returns:
        numpy.ndarray: The preprocessed image, or None on error
    """
    resized_image = resize_image(image, target_width, target_height)
    if resized_image is None:
        return None

    gray_image = apply_grayscale(resized_image)
    if gray_image is None:
        return None

    blurred_image = apply_gaussian_blur(gray_image)
    if blurred_image is None:
        return None

    if use_clahe:
        clahe_image = apply_clahe(blurred_image)
        if clahe_image is None:
            return None
        processed_image = clahe_image
    else:
        processed_image = blurred_image

    if use_unsharp:
        sharpened_image = apply_unsharp_masking(resized_image) # Apply to color image.
        if sharpened_image is None:
            return None
        processed_image = sharpened_image

    return processed_image

def display_image(image, title="Image"):
    """
    Displays an image using Streamlit.

    Args:
        image (numpy.ndarray): The image to display.
        title (str): The title of the image display.
    """
    if image is not None:
        st.image(image, caption=title, use_column_width=True)
    else:
        st.warning(f"No image to display for {title}")

def main():
    """
    Main function to run the damaged car image preprocessing pipeline using Streamlit.
    """
    st.title("Damaged Car Image Preprocessing Pipeline")

    # Upload image using Streamlit
    uploaded_file = st.file_uploader("Upload a damaged car image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Error: Could not decode the image. Please ensure it is a valid image file.")
            return  # Stop processing if the image cannot be decoded.

        # Set target width and height
        target_width = st.slider("Target Width", min_value=100, max_value=1024, value=512)
        target_height = st.slider("Target Height", min_value=100, max_value=1024, value=512)
        use_clahe = st.checkbox("Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)", value=True)
        use_unsharp = st.checkbox("Apply Unsharp Masking (Sharpening)", value=False)


        # Display original image
        display_image(image, "Original Image")

        # Preprocess the image
        preprocessed_image = preprocess_image(image, target_width, target_height, use_clahe, use_unsharp)

        # Display the preprocessed image
        if preprocessed_image is not None:
            display_image(preprocessed_image, "Preprocessed Image")
    else:
        st.text("Please upload an image to start the preprocessing.")

if __name__ == "__main__":
    main()
