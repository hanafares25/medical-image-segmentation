from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from predict import load_model, predict_image
from src.utils import get_device, overlay_mask

st.set_page_config(page_title='Medical Image Segmentation', page_icon='🩻', layout='wide')
st.title('🩻 Medical Image Segmentation App')
st.caption('PyTorch + U-Net + OpenCV | Binary segmentation demo')

checkpoint = Path('checkpoints/unet_best.pt')
threshold = st.sidebar.slider('Segmentation threshold', 0.1, 0.9, 0.5, 0.05)
st.sidebar.info('For education and portfolio use only. Not for clinical diagnosis.')

uploaded = st.file_uploader('Upload a grayscale medical image', type=['png', 'jpg', 'jpeg'])

if not checkpoint.exists():
    st.warning('No trained checkpoint found. Run `python generate_sample_data.py` and then `python train.py` first.')

if uploaded is not None:
    pil = Image.open(uploaded).convert('L')
    image = np.array(pil)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Input image')
        st.image(image, clamp=True, use_container_width=True)

    if checkpoint.exists():
        device = get_device()
        model, size = load_model(checkpoint, device)
        mask, probability = predict_image(model, image, device, size, threshold)
        overlay = overlay_mask(image, mask)

        with col2:
            st.subheader('Predicted mask')
            st.image(mask, clamp=True, use_container_width=True)

        with col3:
            st.subheader('Overlay')
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        area = int((mask > 0).sum())
        coverage = area / mask.size * 100
        st.metric('Segmented area', f'{coverage:.2f}% of image')

        mask_png = cv2.imencode('.png', mask)[1].tobytes()
        st.download_button('Download mask', mask_png, file_name='predicted_mask.png', mime='image/png')
