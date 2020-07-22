import cv2
import numpy as np
import pytest

from paz.backend.image import opencv_image

# TODO:
# Add tests for the following random functions:
# random_saturation
# random_brightness
# random_contrast
# random_hue
# random_plain_background
# show_image


@pytest.fixture
def load_image():
    def call(shape, rgb_channel, with_mask=True):
        image = np.ones(shape)
        image[:, :, 0] = rgb_channel[0]
        image[:, :, 1] = rgb_channel[1]
        image[:, :, 2] = rgb_channel[2]
        if with_mask:
            image[10:50, 50:120] = 100
        return image.astype(np.uint8)
    return call


@pytest.fixture(params=[(128, 128, 3)])
def image_shape(request):
    return request.param


@pytest.fixture(params=[[50, 120, 201]])
def rgb_channel(request):
    return request.param


@pytest.fixture(params=[60])
def resized_shape(request):
    def call(image):
        width = int(image.shape[1] * request.param / 100)
        height = int(image.shape[0] * request.param / 100)
        size = width * height * 3
        return (width, height, size)
    return call


@pytest.mark.parametrize("rgb_channel", [[50, 120, 201]])
def test_cast_image(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    casted_image = opencv_image.cast_image(test_image, dtype=np.float32)
    assert casted_image.dtype == np.float32


def test_resize_image(load_image, image_shape, rgb_channel, resized_shape):
    test_image = load_image(image_shape, rgb_channel)
    width, height, size = resized_shape(test_image)
    resized_image = opencv_image.resize_image(test_image, (width, height))
    assert resized_image.size == size


def test_convert_color_space(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    converted_colorspace = opencv_image.convert_color_space(
        test_image, cv2.COLOR_RGB2BGR)
    rgb_to_bgr = test_image[..., ::-1]
    assert np.all(converted_colorspace == rgb_to_bgr)


def test_flip_left_right(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    image_filp = opencv_image.flip_left_right(test_image)
    flipped_image = test_image[:, ::-1]
    assert np.all(image_filp == flipped_image)


def test_gaussian_blur_output_shape(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    blurred = opencv_image.gaussian_blur(test_image)
    assert test_image.shape == blurred.shape


def test_split_alpha_channel(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    b_channel, g_channel, r_channel = cv2.split(test_image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
    masked_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    split_alpha_img, alpha_mode = opencv_image.split_alpha_channel(
        masked_image)
    assert np.all(split_alpha_img == test_image)


@pytest.mark.parametrize("rgb", [[50, 120, 201]])
def test_alpha_blend(load_image, image_shape, rgb):
    test_image = load_image(image_shape, rgb)
    background_image = load_image(image_shape, rgb, with_mask=False).astype(float)
    foreground = load_image(image_shape, [0, 0, 0])
    alpha = load_image(image_shape, [0, 0, 0])
    alpha[10:50, 50:120] = 255
    alpha = alpha.astype(float) / 255.
    alpha_blend_image = opencv_image.alpha_blend(
        foreground, background_image, alpha)
    assert np.all(alpha_blend_image == test_image)
