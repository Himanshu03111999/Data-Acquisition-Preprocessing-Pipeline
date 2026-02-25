import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import graycomatrix, graycoprops
from skimage import exposure

class SARProcessor:
    @staticmethod
    def apply_lee_filter(img, size=5):
        """
        Standard Lee Filter for SAR speckle reduction.
        Preserves edges while smoothing flat areas.
        """
        img = img.astype(np.float32)
        img_mean = uniform_filter(img, (size, size))
        img_sqr_mean = uniform_filter(img**2, (size, size))
        img_variance = img_sqr_mean - img_mean**2

        overall_variance = np.var(img)
        
        # Lee filter weighting: var / (var + noise_var)
        # We use overall_variance as a proxy for the noise variance
        weighting = img_variance / (img_variance + overall_variance + 1e-10)
        refined_img = img_mean + weighting * (img - img_mean)
        return refined_img

    @staticmethod
    def generate_glcm_texture(image_array, window_size=7):
        """
        Generates a Contrast texture band from SAR backscatter.
        1. Quantizes image to 8-bit (uint8).
        2. Calculates GLCM on a sliding window.
        """
        # Robust scaling to 8-bit (0-255) for GLCM
        p2, p98 = np.percentile(image_array, (2, 98))
        img_8bit = exposure.rescale_intensity(
            image_array, in_range=(p2, p98), out_range=(0, 255)
        ).astype(np.uint8)

        h, w = img_8bit.shape
        pad = window_size // 2
        texture_band = np.zeros_like(img_8bit, dtype=np.float32)

        # Sliding window for texture (Pro-tip: avoid large images here)
        for i in range(pad, h - pad, 2): # Stepping by 2 to speed up for assignment
            for j in range(pad, w - pad, 2):
                window = img_8bit[i-pad:i+pad+1, j-pad:j+pad+1]
                glcm = graycomatrix(window, distances=[1], angles=[0], 
                                    levels=256, symmetric=True, normed=True)
                texture_band[i, j] = graycoprops(glcm, 'contrast')[0, 0]
        
        return texture_band

if __name__ == "__main__":
    # Mock test
    mock_sar = np.random.rand(100, 100).astype(np.float32)
    proc = SARProcessor()
    filtered = proc.apply_lee_filter(mock_sar)
    texture = proc.generate_glcm_texture(filtered)
    print(f"SAR Processing test complete. Texture mean: {np.mean(texture)}")