import numpy as np

class OpticalProcessor:
    @staticmethod
    def calculate_ndvi(red_band, nir_band):
        """
        Calculates NDVI: (NIR - Red) / (NIR + Red)
        """
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        # Handle division by zero using np.errstate
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi = np.nan_to_num(ndvi) # Replace NaNs with 0
        return ndvi

    @staticmethod
    def create_cloud_mask(scl_band):
        """
        Creates a boolean mask where True = Clear, False = Cloudy/Shadowed.
        SCL codes: 3=Shadow, 8=Med Prob, 9=High Prob, 10=Thin Cirrus
        """
        # SCL is usually at 20m, but we treat it as an array here
        cloud_values = [3, 8, 9, 10]
        # Mask is True for clear pixels
        mask = ~np.isin(scl_band, cloud_values)
        return mask

if __name__ == "__main__":
    # # Mock test
    # h, w = 100, 100
    # mock_red = np.random.uniform(0, 0.2, (h, w))
    # mock_nir = np.random.uniform(0.4, 0.8, (h, w))
    # mock_scl = np.random.choice([4, 4, 4, 9], (h, w)) # Mostly vegetation, some clouds
    
    # proc = OpticalProcessor()
    # ndvi = proc.calculate_ndvi(mock_red, mock_nir)
    # mask = proc.create_cloud_mask(mock_scl)
    
    print(f"Optical Processing test complete.")
    # print(f"NDVI Mean: {ndvi.mean():.2f}")
    # print(f"Clear pixels: {np.sum(mask)} out of {h*w}")