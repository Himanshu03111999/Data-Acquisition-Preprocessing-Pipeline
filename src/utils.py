import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

class GeoUtils:
    @staticmethod
    def save_geotiff(filename, data, bbox, shape, crs="EPSG:4326"):
        """
        Saves a numpy array as a geospatial TIFF.
        Theory: Combines raw pixel math with a 'Transform' matrix so the 
        image knows its exact GPS location.
        """
        h, w = shape
        # Create the affine transform (maps pixels to coordinates)
        new_transform = transform_from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

        meta = {
            'driver': 'GTiff',
            'height': h,
            'width': w,
            'count': 1,
            'dtype': 'float32',
            'crs': crs,
            'transform': new_transform,
            'nodata': np.nan,
            'compress': 'lzw'
        }

        # Ensure the outputs folder exists
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        path = os.path.join(output_dir, filename)
        
        with rasterio.open(path, 'w', **meta) as dst:
            dst.write(data.astype('float32'), 1)
        
        return path

    @staticmethod
    def print_summary(ndvi_mean, sar_mean):
        """Prints a clean status report to the console."""
        print("-" * 30)
        print(f"{'METRIC':<15} | {'VALUE':<10}")
        print("-" * 30)
        print(f"{'Mean NDVI':<15} | {ndvi_mean:<10.2f}")
        print(f"{'Mean Texture':<15} | {sar_mean:<10.2f}")
        print("-" * 30)