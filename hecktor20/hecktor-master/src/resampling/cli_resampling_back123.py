import os
from multiprocessing import Pool
from pathlib import Path

import click
import logging
import pandas as pd
import numpy as np
from scipy.ndimage import affine_transform
from scipy.interpolate import RegularGridInterpolator
import SimpleITK as sitk

from src.resampling.resampling import Resampler
from src.resampling.utils import (get_sitk_volume_from_np,
                                  get_np_volume_from_sitk)

path_input = "data/hecktor_test/predicted"
path_output = "data/hecktor_test/submit"
path_bb = "data/hecktor_test/bbox_test.csv"
path_res = "data/original_resolution_ct.csv"


@click.command()
@click.argument('input_folder',
                type=click.Path(exists=True),
                default=path_input)
@click.argument('output_folder', type=click.Path(), default=path_output)
@click.argument('bounding_boxes_file', type=click.Path(), default=path_bb)
@click.argument('original_resolution_file',
                type=click.Path(),
                default=path_res)
@click.option('--cores', type=click.INT, default=1)

def main(input_folder, output_folder, bounding_boxes_file,
         original_resolution_file, cores):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    resolution_df = pd.read_csv(original_resolution_file)
    resolution_df = resolution_df.set_index('PatientID')
    files_list = [
        str(f.resolve()) for f in Path(input_folder).rglob('*.nii.gz')
    ]
    patient_list = [
        f.name[:7] for f in Path(input_folder).rglob('*.nii.gz')
    ]
    #print(output_folder)
    #print(patient_list)
    resampler = Resampler(bb_df, output_folder, order='nearest')
    print(resampler.bb_df, resampler.output_folder, resampler.order )

    resolution_list = [(resolution_df.loc[k, 'Resolution_x'],
                        resolution_df.loc[k, 'Resolution_y'],
                        resolution_df.loc[k, 'Resolution_z'])
                       for k in patient_list]

    #print(files_list)
    for i in range(len(files_list)):
        xxx_resample(bb_df,output_folder, f= files_list[i], order='nearest', resampling = resolution_list[i])
    #with Pool(cores) as p:
        #p.starmap(resampler, zip(files_list, resolution_list))



def xxx_resample(bb_df, output_folder, f, order, resampling ):
    print(f, resampling)
    if resampling is None:
        resampling = resampling
    patient_name = f.split('/')[-1][:7]
    
    # patient_folder = os.path.join(self.output_folder, patient_name)
    # if not os.path.exists(patient_folder):
    #     os.mkdir(patient_folder)
    # output_file = os.path.join(patient_folder, f.split('/')[-1])
    output_file = os.path.join(output_folder, f.split('/')[-1])
    bb = (bb_df.loc[patient_name, 'x1'], bb_df.loc[patient_name,
                                                             'y1'],
          bb_df.loc[patient_name, 'z1'], bb_df.loc[patient_name,
                                                             'x2'],
          bb_df.loc[patient_name, 'y2'], bb_df.loc[patient_name,
                                                             'z2'])
    print('Resampling patient {}'.format(patient_name))

    resample_and_crop(f,
                      output_file,
                      bb,
                      resampling=resampling,
                      order=order)


def resample_and_crop(input_file,
                      output_file,
                      bounding_box,
                      resampling=(1.0, 1.0, 1.0),
                      order=3):
    np_volume, pixel_spacing, origin = get_np_volume_from_sitk(
        sitk.ReadImage(input_file))
    resampling = np.asarray(resampling)
    # If one value of resampling is -1 replace it with the original value
    for i in range(len(resampling)):
        if resampling[i] == -1:
            resampling[i] = pixel_spacing[i]
        elif resampling[i] < 0:
            raise ValueError(
                'Resampling value cannot be negative, except for -1')

    if ('gtv' in input_file or 'GTV' in input_file or order == 'nearest'):
        np_volume = resample_np_binary_volume(np_volume, origin, pixel_spacing,
                                              resampling, bounding_box)

    else:
        np_volume = resample_np_volume(np_volume,
                                       origin,
                                       pixel_spacing,
                                       resampling,
                                       bounding_box,
                                       order=order)

    origin = np.asarray([bounding_box[0], bounding_box[1], bounding_box[2]])
    sitk_volume = get_sitk_volume_from_np(np_volume, resampling, origin)
    writer = sitk.ImageFileWriter()
    print("writing:", output_file)
    writer.SetFileName(output_file)
    writer.SetImageIO("NiftiImageIO")
    writer.Execute(sitk_volume)


def resample_np_volume(np_volume,
                       origin,
                       current_pixel_spacing,
                       resampling_px_spacing,
                       bounding_box,
                       order=3):

    zooming_matrix = np.identity(3)
    zooming_matrix[0, 0] = resampling_px_spacing[0] / current_pixel_spacing[0]
    zooming_matrix[1, 1] = resampling_px_spacing[1] / current_pixel_spacing[1]
    zooming_matrix[2, 2] = resampling_px_spacing[2] / current_pixel_spacing[2]

    offset = ((bounding_box[0] - origin[0]) / current_pixel_spacing[0],
              (bounding_box[1] - origin[1]) / current_pixel_spacing[1],
              (bounding_box[2] - origin[2]) / current_pixel_spacing[2])

    output_shape = np.ceil([
        bounding_box[3] - bounding_box[0],
        bounding_box[4] - bounding_box[1],
        bounding_box[5] - bounding_box[2],
    ]) / resampling_px_spacing

    np_volume = affine_transform(np_volume,
                                 zooming_matrix,
                                 offset=offset,
                                 mode='mirror',
                                 order=order,
                                 output_shape=output_shape.astype(int))

    return np_volume


def grid_from_spacing(start, spacing, n):
    return np.asarray([start + k * spacing for k in range(n)])


def resample_np_binary_volume(np_volume, origin, current_pixel_spacing,
                              resampling_px_spacing, bounding_box):
    print("running np binary volume......")
    x_old = grid_from_spacing(origin[0], current_pixel_spacing[0],
                              np_volume.shape[0])
    y_old = grid_from_spacing(origin[1], current_pixel_spacing[1],
                              np_volume.shape[1])
    z_old = grid_from_spacing(origin[2], current_pixel_spacing[2],
                              np_volume.shape[2])

    output_shape = (np.ceil([
        bounding_box[3] - bounding_box[0],
        bounding_box[4] - bounding_box[1],
        bounding_box[5] - bounding_box[2],
    ]) / resampling_px_spacing).astype(int)

    x_new = grid_from_spacing(bounding_box[0], resampling_px_spacing[0],
                              output_shape[0])
    y_new = grid_from_spacing(bounding_box[1], resampling_px_spacing[1],
                              output_shape[1])
    z_new = grid_from_spacing(bounding_box[2], resampling_px_spacing[2],
                              output_shape[2])
    interpolator = RegularGridInterpolator((x_old, y_old, z_old),
                                           np_volume,
                                           method='nearest',
                                           bounds_error=False,
                                           fill_value=0)
    x, y, z = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    pts = np.array(list(zip(x.flatten(), y.flatten(), z.flatten())))

    return interpolator(pts).reshape(output_shape)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
