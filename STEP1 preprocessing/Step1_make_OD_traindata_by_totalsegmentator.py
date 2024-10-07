import os 
import numpy as np 
import SimpleITK as sitk 
from PIL import Image 
def resample(sitk_volume, new_spacing, new_size, default_value=0, is_label=False):
    """1) Create resampler"""
    resample = sitk.ResampleImageFilter() 
    
    """2) Set parameters"""
    #set interpolation method, output direction, default pixel value
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(sitk_volume.GetDirection())
    resample.SetDefaultPixelValue(default_value)
    
    #set output spacing
    new_spacing = np.array(new_spacing)
    resample.SetOutputSpacing(new_spacing)
    
    #set output size and origin
    old_size = np.array(sitk_volume.GetSize())
    old_spacing = np.array(sitk_volume.GetSpacing())
    new_size_no_shift = np.int16(np.ceil(old_size*old_spacing/new_spacing))
    old_origin = np.array(sitk_volume.GetOrigin())
    
    shift_amount = np.int16(np.floor((new_size_no_shift - new_size)/2))*new_spacing
    new_origin = old_origin + shift_amount
    
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetOutputOrigin(new_origin)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        pass


    """3) execute"""
    new_volume = resample.Execute(sitk_volume)
    return new_volume


import os 
import SimpleITK as sitk 
import numpy as np
import matplotlib.pyplot as plt 

def find_bbox_of_slice(la_arr_2d):
    # make 0-1 arr 
    la_arr_2d = la_arr_2d.astype("uint8")
    la_coord = np.where(la_arr_2d == 1)
    if len(la_coord[0]) == 0:
        return (-1, -1, -1, -1)
    y1, x1 = np.sort(la_coord[0])[0], np.sort(la_coord[1])[0]
    y2, x2 = np.sort(la_coord[0])[-1], np.sort(la_coord[1])[-1]
    # plt.imshow(la_arr_2d)
    # plt.show()
    return (x1, y1, x2, y2)

def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy 
    x = (x1 + x2)//2
    y = (y1 + y2)//2 
    w, h = x2-x1, y2-y1
    return (x, y, w, h)

def give_margin_to_xywh(xywh):
    margin_w = 20
    margin_h = 20
    x, y, w, h = xywh 
    w += margin_w
    h += margin_h
    return (x, y, w, h)

def normalized_bbox(xywh, size):
    x, y, w, h = xywh 
    norm_x = float(x/size)
    norm_y = float(y/size)
    norm_w = float(w/size)
    norm_h = float(h/size)
    return (norm_x, norm_y, norm_w, norm_h)


im_basepath = r"/media/cwl/84DC5431DC54202A/! project/2024- AortaSeg/data/processed_data/nifti/images/"
la_basepath = r"/media/cwl/84DC5431DC54202A/! project/2024- AortaSeg/data/processed_data/nifti/labels_normal/"
#txt_savepath = r"/media/cwl/84DC5431DC54202A/! project/2024- AortaSeg/data/processed_data/od/fold4_normal_od/labels/"

im_dirs_list = sorted(os.listdir(im_basepath))
la_dirs_list = sorted(os.listdir(la_basepath))
print(im_dirs_list)
print(la_dirs_list)
patient_dict = {}
savebasepath =r"/media/cwl/84DC5431DC54202A/! project/2024- AortaSeg/data/processed_data/od_normal/fold_1/"
im_t_path = r"/media/cwl/84DC5431DC54202A/! project/2024- AortaSeg/data/processed_data/od_normal/fold_1/images/train_im"
im_v_path = r"/media/cwl/84DC5431DC54202A/! project/2024- AortaSeg/data/processed_data/od_normal/fold_1/images/valid_im"
la_t_path = r"/media/cwl/84DC5431DC54202A/! project/2024- AortaSeg/data/processed_data/od_normal/fold_1/labels/train_im"
la_v_path = r"/media/cwl/84DC5431DC54202A/! project/2024- AortaSeg/data/processed_data/od_normal/fold_1/labels/valid_im"
os.makedirs(im_t_path, exist_ok=True)
os.makedirs(im_v_path, exist_ok=True)
os.makedirs(la_t_path, exist_ok=True)
os.makedirs(la_v_path, exist_ok=True)


# for k, v in la_dict.items():
#     print(k)
# quit()


name_dict = {"sternum":1, "scapula_left":2, "scapula_right":3, "heart":4, "kidney_left":5, "kidney_right":6, "hip_left":7, "hip_right":8}

for idx in range(len(im_dirs_list)):
    la_dict = {"sternum":None, "scapula_left":None, "scapula_right":None, "heart":None, "kidney_left":None, "kidney_right":None, "hip_left":None, "hip_right":None}
    print(idx)
    im_itk = sitk.ReadImage(os.path.join(im_basepath, im_dirs_list[idx]))
    la_nii_dirs_list = sorted(os.listdir(os.path.join(la_basepath, la_dirs_list[idx])))
    la_nii_list = []
    
    for jdx in range(len(la_nii_dirs_list)):
        for key, val in name_dict.items():
            if key in la_nii_dirs_list[jdx]:
                la_itk = sitk.ReadImage(os.path.join(la_basepath, la_dirs_list[idx], la_nii_dirs_list[jdx]))
                la_itk = resample(la_itk, la_itk.GetSpacing(), (512, 512, la_itk.GetSize()[2]), -1024, is_label=True)
                la_dict[key] = la_itk
                break 
    
                
   

    im_itk = resample(im_itk, im_itk.GetSpacing(), (512, 512, im_itk.GetSize()[2]), -1024)
    spacing = im_itk.GetSpacing()
    origin = im_itk.GetOrigin()
    direction = im_itk.GetDirection()
    temp_la_arr_list = []
    for key, la_itk in la_dict.items():
        la_arr = sitk.GetArrayFromImage(la_itk)
        temp_la_arr_list.append(la_arr)
        
    ct_arr = sitk.GetArrayFromImage(im_itk)
    ct_size = im_itk.GetSize()

    area_num_list = [1, 2, 3, 4, 5, 6, 7, 8]
    
    coord_arr = np.ones((len(ct_arr), len(area_num_list), 4)) * -1 # (ct의 z축 길이, label의 갯수(1부터 순서대로), normalized된 xywh 좌표 들어갈 4개)
    for i in range(len(ct_arr)): # z축 길이로 순회
        # bbox_dict = {}
        for temp_la_arr, label_num in zip(temp_la_arr_list, area_num_list):
            bbox_xyxy = find_bbox_of_slice(temp_la_arr[i])
            if bbox_xyxy == (-1, -1, -1, -1):
                continue
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)
            bbox_xywh = give_margin_to_xywh(bbox_xywh)
            norm_xywh = normalized_bbox(bbox_xywh, ct_size[0])
            # bbox_dict[label_num] = norm_xywh
            coord_arr[i][label_num-1] = norm_xywh
    
    margin_z = 2
    for i in range(coord_arr.shape[1]):
        
        temp = coord_arr[:, i, :] #한 OAR에대한 그 CT에 대해 전체 좌표 (z축으로)
        get_z_done = np.where(temp != -1)
        if len(get_z_done[0]) == 0:
            continue

        z1 = np.sort(get_z_done[0])[0]
        z2 = np.sort(get_z_done[0])[-1]

        z1_bbox = coord_arr[z1, i, :]
        z2_bbox = coord_arr[z2, i, :]
        
        z1_m = z1 - margin_z
        z2_m = z2 + margin_z
        if z1_m < 0:
            z1_m = 0
        if z2_m > coord_arr.shape[0] - 1:
            z2_m = coord_arr.shape[0] - 1

        coord_arr[z1_m:z1, i, :] = z1_bbox
        coord_arr[z2:z2_m+1, i,:] = z2_bbox

    for i in range(coord_arr.shape[0]):
        if 30 <= idx < 40:

            txt_file = open(os.path.join(la_v_path, "P%03d_%03d.txt" %(idx, i)), "w+")
            
            for j in range(coord_arr.shape[1]):
                if np.all(coord_arr[i, j, :] == -1):
                    continue
                
                txt_file.write("%d %f %f %f %f\n" %(j, *coord_arr[i, j, :]))
            txt_file.close()
        else:
            txt_file = open(os.path.join(la_t_path, "P%03d_%03d.txt" %(idx, i)), "w+")
            
            for j in range(coord_arr.shape[1]):
                if np.all(coord_arr[i, j, :] == -1):
                    continue
                
                txt_file.write("%d %f %f %f %f\n" %(j, *coord_arr[i, j, :]))
            txt_file.close()


            # print(np.max(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_path, file_dir_list[jdx])))))
    
    
    img_arr = np.copy(ct_arr)
    win_min = -500
    win_max = 1000
    img_arr[img_arr < win_min] = win_min 
    img_arr[img_arr > win_max] = win_max 
    img_arr = np.uint8(255 * (img_arr.astype("float64") - win_min)/(win_max - win_min))
    for i in range(len(img_arr)):
        if 30 <= idx < 40:
            img = Image.fromarray(img_arr[i])
            img.save(os.path.join(im_v_path, "P%03d_%03d.png" %(idx, i)))
        else:
            img = Image.fromarray(img_arr[i])
            img.save(os.path.join(im_t_path, "P%03d_%03d.png" %(idx, i)))





train_path_list = sorted(os.listdir(im_t_path))
valid_path_list = sorted(os.listdir(im_v_path))

lines = []


for i in range(len(train_path_list)):
    lines.append("./images/train_im/" + train_path_list[i] +"\n")

with open(os.path.join(savebasepath, "train_im.txt"), "w") as file:
    file.writelines(lines)

lines = []


for i in range(len(valid_path_list)):
    lines.append("./images/valid_im/" + valid_path_list[i] +"\n")

with open(os.path.join(savebasepath, "valid_im.txt"), "w") as file:
    file.writelines(lines)