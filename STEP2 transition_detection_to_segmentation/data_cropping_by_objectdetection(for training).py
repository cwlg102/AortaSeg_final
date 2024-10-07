import os
import argparse
import time
import SimpleITK as sitk
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
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

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    nii_basepath = opt.nii_source_path 
    nii_dirs_list = os.listdir(nii_basepath)
    label_basepath = opt.nii_label_source_path
    label_dirs_list = os.listdir(label_basepath)

    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16


    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    name_dict = {"sternum":0, "scapula_left":1, "scapula_right":2, "heart":3, "kidney_left":4, "kidney_right":5, "hip_left":6, "hip_right":7}
    name_rev_dict = {v:k for k,v in name_dict.items()}
    
    for nii_idx in range(len(nii_dirs_list)):
        print(nii_dirs_list[nii_idx])
        im_itk = sitk.ReadImage(os.path.join(nii_basepath, nii_dirs_list[nii_idx]))
        im_itk = resample(im_itk, im_itk.GetSpacing(), (512, 512, im_itk.GetSize()[2]), -1024)
        im_arr = sitk.GetArrayFromImage(im_itk)
        win_min = -500
        win_max = 1000
        im_arr[im_arr < win_min] = win_min 
        im_arr[im_arr > win_max] = win_max 
        im_arr = np.uint8(255 * (im_arr.astype(np.double) - win_min)/(win_max - win_min))
        im_arr = np.stack((im_arr, )*3 ,axis = -1)
        im_arr = np.transpose(im_arr, (3, 0, 1, 2))
        im_arr = im_arr[np.newaxis, ...]
        pic_idx = -1

        coord_oar_arr = np.ones((im_itk.GetSize()[2], len(name_dict), 4)) * -1 # z축, oar갯수, x1 y1 x2 y2

        for img_ind in range(im_arr.shape[2]):
            img = im_arr[:, :, img_ind, :, :]
            # print(img.shape)
            pic_idx += 1
            # name_ = path.split("_im")[-1][1:]
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            # print(opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms)
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                coord_list = []
                 #x1 y1 x2 y2 class
                # if webcam:  # batch_size >= 1
                #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                # else:
                #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                # p = Path(p)  # to Path

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (512, 512, 3)).round()

                    # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        coord_arr = []
                        # if save_txt:  # Write to file
                            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            # with open(txt_path + '.txt', 'a') as f:
                                # f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # if save_img or view_img:  # Add bbox to image
                            # label = f'{names[int(cls)]} {conf:.2f}'
                            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                        # print(x1, y1, x2, y2)
                        class_num = cls.item()
                        coord_arr=[x1, y1, x2, y2, class_num]

                        coord_list.append(coord_arr)
                coord_np = np.array(coord_list)
                coord_np = coord_np.astype("uint16")
                # print(i, coord_np)

                # Print time (inference + NMS)
                # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            for xy_cls_arr in coord_np:
                xy = np.int32(xy_cls_arr[:4])
                oar_cls = xy_cls_arr[4] 
                coord_oar_arr[img_ind, oar_cls] = xy
        
        ######################################Zone 1 Start #############################################
        #setting for zone 1 oar
        sternum_co = coord_oar_arr[:, 0, :][np.where(coord_oar_arr[:, 0, :] != -1)[0], ...]
        scapula_left_co = coord_oar_arr[:, 1, :][np.where(coord_oar_arr[:, 1, :] != -1)[0], ...]
        scapula_right_co = coord_oar_arr[:, 2, :][np.where(coord_oar_arr[:, 2, :] != -1)[0], ...]
        heart_co = coord_oar_arr[:, 3, :][np.where(coord_oar_arr[:, 3, :] != -1)[0], ...]
        # Z min_max setting

        scapula_left_Z = np.where(coord_oar_arr[:, name_dict['scapula_left'], :] != [-1, -1, -1, -1])
        scapula_right_Z = np.where(coord_oar_arr[:, name_dict['scapula_right'], :] != [-1, -1, -1, -1])
        
        scapula_left_superior = np.percentile(scapula_left_Z[0], 95, method="nearest")
        scapula_right_superior = np.percentile(scapula_right_Z[0], 95, method="nearest")
        zone1_Z_max = int(max(scapula_left_superior, scapula_right_superior))
        
        heart_Z = np.where(coord_oar_arr[:, name_dict["heart"], :] != [-1, -1, -1, -1])
        zone1_Z_min = int(np.percentile(heart_Z[0], 10, method="nearest"))
        
        # Y min_max setting
        scapula_left_posterior = np.percentile(scapula_left_co[:, 3], 99, method="nearest")
        scapula_right_posterior = np.percentile(scapula_right_co[:, 3], 99, method="nearest")
        zone1_Y_max =  int(max(scapula_left_posterior, scapula_right_posterior))
        
        
        sternum_anterior = np.percentile(sternum_co[:, 1], 1, method="nearest")
        zone1_Y_min = int(sternum_anterior)

        # X min_max setting
        scapula_left_left = np.percentile(scapula_left_co[:, 2], 99, method="nearest")
        zone1_X_max = int(scapula_left_left)
        scapula_right_right = np.percentile(scapula_right_co[:, 0], 1, method="nearest")
        zone1_X_min = int(scapula_right_right)
        ######################################Zone 1 End #############################################

        ######################################Zone 2 Start ###########################################
        sternum_co = coord_oar_arr[:, 0, :][np.where(coord_oar_arr[:, 0, :] != -1)[0], ...]
        scapula_left_co = coord_oar_arr[:, 1, :][np.where(coord_oar_arr[:, 1, :] != -1)[0], ...]
        scapula_right_co = coord_oar_arr[:, 2, :][np.where(coord_oar_arr[:, 2, :] != -1)[0], ...]
        heart_co = coord_oar_arr[:, 3, :][np.where(coord_oar_arr[:, 3, :] != -1)[0], ...]
        kidney_left_co = coord_oar_arr[:, 4, :][np.where(coord_oar_arr[:, 4, :] != -1)[0], ...]
        kidney_right_co = coord_oar_arr[:, 5, :][np.where(coord_oar_arr[:, 5, :] != -1)[0], ...]

        heart_Z = np.where(coord_oar_arr[:, name_dict["heart"], :] != [-1, -1, -1, -1])
        zone2_Z_max = int(np.percentile(heart_Z[0], 99, method="nearest"))

        hip_left_Z = np.where(coord_oar_arr[:, name_dict['hip_left'], :] != [-1, -1, -1, -1])
        hip_right_Z = np.where(coord_oar_arr[:, name_dict['hip_right'], :] != [-1, -1, -1, -1])
        hip_left_superior = np.percentile(hip_left_Z[0], 90, method="nearest")
        hip_right_superior = np.percentile(hip_right_Z[0], 90, method="nearest")
        
        zone2_Z_min = int(max(hip_left_superior, hip_right_superior))
        
        zone2_Y_max = zone1_Y_max
        zone2_Y_min = zone1_Y_min
        kidney_left_left = np.percentile(kidney_left_co[:, 2], 99, method="nearest")
        zone2_X_max = int(kidney_left_left)
        kidney_right_right = np.percentile(kidney_right_co[:, 0], 1, method="nearest")
        zone2_X_min = int(kidney_right_right)
        ######################################Zone 2 End #############################################

        ######################################Zone 3 Start ###########################################
        hip_left_co = coord_oar_arr[:, 6, :][np.where(coord_oar_arr[:, 6, :] != -1)[0], ...]
        hip_right_co = coord_oar_arr[:, 7, :][np.where(coord_oar_arr[:, 7, :] != -1)[0], ...]

        kidney_left_Z = np.where(coord_oar_arr[:, name_dict['kidney_left'], :] != [-1, -1, -1, -1])
        kidney_right_Z = np.where(coord_oar_arr[:, name_dict['kidney_right'], :] != [-1, -1, -1, -1])
        kidney_left_superior = np.percentile(kidney_left_Z[0], 95, method="nearest")
        kidney_right_superior = np.percentile(kidney_right_Z[0], 95, method="nearest")
        zone3_Z_max = int(min(kidney_left_superior, kidney_right_superior))
        
        hip_left_Z = np.where(coord_oar_arr[:, name_dict['hip_left'], :] != [-1, -1, -1, -1])
        hip_right_Z = np.where(coord_oar_arr[:, name_dict['hip_right'], :] != [-1, -1, -1, -1])
        hip_left_inferior = np.percentile(hip_left_Z[0], 1, method="nearest")
        hip_right_inferior = np.percentile(hip_right_Z[0], 1, method="nearest")
        zone3_Z_min = int(min(hip_left_inferior, hip_right_inferior))
        zone3_Y_max = zone1_Y_max 
        zone3_Y_min = zone1_Y_min
        
        hip_left_left = np.percentile(hip_left_co[:, 2], 99, method="nearest")
        zone3_X_max = int(hip_left_left)
        hip_right_right = np.percentile(hip_right_co[:, 0], 1, method="nearest")
        zone3_X_min = int(hip_right_right)
        ######################################Zone 3 End #############################################
        # im_itk = sitk.ReadImage(os.path.join(nii_basepath, nii_dirs_list[nii_idx]))
        la_itk = sitk.ReadImage(os.path.join(label_basepath, label_dirs_list[nii_idx]))
        la_itk = resample(la_itk, la_itk.GetSpacing(), (512, 512, la_itk.GetSize()[2]), 0, is_label=True)

        zone1_im_arr = sitk.GetArrayFromImage(im_itk)[zone1_Z_min:zone1_Z_max, zone1_Y_min:zone1_Y_max, zone1_X_min:zone1_X_max]
        zone1_la_arr = sitk.GetArrayFromImage(la_itk)[zone1_Z_min:zone1_Z_max, zone1_Y_min:zone1_Y_max, zone1_X_min:zone1_X_max]
        zone1_la_arr[zone1_la_arr >= 9] = 0
        # print(zone2_Z_max, zone2_Z_min, zone2_Y_max, zone2_Y_min, zone2_X_max, zone2_X_min)
        zone2_im_arr = sitk.GetArrayFromImage(im_itk)[zone2_Z_min:zone2_Z_max, zone2_Y_min:zone2_Y_max, zone2_X_min:zone2_X_max]
        zone2_la_arr = sitk.GetArrayFromImage(la_itk)[zone2_Z_min:zone2_Z_max, zone2_Y_min:zone2_Y_max, zone2_X_min:zone2_X_max].astype("int32")
        zone2_la_arr[zone2_la_arr < 9] = 0
        zone2_la_arr[zone2_la_arr >= 17] = 0
        zone2_la_arr -= 8
        zone2_la_arr[zone2_la_arr < 0] = 0

        zone3_im_arr = sitk.GetArrayFromImage(im_itk)[zone3_Z_min:zone3_Z_max, zone3_Y_min:zone3_Y_max, zone3_X_min:zone3_X_max]
        zone3_la_arr = sitk.GetArrayFromImage(la_itk)[zone3_Z_min:zone3_Z_max, zone3_Y_min:zone3_Y_max, zone3_X_min:zone3_X_max].astype("int32")
        
        zone3_la_arr[zone3_la_arr < 17] = 0
        zone3_la_arr -= 16
        zone3_la_arr[zone3_la_arr < 0] = 0

        zone1_im_itk = sitk.GetImageFromArray(zone1_im_arr)
        zone1_la_itk = sitk.GetImageFromArray(zone1_la_arr.astype("uint8"))
        zone1_la_itk.SetOrigin(zone1_im_itk.GetOrigin())
        zone2_im_itk = sitk.GetImageFromArray(zone2_im_arr)
        zone2_la_itk = sitk.GetImageFromArray(zone2_la_arr.astype("uint8"))
        zone2_la_itk.SetOrigin(zone2_im_itk.GetOrigin())
        zone3_im_itk = sitk.GetImageFromArray(zone3_im_arr)
        zone3_la_itk = sitk.GetImageFromArray(zone3_la_arr.astype("uint8"))
        zone3_la_itk.SetOrigin(zone3_im_itk.GetOrigin())
        sitk.WriteImage(zone1_im_itk, os.path.join(r"D:\! project\2024- AortaSeg\data\new_final\zone1\image", nii_dirs_list[nii_idx][:-4] + ".nii.gz"))
        sitk.WriteImage(zone1_la_itk, os.path.join(r"D:\! project\2024- AortaSeg\data\new_final\zone1\label", label_dirs_list[nii_idx][:-4] + ".nii.gz"))
        sitk.WriteImage(zone2_im_itk, os.path.join(r"D:\! project\2024- AortaSeg\data\new_final\zone2\image", nii_dirs_list[nii_idx][:-4] + ".nii.gz"))
        sitk.WriteImage(zone2_la_itk, os.path.join(r"D:\! project\2024- AortaSeg\data\new_final\zone2\label", label_dirs_list[nii_idx][:-4] + ".nii.gz"))
        sitk.WriteImage(zone3_im_itk, os.path.join(r"D:\! project\2024- AortaSeg\data\new_final\zone3\image", nii_dirs_list[nii_idx][:-4] + ".nii.gz"))
        sitk.WriteImage(zone3_la_itk, os.path.join(r"D:\! project\2024- AortaSeg\data\new_final\zone3\label", label_dirs_list[nii_idx][:-4] + ".nii.gz"))





        # win_min = -500
        # win_max = 1000
        # new_im_arr[new_im_arr < win_min] = win_min
        # new_im_arr[new_im_arr > win_max] = win_max 
        # new_im_arr = np.uint8(255 * (new_im_arr.astype(np.double) - win_min)/(win_max - win_min))
        # from PIL import Image 

        # for ii, arr in enumerate(new_im_arr):
        #     img = Image.fromarray(arr)
        #     img.save(os.path.join(r"D:\! project\2024- AortaSeg\data\validation_data\fold_0_cut_debug", "P%03d_%03d.png" %(nii_idx, ii)))
        

name_dict = {"sternum":0, 
             "scapula_left":1, 
             "scapula_right":2, 
             "heart":3, 
             "kidney_left":4, 
             "kidney_right":5, 
             "hip_left":6, 
             "hip_right":7}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--coord_path', type=str, default=None, help="coord save path")
    parser.add_argument('--nii_savepath', type=str, default=None, help="cropped_val_nii_savepath")
    parser.add_argument('--nii_source_path',  type=str, default=None, help="cropped_val_nii_savepath")
    parser.add_argument('--nii_label_source_path',  type=str, default=None, help="cropped_val_nii_label_savepath")
    opt = parser.parse_args()
    
    #check_requirements(exclude=('pycocotools', 'thop'))
    opt.weights = r"D:\! project\2024- AortaSeg\ckpt\0028_best_model_normalorgan_noflip.pt"
    opt.conf_thres = 0.25
    opt.img_size = 512
    opt.source = r"D:\! project\2024- AortaSeg\data\training\images"
    opt.coord_path = r"D:\! project\2024- AortaSeg\results\coord_path"
    opt.nii_source_path = r"D:\! project\2024- AortaSeg\data\training\images"
    opt.nii_label_source_path = r"D:\! project\2024- AortaSeg\data\training\masks"
    print(opt)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
