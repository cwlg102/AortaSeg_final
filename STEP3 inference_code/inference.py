import os 
import time
import SimpleITK as sitk 
import numpy as np 
import torch
from inference_detect import detect
from monai.networks.nets import (SwinUNETR, UNETR, UNet, DynUNet, SegResNet)
from monai.inferers import sliding_window_inference
def Make512Array(im_arr, default_value):
    im_ori_arr= np.copy(im_arr)
    change_state_dict = {"x_pad" : 0, "x_crop" : 0, "y_pad" : 0, "y_crop" : 0}
    meta_state_dict = {"x_pad" : None, "x_crop" : None, "y_pad" : None, "y_crop" : None}

    # X axis determine 
    
    if im_ori_arr.shape[2] < 512:# 이미지가 512보다 작으면 padding mode로
        x_pad_value_1 = (512 - im_ori_arr.shape[2])//2
        x_pad_value_2 = 512 - im_ori_arr.shape[2] - x_pad_value_1
        im_arr = np.pad(im_arr, ((0, 0), (0, 0), (x_pad_value_1, x_pad_value_2)), mode="constant", constant_values=default_value)
        change_state_dict["x_pad"] = 1
        meta_state_dict["x_pad"] = (x_pad_value_1, x_pad_value_2)
    elif im_ori_arr.shape[2] > 512: #  이미지가 512보다 크면 cropping mode로
        x_crop_value_1 = (im_ori_arr.shape[2] - 512)//2
        x_crop_value_2 = im_ori_arr.shape[2] - 512 - x_crop_value_1
        im_arr = im_arr[:, :, x_crop_value_1:(im_ori_arr.shape[2] - x_crop_value_2)]
        change_state_dict["x_crop"] = 1
        meta_state_dict["x_crop"] = (x_crop_value_1, x_crop_value_2)
    # Y axis determine
    
    if im_ori_arr.shape[1] < 512: #  이미지가 512보다 작으면 padding mode로
        y_pad_value_1 = (512 - im_ori_arr.shape[1])//2
        y_pad_value_2 = 512 - im_ori_arr.shape[1] - y_pad_value_1
        im_arr = np.pad(im_arr, ((0, 0), (y_pad_value_1, y_pad_value_2), (0, 0)), mode="constant", constant_values=default_value)
        change_state_dict["y_pad"] = 1
        meta_state_dict["y_pad"] = (y_pad_value_1, y_pad_value_2)
    elif im_ori_arr.shape[1] > 512: #  이미지가 512보다 크면 cropping mode로
        y_crop_value_1 = (im_ori_arr.shape[1] - 512)//2
        y_crop_value_2 = im_ori_arr.shape[1] - 512 - y_crop_value_1
        im_arr = im_arr[:, y_crop_value_1:(im_ori_arr.shape[1] - y_crop_value_2), :]
        change_state_dict["y_crop"] = 1
        meta_state_dict["y_crop"] = (y_crop_value_1, y_crop_value_2)
        
    ##### example restoring #####
    # if change_state_dict["y_pad"]: # pad면 crop해야하고
    #     pad_val_1, pad_val_2 = meta_state_dict["y_pad"]
    #     im_restore_arr = im_arr[:, pad_val_1: 512-pad_val_2, :]
    # if change_state_dict["y_crop"]: # crop이면 pad해줘야
    #     crop_val_1, crop_val_2 = meta_state_dict["y_crop"]
    #     im_restore_arr = np.pad(im_arr, ((0, 0), (crop_val_1, crop_val_2), (0, 0)), mode="constant", constant_values=-1024)
    return im_arr, change_state_dict, meta_state_dict
def get_kernels_strides(patch_size, spacing):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    sizes, spacings = patch_size, spacing
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

print("working...!")
# ROOT_DIR= r"/input/images/ct-angiography/"
MODEL_PATH_FOLD_0 = r"/opt/app/model_fold_0/"
MODEL_PATH_FOLD_1 = r"/opt/app/model_fold_1/"
MODEL_PATH_FOLD_2 = r"/opt/app/model_fold_2/"
MODEL_PATH_FOLD_3 = r"/opt/app/model_fold_3/"
MODEL_PATH_FOLD_4 = r"/opt/app/model_fold_4/"
RESULT_DIR = r"/output/images/aortic-branches/"
os.makedirs(RESULT_DIR, exist_ok=True)
image_basepath = r"/input/images/ct-angiography/"
image_dirs_list = os.listdir(image_basepath)
model1_fold0_path = os.path.join(MODEL_PATH_FOLD_0, "zone1_fold0.pth")
model2_fold0_path = os.path.join(MODEL_PATH_FOLD_0, "zone2_fold0.pth")
model3_fold0_path = os.path.join(MODEL_PATH_FOLD_0, "zone3_fold0.pth")
model_fold0_path_list = [model1_fold0_path, model2_fold0_path, model3_fold0_path]

model1_fold1_path = os.path.join(MODEL_PATH_FOLD_1, "zone1_fold1.pth")
model2_fold1_path = os.path.join(MODEL_PATH_FOLD_1, "zone2_fold1.pth")
model3_fold1_path = os.path.join(MODEL_PATH_FOLD_1, "zone3_fold1.pth")
model_fold1_path_list = [model1_fold1_path, model2_fold1_path, model3_fold1_path]

model1_fold2_path = os.path.join(MODEL_PATH_FOLD_2, "zone1_fold2.pth")
model2_fold2_path = os.path.join(MODEL_PATH_FOLD_2, "zone2_fold2.pth")
model3_fold2_path = os.path.join(MODEL_PATH_FOLD_2, "zone3_fold2.pth")
model_fold2_path_list = [model1_fold2_path, model2_fold2_path, model3_fold2_path]

model1_fold3_path = os.path.join(MODEL_PATH_FOLD_3, "zone1_fold3.pth")
model2_fold3_path = os.path.join(MODEL_PATH_FOLD_3, "zone2_fold3.pth")
model3_fold3_path = os.path.join(MODEL_PATH_FOLD_3, "zone3_fold3.pth")
model_fold3_path_list = [model1_fold3_path, model2_fold3_path, model3_fold3_path]

model1_fold4_path = os.path.join(MODEL_PATH_FOLD_4, "zone1_fold4.pth")
model2_fold4_path = os.path.join(MODEL_PATH_FOLD_4, "zone2_fold4.pth")
model3_fold4_path = os.path.join(MODEL_PATH_FOLD_4, "zone3_fold4.pth")
model_fold4_path_list = [model1_fold4_path, model2_fold4_path, model3_fold4_path]

od_model_path = os.path.join(MODEL_PATH_FOLD_0, "0028_best_model_normalorgan_noflip.pt")
spatial_size_xyz_zone1 = (128, 128, 112)
spatial_size_xyz_zone2 = (128, 128, 112)
spatial_size_xyz_zone3 = (128, 112, 160)
patch_size_list = [spatial_size_xyz_zone1, spatial_size_xyz_zone2, spatial_size_xyz_zone3]
spacing = [1.0, 1.0, 1.0]
ks1, st1 = get_kernels_strides(spatial_size_xyz_zone1, spacing)
uks1 = st1[1:]
ks2, st2 = get_kernels_strides(spatial_size_xyz_zone2, spacing)
uks2 = st2[1:]
ks3, st3 = get_kernels_strides(spatial_size_xyz_zone3, spacing)
uks3 = st3[1:]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1_fold0 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks1,
        strides=st1,
        upsample_kernel_size=uks1,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model2_fold0 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks2,
        strides=st2,
        upsample_kernel_size=uks2,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model3_fold0 = DynUNet(
        spatial_dims=3,
        in_channels=5,
        out_channels=9,
        kernel_size=ks3,
        strides=st3,
        upsample_kernel_size=uks3,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)

model1_fold1 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks1,
        strides=st1,
        upsample_kernel_size=uks1,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model2_fold1 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks2,
        strides=st2,
        upsample_kernel_size=uks2,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model3_fold1 = DynUNet(
        spatial_dims=3,
        in_channels=5,
        out_channels=9,
        kernel_size=ks3,
        strides=st3,
        upsample_kernel_size=uks3,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)

model1_fold2 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks1,
        strides=st1,
        upsample_kernel_size=uks1,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model2_fold2 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks2,
        strides=st2,
        upsample_kernel_size=uks2,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model3_fold2 = DynUNet(
        spatial_dims=3,
        in_channels=5,
        out_channels=9,
        kernel_size=ks3,
        strides=st3,
        upsample_kernel_size=uks3,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)

model1_fold3 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks1,
        strides=st1,
        upsample_kernel_size=uks1,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model2_fold3 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks2,
        strides=st2,
        upsample_kernel_size=uks2,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model3_fold3 = DynUNet(
        spatial_dims=3,
        in_channels=5,
        out_channels=9,
        kernel_size=ks3,
        strides=st3,
        upsample_kernel_size=uks3,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)

model1_fold4 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks1,
        strides=st1,
        upsample_kernel_size=uks1,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model2_fold4 = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=9,
        kernel_size=ks2,
        strides=st2,
        upsample_kernel_size=uks2,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)
model3_fold4 = DynUNet(
        spatial_dims=3,
        in_channels=5,
        out_channels=9,
        kernel_size=ks3,
        strides=st3,
        upsample_kernel_size=uks3,
        dropout=0.1,
        act_name= "LEAKYRELU",
        deep_supervision=False
    ).to(device)

model_fold0_list = [model1_fold0, model2_fold0, model3_fold0]
for i in range(len(model_fold0_path_list)):
    model_fold0_list[i].load_state_dict(torch.load(model_fold0_path_list[i])["model_state_dict"])
    model_fold0_list[i].eval()
model_fold1_list = [model1_fold1, model2_fold1, model3_fold1]
for i in range(len(model_fold1_path_list)):
    model_fold1_list[i].load_state_dict(torch.load(model_fold1_path_list[i])["model_state_dict"])
    model_fold1_list[i].eval()
model_fold2_list = [model1_fold2, model2_fold2, model3_fold2]
for i in range(len(model_fold2_path_list)):
    model_fold2_list[i].load_state_dict(torch.load(model_fold2_path_list[i])["model_state_dict"])
    model_fold2_list[i].eval()
model_fold3_list = [model1_fold3, model2_fold3, model3_fold3]
for i in range(len(model_fold3_path_list)):
    model_fold3_list[i].load_state_dict(torch.load(model_fold3_path_list[i])["model_state_dict"])
    model_fold3_list[i].eval()
model_fold4_list = [model1_fold4, model2_fold4, model3_fold4]
for i in range(len(model_fold4_path_list)):
    model_fold4_list[i].load_state_dict(torch.load(model_fold4_path_list[i])["model_state_dict"])
    model_fold4_list[i].eval()
# model_fold0_list = [model1_fold0, model2_fold0, model3_fold0]

for idx in range(len(image_dirs_list)):
    # uuid = None
    # mode = 0
    # if ".mha" in image_dirs_list[idx]:
    #     uuid = image_dirs_list[idx].split(".mha")[0]
    #     mode = 0
    # elif ".tif" in image_dirs_list[idx]:
    #     uuid = image_dirs_list[idx].split(".tif")[0]
    #     mode = 1
    im_itk = sitk.ReadImage(os.path.join(image_basepath, image_dirs_list[idx]))
    print(im_itk.GetDirection())
    print(im_itk.GetSize())
    print(im_itk.GetSpacing())
    x_filp_flag = 0
    y_flip_flag = 0
    z_flip_flag = 0
    if int(im_itk.GetDirection()[0])  == -1:
        print("Xflip")
        im_itk = sitk.Flip(im_itk, [True, False, False])
        x_filp_flag = 1
    if int(im_itk.GetDirection()[4])  == -1:
        print("Yflip")
        im_itk = sitk.Flip(im_itk, [ False,True, False])
        y_flip_flag = 1
    if int(im_itk.GetDirection()[8])  == -1:
        print("Zflip")
        im_itk = sitk.Flip(im_itk, [ False, False,True])
        z_flip_flag = 1
    im_arr = sitk.GetArrayFromImage(im_itk)
    im_mod_arr, change_state_dict, meta_state_dict = Make512Array(im_arr, -1024)
    im_arr_list, od_cut_meta_list = detect(im_mod_arr, od_model_path)
    prediction_list = []
    iteration_oar_list = [9, 9, 8]
    i_list = [0, 1, 3]
    with torch.no_grad():
        for ii, val_inputs, model_fold0, model_fold1, model_fold2, model_fold3, model_fold4, patchsize, iternum in zip(
                                                                                i_list,
                                                                                im_arr_list, 
                                                                                model_fold0_list, 
                                                                                model_fold1_list, 
                                                                                model_fold2_list, 
                                                                                model_fold3_list,
                                                                                model_fold4_list,
                                                                                patch_size_list,
                                                                                iteration_oar_list):
            start = time.time()
            val_inputs = np.transpose(val_inputs, (2, 1, 0))
            val_inputs = val_inputs[np.newaxis, np.newaxis, ...].astype("float32")
            val_inputs = torch.tensor(val_inputs).to(device)
            
            total_min = -1024
            total_max = 3071
            bone_min = -1000
            bone_max = 2000
            soft_min = -160
            soft_max = 350
            brain_min = -5
            brain_max = 65
            stroke_min = 15
            stroke_max = 45
            tx =  torch.reshape(val_inputs[:, 0, :, :, :].clone().detach(), (val_inputs.shape[0], 1, val_inputs.shape[2], val_inputs.shape[3], val_inputs.shape[4]))
            x1 =  torch.reshape(val_inputs[:, 0, :, :, :].clone().detach(), (val_inputs.shape[0], 1, val_inputs.shape[2], val_inputs.shape[3], val_inputs.shape[4]))
            x2 =  torch.reshape(val_inputs[:, 0, :, :, :].clone().detach(), (val_inputs.shape[0], 1, val_inputs.shape[2], val_inputs.shape[3], val_inputs.shape[4]))
            x3 =  torch.reshape(val_inputs[:, 0, :, :, :].clone().detach(), (val_inputs.shape[0], 1, val_inputs.shape[2], val_inputs.shape[3], val_inputs.shape[4]))
            x4 =  torch.reshape(val_inputs[:, 0, :, :, :].clone().detach(), (val_inputs.shape[0], 1, val_inputs.shape[2], val_inputs.shape[3], val_inputs.shape[4]))
            tx[tx<total_min] = total_min
            tx[tx>total_max] = total_max
            tx = (tx-total_min)/(total_max-total_min)
            x1[x1<bone_min] = bone_min
            x1[x1>bone_max] = bone_max
            x1 = (x1-bone_min)/(bone_max-bone_min)
            x2[x2<soft_min] = soft_min
            x2[x2>soft_max] = soft_max
            x2 = (x2-soft_min)/(soft_max-soft_min)
            x3[x3<brain_min] = brain_min
            x3[x3>brain_max] = brain_max 
            x3 = (x3-brain_min)/(brain_max-brain_min)
            x4[x4<stroke_min] = stroke_min
            x4[x4>stroke_max] = stroke_max
            x4 = (x4 - stroke_min)/(stroke_max - stroke_min)
            if ii == 0:
                tx *= 2 
                tx -= 1
                x2 *= 2 
                x2 -= 1
                val_inputs = torch.cat((tx, x2), 1)
            elif ii == 1:
                val_inputs = torch.cat((tx, x2), 1)
            else:
                tx *= 2 
                tx -= 1
                x2 *= 2
                x2 -= 1
                val_inputs = torch.cat((tx, x1, x2, x3, x4), 1)

            with torch.cuda.amp.autocast():
                val_outputs_fold0 = sliding_window_inference(val_inputs, patchsize, 4, model_fold0, overlap=0.56)
            last_outputs_fold0 = torch.argmax(val_outputs_fold0, dim=1).detach().cpu()[0].numpy()
            last_outputs_fold0 = np.transpose(last_outputs_fold0, (2, 1, 0))
            del val_outputs_fold0

            with torch.cuda.amp.autocast():
                val_outputs_fold1 = sliding_window_inference(val_inputs, patchsize, 4, model_fold1, overlap=0.56)
            last_outputs_fold1 = torch.argmax(val_outputs_fold1, dim=1).detach().cpu()[0].numpy()
            last_outputs_fold1 = np.transpose(last_outputs_fold1, (2, 1, 0))
            del val_outputs_fold1
            
            with torch.cuda.amp.autocast():
                val_outputs_fold2 = sliding_window_inference(val_inputs, patchsize, 4, model_fold2, overlap=0.56)
            last_outputs_fold2 = torch.argmax(val_outputs_fold2, dim=1).detach().cpu()[0].numpy()
            last_outputs_fold2 = np.transpose(last_outputs_fold2, (2, 1, 0))
            del val_outputs_fold2

            with torch.cuda.amp.autocast():
                val_outputs_fold3 = sliding_window_inference(val_inputs, patchsize, 4, model_fold3, overlap=0.56)
            last_outputs_fold3 = torch.argmax(val_outputs_fold3, dim=1).detach().cpu()[0].numpy()
            last_outputs_fold3 = np.transpose(last_outputs_fold3, (2, 1, 0))
            del val_outputs_fold3

            with torch.cuda.amp.autocast():
                val_outputs_fold4 = sliding_window_inference(val_inputs, patchsize, 4, model_fold4, overlap=0.56)
            last_outputs_fold4 = torch.argmax(val_outputs_fold4, dim=1).detach().cpu()[0].numpy()
            last_outputs_fold4 = np.transpose(last_outputs_fold4, (2, 1, 0))
            del val_outputs_fold4

            last_outputs = np.zeros_like(last_outputs_fold0).astype("uint8")
            for oar in range(1, iternum):
                ensemble_arr = np.zeros_like(last_outputs_fold0).astype("uint8")
                ensemble_arr += np.where(last_outputs_fold0 == oar, 1, 0).astype("uint8")
                ensemble_arr += np.where(last_outputs_fold1 == oar, 1, 0).astype("uint8")
                ensemble_arr += np.where(last_outputs_fold2 == oar, 1, 0).astype("uint8")
                ensemble_arr += np.where(last_outputs_fold3 == oar, 1, 0).astype("uint8")
                ensemble_arr += np.where(last_outputs_fold4 == oar, 1, 0).astype("uint8")
                last_outputs = np.where(ensemble_arr >= 3, oar, last_outputs).astype("uint8")
            prediction_list.append(np.copy(last_outputs))
            print(time.time() - start)
    pred_restore_list = []
    
    for pred_arr, od_cut_meta in zip(prediction_list, od_cut_meta_list):
        zmin, zmax, ymin, ymax, xmin, xmax = od_cut_meta
        # print(im_mod_arr.shape, od_cut_meta)
        pred_temp_arr = np.pad(pred_arr.astype("uint8"), ((zmin, im_mod_arr.shape[0]-zmax), (ymin, im_mod_arr.shape[1]-ymax), (xmin, im_mod_arr.shape[2]-xmax)), mode="constant", constant_values=0).astype("uint8")
        # print(change_state_dict, meta_state_dict)
        for key, val in change_state_dict.items():
            if val == 1 and key == "x_pad":
                val_1, val_2 = meta_state_dict["x_pad"]
                pred_temp_arr = pred_temp_arr[:, :, val_1: 512-val_2].astype("uint8")
            if val == 1 and key == "x_crop":
                val_1, val_2 = meta_state_dict["x_crop"]
                pred_temp_arr = np.pad(pred_temp_arr, ((0, 0), (0, 0), (val_1, val_2)), mode="constant", constant_values=0).astype("uint8")
            if val == 1 and key == "y_pad":
                val_1, val_2 = meta_state_dict["y_pad"]
                pred_temp_arr = pred_temp_arr[:, val_1: 512-val_2, :].astype("uint8")
            if val == 1 and key == "y_crop":
                val_1, val_2 = meta_state_dict["y_crop"]
                pred_temp_arr = np.pad(pred_temp_arr, ((0, 0), (val_1, val_2),(0, 0)), mode="constant", constant_values=0).astype("uint8")
            
        # print(im_arr.shape, pred_temp_arr.shape)
        pred_restore_list.append(np.copy(pred_temp_arr.astype("uint8")))

    final_pred_arr = np.zeros_like(im_arr).astype("uint8")
    
    session_1 = [i+1 for i in range(8)]
    session_2 = [i+9 for i in range(8)]
    session_3 = [i+17 for i in range(7)]
    group_list = [session_1, session_2, session_3]
    for idx_list, pred_arr in zip(group_list, pred_restore_list):
        for pre_oarnum, num in enumerate(idx_list):
            final_pred_arr = np.where(pred_arr == pre_oarnum+1, num, final_pred_arr).astype("uint8")
    # print(np.max(pred_restore_list[0]), np.min(pred_restore_list[0]))

    final_itk = sitk.GetImageFromArray(final_pred_arr)
    final_itk.SetSpacing(im_itk.GetSpacing())
    final_itk.SetOrigin(im_itk.GetOrigin())
    final_itk.SetDirection(im_itk.GetDirection())
    if x_filp_flag:
        final_itk = sitk.Flip(final_itk, [True, False, False])
    if y_flip_flag:
        final_itk = sitk.Flip(final_itk, [False, True, False])
    if z_flip_flag:
        final_itk = sitk.Flip(final_itk, [False, False, True])
    
    final_itk= sitk.Cast(final_itk, sitk.sitkUInt8)
    sitk.WriteImage(final_itk, os.path.join(RESULT_DIR, image_dirs_list[idx]), useCompression=True)
    
    