import numpy as np
import torch, cv2, time
from torch.utils.data import DataLoader
from metrics import *
from test_Dataloader import M3D
from tqdm import tqdm
from M3DNet import Network


def DepthtoFused(DepthImg, imgarr):
    h, w = DepthImg.shape
    imgarr = imgarr.cpu().numpy()
    imgarr = imgarr[0, :h, :w, :, :]
    rows, cols, channels, count = imgarr.shape
    Fused = np.zeros((h, w, channels))
    for k in range(w):
        for j in range(h):
            Dep = DepthImg[j, k]
            Res = count - Dep + 1
            if Res < 1:
                Res = 1
            if Res > count:
                Res = count
            Fused[j, k, :] = imgarr[j, k, :, int(Dep-1)]
    return Fused

path = 'D:/M3D/M3D_bestmodel.pth'


model = torch.load(path)
model = torch.nn.DataParallel(model, device_ids=[0])
test_dataset = M3D()
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
num_val = len(test_dataloader)

model.eval()
with torch.no_grad():
    Avg_abs_rel=0.0
    Avg_sq_rel=0.0
    Avg_mse=0.0
    Avg_mae=0.0
    Avg_rmse=0.0
    Avg_rmse_log=0.0
    Avg_accuracy_1=0.0
    Avg_accuracy_2=0.0
    Avg_accuracy_3=0.0
    val_time = 0
    for idx, samples in enumerate(tqdm(test_dataloader, desc="test")):
        inputs, dpt, focus_dists, test_mask, img = samples

        inputs = inputs.to('cuda:0', non_blocking=True)
        focus_dists = focus_dists.to('cuda:0', non_blocking=True)
        dpt = np.squeeze(dpt.numpy())
        test_mask = np.squeeze(test_mask.data.cpu().numpy())

        start = time.time()
        _, _, _, test_est = model(inputs, focus_dists)
        val_time += (time.time() - start)
        test_est = test_est.cpu().numpy()
        test_est = np.squeeze(test_est)
        _, h, w, _, _ = img.shape
        test_est = test_est[:h, :w]
        AiF = DepthtoFused(np.around(test_est), img)

        path11 = "D:/M3D/result/"

        cv2.imwrite(path11 + "%s.bmp" % (idx + 1), np.around(test_est).astype(np.uint8))
        convertPNG(test_est).save(path11 + "%s_map.bmp" % (idx + 1))
        cv2.imwrite(path11 + "%s_AiF.jpg" % (idx + 1), AiF.astype(np.uint8))

        Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_est, dpt, test_mask)
        Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_est, dpt, test_mask)
        Avg_mse = Avg_mse + mask_mse(test_est, dpt, test_mask)
        Avg_mae = Avg_mae + mask_mae(test_est, dpt, test_mask)
        Avg_rmse = Avg_rmse + mask_rmse(test_est, dpt, test_mask)
        Avg_rmse_log = Avg_rmse_log + mask_rmse_log(test_est, dpt, test_mask)
        Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_est, dpt, 1, test_mask)
        Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_est, dpt, 2, test_mask)
        Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_est, dpt, 3, test_mask)
    print("Avg_abs_rel: ", Avg_abs_rel / num_val)
    print("Avg_sq_rel: ", Avg_sq_rel / num_val)
    print("Avg_mse: ", Avg_mse / num_val)
    print("Avg_mae: ", Avg_mae / num_val)
    print("Avg_rmse: ", Avg_rmse / num_val)
    print("Avg_rmse_log: ", Avg_rmse_log / num_val)
    print("Avg_accuracy_1: ", Avg_accuracy_1 / num_val)
    print("Avg_accuracy_2: ", Avg_accuracy_2 / num_val)
    print("Avg_accuracy_3: ", Avg_accuracy_3 / num_val)
    print("AVG_time:", val_time / num_val)