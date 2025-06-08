import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

from model.IFNet_HDv3 import IFNet 

def load_model(model_path):
    model = IFNet().eval()
    ckpt = torch.load(model_path, map_location='cpu')
    # Xử lý nếu checkpoint có prefix 'module.'
    new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(new_ckpt, strict=True)
    model = model.cuda() if torch.cuda.is_available() else model
    return model

def pad_image(img, tmp=32, scale=1):
    h, w = img.shape[-2:]
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    img = F.pad(img, padding)
    return img, h, w

def inference(model, I0, I1, exp=2, scale=1.0):
    # exp = số lần tăng fps (exp=2: 4X, exp=3: 8X)
    with torch.no_grad():
        def recur(I0, I1, n):
            middle = model(torch.cat((I0, I1), 1), [4/scale, 2/scale, 1/scale])[2][2]
            if n == 1:
                return [middle]
            first_half = recur(I0, middle, n // 2)
            second_half = recur(middle, I1, n // 2)
            if n % 2:
                return first_half + [middle] + second_half
            else:
                return first_half + second_half
        return recur(I0, I1, 2**exp-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str, default='train_log/flownet.pkl')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--exp', type=int, default=2)
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model)
    model = model.to(device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outname = args.output or f"{os.path.splitext(args.video)[0]}_interp_hd3_{2**args.exp}X.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outname, fourcc, fps * (2 ** args.exp), (width, height))

    ret, lastframe = cap.read()
    if not ret:
        print("Cannot read video!")
        exit()

    pbar = tqdm(total=frame_count - 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Chuẩn hóa và chuyển tensor
        I0 = torch.from_numpy(lastframe).permute(2,0,1).float().unsqueeze(0) / 255.
        I1 = torch.from_numpy(frame).permute(2,0,1).float().unsqueeze(0) / 255.
        I0 = I0.to(device)
        I1 = I1.to(device)

        I0_pad, h, w = pad_image(I0, scale=args.scale)
        I1_pad, _, _ = pad_image(I1, scale=args.scale)

        mids = inference(model, I0_pad, I1_pad, exp=args.exp, scale=args.scale)
        # Ghi frame gốc + frame nội suy
        out.write(lastframe)
        for mid in mids:
            mid = (mid[0, :, :h, :w].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            out.write(mid)
        lastframe = frame
        pbar.update(1)
    out.write(lastframe)
    out.release()
    pbar.close()
    print(f"Done! Output saved: {outname}")
