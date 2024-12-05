import argparse
import os
import torch
from model.autoencoder import *
from PIL import Image
import numpy as np
from torchvision import transforms
from model.autoencoder import *

@torch.no_grad()
def apply_nail_art_and_save(args, enc, dec, test_image):
    enc.eval()
    dec.eval()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    test_image = test_image.to(device)

    # Forward Pass
    latent, size = enc(test_image)  # Feature extraction
    output = dec(latent, size)  # Segmentation prediction

    output = output.squeeze(0).squeeze(0)  # (H, W)

    # Load nail art image
    nail_art_path = f"/Users/Han/Desktop/capstone/JaPyGuri_dummy/nail_art/preimage_{args.nail_art_id}.jpg"
    nail_art = Image.open(nail_art_path).convert("RGB")

    # Resize nail art image to match test_image size
    test_image_pil = transforms.ToPILImage()(test_image.squeeze(0).cpu())
    nail_art_resized = nail_art.resize(test_image_pil.size)

    # Convert outputs to masks
    segmentation_mask = (output > 0.5).cpu().numpy().astype(np.uint8)  # Binary mask (0 or 1)
    segmentation_mask = Image.fromarray((segmentation_mask * 255).astype(np.uint8))

     # Alpha blending for transparency
    alpha = 0.95  # Transparency level for nail art
    segmentation_mask_tensor = transforms.ToTensor()(segmentation_mask)
    test_image_array = np.array(test_image_pil).astype(np.float32)
    nail_art_array = np.array(nail_art_resized).astype(np.float32)

    # Apply transparency to nail art only on the segmented region
    blended_image_array = (
        segmentation_mask_tensor.numpy()[0, :, :, None] * (
            alpha * nail_art_array + (1 - alpha) * test_image_array
        ) + (1 - segmentation_mask_tensor.numpy()[0, :, :, None]) * test_image_array
    ).astype(np.uint8)

    # Convert array to PIL Image
    blended_image_pil = Image.fromarray(blended_image_array)

    # Save the blended image
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, "nail_art_transparent_applied.png")
    blended_image_pil.save(save_path)
    print(save_path)

def delete_all_files_in_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        

        elif os.path.isdir(file_path):
            delete_all_files_in_directory(file_path)
            os.rmdir(file_path)
            print(f"Deleted directory: {file_path}")

def load_single_image_from_directory(directory_path):
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    image_files = [
        f for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and os.path.splitext(f)[1].lower() in supported_extensions
    ]

    if len(image_files) == 0:
        raise Exception(f"디렉토리 '{directory_path}'에 이미지 파일이 없습니다.")
    elif len(image_files) > 1:
        raise Exception(f"디렉토리 '{directory_path}'에 이미지 파일이 2개 이상 존재합니다.")

    image_path = os.path.join(directory_path, image_files[0])

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)
    return image

def run(args):

    delete_all_files_in_directory(args.save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 정의 및 파라미터 로드
    enc = encoder(in_channel=3)
    path = "/Users/Han/Desktop/capstone/JaPyGuri_dummy/downstream_segmentor/noise_patch_449/checkpoints/e_191.pth"
    enc.load_state_dict(torch.load(path, map_location=torch.device(device)))

    dec = decoder(out_channel=1)
    path = "/Users/Han/Desktop/capstone/JaPyGuri_dummy/downstream_segmentor/noise_patch_449/checkpoints/d_191.pth"
    dec.load_state_dict(torch.load(path, map_location=torch.device(device)))

    # 모델을 GPU로 보내기
    enc = enc.to(device)
    dec = dec.to(device)

    test_image = load_single_image_from_directory(directory_path=args.load_path)
    apply_nail_art_and_save(args, enc, dec, test_image=test_image)
    # delete_all_files_in_directory(args.load_path)

if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # path 설정
    parser.add_argument('--save_path', default='/Users/Han/Desktop/capstone/JaPyGuri_dummy/output_image',
                         type=str, help='path to save inference image')
    parser.add_argument('--load_path', default='/Users/Han/Desktop/capstone/JaPyGuri_dummy/input_image',
                         type=str, help='path to load input image')
    parser.add_argument('--nail_art_id', default=10, type=int, help='num of desired nail art image')
    args = parser.parse_args()

    run(args)

