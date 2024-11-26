import argparse
import os
import random
import time
import torch
from model.autoencoder import *
from PIL import Image
from torchvision import transforms

@torch.no_grad()
def test_and_save(args, enc, dec,test_image):
    enc.eval()
    dec.eval()

    print(test_image.shape)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    test_image = test_image.to(device)

    # Forward Pass
    latent, size = enc(test_image)  # Feature extraction
    output = dec(latent, size)  # Segmentation prediction
    print(output.shape)

    # output tensor를 이미지로 바꾸고 args.save_path의 경로로 해당 텐서를 이미지로 저장.
    # output tensor는 항상 (1,1,H,W)의 흑백 이미지임. 이 흑백이미지를 지정된 경로에 저장
        # output의 크기는 (1, 1, H, W), 따라서 squeeze로 차원 축소 후 (H, W) 크기 2D 텐서로 만듬
    output = output.squeeze(0).squeeze(0)  # (H, W)

    # Tensor -> PIL Image로 변환
    transform = transforms.ToPILImage()
    output_image = transform(output.cpu())  # .cpu()를 호출하여 Tensor를 CPU로 옮기기

    # 경로가 존재하지 않으면 디렉토리 생성
    os.makedirs(args.save_path, exist_ok=True)

    # 이미지 저장
    save_path = os.path.join(args.save_path, "segmentation_output.png")
    output_image.save(save_path)

    
def delete_all_files_in_directory(directory_path):
    # 디렉토리가 존재하는지 확인
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    # 디렉토리 내 모든 파일과 서브디렉토리 삭제
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # 파일인 경우 삭제
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        
        # 디렉토리인 경우 삭제 (하위 디렉토리 포함)
        elif os.path.isdir(file_path):
            # 디렉토리 내의 파일을 먼저 삭제 후 디렉토리 삭제
            delete_all_files_in_directory(file_path)
            os.rmdir(file_path)  # 디렉토리 삭제
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

    image = Image.open(image_path)
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
    path = "/Users/Han/Desktop/capstone/JaPyGuri_dummy/pre_trained_parameter/f_99.pth"
    enc.load_state_dict(torch.load(path, map_location=torch.device(device)))

    dec = decoder(out_channel=1)
    path = "/Users/Han/Desktop/capstone/JaPyGuri_dummy/pre_trained_parameter/c_99.pth"
    dec.load_state_dict(torch.load(path, map_location=torch.device(device)))

    # 모델을 GPU로 보내기
    enc = enc.to(device)
    dec = dec.to(device)

    test_image = load_single_image_from_directory(directory_path=args.load_path)
    test_and_save(args, enc, dec, test_image=test_image)
    delete_all_files_in_directory(args.load_path)



if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # path 설정
    parser.add_argument('--save_path', default='/Users/Han/Desktop/capstone/JaPyGuri_dummy/output_image',
                         type=str, help='path to save inference image')
    parser.add_argument('--load_path', default='/Users/Han/Desktop/capstone/JaPyGuri_dummy/input_image',
                         type=str, help='path to load input image')
    args = parser.parse_args()

    run(args)