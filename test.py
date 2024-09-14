import os
import argparse
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SRCNN
from utils import calculate_psnr

def parse_args():
    parser = argparse.ArgumentParser(description='Test SRCNN for Image Super-Resolution')
    parser.add_argument('--test_dir', type=str, required=False, help='Directory with test HR images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--scale_factor', type=int, default=2, help='Scale factor for super-resolution')
    parser.add_argument('--patch_size', type=int, default=32, help='Patch size for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for testing (cuda or cpu)')
    parser.add_argument('--image_path', type=str, default=None, help='Path to a single image to super-resolve and display')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the super-resolved image')
    parser.add_argument('--save_results_dir', type=str, default='unseen_images/results/', help='Directory to save super-resolved images')
    return parser.parse_args()

def super_resolve_image(model, image_path, scale_factor, device):
    """
    Super-resolve a single image using the trained SRCNN model.

    Args:
        model (torch.nn.Module): Trained SRCNN model.
        image_path (str): Path to the input image.
        scale_factor (int): Upscaling factor.
        device (torch.device): Device to perform computation on.

    Returns:
        sr_image (PIL.Image.Image): Super-resolved image.
        lr_image (PIL.Image.Image): Low-resolution image (input to the model).
        hr_image (PIL.Image.Image): High-resolution ground truth image (if available).
    """
    model.eval()
    with torch.no_grad():
        hr_image = Image.open(image_path).convert('RGB')
        
        # Generate low-resolution image
        lr_image = hr_image.resize(
            (hr_image.width // scale_factor, hr_image.height // scale_factor),
            Image.BICUBIC
        )
        lr_image_upsampled = lr_image.resize(
            (hr_image.width, hr_image.height),
            Image.BICUBIC
        )

        lr_tensor = ToTensor()(lr_image_upsampled).unsqueeze(0).to(device)

        sr_tensor = model(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)
        
        sr_image = sr_tensor.squeeze(0).cpu()
        sr_image = ToPILImage()(sr_image)
        
        return sr_image, lr_image_upsampled, hr_image

def process_directory(model, test_dir, scale_factor, device, save_results_dir):

    if not os.path.isdir(test_dir):
        print(f"Test directory '{test_dir}' does not exist. Please provide a valid directory.")
        return
    

    os.makedirs(save_results_dir, exist_ok=True)
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = [f for f in os.listdir(test_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in '{test_dir}'. Please check the directory and try again.")
        return
    
    total_psnr = 0
    num_images = 0
    
    for img_name in tqdm(image_files, desc='Processing Images'):
        img_path = os.path.join(test_dir, img_name)
        sr_image, lr_image, hr_image = super_resolve_image(model, img_path, scale_factor, device)

        sr_tensor = ToTensor()(sr_image).unsqueeze(0).to(device)
        hr_tensor = ToTensor()(hr_image).unsqueeze(0).to(device)
        psnr = calculate_psnr(sr_tensor, hr_tensor)
        total_psnr += psnr
        num_images += 1
        
        sr_save_path = os.path.join(save_results_dir, f'sr_{img_name}')
        sr_image.save(sr_save_path)
        
        lr_save_path = os.path.join(save_results_dir, f'lr_{img_name}')
        lr_image.save(lr_save_path)
    
    avg_psnr = total_psnr / num_images
    print(f'Average PSNR on Test Set: {avg_psnr:.2f} dB')
    print(f'Super-resolved images saved in: {save_results_dir}')

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = SRCNN(num_channels=3).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    if args.image_path:
        if not os.path.isfile(args.image_path):
            print(f"Image path '{args.image_path}' does not exist. Please provide a valid image path.")
            return
        
        sr_image, lr_image, hr_image = super_resolve_image(model, args.image_path, args.scale_factor, device)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(lr_image)
        plt.title('Low Resolution (Input)')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(sr_image)
        plt.title('Super Resolution (Output)')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(hr_image)
        plt.title('High Resolution (Ground Truth)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
     
        if args.save_path:
            sr_image.save(args.save_path)
            print(f'Super-resolved image saved at {args.save_path}')
    

    elif args.test_dir:
        process_directory(model, args.test_dir, args.scale_factor, device, args.save_results_dir)
    
    else:
        print("No action specified. Please provide either --image_path for a single image or --test_dir for a directory of images.")
        print("Use -h or --help for more information.")

if __name__ == '__main__':
    main()
