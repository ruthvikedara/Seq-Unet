import torch
import argparse
from tqdm import tqdm
from model import UNet
from dataloader import get_dataloaders
from metrics import calculate_segmentation_metrics
from torchvision.transforms import v2

def transform_blur(image):  
  return v2.GaussianBlur(31, sigma=15)(image)

def inference(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet(in_channels=2).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Get test dataloader
    _, _, test_dataloader = get_dataloaders(args.datapath, args.batch_size, args.block_rate)

    dice = {}

    for patient_num in range(1, 11):
        count = 0
        prior_mask = None
        prior = None
        score_list = []

        with torch.no_grad():
            for n, data in enumerate(tqdm(test_dataloader, desc=f"Patient {patient_num}")):
                if count < 300 * (patient_num - 1):
                    count += 1
                    continue
                image, mask, id, num = data
                mask = mask.to(device)
                inp_image = image[0, 0].unsqueeze(0).to(device)
                if prior_mask is not None:
                    prior_mask = transform_blur(torch.clamp(prior_mask.unsqueeze(0) + 0.1, 0, 1))
                    prior = prior_mask * inp_image
                else:
                    prior = torch.zeros_like(inp_image).to(device)
                mod_input = torch.stack((inp_image, prior), 1)
                mod_out = model(mod_input)

                precision, recall, dice_score = calculate_segmentation_metrics(mod_out, mask)
                score_list.append(dice_score)

                prior_mask = mod_out[0, 0]
                count += 1
                if count == 300 * patient_num:
                    break

        dice[patient_num] = sum(score_list) / len(score_list)

    print("Dice scores per patient:")
    for patient, score in dice.items():
        print(f"Patient {patient}: {score:.4f}")
    print(f"Average Dice score: {sum(dice.values()) / len(dice):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for U-Net model")
    parser.add_argument("--datapath", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--block_rate", type=float, default=0.01, help="Block rate")

    args = parser.parse_args()
    inference(args)
