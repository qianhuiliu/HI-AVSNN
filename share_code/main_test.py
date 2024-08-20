import argparse
import torch
import torch.optim
import os
import torch.nn.parallel
from src.data.dvs_audio_lip import MultiDataset
from src.models.visual_front import Spiking_Visual_front
from src.models.fuse_cross import Fuse
from src.data.utils import *
from src.functions import seed_all, get_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_root', default="/home/qianhui/lip-reading/share_audio-Lip")
    parser.add_argument("--checkpoint_dir", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--seq_len', type=int, default=28)
    parser.add_argument('--num_bins', type=str2list, default='1+4')
    parser.add_argument('--nb_hiddens', type=int, default=256)
    args = parser.parse_args()
    return args

@torch.no_grad()
def test(v_front, fuse, test_loader, device):
    v_front.eval()
    fuse.eval()
    total = 0
    correct = 0
    for i, batch in enumerate(test_loader):
        a_in, v_in, labels, part = batch
        
        v_feat = v_front(v_in.to(device))  
        outputs = fuse(v_feat, a_in.to(device))
        mean_out = outputs.mean(1)
        labels = labels.to(device)
        
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())

    final_acc = 100 * correct / total
    return final_acc

if __name__ == "__main__":
    args = parse_args()
    seed_all(args.seed)

    #load data
    test_data = MultiDataset('test', args)
    test_loader = dataset2dataloader(test_data, args.batch_size, args.workers, shuffle=False)     

    #define model
    v_front = Spiking_Visual_front(in_channels=1)  
    fuse = Fuse()
    
    for param in v_front.parameters():
        param.requires_grad = False

    for param in fuse.parameters():
       param.requires_grad = False

    #load weight
    visual_state_dict = torch.load('./data/ours.ckpt', map_location=torch.device('cpu'))
    v_front.load_state_dict(visual_state_dict['v_front_state_dict'])
    fuse.load_state_dict(visual_state_dict['fuse_state_dict'])
    
    v_front = torch.nn.DataParallel(v_front)
    fuse = torch.nn.DataParallel(fuse)
    
    v_front.to(device)
    fuse.to(device)

    facc = test(v_front, fuse, test_loader, device)
    print (facc)

