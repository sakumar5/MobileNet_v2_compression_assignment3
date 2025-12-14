"""----------------------------------------------------------------
Modules:
    argparse : Parses command-line arguments for training configuration.
    torch    : Core PyTorch library for tensor operations and training.
    nn       : Neural network layers and loss functions.
    optim    : Optimization algorithms for model training.
    data     : Dataset loading and data preprocessing utilities.
    models   : Model architecture definitions used in training.
    utils    : Helper functions for reproducibility, metrics, and scheduling.
    wandb    : Experiment tracking and metric logging.
----------------------------------------------------------------"""
import argparse
import os
from typing import Tuple
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from config import TrainConfig
from data import get_cifar10_dataloaders
from utils import set_seed, accuracy, save_checkpoint, cosine_lr_scheduler
from models.mobilenetv2_cifar import mobilenet_v2_cifar
from compression.quant_ops import fake_quantize_tensor
from compression.activation_wrappers import wrap_activations_in_model

try:
    import wandb
except ImportError:
    wandb = None

"""---------------------------------------------
* def name :
*       parse_args
*
* purpose:
*       Parses command-line arguments for
*       training, quantization, and logging.
*
* Input parameters:
*       None : arguments are read from command line
*
* return:
*       Parsed arguments object
---------------------------------------------"""
def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    # Data & model
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_schedule", type=str, default="cosine")  # or 'step'

    # Compression
    parser.add_argument("--weight_bits", type=int, default=32)
    parser.add_argument("--activation_bits", type=int, default=32)
    parser.add_argument("--symmetric", action="store_true", default=True)
    parser.add_argument("--per_channel", action="store_true", default=False)
    parser.add_argument("--quantize_weights_during_forward", action="store_true")
    parser.add_argument("--quantize_activations_during_train", action="store_true")

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs6886_assignment3")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")


    args = parser.parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        width_mult=args.width_mult,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        symmetric=args.symmetric,
        per_channel=args.per_channel,
        quantize_weights_during_forward=args.quantize_weights_during_forward,
        quantize_activations_during_train=args.quantize_activations_during_train,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_interval=args.log_interval,
        out_dir=args.out_dir,
        seed=args.seed,
        resume=args.resume,
    )
    return cfg

"""---------------------------------------------
* def name :
*       load_checkpoint_compat
*
* purpose:
*       Loads a saved checkpoint into the model
*       while handling compatibility between
*       different checkpoint formats.
*
* Input parameters:
*       checkpoint_path : file path of the saved checkpoint
*       model           : model into which weights are loaded
*       optimizer       : optimizer to restore state (optional)
*       device          : device used to map checkpoint tensors
*
* return:
*       Loaded checkpoint dictionary or restored state
---------------------------------------------"""
def load_checkpoint_compat(model: torch.nn.Module,
                           ckpt_path: str,
                           device: torch.device):
    """
    Load a checkpoint that may have been saved with wrappers around
    `features` / `classifier` (e.g. .module, activation stats).

    - Strips 'features.module.' -> 'features.'
    - Strips 'classifier.module.' -> 'classifier.'
    - Drops activation statistics keys like 'activation_bits_accum', 'num_samples'
    - Loads with strict=False.
    """
    print(f">>> Loading checkpoint (compat) from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    new_state_dict: Dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        # Skip activation-stat bookkeeping tensors
        if ("activation_bits_accum" in k) or ("num_samples" in k):
            continue

        new_k = k

        # Strip wrapper "module" under features / classifier
        if new_k.startswith("features.module."):
            new_k = "features." + new_k[len("features.module."):]
        if new_k.startswith("classifier.module."):
            new_k = "classifier." + new_k[len("classifier.module."):]
        
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f">>> Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0:
        print("    (missing)", missing[:5], "...")
    if len(unexpected) > 0:
        print("    (unexpected)", unexpected[:5], "...")

    return checkpoint

"""---------------------------------------------
* def name :
*       main
*
* purpose:
*       Coordinates the full training and
*       evaluation pipeline.
*
* Input parameters:
*       None : configuration is parsed internally
*
* return:
*       None
---------------------------------------------"""
def main():
    cfg = parse_args()
    print(">>> main() started with config:", cfg)  # debug
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(">>> CIFAR-10 dataloaders ready")

    model = mobilenet_v2_cifar(
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    ).to(device)
    print(">>> Model created")
    
    # # Resume from pretrained FP32 checkpoint
    # if cfg.resume is not None:
        # print(f">>> Loading checkpoint from {cfg.resume}")
        # checkpoint = torch.load(cfg.resume, map_location=device)
        # model.load_state_dict(checkpoint["state_dict"])
        # print(">>> Loaded pretrained weights")
        
    checkpoint = None
    if cfg.resume is not None:
        checkpoint = load_checkpoint_compat(model, cfg.resume, device)
        print(">>> Loaded pretrained weights (compat mode)")

    # Optionally wrap activations for fake quant
    if cfg.quantize_activations_during_train and cfg.activation_bits < 32:
        wrap_activations_in_model(
            model,
            num_bits=cfg.activation_bits,
            symmetric=cfg.symmetric,
            per_channel=cfg.per_channel,
        )
        print(">>> Activation quantization wrappers enabled")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    if cfg.use_wandb and wandb is not None:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.as_dict(),
        )
        print(">>> WandB logging enabled")

    best_acc1 = 0.0

    print(">>> Starting training loop...")
    for epoch in range(cfg.epochs):
        if cfg.lr_schedule == "cosine":
            lr = cosine_lr_scheduler(optimizer, cfg.lr, epoch, cfg.epochs)
        else:
            lr = optimizer.param_groups[0]["lr"]

        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg)
        acc1, acc5 = evaluate(model, test_loader, criterion, device)

        if cfg.use_wandb and wandb is not None:
            wandb.log({"val/acc1": acc1, "val/acc5": acc5, "epoch": epoch, "lr": lr})

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_path = os.path.join(cfg.out_dir, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "config": cfg.as_dict(),
            },
            save_path,
        )

    if cfg.use_wandb and wandb is not None:
        wandb.finish()

"""---------------------------------------------
* def name :
*       maybe_fake_quantize_weights
*
* purpose:
*       Applies fake quantization to model weights
*       based on the specified bit-width and
*       quantization configuration.
*
* Input parameters:
*       weight_tensor : input weight tensor to be quantized
*       num_bits      : number of bits used for quantization
*       enabled       : flag indicating whether quantization is applied
*
* return:
*       Quantized (or original) weight tensor
---------------------------------------------"""
def maybe_fake_quantize_weights(model: nn.Module, cfg: TrainConfig):
    """
    Fake quantize Conv and Linear weights before forward.
    """
    if cfg.weight_bits >= 32 or not cfg.quantize_weights_during_forward:
        return

    from compression.quant_ops import fake_quantize_tensor

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                w = m.weight
                q_w = fake_quantize_tensor(
                    w,
                    num_bits=cfg.weight_bits,
                    symmetric=cfg.symmetric,
                    per_channel=cfg.per_channel,
                    ch_axis=0,
                )
                m.weight.copy_(q_w)


"""---------------------------------------------
* def name :
*       train_one_epoch
*
* purpose:
*       Trains the model for one complete epoch
*       using the training dataset.
*
* Input parameters:
*       model        : neural network model being trained
*       train_loader : dataloader providing training batches
*       criterion    : loss function used for optimization
*       optimizer    : optimizer used to update model weights
*       device       : computation device (CPU or GPU)
*       epoch        : current training epoch index
*       cfg          : configuration object with training settings
*
* return:
*       None
---------------------------------------------"""
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg):
    model.train()
    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    total_samples = 0

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # forward
        outputs = model(images)  # MUST be in grad mode

        # standard supervised loss, keep as tensor (NO .item(), NO .detach())
        loss = criterion(outputs, targets)

        # backward
        loss.backward()  # loss.requires_grad should be True here

        # (optional but safe for QAT)
        # import torch.nn.utils
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # stats (use .item() only for logging)
        batch_size = images.size(0)
        total_samples += batch_size
        running_loss += loss.item() * batch_size

        with torch.no_grad():
            # compute accuracy for logging only
            _, pred = outputs.topk(5, 1, True, True)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))
            correct1 = correct[:, :1].reshape(-1).float().sum(0, keepdim=True)
            correct5 = correct.reshape(-1).float().sum(0, keepdim=True)
            running_acc1 += correct1.item()
            running_acc5 += correct5.item()

        if (i + 1) % cfg.log_interval == 0:
            avg_loss = running_loss / total_samples
            avg_acc1 = 100.0 * running_acc1 / total_samples
            avg_acc5 = 100.0 * running_acc5 / total_samples
            print(
                f"Epoch [{epoch}] Step [{i+1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f} Acc@1: {avg_acc1:.2f} Acc@5: {avg_acc5:.2f}"
            )

    return (
        running_loss / total_samples,
        100.0 * running_acc1 / total_samples,
        100.0 * running_acc5 / total_samples,
    )

"""---------------------------------------------
* def name :
*       evaluate
*
* purpose:
*       Evaluates the trained model on the
*       validation or test dataset.
*
* Input parameters:
*       model       : trained neural network model
*       data_loader : dataloader for evaluation dataset
*       criterion   : loss function used during evaluation
*       device      : computation device (CPU or GPU)
*
* return:
*       acc1 : top-1 classification accuracy
*       acc5 : top-5 classification accuracy
*       loss : average evaluation loss
---------------------------------------------"""
def evaluate(
    model: nn.Module,
    data_loader,
    criterion,
    device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_acc1 += acc1.item() * batch_size
            running_acc5 += acc5.item() * batch_size
            total_samples += batch_size

    avg_acc1 = running_acc1 / total_samples
    avg_acc5 = running_acc5 / total_samples
    avg_loss = running_loss / total_samples

    print(
        f"Eval: Loss {avg_loss:.4f} Acc@1 {avg_acc1:.2f} Acc@5 {avg_acc5:.2f}"
    )
    return avg_acc1, avg_acc5


"""---------------------------------------------
* execution block :
*       __main__
*
* purpose:
*       Ensures training starts only when the
*       script is executed directly.
---------------------------------------------"""
if __name__ == "__main__":
    main()
