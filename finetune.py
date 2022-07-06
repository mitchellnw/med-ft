import argparse
import os
import torch
import clip
import os
from tqdm import tqdm
import time

from PIL import Image

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from torchvision.datasets import ImageFolder
from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr

def _convert_to_rgb(image):
    return image.convert('RGB')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/tmp123'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--custom-template", action="store_true", default=False,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--model",
        default='ViT-B/32',
        help='Model to use -- you can try another like ViT-L/14'
    )
    parser.add_argument(
        "--name",
        default='finetune_cp',
        help='Filename for the checkpoints.'
    )
    return parser.parse_args()

def zeroshot_classifier(model, classnames, templates, device):
    print('Building zero-shot classifier.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return 100*zeroshot_weights.t()

if __name__ == '__main__':
    args = parse_arguments()
    DEVICE = 'cuda'

    template = [lambda c: f'a photo of a syringe containing the drug: "{c}".',]

    base_model, preprocess = clip.load(args.model, 'cuda', jit=False)

    train_preprocess = preprocess

    traindir = os.path.join(args.data_location, 'train')
    valdir = os.path.join(args.data_location, 'test')

    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    train_preprocess = Compose([
            RandomResizedCrop(base_model.visual.input_resolution, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

    train_dataset = ImageFolder(
        traindir, transform=train_preprocess)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_dataset = ImageFolder(valdir, transform=preprocess)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    idx_to_class = dict((v, k)
                        for k, v in train_dataset.class_to_idx.items())
    classnames = [idx_to_class[i].replace(
        '_', ' ') for i in range(len(idx_to_class))]
    print('classnames are', classnames)
    print(classnames)

    clf = zeroshot_classifier(base_model, classnames, template, DEVICE)
    NUM_CLASSES = len(classnames)
    feature_dim = base_model.visual.output_dim

    model = ModelWrapper(base_model, feature_dim, NUM_CLASSES, normalize=True, initial_weights=clf)
    for p in model.parameters():
        p.data = p.data.float()

    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model,  device_ids=devices)

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)

    num_batches = len(train_loader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    loss_fn = torch.nn.CrossEntropyLoss()

    if os.path.exists(args.model_location):
        model_path = os.path.join(args.model_location, f'{args.name}_0.pt')
        print('Saving model to', model_path)
        torch.save(model.module.state_dict(), model_path)

    for epoch in range(args.epochs):
        # Train
        model.train()
        end = time.time()
        for i, batch in enumerate(train_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)
            data_time = time.time() - end

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if i % 20 == 0:
                percent_complete = 100.0 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        # #Evaluate
        test_loader = test_loader
        model.eval()
        with torch.no_grad():
            print('*'*80)
            print('Starting eval')
            correct, count = 0.0, 0.0
            pbar = tqdm(test_loader)
            for batch in pbar:
                batch = maybe_dictionarize_batch(batch)
                inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)

                logits = model(inputs)

                loss = loss_fn(logits, labels)

                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                count += len(logits)
                pbar.set_description(
                    f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}")
            top1 = correct / count
        print(f'Val acc at epoch {epoch}: {100*top1:.2f}')

        if os.path.exists(args.model_location):
            model_path = os.path.join(args.model_location, f'{args.name}_{epoch + 1}.pt')
            print('Saving model to', model_path)
            torch.save(model.module.state_dict(), model_path)

