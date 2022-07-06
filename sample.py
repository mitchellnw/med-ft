from PIL import Image
import clip
import torch
from utils import ModelWrapper

classnames = ['cisatracurium', 'dexamethasone', 'ephedrine', 'epinephrine', 
    'etomidate', 'fentanyl', 'glycopyrrolate', 'hydromorphone', 'ketamine',
    'ketorolac', 'lidocaine', 'midazolam', 'neostigmine', 'ondansetron',
    'phenylephrine', 'propofol', 'rocuronium', 'succinylcholine',
    'sugammadex', 'vecuronium']

if __name__ == '__main__':
    base_model, preprocess = clip.load('ViT-L/14', 'cuda', jit=False)
    NUM_CLASSES = 20
    model = ModelWrapper(base_model, base_model.visual.output_dim, NUM_CLASSES, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()

    # TODO change to where model is saved -- this is the model i've give you via gdrive
    saved_model_path = '/gscratch/efml/mitchnw/cps/med_data_cp.pt' 
    model.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
    for p in model.parameters():
        p.data = p.data.float()

    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model,  device_ids=devices)

    image = Image.open('glycopyrrolate_test.jpg').convert("RGB")
    image = preprocess(image).unsqueeze(0)
    image = image.cuda()

    out = model(image)
    prediction = out.argmax(dim=1, keepdim=True)

    print('Prediction:', classnames[prediction.item()])