import torch
import torch.nn as nn

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
        

def save_model(model, cfg, args, name):
    if len(cfg.DEVICES.GPU_ID) == 1:
        torch.save(model.state_dict(), 'checkpoints/%s/models/'+name+'.t7' % (args.exp_name))
    else:
        torch.save(model.module.state_dict(), 'checkpoints/%s/models/'+name+'.t7' % (args.exp_name))
        
        
def load_model(args, cfg, model):
      model.load_state_dict(torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH))
      if len(cfg.DEVICES.GPU_ID) > 1:
          model = nn.DataParallel(model, device_ids=cfg.DEVICES.GPU_ID)
      return model
