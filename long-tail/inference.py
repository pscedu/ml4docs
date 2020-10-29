import os
import argparse
import pprint
import torch
from torch.autograd import Variable
from data import dataloader
from run_networks import model
import warnings
from utils import source_import

# ================
# LOAD CONFIGURATIONS

data_root = {'ImageNet': '/MLStamps/long-tail/OpenLongTailRecognition-OLTR/OLTRDataset/OLTRDataset_1',
             'Places': '/home/public/dataset/Places365'}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='/MLStamps/long-tail/OpenLongTailRecognition-OLTR/config/Imagenet_LT/Stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
args = parser.parse_args()

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits

config = source_import(args.config).config
training_opt = config['training_opt']
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']
training_opt['batch_size'] = 1

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
pprint.pprint(config)


print('Under testing phase, we load training data simply to calculate training data number for each class.')

data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                batch_size=training_opt['batch_size'],
                                sampler_dic=None, 
                                test_open=test_open,
                                num_workers=training_opt['num_workers'],
                                shuffle=False)
        for x in ['train', 'test']}

#testsample="/MLStamps/long-tail/OpenLongTailRecognition-OLTR/OLTRDataset/OLTRDataset_1/campaign3to5/arita/000008927.jpg"

training_model = model(config, data, test=True)
training_model.load_model()
#for training_model in training_model.networks.values():
#    training_model.eval()
#training_model.eval()

memory = config['memory']
training_model.eval(phase='test', openset=test_open)

#for inputs, labels, paths in (data['test']):
#    with torch.no_grad():
#        inputs = Variable(inputs)
#        labels = Variable(labels)
        #print(inputs)
#        print(labels)
#        print(paths)
#        preds = training_model(inputs, centroids=memory['centroids'])
#        print(preds)
        


#training_model.eval(phase='test', openset=test_open)

    
if output_logits:
    training_model.output_logits(openset=test_open)
    
print('ALL COMPLETED.')
