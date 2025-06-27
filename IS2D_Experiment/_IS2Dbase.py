import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from torch.utils.data import DataLoader

from IS2D_models import IS2D_model
from dataset.BioMedicalDataset.PolypSeg import *
from dataset.BioMedicalDataset.DataScienceBowl2018Dataset import *
from dataset.BioMedicalDataset.MonuSegDataset import *
from dataset.BioMedicalDataset.SkinSegmentation2018Dataset import *
from dataset.BioMedicalDataset.PH2Dataset import *
from dataset.BioMedicalDataset.Covid19CTScanDataset import *
from dataset.BioMedicalDataset.Covid19CTScan2Dataset import *
from dataset.BioMedicalDataset.BUSISegmentationDataset import *
from dataset.BioMedicalDataset.STUSegmentationDataset import *
from dataset.BioMedicalDataset.BUSUCLMSegmentationDataset import *
from dataset.BioMedicalDataset.BUSICombinedDataset import *
from utils.get_functions import *
from torch.utils.data import ConcatDataset


class BaseSegmentationExperiment(object):
    def __init__(self, args):
        super(BaseSegmentationExperiment, self).__init__()

        self.args = args

        self.args.device = get_deivce()
        if args.seed_fix: self.fix_seed(self.args.device)

        print("STEP1. Load {} Test Dataset Loader...".format(args.test_data_type))
        test_image_transform, test_target_transform = self.transform_generator()
        if args.test_data_type in ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']: test_dataset = PolypImageSegDataset(args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif args.test_data_type == 'DSB2018': test_dataset = DataScienceBowl2018Dataset(args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif args.test_data_type == 'MonuSeg2018': test_dataset = MonuSeg2018Dataset(args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif args.test_data_type == 'ISIC2018': test_dataset = ISIC2018Dataset(args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif args.test_data_type == 'PH2': test_dataset = PH2Dataset(args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif args.test_data_type == 'COVID19': test_dataset = Covid19CTScanDataset(args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif args.test_data_type == 'COVID19_2': test_dataset = Covid19CTScan2Dataset(args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif args.test_data_type == 'BUSI': test_dataset = BUSISegmentationDataset(args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif args.test_data_type == 'BUS-UCLM': test_dataset = BUSUCLMSegmentationDataset(args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif args.test_data_type == 'STU': test_dataset = STUSegmentationDataset(args.test_dataset_dir, mode='test',  transform=test_image_transform, target_transform=test_target_transform)
        else:
            print("Wrong Dataset")
            sys.exit()

        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        print("STEP2. Load MADGNet ...")
        self.model = IS2D_model(args)
        self.model.to(self.args.device)  # <-- Add this line

    def fix_seed(self):
        random.seed(4321)
        np.random.seed(4321)
        torch.manual_seed(4321)
        torch.cuda.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)

    def forward(self, image, target, mode):
        image, target = image.to(self.args.device), target.to(self.args.device)

        with torch.cuda.amp.autocast(enabled=True):
            output = self.model(image, mode)
            loss = self.model._calculate_criterion(output, target)

            return loss, output, target
        
    # Add this to BaseSegmentationExperiment

    def setup_train_loader(self):
        print("STEP1. Load {} Train Dataset Loader...".format(self.args.train_data_type))
        train_image_transform, train_target_transform = self.transform_generator()
        if self.args.train_data_type == 'BUS-UCLM':
            train_dataset = BUSUCLMSegmentationDataset(self.args.train_dataset_dir, mode='train', transform=train_image_transform, target_transform=train_target_transform)
        elif self.args.train_data_type == 'BUSI':
            train_dataset = BUSISegmentationDataset(self.args.train_dataset_dir, mode='train', transform=train_image_transform, target_transform=train_target_transform)
        elif self.args.train_data_type == 'BUSI-Combined':
            train_dataset = BUSICombinedDataset(self.args.train_dataset_dir, mode='train', transform=train_image_transform, target_transform=train_target_transform)
        # Add other datasets as needed
        elif self.args.train_data_type == 'BUSIBUSUCLM':
            busi_dataset = BUSISegmentationDataset('dataset/BioMedicalDataset/BUSI', mode='train', transform=train_image_transform, target_transform=train_target_transform)
            bus_uclm_dataset = BUSUCLMSegmentationDataset('dataset/BioMedicalDataset/BUS-UCLM', mode='train', transform=train_image_transform, target_transform=train_target_transform)
            train_dataset = ConcatDataset([busi_dataset, bus_uclm_dataset])
            
            # combined_dataset = ConcatDataset([busi_dataset, bus_uclm_dataset])
            # train_ = DataLoader(combined_dataset, batch_size=..., shuffle=True, ...)


        else:
            print("Wrong Train Dataset")
            sys.exit()
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size if hasattr(self.args, 'batch_size') else 4, shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)