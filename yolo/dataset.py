import os
import shutil

class YoloDataset:
    """
    A utility class to prepare and convert object detection datasets into YOLO format.

    This class processes a DataFrame containing labeled object detection data
    and converts it into the format required by the YOLO training pipeline.
    It also copies the corresponding images and creates YOLO-compatible label files
    in the specified directory structure.

    It supports splitting the dataset into training and validation sets using an index range.

    """

    def __init__(self, df, index, img_dir='../dataset/images/images/', out_dir='../yolo_dataset/', mode='train'):
        '''    
        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing the annotated object detection data with columns:
            'ImageID', 'LabelName', 'XMin', 'YMin', 'XMax', 'YMax'.
        
        index : tuple(float, float)
            A tuple indicating the start and end proportion of the dataset to include.
            For example, (0.0, 0.8) would take the first 80% of the dataset (used for training),
            while (0.8, 1.0) would take the last 20% (used for validation).

        img_dir : str, optional
            Path to the folder containing the original images. Default is '../dataset/images/images/'.

        out_dir : str, optional
            Path to the root output directory where the YOLO-formatted dataset will be saved.
            Default is '../yolo_dataset/'.

        mode : str, optional
            Dataset split mode, must be either 'train' or 'val'. Default is 'train'.
        '''
        
        if not mode in ('train', 'val'):
            raise ValueError('this Mode is not an Option. Try -> train or val')
        
        self.df = df
        self.img_dir = img_dir
        self.unique_images = self.df['ImageID'].unique()
        self.labels = {'Bus':0, 'Truck':1}
        self.index = index

        self.out_dir = out_dir
        self.labels_dir = f'{out_dir}labels/{mode}/'
        self.new_img_dir = f'{out_dir}images/{mode}/'

        # remove yolo_dataset and initialize the new
        self._reset()
        
        self._make_dirs()
        self.convert_to_yolo()

    def __len__(self):
        return len(self.unique_images)
    
    def convert_to_yolo(self):
        # read the Data
        for idx in range(int(self.index[0]*len(self.unique_images)), int(self.index[1]*len(self.unique_images))):
            img_id = self.unique_images[idx]
            df  = self.df[self.df['ImageID']==img_id]

            boxes = df['LabelName,XMin,YMin,XMax,YMax'.split(',')].values

            # convert bbxs from xyxy to cxcywh
            boxes = [f'{self.labels[label]} {(xmax+xmin)/2} {(ymax+ymin)/2} {xmax-xmin} {ymax-ymin}'
                    for label, xmin, ymin, xmax, ymax in boxes] 
            
            # Write to .txt file
            txt_filename = img_id + '.txt'
            txt_path = os.path.join(self.labels_dir, txt_filename)
            try:
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(boxes))
            except Exception as e:
                print(f"Error writing {txt_path}: {e}")

            # Copy the images into images dir for yolo
            try:
                img_file = f'{img_id}.jpg'
                img_file = os.path.join(self.img_dir,img_file)
                dest = os.path.join(self.new_img_dir, f'{img_id}.jpg')
                os.link(img_file, dest)
            except FileNotFoundError as e:    
                print(e)
                
    def _reset(self):
        if os.path.exists(self.labels_dir):
            shutil.rmtree(self.labels_dir)
        if os.path.exists(self.new_img_dir):
            shutil.rmtree(self.new_img_dir)
    
    def _make_dirs(self):
        os.makedirs(self.new_img_dir)
        os.makedirs(self.labels_dir)
