
# CRMNet Project

## Environment Setup

Clone the repository and set up the environment:

```bash
git clone -b main https://github.com/godcodehand/CRMNet.git
cd CRMNet
conda create -n CRMNet python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate CRMNet
python setup.py
```

## Testing

1. **Download the Dataset**:
   Refer to the "Data availability" section in the paper for the link to download the "dataset" file.

2. **Get Configuration File and Checkpoints**:
   Download the "config_and_checkpoint" file, as mentioned in the paper's "Data availability" section.

3. **Run the Test**:
   Modify and run `simplified_test.py` with the following parameters:
   - `config_path`: Path to the configuration `.py` file.
   - `checkpoint_path`: Path to the checkpoint `.pth` file.
   - `dataset_root`: Root path of the dataset (e.g., `path_to/Rat_Ulcer`), include:
     - `--ann_dir`: Annotation directory path .
     - `--img_dir`: Image directory path .
   - `dataset_type`: Specify the dataset type, options are:
     - "MyDatasetUlcer" for "Rat_Ulcer"
     - 'MyDatasetCerebralInfarction' for "Rat_Cerebral_Infarction"
     - 'MyDatasetPolyp' for various Polyp datasets."CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB" and "Kvasir-SEG".
     - 'MyDatasetISIC2018Task1' for "ISIC_2018_Task1"
   - `save_path`: Path to save the outputs.
   - `show_dir`: Path to save visualization results.

   ```bash
   python path_to/simplified_test.py
   ```

## Training

1. **Download the Dataset**:
   Same as testing.

2. **Download Pretrained Model and Config File**:
   Refer to the "Data availability" section in the paper for the link to download the "pretrained_model" file.

3. **Run the Training**:
   Modify and run `simplified_train.py` with the following parameters:
   - `config_path`: Path to the configuration `.py` file.
   - `pretrain_checkpoint`: Path to the pretrained model checkpoint `.pth` file.
   - `dataset_root`: Path to the dataset directories, same as specified in the testing section.
   - `dataset_type`: Choose dataset type as specified in testing.
   - `save_path`: Path to save the training outputs.

   ```bash
   python path_to/simplified_train.py
   ```

## Inference

1. **Download the Dataset**:
   Same as testing.

2. **Get Configuration File and Checkpoints**:
   Same as testing.

3. **Perform Inference**:
   Modify and run `simplified_infer.py` with the following parameters:
   - `config_path`: Path to the configuration `.py` file.
   - `pretrain_checkpoint`: Path to the checkpoint `.pth` file.
   - `img_folder_path`: Path to the folder containing images for inference.
   - `save_path`: Path to save inference results.

   The output will be in the same format as the annotation, with pixel values representing the class. Visualization of the predictions can be found in `save_path/visual`.

## Customization

You can modify the configuration files as per your requirements based on the mmsegmentation guidelines. For more details, visit [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

## GPAB Usage

1. **Download GPAB Code**:
   Refer to the "Data availability" section in the paper for the link to download the "GPAB" file.

2. **Demo**:
   Modify the paths in the code:
   - `img_path`: Path to the .jpg image.
   - `save_folder_path`: Path to save the binarized results.
