import os, cv2
import numpy as np
import pandas as pd
import io
import random, tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as u
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import operator
import argparse

DATA_DIR = '/home6/m_imm_freedata/Roads/DeepGlobe_Roads'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def visualize(file_writer, name, **images):
    """
    Plot images in one row
    """
    n_images = len(images)
    figure = plt.figure(figsize=(20, 8))
    for idx, (nm, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        title = nm.replace('_', ' ').title()
        plt.title(title, fontsize=20)
        plt.imshow(image)
    with file_writer.as_default():
        tf.summary.image(name, plot_to_image(figure), step=0)


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


class RoadsDataset(torch.utils.data.Dataset):
    """DeepGlobe Road Extraction Challenge Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            df,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None
    ):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.image_paths)


class KolomnaRoadsDataset(torch.utils.data.Dataset):
        def __init__(
                self,
                image_path, mask_path,
                class_rgb_values=None,
                preprocessing=None,
                tile_size=1024,
                res_size=224
        ):
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

            self.class_rgb_values = class_rgb_values
            self.preprocessing = preprocessing

            self.tile_size = step = tile_size
            self.res_size = res_size
            img = np.asarray(img)
            mask = np.asarray(mask)
            w, h, _ = img.shape

            self.w_new = res_size * (w // tile_size + 1)
            self.h_new = res_size * (h // tile_size + 1)
            i = j = 0
            self.img_frags = []
            self.msk_frags = []
            self.res = np.zeros((self.w_new, self.h_new, 3), dtype=np.float32)
            image_new = np.zeros(((w // tile_size + 1) * tile_size,
                                 (h // tile_size + 1) * tile_size, 3), dtype=np.float32)
            image_new[0:w, 0:h, :] = img
            img = image_new
            mask_new = np.zeros(((w // tile_size + 1) * (tile_size // 2),
                                 (h // tile_size + 1) * (tile_size // 2), 3), dtype=np.float32)
            mask_new[0:mask.shape[0], 0:mask.shape[1], :] = mask
            mask = mask_new
            while i < w // step + 1:
                j = 0
                while j < h // step + 1:
                    frag = img[i*tile_size:(i + 1)*tile_size, j*tile_size:(j+1)*tile_size, :]
                    self.img_frags.append(frag)
                    frag = mask[i*(tile_size // 2):(i+1)*(tile_size // 2),
                           j*(tile_size // 2):(j+1)*(tile_size // 2), :]
                    self.msk_frags.append(frag)
                    j += 1
                i += 1


        def __getitem__(self, i):
            image = self.img_frags[i]
            mask = one_hot_encode(self.msk_frags[i], self.class_rgb_values).astype('float')
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            return image, mask

        def throw_pred(self, index, pred):
            i = self.res_size * (index // (self.w_new // self.res_size))
            j = self.res_size * (index % (self.h_new // self.tile_size))
            self.res[i:i + self.res_size, j:j + self.res_size, :] = pred

        def result(self):
            max = np.max(self.res)
            min = np.min(self.res)

            img = 255.0 * (self.res - min) / (max - min)
            return np.uint8(img)

        def __len__(self):
            return len(self.msk_frags)


class TestRoadsDataset(torch.utils.data.Dataset):
    """DeepGlobe Road Extraction Challenge Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            df,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_paths = df['image_id'].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentation:
            image = self.augmentation(image=image)['image']

        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']
        return image

    def __len__(self):
        return len(self.image_paths)


def main(epochs=1):
    exp_dir = './experiments/' + f'deepglobe-{datetime.now().strftime("%m-%d-%Y, %H:%M:%S")}'
    logs_path = './logs' + exp_dir[1:]

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    writer = SummaryWriter(logs_path)

    metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
    metadata_df = metadata_df[metadata_df['split'] == 'train']
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))

    # drop
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    # shuffle & frac
    valid_df = metadata_df.sample(frac=0.1, random_state=42)
    train_df = metadata_df.drop(valid_df.index)

    class_dict = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
    class_names = class_dict['name'].tolist()
    class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
    print(class_rgb_values)

    select_classes = ['background', 'road']
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    dataset = RoadsDataset(train_df, class_rgb_values=select_class_rgb_values)
    random_idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[random_idx]

    file_writer = tf.summary.create_file_writer(logs_path)

    visualize(
        file_writer=file_writer,
        name='one sample before',
        original_image=image,
        ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask=reverse_one_hot(mask)
    )

    #augmentation
    augmented_dataset = RoadsDataset(
        train_df,
        augmentation=get_training_augmentation(),
        class_rgb_values=select_class_rgb_values,
    )

    random_idx = random.randint(0, len(augmented_dataset) - 1)

    for idx in range(3):
        image, mask = augmented_dataset[idx]
        visualize(
            file_writer=file_writer,
            name=f'{idx} sample of augmented',
            original_image=image,
            ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
            one_hot_encoded_mask=reverse_one_hot(mask)
        )


    # parametres
    ENCODER = 'resnet50' #resnet101
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = select_classes
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

    print('ENCODER', ENCODER)
    print('ENCODER_WEIGHTS', ENCODER_WEIGHTS)
    print('CLASSES', CLASSES)
    print('ACTIVATION', ACTIVATION)

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = RoadsDataset(
        train_df,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    valid_dataset = RoadsDataset(
        valid_df,
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    print('len train_dataset', len(train_dataset))
    print('len valid_dataset', len(valid_dataset))

    # epochs
    EPOCHS = epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # batches & workers
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=DEVICE == 'cuda')
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=DEVICE == 'cuda')

    print('batch_size', 4)
    print('num_workers', 4)

    TRAINING = False

    print('EPOCHS', EPOCHS)

    # loss function
    loss = u.losses.DiceLoss()

    print('DiceLoss')

    # metrics, thereshold
    metrics = [
        u.metrics.IoU(threshold=0.5),
    ]

    print('IoU(threshold=0.5)')

    # optimizer, lr
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.00008),
    ])

    print('Adam([dict(params=model.parameters(), lr=0.00008)')
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    print('CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5,')

    if os.path.exists('./best_model.pth'):
        model = torch.load('./best_model.pth')
        print('Loaded DeepLabV3+ model from last run.')
    else:
        epoch = 1

    train_epoch = u.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = u.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    print('TRAINING')
    if TRAINING:

        for i in range(epoch, epoch + EPOCHS):
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            writer.add_scalar('Accuracy/train', train_logs['iou_score'], i)
            writer.add_scalar('Accuracy/valid', valid_logs['iou_score'], i)
            writer.add_scalar('Loss/train', train_logs['dice_loss'], i)
            writer.add_scalar('Loss/valid', valid_logs['dice_loss'], i)

            print('EPOCHS', i)
            print(valid_logs['iou_score'])

            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'encoder_name': ENCODER,
                'encoder_weights': ENCODER_WEIGHTS,
                'activation': ACTIVATION
            }, './best_model_g.pth')
            print('Saved model!')

            #lr_scheduler.step()
            #print(lr_scheduler.get_last_lr())

    print('END TRAINING')

    model.eval()

    sample_preds_folder = logs_path + '/sample_predictions/'
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)

    kolomna_rgb_values = [[0, 0, 0], [110, 110, 110], [100, 100, 100]]

    kolomna_dataset = KolomnaRoadsDataset(
        'kolomna_big.png', 'mask_kolomna.png',
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=kolomna_rgb_values,
    )

    kolomna_dataset_vis = KolomnaRoadsDataset(
        'kolomna_big.png', 'mask_kolomna.png',
        class_rgb_values=kolomna_rgb_values,
    )

    for idx in range(len(kolomna_dataset)):
        image, gt_mask = kolomna_dataset[idx]
        image_vis = crop_image(kolomna_dataset_vis[idx][0].astype('uint8'))
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_building_heatmap = pred_mask[:, :, select_classes.index('road')]
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))

        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), kolomna_rgb_values))
        kolomna_dataset.throw_pred(idx, pred_mask)
        cv2.imwrite(os.path.join(sample_preds_folder, f"kolomna_pred_{idx}.png"),
                    np.hstack([image_vis, gt_mask, pred_mask])[:, :, ::-1])

        visualize(
            file_writer=file_writer,
            name=f'Kolomna prediction {idx}',
            original_image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pred_mask,
            predicted_building_heatmap=pred_building_heatmap
        )
    pred = kolomna_dataset.result()
    w, h, _ = pred.shape
    image = cv2.resize(cv2.imread('kolomna_big.png'), (h, w))
    mask = cv2.resize(cv2.imread('mask_kolomna.png'),  (h, w))
    mask = colour_code_segmentation(reverse_one_hot(one_hot_encode(mask, kolomna_rgb_values)), kolomna_rgb_values)

    print(image.shape, mask.shape, pred.shape)

    cv2.imwrite(os.path.join(sample_preds_folder, f"kolomna_pred_total.png"),
                np.hstack([image, mask, pred])[:, :, ::-1])

    visualize(
        file_writer=file_writer,
        name=f'Kolomna prediction total',
        original_image=image,
        ground_truth_mask=mask,
        predicted_mask=pred
    )

    writer.add_scalar('Accuracy/kolomna', get_iou(mask, pred))

    metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
    metadata_df = metadata_df[(metadata_df['split'] == 'val') | (metadata_df['split'] == 'test')]
    metadata_df = metadata_df[['image_id']]
    metadata_df['image_id'] = metadata_df['image_id'].apply(lambda img_pth: os.path.join(DATA_DIR, f"{img_pth}_sat.jpg"))
    valid_df = metadata_df.sample(frac=1).reset_index(drop=True)
    test_dataset = TestRoadsDataset(
        valid_df,
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    test_dataset_vis = TestRoadsDataset(
        valid_df,
        class_rgb_values=select_class_rgb_values,
    )

    for idx in range(len(test_dataset)):
        image = test_dataset[idx]
        image_vis = test_dataset_vis[idx].astype('uint8')
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_road_heatmap = pred_mask[:, :, select_classes.index('road')]
        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)

        cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"),
                    np.hstack([image_vis, pred_mask])[:,:,::-1])
        visualize(
            file_writer=file_writer,
            name=f'Prediction {idx}',
            original_image=image_vis,
            predicted_mask=pred_mask,
            pred_road_heatmap=pred_road_heatmap
        )

    writer.close()
    print("i'm done")


def crop_image(image, target_image_dims=[224, 224, 3]):
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
           padding:image_size - padding,
           padding:image_size - padding,
           :,
           ]


def get_iou(img, result):
    img = img.sum(axis=2)
    result = result.sum(axis=2)
    component1 = img
    component2 = result

    overlap = component1 * component2  # Logical AND
    union = component1 + component2  # Logical OR

    return overlap.sum() / float(union.sum())


# augmentation
def get_training_augmentation():
    train_transform = [
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ]
    return album.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='deeplabv3+')
    parser.add_argument('-e', type=int) #epoch
    parser.add_argument('-enc', type=str) #encoder
    parser.add_argument('-a', type=str) #activation
    parser.add_argument('-b', type=int) #batch
    parser.add_argument('-lr', type=int) #lr
    parser.add_argument('-mt', type=int) #thereshold metrics
    parser.add_argument('-l', type=str) #loss

    args = parser.parse_args()
    main()


#sbatch -p debug  --mem=32000 --gres=gpu:k40m:3 --cpus-per-task=5 -t 00:30:00 --job-name=road --output='test_cs %j' --wrap="python deepglobe_picture.py"
#sbatch  --mem=32000 --gres=gpu:k40m:3 --cpus-per-task=5 -t 20:00:00 --job-name=road --output='./logs/test_cs %j' --wrap="python deepglobe_raw_working.py"