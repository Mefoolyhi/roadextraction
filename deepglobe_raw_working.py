import os, cv2
import numpy as np
import pandas as pd
import io
import random, tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as u
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse

DATA_DIR = '/home6/m_imm_freedata/Roads/DeepGlobe_Roads'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def visualize(file_writer, name, **images):
    """
    Plot images in one row
    """
    n_images = len(images)
    figure = plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        title = name.replace('_', ' ').title()
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
            preprocessing=None,
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


def main(best_model_path, order, epochs=1):
    exp_dir = './experiments/' + f'deepglobe-{best_model_path}-{order}-{datetime.now().strftime("%m-%d-%Y, %H:%M:%S")}'
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

    select_classes = ['background', 'road']
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    dataset = RoadsDataset(train_df, class_rgb_values=select_class_rgb_values)
    random_idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[2]

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

    TRAINING = True

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

    if os.path.exists('./best_model_g.pth'):
        checkpoint = torch.load('./best_model_g.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
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
            print('order', order)

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

    test_dataset = RoadsDataset(
        valid_df,
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    test_dataloader = DataLoader(test_dataset)
    test_dataset_vis = RoadsDataset(
        valid_df,
        class_rgb_values=select_class_rgb_values,
    )

    random_idx = random.randint(0, len(test_dataset_vis) - 1)
    image, mask = test_dataset_vis[random_idx]

    visualize(
        file_writer=file_writer,
        name='after training',
        original_image=image,
        ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask=reverse_one_hot(mask)
    )

    sample_preds_folder = logs_path + '/sample_predictions/'
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)

    image = 'kolomna.png'
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pred_mask = model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    pred_road_heatmap = pred_mask[:, :, select_classes.index('road')]
    pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
    cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_kolomna.png"),
                np.hstack([image, pred_mask])[:, :, ::-1])

    visualize(
        file_writer=file_writer,
        name='kolomna',
        original_image=image,
        predicted_mask=pred_mask,
        pred_road_heatmap=pred_road_heatmap
    )

    for idx in range(len(test_dataset)):
        image, gt_mask = test_dataset[idx]
        image_vis = test_dataset_vis[idx][0].astype('uint8')
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_road_heatmap = pred_mask[:, :, select_classes.index('road')]
        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        gt_mask = colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values)
        cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])

        visualize(
            file_writer=file_writer,
            name=f'Prediction {idx}',
            original_image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pred_mask,
            pred_road_heatmap=pred_road_heatmap
        )

    test_epoch = u.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    valid_logs = test_epoch.run(test_dataloader)
    writer.add_scalar('Accuracy/test', valid_logs['iou_score'])
    writer.add_scalar('Loss/test', valid_logs['dice_loss'])

    writer.close()


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
    parser.add_argument('-pm', type=str) #pretrained model
    parser.add_argument('-e', type=int) #epoch
    parser.add_argument('-o', type=int) #order
    parser.add_argument('-enc', type=str) #encoder
    parser.add_argument('-a', type=str) #activation
    parser.add_argument('-b', type=int) #batch
    parser.add_argument('-lr', type=int) #lr
    parser.add_argument('-mt', type=int) #thereshold metrics
    parser.add_argument('-l', type=str) #loss

    args = parser.parse_args()
    main(args.pm, args.o)


#sbatch -p debug  --mem=32000 --gres=gpu:k40m:3 --cpus-per-task=5 -t 00:30:00 --job-name=road --output='test_cs %j' --wrap="python deepglobe_raw_working.py -o 1"
#sbatch  --mem=32000 --gres=gpu:k40m:3 --cpus-per-task=5 -t 20:00:00 --job-name=road --output='./logs/test_cs %j' --wrap="python deepglobe_raw_working.py -o 1"