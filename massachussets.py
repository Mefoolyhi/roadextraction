import argparse
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as u
import io
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

DATA_DIR = '/misc/home6/s0106'
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
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
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

    def __init__(
            self,
            images_dir,
            masks_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)[0:1536, 0:1536, :]
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)[0:1536, 0:1536, :]

        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.image_paths)


# augmentation
def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def crop_image(image, target_image_dims=[1500, 1500, 3]):
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
           padding:image_size - padding,
           padding:image_size - padding,
           :,
           ]


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


def main(epochs, encoder):
    exp_dir = './experiments/' + f'massachussets-{encoder}-{datetime.now().strftime("%m-%d-%Y, %H:%M:%S")}'
    logs_path = './logs' + exp_dir[1:]

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    writer = SummaryWriter(logs_path)
    x_train_dir = os.path.join(DATA_DIR, 'tiff/train')
    y_train_dir = os.path.join(DATA_DIR, 'tiff/train_labels')

    x_valid_dir = os.path.join(DATA_DIR, 'tiff/val')
    y_valid_dir = os.path.join(DATA_DIR, 'tiff/val_labels')

    x_test_dir = os.path.join(DATA_DIR, 'tiff/test')
    y_test_dir = os.path.join(DATA_DIR, 'tiff/test_labels')

    class_dict = pd.read_csv(os.path.join(DATA_DIR, 'label_class_dict.csv'))
    class_names = class_dict['name'].tolist()
    class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

    print('All dataset classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)

    select_classes = ['background', 'road']

    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    print('Selected classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)

    dataset = RoadsDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values)
    random_idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[random_idx]

    file_writer = tf.summary.create_file_writer(logs_path)

    visualize(
        file_writer=file_writer,
        name='random image before',
        original_image=image,
        ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask=reverse_one_hot(mask)
    )

    # аугментация
    augmented_dataset = RoadsDataset(
        x_train_dir, y_train_dir,
        augmentation=get_training_augmentation(),
        class_rgb_values=select_class_rgb_values,
    )

    random_idx = random.randint(0, len(augmented_dataset) - 1)

    for i in range(3):
        image, mask = augmented_dataset[random_idx]
        visualize(
            file_writer=file_writer,
            name=f'{i} sample of augmented',
            original_image=image,
            ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
            one_hot_encoded_mask=reverse_one_hot(mask)
        )

    # parametres
    ENCODER = encoder
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = class_names
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    print('ENCODER', ENCODER)
    print('ENCODER_WEIGHTS', ENCODER_WEIGHTS)
    print('CLASSES', CLASSES)
    print('ACTIVATION', ACTIVATION)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = RoadsDataset(
        x_train_dir, y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    valid_dataset = RoadsDataset(
        x_valid_dir, y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    # batch, worers
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=5)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    TRAINING = True

    # Set num of epochs
    EPOCHS = epochs

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loss function
    loss = u.losses.DiceLoss()

    #  metrics
    metrics = [
        u.metrics.IoU(threshold=0.5),
    ]

    # optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0008),
    ])

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    print('EPOCHS', EPOCHS)
    print('DiceLoss')
    print('Adam([dict(params=model.parameters(), lr=0.00008)')
    print('CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5,')
    print('IoU(threshold=0.5)')
    print('batch_size', 4)
    print('num_workers', 4)

    # if os.path.exists('./best_model_m.pth'):
    #     checkpoint = torch.load('./best_model_m.pth')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print('Loaded DeepLabV3+ model from last run.')
    # else:
    start_epoch = 1

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
    print("start & end epoch numb", start_epoch, start_epoch + EPOCHS)

    print('TRAINING')
    if TRAINING:

        best_iou_score = 0.0

        for i in range(start_epoch, start_epoch + EPOCHS):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            writer.add_scalar('Accuracy/train', train_logs['iou_score'], i)
            writer.add_scalar('Accuracy/valid', valid_logs['iou_score'], i)
            writer.add_scalar('Loss/train', train_logs['dice_loss'], i)
            writer.add_scalar('Loss/valid', valid_logs['dice_loss'], i)

            print('EPOCHS', i)
            print(valid_logs['iou_score'])

            if best_iou_score < valid_logs['iou_score']:
                torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'encoder_name': ENCODER,
                    'encoder_weights': ENCODER_WEIGHTS,
                    'activation': ACTIVATION
                }, f'./best_model_m_{ENCODER}.pth')
                print('Saved model!')

    print('END TRAINING')
    sample_preds_folder = logs_path + '/sample_predictions/'
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)

    model.eval()

    x_kolomna_dir = os.path.join(DATA_DIR, 'tiff/kolomna')
    y_kolomna_dir = os.path.join(DATA_DIR, 'tiff/kolomna_labels')

    kolomna_rgb_values = [[0, 0, 0], [110, 110, 110]]

    kolomna_dataset = RoadsDataset(
        x_kolomna_dir,
        y_kolomna_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=kolomna_rgb_values,
    )

    kolomna_dataloader = DataLoader(kolomna_dataset, batch_size=1, shuffle=False)

    kolomna_dataset_vis = RoadsDataset(
        x_kolomna_dir,
        y_kolomna_dir,
        augmentation=get_validation_augmentation(),
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

    kolomna_epoch = u.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    kolomna_logs = kolomna_epoch.run(kolomna_dataloader)
    writer.add_scalar('Accuracy/kolomna', kolomna_logs['iou_score'])
    writer.add_scalar('Loss/kolomna', kolomna_logs['dice_loss'])

    test_dataset = RoadsDataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_dataset_vis = RoadsDataset(
        x_test_dir, y_test_dir,
        augmentation=get_validation_augmentation(),
        class_rgb_values=select_class_rgb_values,
    )

    random_idx = random.randint(0, len(test_dataset_vis) - 1)
    image, mask = test_dataset_vis[random_idx]

    visualize(
        file_writer=file_writer,
        name='1 from test before testing',
        original_image=image,
        ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask=reverse_one_hot(mask)
    )

    for idx in range(len(test_dataset)):
        image, gt_mask = test_dataset[idx]
        image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_building_heatmap = pred_mask[:, :, select_classes.index('road')]
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))

        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))
        cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"),
                    np.hstack([image_vis, gt_mask, pred_mask])[:, :, ::-1])

        visualize(
            file_writer=file_writer,
            name=f'Prediction {idx}',
            original_image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pred_mask,
            predicted_building_heatmap=pred_building_heatmap
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
    print("i'm done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='deeplabv3+')
    parser.add_argument('-e', type=int, default=20)  # epoch
    parser.add_argument('-enc', type=str, default='resnet101') #encoder
    # resnet152, timm-res2net101_26w_4s, timm-gernet_l,
    # senet154, densenet161, efficientnet-b7, timm-efficientnet-l2 с весами noisy-student,
    # dpn98, mit_b5, vgg19_bn
    parser.add_argument('-a', type=str) #activation
    #“sigmoid”, “softmax”, “logsoftmax”, “tanh”, “identity”
    parser.add_argument('-b', type=int) #batch
    parser.add_argument('-lr', type=int) #lr
    parser.add_argument('-mt', type=int) #thereshold metrics
    parser.add_argument('-l', type=str) #loss

    args = parser.parse_args()
    main(args.e, args.enc)

#sbatch -p debug  --mem=32000 --gres=gpu:k40m:3 --cpus-per-task=5 -t 00:30:00 --job-name=road --output='test_cs %j' --wrap="python massachussets.py -e 0"
# sbatch  --mem=32000 --gres=gpu:k40m:3 --cpus-per-task=5 -t 20:00:00 --job-name=road --output='./logs/test_cs %j' --wrap="python massachussets.py -e 1"