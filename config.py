# Shared configuration and constants for teachers and students

BASE_PATH = 'preprocessing_crop_3slices128'  # .npy files with shape (224, 224, 3)
TEACHER_DIR = 'checkpoints_teachers_equals'  # teacher checkpoints
CKPT_DIR = 'checkpoints_students'

# Student Training Configuration
EPOCHS = 100
LR = 1e-4
PATIENCE = 8
BATCH = 16

TEMPERATURES = [4, 10, 30]
ALPHA = 0.5
FOCAL_GAMMA = 2.0

TEACHERS = {
    'resnet50': 'resnet50',
    'densenet121': 'densenet121',
    'vit_small': 'vit_small_patch16_224',
    'swin_transformer': 'swin_base_patch4_window7_224',
    'maxvit_tiny': 'maxvit_tiny_tf_224',
}

STUDENTS = {
    'efficientnet_b0': 'efficientnet_b0',
    'mobilenetv2': 'mobilenetv2_100',
}

LOSS_VARIANTS = ['classic', 'kl_mse', 'kl_focal', 'kl_mse_focal']
VARIANT_LABEL = {
    'classic': 'D_L  (KL)',
    'kl_mse': 'D_L1 (KL+MSE)',
    'kl_focal': 'D_L2 (KL+Focal)',
    'kl_mse_focal': 'D_L3 (KL+MSE+Focal)',
}