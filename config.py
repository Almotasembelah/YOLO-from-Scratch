# config.py

MODEL_SIZES = {
    'n': {'W': 0.25, 'D': 0.33, 'R': 2},
    's': {'W': 0.50, 'D': 0.33, 'R': 2},
    'm': {'W': 0.75, 'D': 0.67, 'R': 1.5},
    'l': {'W': 1.00, 'D': 1.00, 'R': 1},
    'x': {'W': 1.25, 'D': 1.00, 'R': 1},
}


def get_yolo_config(model_size='n', num_classes=2, num_boxes=16):
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Invalid model size '{model_size}'. Choose from {list(MODEL_SIZES.keys())}")

    W = MODEL_SIZES[model_size]['W']
    D = MODEL_SIZES[model_size]['D']
    R = MODEL_SIZES[model_size]['R']

    def d(x): return max(round(x * D), 1)

    # BACKBONE
    BACKBONE_CONFIG = [
        [ 
            ('conv', {'in_channels': 3, 'out_channels': int(64 * W), 'kernel_size': 3, 'stride': 2, 'padding': 1}),
            ('conv', {'in_channels': int(64 * W), 'out_channels': int(128 * W), 'kernel_size': 3, 'stride': 2, 'padding': 1}),
            ('c2f', {'in_channels': int(128 * W), 'out_channels': int(128 * W), 'shortcut': True, 'n': d(3)}),
            ('conv', {'in_channels': int(128 * W), 'out_channels': int(256 * W), 'kernel_size': 3, 'stride': 2, 'padding': 1}),
            ('c2f', {'in_channels': int(256 * W), 'out_channels': int(256 * W), 'shortcut': True, 'n': d(6)}),
        ],
        [
            ('conv', {'in_channels': int(256 * W), 'out_channels': int(512 * W), 'kernel_size': 3, 'stride': 2, 'padding': 1}),
            ('c2f', {'in_channels': int(512 * W), 'out_channels': int(512 * W), 'shortcut': True, 'n': d(6)}),
        ],
        [
            ('conv', {'in_channels': int(512 * W), 'out_channels': int(512 * W), 'kernel_size': 3, 'stride': 2, 'padding': 1}),
            ('c2f', {'in_channels': int(512 * W), 'out_channels': int(512 * W), 'shortcut': True, 'n': d(6)}),
            ('sppf', {'in_channels': int(512 * W), 'out_channels': int(512 * W * R)}),
        ]
    ]

    # NECK
    NECK_CONFIG = [
        # FPN Down
        [
            [
                ('u', {'scale_factor': 2, 'mode': 'nearest'}),
                ('c', {'dim': 1}),
                ('c2f', {'in_channels': int(512 * W * (1 + R)), 'out_channels': int(512 * W), 'shortcut': False, 'n': d(3)}),
            ],
            [
                ('u', {'scale_factor': 2, 'mode': 'nearest'}),
                ('c', {'dim': 1}),
                ('c2f', {'in_channels': int(768 * W), 'out_channels': int(256 * W), 'shortcut': False, 'n': d(3)}),
            ]
        ],
        # FPN Up
        [
            [
                ('conv', {'in_channels': int(256 * W), 'out_channels': int(256 * W), 'kernel_size': 3, 'stride': 2, 'padding': 1}),
                ('c', {'dim': 1}),
                ('c2f', {'in_channels': int(768 * W), 'out_channels': int(512 * W), 'shortcut': False, 'n': d(3)}),
            ],
            [
                ('conv', {'in_channels': int(512 * W), 'out_channels': int(512 * W), 'kernel_size': 3, 'stride': 2, 'padding': 1}),
                ('c', {'dim': 1}),
                ('c2f', {'in_channels': int(512 * W * (1 + R)), 'out_channels': int(512 * W * R), 'shortcut': False, 'n': d(3)}),
            ]
        ]
    ]

    # HEAD
    def head_block(in_ch):
        return [
            [   # Classification
                ('conv', {'in_channels': in_ch, 'out_channels': in_ch, 'kernel_size': 3, 'stride': 1, 'padding': 1}),
                ('conv', {'in_channels': in_ch, 'out_channels': in_ch, 'kernel_size': 3, 'stride': 1, 'padding': 1}),
                ('conv2d', {'in_channels': in_ch, 'out_channels': num_classes, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True}),
            ],
            [   # Regression
                ('conv', {'in_channels': in_ch, 'out_channels': in_ch, 'kernel_size': 3, 'stride': 1, 'padding': 1}),
                ('conv', {'in_channels': in_ch, 'out_channels': in_ch, 'kernel_size': 3, 'stride': 1, 'padding': 1}),
                ('conv2d', {'in_channels': in_ch, 'out_channels': 4 * num_boxes, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True}),
            ]
        ]

    HEAD_CONFIG = [
        head_block(int(256 * W)),        # 80x80
        head_block(int(512 * W)),        # 40x40
        head_block(int(512 * W * R)),    # 20x20
    ]

    return [BACKBONE_CONFIG, NECK_CONFIG, HEAD_CONFIG]

yolo8vn = get_yolo_config('n')
yolo8vs = get_yolo_config('s')
yolo8vm = get_yolo_config('m')