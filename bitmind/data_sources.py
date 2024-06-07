# datasets used in miner training and validator challenges, more to come

DATASET_META = {
    'real': [
        {'name': 'dalle-mini/open-images', 'create_splits': False},
        {'name': 'merkol/ffhq-256', 'create_splits': True},
        {'name': 'jlbaker361/flickr_humans_20k', 'create_splits': True},
        {'name': 'saitsharipov/CelebA-HQ', 'create_splits': True}
    ],
    'fake': [
        {'name': 'bitmind/RealVisXL_V4.0_images', 'create_splits': True}
    ]
}
