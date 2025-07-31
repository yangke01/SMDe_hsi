from collections import namedtuple

TrainSchedule = namedtuple('TrainSchedule', ['base_lr', 'stage1_epoch', 'lr_schedule', 'max_epochs'])


denoise_yknet_complex = TrainSchedule(
    max_epochs=150,
    stage1_epoch=30,
    base_lr=1e-3,
    lr_schedule={
        20: 1e-4,
        31: 5e-5,
        40: 1e-5,
        48: 5e-6,
        55: 1e-6,
    },
)

denoise_ssumamba_complex = TrainSchedule(
    max_epochs=80,
    stage1_epoch=30,
    base_lr=1e-4,
    lr_schedule={
        20: 1e-4,
        40: 1e-4,
        60: 1e-5,
        70: 1e-6
    },
)


denoise_hybrid_complex = TrainSchedule(
    max_epochs=140,
    stage1_epoch=30,
    base_lr=1e-4,
    lr_schedule={
        20: 1e-4,
        29: 5e-5,
        36: 1e-5,
        60: 5e-6,
        90: 1e-6,
        # 121: 5e-7,
        106: 1e-7,
    },
)

denoise_tm_complex = TrainSchedule(
    max_epochs=120,
    stage1_epoch=30,
    base_lr=1e-4,
    lr_schedule={
        20: 1e-4,
        40: 1e-4,
        60: 1e-4,
        80: 1e-4,
        90: 5e-5,
        95: 1e-5,
        100: 5e-6,
        105: 1e-6,
    },
)

denoise_mixer_complex = TrainSchedule(
    max_epochs=120,
    stage1_epoch=30,
    base_lr=1e-4,
    lr_schedule={
        20: 1e-4,
        40: 1e-4,
        80: 5e-5,
        95: 1e-5,
        105: 5e-6,
        110: 1e-6,
    },
)

denoise_default = TrainSchedule(
    max_epochs=80,
    stage1_epoch=30,
    base_lr=1e-3,
    lr_schedule={
        0: 1e-3,
        20: 1e-4,
        30: 1e-3,
        45: 1e-4,
        55: 5e-5,
        60: 1e-5,
        65: 5e-6,
        75: 1e-6,
    },
)

denoise_complex_default = TrainSchedule(
    max_epochs=110,
    stage1_epoch=30,
    base_lr=1e-3,
    lr_schedule={
        80: 1e-3,
        90: 5e-4,
        95: 1e-4,
        100: 5e-5,
        105: 1e-5,
    },
)

denoise_after = TrainSchedule(
    max_epochs=200,
    stage1_epoch=30,
    base_lr=1e-6,
    lr_schedule={
        # 0: 1e-6,
        # 80: 1e-3,
        # 90: 5e-4,
        # 95: 1e-4,
        # 100: 5e-5,
        # 105: 1e-5,
        96: 1e-6,
    },
)