import math
import numpy as np
from collections import OrderedDict, _OrderedDictItemsView

from omni.utils.loguru import logger

# computed from above code
# S = 8294400
ASPECT_RATIO_4K = {
    "0.38": (1764, 4704),
    "0.43": (1886, 4400),
    "0.48": (1996, 4158),
    "0.50": (2036, 4072),
    "0.53": (2096, 3960),
    "0.54": (2118, 3918),
    "0.62": (2276, 3642),
    "0.56": (2160, 3840),  # base
    "0.67": (2352, 3528),
    "0.75": (2494, 3326),
    "1.00": (2880, 2880),
    "1.33": (3326, 2494),
    "1.50": (3528, 2352),
    "1.78": (3840, 2160),
    "1.89": (3958, 2096),
    "2.00": (4072, 2036),
    "2.08": (4156, 1994),
}

# S = 3686400
ASPECT_RATIO_2K = {
    "0.38": (1176, 3136),
    "0.43": (1256, 2930),
    "0.48": (1330, 2770),
    "0.50": (1358, 2716),
    "0.53": (1398, 2640),
    "0.54": (1412, 2612),
    "0.56": (1440, 2560),  # base
    "0.62": (1518, 2428),
    "0.67": (1568, 2352),
    "0.75": (1662, 2216),
    "1.00": (1920, 1920),
    "1.33": (2218, 1664),
    "1.50": (2352, 1568),
    "1.78": (2560, 1440),
    "1.89": (2638, 1396),
    "2.00": (2716, 1358),
    "2.08": (2772, 1330),
}

# S = 2073600
ASPECT_RATIO_1080P = {
    "0.38": (882, 2352),
    "0.43": (942, 2198),
    "0.48": (998, 2080),
    "0.50": (1018, 2036),
    "0.53": (1048, 1980),
    "0.54": (1058, 1958),
    "0.56": (1080, 1920),  # base
    "0.62": (1138, 1820),
    "0.67": (1176, 1764),
    "0.75": (1248, 1664),
    "1.00": (1440, 1440),
    "1.33": (1662, 1246),
    "1.50": (1764, 1176),
    "1.78": (1920, 1080),
    "1.89": (1980, 1048),
    "2.00": (2036, 1018),
    "2.08": (2078, 998),
}

# S = 921600
ASPECT_RATIO_720P = {
    "0.38": (588, 1568),
    "0.43": (628, 1466),
    "0.48": (666, 1388),
    "0.50": (678, 1356),
    "0.53": (698, 1318),
    "0.54": (706, 1306),
    "0.56": (720, 1280),  # base
    "0.62": (758, 1212),
    "0.67": (784, 1176),
    "0.75": (832, 1110),
    "1.00": (960, 960),
    "1.33": (1108, 832),
    "1.50": (1176, 784),
    "1.78": (1280, 720),
    "1.89": (1320, 698),
    "2.00": (1358, 680),
    "2.08": (1386, 666),
}

# S = 409920
ASPECT_RATIO_480P = {
    "0.38": (392, 1046),
    "0.43": (420, 980),
    "0.48": (444, 925),
    "0.50": (452, 904),
    "0.53": (466, 880),
    "0.54": (470, 870),
    "0.56": (480, 854),  # base
    "0.62": (506, 810),
    "0.67": (522, 784),
    "0.75": (554, 738),
    "1.00": (640, 640),
    "1.33": (740, 555),
    "1.50": (784, 522),
    "1.78": (854, 480),
    "1.89": (880, 466),
    "2.00": (906, 454),
    "2.08": (924, 444),
}

# S = 230400
ASPECT_RATIO_360P = {
    "0.38": (294, 784),
    "0.43": (314, 732),
    "0.48": (332, 692),
    "0.50": (340, 680),
    "0.53": (350, 662),
    "0.54": (352, 652),
    "0.56": (360, 640),  # base
    "0.62": (380, 608),
    "0.67": (392, 588),
    "0.75": (416, 554),
    "1.00": (480, 480),
    "1.33": (554, 416),
    "1.50": (588, 392),
    "1.78": (640, 360),
    "1.89": (660, 350),
    "2.00": (678, 340),
    "2.08": (692, 332),
}

# S = 102240
ASPECT_RATIO_240P = {
    "0.38": (196, 522),
    "0.43": (210, 490),
    "0.48": (222, 462),
    "0.50": (226, 452),
    "0.53": (232, 438),
    "0.54": (236, 436),
    "0.56": (240, 426),  # base
    "0.62": (252, 404),
    "0.67": (262, 393),
    "0.75": (276, 368),
    "1.00": (320, 320),
    "1.33": (370, 278),
    "1.50": (392, 262),
    "1.78": (426, 240),
    "1.89": (440, 232),
    "2.00": (452, 226),
    "2.08": (462, 222),
}

# S = 36864
ASPECT_RATIO_144P = {
    "0.38": (117, 312),
    "0.43": (125, 291),
    "0.48": (133, 277),
    "0.50": (135, 270),
    "0.53": (139, 262),
    "0.54": (141, 260),
    "0.56": (144, 256),  # base
    "0.62": (151, 241),
    "0.67": (156, 234),
    "0.75": (166, 221),
    "1.00": (192, 192),
    "1.33": (221, 165),
    "1.50": (235, 156),
    "1.78": (256, 144),
    "1.89": (263, 139),
    "2.00": (271, 135),
    "2.08": (277, 132),
}

# from PixArt
# S = 8294400
ASPECT_RATIO_2880 = {
    "0.25": (1408, 5760),
    "0.26": (1408, 5568),
    "0.27": (1408, 5376),
    "0.28": (1408, 5184),
    "0.32": (1600, 4992),
    "0.33": (1600, 4800),
    "0.34": (1600, 4672),
    "0.40": (1792, 4480),
    "0.42": (1792, 4288),
    "0.47": (1920, 4096),
    "0.49": (1920, 3904),
    "0.51": (1920, 3776),
    "0.55": (2112, 3840),
    "0.59": (2112, 3584),
    "0.68": (2304, 3392),
    "0.72": (2304, 3200),
    "0.78": (2496, 3200),
    "0.83": (2496, 3008),
    "0.89": (2688, 3008),
    "0.93": (2688, 2880),
    "1.00": (2880, 2880),
    "1.07": (2880, 2688),
    "1.12": (3008, 2688),
    "1.21": (3008, 2496),
    "1.28": (3200, 2496),
    "1.39": (3200, 2304),
    "1.47": (3392, 2304),
    "1.70": (3584, 2112),
    "1.82": (3840, 2112),
    "2.03": (3904, 1920),
    "2.13": (4096, 1920),
    "2.39": (4288, 1792),
    "2.50": (4480, 1792),
    "2.92": (4672, 1600),
    "3.00": (4800, 1600),
    "3.12": (4992, 1600),
    "3.68": (5184, 1408),
    "3.82": (5376, 1408),
    "3.95": (5568, 1408),
    "4.00": (5760, 1408),
}

# S = 4194304
ASPECT_RATIO_2048 = {
    "0.25": (1024, 4096),
    "0.26": (1024, 3968),
    "0.27": (1024, 3840),
    "0.28": (1024, 3712),
    "0.32": (1152, 3584),
    "0.33": (1152, 3456),
    "0.35": (1152, 3328),
    "0.40": (1280, 3200),
    "0.42": (1280, 3072),
    "0.48": (1408, 2944),
    "0.50": (1408, 2816),
    "0.52": (1408, 2688),
    "0.57": (1536, 2688),
    "0.60": (1536, 2560),
    "0.68": (1664, 2432),
    "0.72": (1664, 2304),
    "0.78": (1792, 2304),
    "0.82": (1792, 2176),
    "0.88": (1920, 2176),
    "0.94": (1920, 2048),
    "1.00": (2048, 2048),
    "1.07": (2048, 1920),
    "1.13": (2176, 1920),
    "1.21": (2176, 1792),
    "1.29": (2304, 1792),
    "1.38": (2304, 1664),
    "1.46": (2432, 1664),
    "1.67": (2560, 1536),
    "1.75": (2688, 1536),
    "2.00": (2816, 1408),
    "2.09": (2944, 1408),
    "2.40": (3072, 1280),
    "2.50": (3200, 1280),
    "2.89": (3328, 1152),
    "3.00": (3456, 1152),
    "3.11": (3584, 1152),
    "3.62": (3712, 1024),
    "3.75": (3840, 1024),
    "3.88": (3968, 1024),
    "4.00": (4096, 1024),
}

# S = 1048576
ASPECT_RATIO_1024 = {
    "0.25": (512, 2048),
    "0.26": (512, 1984),
    "0.27": (512, 1920),
    "0.28": (512, 1856),
    "0.32": (576, 1792),
    "0.33": (576, 1728),
    "0.35": (576, 1664),
    "0.40": (640, 1600),
    "0.42": (640, 1536),
    "0.48": (704, 1472),
    "0.50": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.60": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.00": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "2.00": (1408, 704),
    "2.09": (1472, 704),
    "2.40": (1536, 640),
    "2.50": (1600, 640),
    "2.89": (1664, 576),
    "3.00": (1728, 576),
    "3.11": (1792, 576),
    "3.62": (1856, 512),
    "3.75": (1920, 512),
    "3.88": (1984, 512),
    "4.00": (2048, 512),
}

# S = 262144
ASPECT_RATIO_512 = {
    '0.053': (128, 2432),
    '0.077': (128, 1664),
    '0.167': (192, 1152),
    '0.250': (256, 1024),
    '0.333': (320, 960),
    '0.417': (320, 768),
    '0.500': (384, 768),
    '0.667': (384, 576),
    '1.000': (512, 512),
    '1.500': (576, 384),
    '2.000': (768, 384),
    '2.400': (768, 320),
    '3.000': (960, 320),
    '4.000': (1024, 256),
    '6.000': (1152, 192),
    '13.000': (1664, 128),
    '19.000': (2432, 128),
}

# S = 65536
ASPECT_RATIO_256 = {
    '0.053': (64, 1216),
    '0.077': (64, 832),
    '0.250': (128, 512),
    '0.256': (128, 448),
    '0.500': (192, 384),
    '1.000': (256, 256),
    '2.000': (384, 192),
    '3.500': (448, 128),
    '4.000': (512, 128),
    '13.000': (832, 64),
    '19.000': (1216, 64),
}

# S = 16384
ASPECT_RATIO_128 = {
    '0.250': (64, 256),
    '1.000': (128, 128),
    '4.000': (256, 64),
}


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return closest_ratio


ASPECT_RATIOS = {
    "128": (16384, ASPECT_RATIO_128),
    "144p": (36864, ASPECT_RATIO_144P),
    "256": (65536, ASPECT_RATIO_256),
    "240p": (102240, ASPECT_RATIO_240P),
    "360p": (230400, ASPECT_RATIO_360P),
    "512": (262144, ASPECT_RATIO_512),
    "480p": (409920, ASPECT_RATIO_480P),
    "720p": (921600, ASPECT_RATIO_720P),
    "1024": (1048576, ASPECT_RATIO_1024),
    "1080p": (2073600, ASPECT_RATIO_1080P),
    "2k": (3686400, ASPECT_RATIO_2K),
    "2048": (4194304, ASPECT_RATIO_2048),
    "2880": (8294400, ASPECT_RATIO_2880),
    "4k": (8294400, ASPECT_RATIO_4K),
}


def find_approximate_hw(hw, hw_dict, approx=0.8):
    for k, v in hw_dict.items():
        if hw >= v * approx:
            return k
    return None


def find_closet_smaller_bucket(t, t_dict, frame_interval):
    # process image
    if t == 1:
        if 1 in t_dict:
            return 1
        else:
            return None
    # process video
    for k, v in t_dict.items():
        if t >= v * frame_interval and v != 1:
            return k
    return None


class Bucket:
    def __init__(self, bucket_config):
        logger.info(f"Bucket configs: {bucket_config}")
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        # wrap config with OrderedDict
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)

        for key in bucket_names:
            bucket_probs[key] = bucket_config[key][0]
            bucket_bs[key] = bucket_config[key][1]

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        ar_criteria = dict()
        bucket_id = OrderedDict()
        bucket_id_cnt = 0
        for k1, v1 in bucket_probs.items():
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            ar_criteria[k1] = dict()
            bucket_id[k1] = bucket_id_cnt
            bucket_id_cnt += 1
            for k2, v2 in ASPECT_RATIOS[k1][1].items():
                ar_criteria[k1][k2] = v2
                num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.hw_criteria = hw_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket

    def get_bucket_id(self, H, W, seed=None):
        resolution = H * W
        approx = 0.8

        for hw_id, prob in self.bucket_probs.items():
            if resolution < self.hw_criteria[hw_id] * approx:
                continue

            rng = np.random.default_rng(seed + self.bucket_id[hw_id])
            if isinstance(prob, tuple):
                prob = prob[0]
            if prob >= 1 or rng.random() < prob:
                break

        ar_criteria = self.ar_criteria[hw_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, ar_id

    def get_hw(self, bucket_id):
        assert len(bucket_id) == 2
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]]
        return H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id]

    def __len__(self):
        return self.num_bucket


def closet_smaller_bucket(value, bucket):
    for i in range(1, len(bucket)):
        if value < bucket[i]:
            return bucket[i - 1]
    return bucket[-1]


def find_approximate_hw(hw, hw_dict, approx=0.8):
    for k, v in hw_dict.items():
        if hw >= v * approx:
            return k
    return None


def find_closet_smaller_bucket(t, t_dict, frame_interval):
    # process image
    if t == 1:
        if 1 in t_dict:
            return 1
        else:
            return None
    # process video
    for k, v in t_dict.items():
        if t >= v * frame_interval and v != 1:
            return k
    return None


ASPECT_RATIO_MAP = {
    "0.10": "0.10",
    "0.15": "0.15",
    "0.20": "0.20",
    "0.25": "0.25",
    "0.30": "0.30",
    "0.35": "0.35",
    "3:8": "0.38",
    "9:21": "0.43",
    "12:25": "0.48",
    "1:2": "0.50",
    "9:17": "0.53",
    "27:50": "0.54",
    "9:16": "0.56",
    "5:8": "0.62",
    "2:3": "0.67",
    "3:4": "0.75",
    "1:1": "1.00",
    "4:3": "1.33",
    "3:2": "1.50",
    "8:5": "1.60",
    "16:9": "1.78",
    "50:27": "1.85",
    "17:9": "1.89",
    "2:1": "2.00",
    "25:12": "2.08",
    "21:9": "2.33",
    "8:3": "2.67",
    "1/0.35": "2.86",
    "1/0.30": "3.33",
    "1/0.25": "4.00",
    "1/0.20": "5.00",
    "1/0.15": "6.67",
    "1/0.10": "10.0"
}


def get_h_w(a, ts, eps=1e-4):
    h = (ts * a) ** 0.5
    h = h + eps
    h = math.ceil(h) if math.ceil(h) % 2 == 0 else math.floor(h)
    w = h / a
    w = w + eps
    w = math.ceil(w) if math.ceil(w) % 2 == 0 else math.floor(w)
    return h, w


def get_aspect_ratios_dict(ars, ts=360 * 640):
    est = {f"{a:.2f}": get_h_w(a, ts) for a in ars}
    return est


def get_aspect_ratios_dict2(stride, area, aspect_ratio):
    ret_dict = {}
    max_stride = math.ceil(area / stride)
    for i in range(max_stride):
        h = stride * (i + 1)
        min_w, max_w = h / aspect_ratio, h * aspect_ratio
        min_w, max_w = (min_w, max_w) if min_w < max_w else (max_w, min_w)
        min_w = max(math.ceil(area * 0.8 / h / stride) * stride, min_w)
        max_w = min(math.floor(area * 1.2 / h / stride) * stride, max_w)
        w_list = list(range(int(min_w), int(max_w+1), stride))
        for w in w_list:
            ret_dict[f"{h / w:.3f}"] = (h, w)

    ret_dict = {k: v for k, v in sorted(ret_dict.items(), key=lambda item: float(item[0]))}

    return ret_dict


def custom_print_dict(d):
    print("{")
    for key, value in d.items():
        print(f"'{key}': {value},")
    print("}")


if __name__ == '__main__':
    import pprint

    ret_dict = get_aspect_ratios_dict2(64, 512 ** 2, 0.05)
    
    custom_print_dict(ret_dict)
    print(len(ret_dict))