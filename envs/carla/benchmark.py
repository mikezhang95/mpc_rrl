
import random
import numpy as np


class DrivingTask(object):

    def __init__(self, task_name="straight", domain_random=False):
        self.task_name = task_name
        self.domain_random = domain_random

    # changable environmental parameters
    def get_changable_parameters(self,):
        return ["town_pose",  "tire_friction", "drag_coefficient", "damping_rate", "moi", "final_ratio"]

    def get_test_range(self, param_name):
        assert param_name in self.get_changable_parameters()
        mode = "test"
        get_class = getattr(self, f"_{param_name}")
        return get_class(mode)

    # sample a env setting for training
    def sample_train_setting(self):
        mode = "train"
        param_dict = {}
        for param_name in self.get_changable_parameters():
            get_class = getattr(self, f"_{param_name}")
            values = get_class(mode)
            value = random.choice(values)
            param_dict[param_name] = value
        return param_dict

    # === town, starting point ====
    def _town_pose(self, mode="train"):
        towns_poses = []
        def _towns(mode="train"):
            if mode == "train":
                if self.domain_random:
                    return ["Town01", "Town02", "Town03"]
                else:
                    return ["Town01"]
            elif mode == "test":
                return ["Town04", "Town05", "Town06"]

        towns = _towns(mode)
        towns_poses = [(town, [0,1]) for town in towns]
        # for town_name in towns:
        #     poses_class = getattr(self, f"_poses_{town_name.lower()}")
        #     poses = poses_class(self.task_name)
        #     towns_poses.extend([(town_name, p)for p in poses])
        return towns_poses

    def _road_friction(self, mode="train"):
        if mode == "train":
            return [3.5]
        elif mode == "test":
            return [2.0, 3.0, 3.5, 4.0, 5.0]

    def _car_mass(self, mode="train"):
        if mode == "train":
            if self.domain_random:
                return [500, 1845, 1900]
            else:
                return [1845]
        elif mode == "test":
            return [100, 1000, 2000]
            # return [1200, 1700, 1845, 2000, 2500]

    def _moi(self, mode="train"):
        if mode == "train":
            if self.domain_random:
                return [0.5, 1.0, 1.5]
            else:
                return [1.0]
        elif mode == "test":
            return [0.4, 1.3, 1.9]
            # return [0.1, 1.0, 2.1]


    def _final_ratio(self, mode="train"):
        if mode == "train":
            if self.domain_random:
                return [3.0, 4.0, 8.0]
            else:
                return [4.0]
        elif mode == "test":
            return [2.0, 5.0, 10.0]
            # return [2.0, 4.0, 25.0]

    def _radius(self, mode="train"):
        if mode == "train":
            return [37]
        elif mode == "test":
            return [34, 36, 37, 38, 40]

    def _drag_coefficient(self, mode="train"):
        if mode == "train":
            if self.domain_random:
                return [0.0015, 0.15, 15]
            else:
                return [0.15]
        elif mode == "test":
            return [1e-4, 0.2, 100]
            # return [1e-5, 0.15, 100]

    def _tire_friction(self, mode="train"):
        if mode == "train":
            if self.domain_random:
                return [2.0, 3.5, 3.75]
            else:
                return [3.5]
        elif mode == "test":
            return [0.5, 2.25, 4.0]
            # return [0.3, 3.5, 3.9]

    def _damping_rate(self, mode="train"):
        if mode == "train":
            if self.domain_random:
                return [0.1, 0.25, 1.1]
            else:
                return [0.25]
        elif mode == "test":
            return [5e-3, 0.5, 5.0]
            # return [1e-4, 0.25, 50.0]

    def _car_type(self, mode="train"):
        if mode == "train":
            if self.domain_random:
                return ["model3", "wrangler_rubicon", "audi"]
            else:
                return ["model3"]
        elif mode == "test":
            return ["low_rider", "ambulance", "mkz_2017"]
            # return ["model3", "mkz_2017", "wrangler_rubicon", "low_rider", "ambulance"] # "firetruck", "crossbike"


    def _poses_town01(self, task_name="straight"):
        """
        Each matrix is a new task. We have all the four tasks
        """
        def _poses_straight():
            return [[36, 40], [39, 35], [110, 114], [7, 3], [0, 4],
                    [68, 50], [61, 59], [47, 64], [147, 90], [33, 87],
                    [26, 19], [80, 76], [45, 49], [55, 44], [29, 107],
                    [95, 104], [84, 34], [53, 67], [22, 17], [91, 148],
                    [20, 107], [78, 70], [95, 102], [68, 44], [45, 69]]

        def _poses_one_curve():
            return [[138, 17], [47, 16], [26, 9], [42, 49], [140, 124],
                    [85, 98], [65, 133], [137, 51], [76, 66], [46, 39],
                    [40, 60], [0, 29], [4, 129], [121, 140], [2, 129],
                    [78, 44], [68, 85], [41, 102], [95, 70], [68, 129],
                    [84, 69], [47, 79], [110, 15], [130, 17], [0, 17]]

        def _poses_navigation():
            return [[105, 29], [27, 130], [102, 87], [132, 27], [24, 44],
                    [96, 26], [34, 67], [28, 1], [140, 134], [105, 9],
                    [148, 129], [65, 18], [21, 16], [147, 97], [42, 51],
                    [30, 41], [18, 107], [69, 45], [102, 95], [18, 145],
                    [111, 64], [79, 45], [84, 69], [73, 31], [37, 81]]

        if task_name == "straight": 
            return _poses_straight()
        elif task_name == "turn": 
            return _poses_one_curve()
        elif task_name == "navigation":
            return _poses_navigation(),
        else:
            raise notimplementederror

    def _poses_town02(self, task_name="straight"):

        def _poses_straight():
            return [[38, 34], [4, 2], [12, 10], [62, 55], [43, 47],
                    [64, 66], [78, 76], [59, 57], [61, 18], [35, 39],
                    [12, 8], [0, 18], [75, 68], [54, 60], [45, 49],
                    [46, 42], [53, 46], [80, 29], [65, 63], [0, 81],
                    [54, 63], [51, 42], [16, 19], [17, 26], [77, 68]]

        def _poses_one_curve():
            return [[37, 76], [8, 24], [60, 69], [38, 10], [21, 1],
                    [58, 71], [74, 32], [44, 0], [71, 16], [14, 24],
                    [34, 11], [43, 14], [75, 16], [80, 21], [3, 23],
                    [75, 59], [50, 47], [11, 19], [77, 34], [79, 25],
                    [40, 63], [58, 76], [79, 55], [16, 61], [27, 11]]

        def _poses_navigation():
            return [[19, 66], [79, 14], [19, 57], [23, 1],
                    [53, 76], [42, 13], [31, 71], [33, 5],
                    [54, 30], [10, 61], [66, 3], [27, 12],
                    [79, 19], [2, 29], [16, 14], [5, 57],
                    [70, 73], [46, 67], [57, 50], [61, 49], [21, 12],
                    [51, 81], [77, 68], [56, 65], [43, 54]]

        if task_name == "straight": 
            return _poses_straight()
        elif task_name == "turn": 
            return _poses_one_curve()
        elif task_name == "navigation":
            return _poses_navigation(),
        else:
            raise NotImplementedError
