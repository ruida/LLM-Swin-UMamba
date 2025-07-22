import os
import json
import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader2D(nnUNetDataLoaderBase):
    def __init__(self, *args, reports_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reports_dir = reports_dir  # absolute or relative path to .json report files

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        reports = []

        for j, current_key in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(current_key)
            case_properties.append(properties)

            # Load report from JSON
            report_text = ""
            if self.reports_dir is not None:
                # Extract the base file name (e.g., "003225_05_01_144.json")
                image_filename = os.path.basename(properties["data_file"])
                base_json = image_filename.replace("_0000.npy", ".json")
                json_path = os.path.join(self.reports_dir, base_json)
                if os.path.isfile(json_path):
                    try:
                        with open(json_path, "r") as f:
                            report_text = json.load(f).get("report", "")
                    except Exception as e:
                        print(f"⚠️ Error loading report for {json_path}: {e}")
                else:
                    print(f"⚠️ Report not found: {json_path}")
            reports.append(report_text)

            if not force_fg:
                selected_class_or_region = self.annotated_classes_key if self.has_ignore else None
            else:
                eligible_classes_or_regions = [i for i in properties['class_locations'].keys()
                                               if len(properties['class_locations'][i]) > 0]
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False
                       for i in eligible_classes_or_regions]
                if any(tmp) and len(eligible_classes_or_regions) > 1:
                    eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                selected_class_or_region = eligible_classes_or_regions[
                    np.random.choice(len(eligible_classes_or_regions))] if eligible_classes_or_regions else None

            selected_slice = np.random.choice(
                properties['class_locations'][selected_class_or_region][:, 1]
            ) if selected_class_or_region is not None else np.random.choice(len(data[0]))

            data = data[:, selected_slice]
            seg = seg[:, selected_slice]

            class_locations = {
                selected_class_or_region: properties['class_locations'][selected_class_or_region][
                    properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
            } if selected_class_or_region is not None else None

            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
                                               class_locations, overwrite_class=selected_class_or_region)

            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {
            'data': data_all,
            'seg': seg_all,
            'properties': case_properties,
            'keys': selected_keys,
            'reports': reports
        }


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset004_Hippocampus/2d'
    reports = '/absolute/path/to/reports'  # ← replace with actual path
    ds = nnUNetDataset(folder, None, 1000)
    dl = nnUNetDataLoader2D(ds, 366, (65, 65), (56, 40), 0.33, None, None, reports_dir=reports)
    a = next(dl)
    print(a['reports'])  # to confirm
