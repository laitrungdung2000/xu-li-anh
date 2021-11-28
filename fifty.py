import fiftyone as fo
dataset = fo.Dataset(
    "data/coco/2017/1.1.0",
    dataset_type=fo.types.COCODetectionDataset,
    name="dataset_info"
)