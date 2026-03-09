from omegaconf import OmegaConf

from ...utils.geo import BoundaryBox

location_to_params = {
    # New Cities
    "sanfrancisco":{
        "bbox": BoundaryBox((37.780074, -122.447399), (37.801918, -122.420895)),
        "filters": {"is_pano": True, "start_captured_at": "2020-01-01T01:00:00Z", "creator_username": "GoProMAX3M"},
        "osm_file": "sanfrancisco.osm",
    },
    "berlin": {
        "bbox": BoundaryBox((52.473920, 13.428936), (52.486693, 13.449853)),
        "filters": {"is_pano": True, "start_captured_at": "2020-01-01T01:00:00Z"},
        "osm_file": "berlin.osm",
    },
    "washington": {
        "bbox": BoundaryBox(
            (38.901344, -77.038570), (38.912748, -77.023916)),
        "filters": {"is_pano": True, "start_captured_at": "2020-01-01T01:00:00Z"},
        "osm_file": "washington.osm",
    },
    "montrouge": {
        "bbox": BoundaryBox((48.808548, 2.321291), (48.829312, 2.360859)),
        "filters": {"is_pano": True,"start_captured_at": "2020-01-01T01:00:00Z"},
        "osm_file": "montrouge.osm",
    },
    "toulouse": {
        "bbox": BoundaryBox((43.591434, 1.429457), (43.61343, 1.456653)),
        "filters": {"is_pano": True,"start_captured_at": "2020-01-01T01:00:00Z"},
        "osm_file": "toulouse.osm",
    },
    "detroit":{
        "bbox": BoundaryBox((42.329514,-83.062563), (42.343041,-83.042753)),
        "filters": {"is_pano": True, "start_captured_at": "2020-01-01T01:00:00Z", "creator_username": "codgis"},
        "osm_file": "detroit.osm",
    },
    "chicago":{
        "bbox": BoundaryBox((41.932461,-87.681438), (41.95358,-87.649561)),
        "filters": {"is_pano": True, "start_captured_at": "2020-01-01T01:00:00Z", "creator_username": "epicspongee"},
        "osm_file": "chicago.osm",
    },

    # Data of Mount Vernon is used to test the generalization of the model trained on the above cities. It is not used for training or validation.
    "MountVernon":{
        "bbox": BoundaryBox((40.891404,-73.854904),(40.919845,-73.816109)),
        "filters": {"is_pano":True},
        "osm_file": "MountVernon.osm",
    },

    # Add any new region/city here:
    # "location_name": {
    #     "bbox": BoundaryBox((lat_min, long_min), (lat_max, long_max)),
    #     "filters": {"is_pano": True},
    #     # or other filters like creator_username, model, etc.
    #     # all described at https://www.mapillary.com/developer/api-documentation#image
    # }
    # Other fields (bbox_val, osm_file) will be deduced automatically.
}

default_cfg = OmegaConf.create(
    {
        "downsampling_resolution_meters": 3,
        "target_num_val_images": 0.2,
        "val_train_margin_meters": 25,
        "max_num_train_images": 50_000,
        "max_image_size": 512,
        "do_legacy_pano_offset": True,
        "min_dist_between_keyframes": 4,
        "tiling": {
            "tile_size": 128,
            "margin": 128,
            "ppm": 2,
        },
    }
)
