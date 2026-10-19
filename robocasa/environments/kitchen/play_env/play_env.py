import os
import numpy as np
from typing import List, Dict, Any

import robocasa
from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU
import robocasa.macros as macros

macros.SHOW_SITES = False


class StyleMixin:
    REQUIRED_STYLE_ID = None
    """Mixin for style environments"""

    def set_ep_meta(self, ep_meta):
        assert (
            self.REQUIRED_STYLE_ID is not None
        ), "REQUIRED_STYLE_ID must be set for style environments"
        if "style_id" in ep_meta:  # Maybe we should always enforce this?
            ep_meta["style_id"] = self.REQUIRED_STYLE_ID
        return super().set_ep_meta(ep_meta)


class LayoutMixin:
    REQUIRED_LAYOUT_ID = None
    """Mixin for layout environments"""

    def set_ep_meta(self, ep_meta):
        assert (
            self.REQUIRED_LAYOUT_ID is not None
        ), "REQUIRED_LAYOUT_ID must be set for layout environments"
        if "layout_id" in ep_meta:
            ep_meta["layout_id"] = self.REQUIRED_LAYOUT_ID
        return super().set_ep_meta(ep_meta)


class BaseEnvForPlay(Kitchen):
    EXCLUDE_LAYOUTS = []

    # Layout-specific constants
    LAYOUTS_WITHOUT_CAB_SINK = [1, 3, 4, 5, 6]

    # Common style IDs for eval_mode
    TRAIN_STYLE_IDS = ["001_l1", "038_l1", "014_l1", "015_l1", "019_l1", "037_l1"]

    # Fixture removal keys
    FIXTURES_TO_REMOVE = [
        "toaster_main_group",
        "toaster_right_group",
        "toaster_left_group",
        "coffee_machine_left_group",
        "coffee_machine_right_group",
        "coffee_machine_main_group",
    ]

    def __init__(self, *args, **kwargs):
        # check what is returned in args and kwargs
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self) -> None:
        """Setup kitchen fixture references"""
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter_stove = self.register_fixture_ref(
            # "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
            "counter_stove",
            dict(id=FixtureType.COUNTER, ref=self.stove),
        )
        self.counter_sink = self.register_fixture_ref(
            # "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=[0.30, 0.40])
            "counter_sink",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        if self.layout_id not in self.LAYOUTS_WITHOUT_CAB_SINK:
            self.cab_sink = self.register_fixture_ref(
                "cab_sink", dict(id=FixtureType.CABINET, ref=self.sink, loc="above")
            )
        else:
            self.cab_sink = None

    def split_type(self) -> str:
        """
        Returns the split type of the environment.
        """
        return "train"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "There is no language here."
        return ep_meta

    def _remove_specific_fixtures(self) -> None:
        self.fixtures = {
            key: value
            for key, value in self.fixtures.items()
            if key not in self.FIXTURES_TO_REMOVE
        }
        self.fixture_cfgs = [
            elem
            for elem in self.fixture_cfgs
            if elem["name"] not in self.FIXTURES_TO_REMOVE
        ]
        return

    def _get_obj_cfgs(self) -> List[Dict[str, Any]]:
        """Get object configurations for the environment"""
        raise NotImplementedError

    def _check_success(self) -> bool:
        """Check if the current episode is successful"""
        return False


class SinkEnvForPlay(BaseEnvForPlay):
    EXCLUDE_LAYOUTS = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_pos = self.sink
        return

    @property
    def fruit_container_counter_loc(self):
        return "left_right"

    @property
    def bread_container_counter_loc(self):
        return "left_right"

    @property
    def packed_food_counter_loc(self):
        return "left_right"

    def _get_obj_cfgs(self):
        split_type = self.split_type()
        cfgs = []
        cfgs.append(
            dict(
                name="fruit",
                obj_groups=f"fruit_set_{split_type}",
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                graspable=True,
                placement=dict(
                    fixture=self.counter_sink,
                    size=(1.0, 1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                    pos=("ref", -1.0),
                    try_to_place_in=f"container_set_{split_type}",
                    container_kwargs=dict(
                        placement=dict(
                            fixture=self.counter_sink,
                            sample_region_kwargs=dict(
                                ref=self.sink,
                                loc=self.fruit_container_counter_loc,
                                top_size=(0.25, 0.25),
                            ),
                            size=(0.35, 0.35),
                            pos=("ref", -1.0),
                        ),
                    ),
                ),
            )
        )
        if self.layout_id not in [7]:
            cfgs.append(
                dict(
                    name="bread",
                    obj_groups=f"bread_set_{split_type}",
                    obj_registries=("objaverse", "aigen"),
                    obj_instance_split=None,
                    graspable=True,
                    placement=dict(
                        fixture=self.counter_sink,
                        size=(1.0, 1.0),
                        rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                        pos=("ref", -1.0),
                        try_to_place_in=f"container_set_{split_type}",
                        container_kwargs=dict(
                            placement=dict(
                                fixture=self.counter_sink,
                                sample_region_kwargs=dict(
                                    ref=self.sink,
                                    loc=self.bread_container_counter_loc,
                                    top_size=(0.25, 0.25),
                                ),
                                size=(0.35, 0.35),
                                pos=("ref", -1.0),
                            ),
                        ),
                    ),
                )
            )
        cfgs.append(
            dict(
                name="packed_food",
                obj_groups=f"packaged_food_{split_type}",
                graspable=True,
                obj_registries=("objaverse", "aigen"),
                obj_instance_split=None,
                placement=dict(
                    fixture=self.counter_sink,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc=self.packed_food_counter_loc,
                        top_size=(0.25, 0.35),
                    ),
                    size=(0.25, 0.35),
                    pos=("ref", -1.0),
                    rotation=[
                        (np.pi / 2 - np.pi / 16, np.pi / 2 + np.pi / 16),
                    ],
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=f"vegetable_set_{split_type}",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(1.0, 1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                    pos=("ref", -1.0),
                    try_to_place_in=f"container_set_{split_type}",
                    container_kwargs=dict(
                        placement=dict(
                            fixture=self.sink,
                            size=(0.3, 0.3),
                            pos=(0.5, 0.4),
                        ),
                    ),
                ),
            )
        )
        if self.cab_sink is not None:
            offset = (-0.15, -0.05)
            size = (0.45, 0.15)
            if self.layout_id == 8:  # make it reachable for each layout
                offset = (-0.5, -0.05)
                size = (0.30, 0.15)
            if self.layout_id == 9:  # make it reachable for each layout
                offset = (0.10, -0.05)
                size = (0.30, 0.15)
            cfgs.append(
                dict(
                    name="meat",
                    obj_groups=[f"meat_set_{split_type}", f"bread_set_{split_type}"],
                    graspable=True,
                    max_size=(0.15, 0.15, None),
                    obj_registries=("objaverse", "aigen"),
                    obj_instance_split=None,
                    placement=dict(
                        fixture=self.cab_sink,
                        size=size,
                        pos=(0.75, 0.05),
                        offset=offset,
                    ),
                )
            )
        else:
            cfgs.append(
                dict(
                    name="meat",
                    obj_groups=f"meat_set_{split_type}",
                    graspable=True,
                    max_size=(0.15, 0.15, None),
                    obj_registries=("objaverse", "aigen"),
                    obj_instance_split=None,
                    placement=dict(
                        fixture=self.counter_sink,
                        sample_region_kwargs=dict(
                            ref=self.sink, loc="left_right", top_size=(0.25, 0.3)
                        ),
                        size=(0.25, 0.30),
                        pos=("ref", -1.0),
                        rotation=[
                            (np.pi / 2 - np.pi / 16, np.pi / 2 + np.pi / 16),
                        ],
                    ),
                )
            )
        return cfgs

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if self.cab_sink is not None:
            self.cab_sink.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        return

    def _check_cab_sink_door_success(
        self, door_side: str, target_state: float, comparison: str = "less"
    ) -> bool:
        """Check if cabinet door is in desired state"""
        if not hasattr(self, "cab_sink") or self.cab_sink is None:
            raise ValueError(
                "cab_sink is not set. We should not be calling this function."
            )

        cab_state = self.cab_sink.get_door_state(self)
        door_value = cab_state[f"{door_side}_door"]

        if comparison == "less":
            return door_value < target_state
        elif comparison == "greater":
            return door_value > target_state
        else:
            raise ValueError(f"Unknown comparison: {comparison}")
        return False

    def _check_place_on_fixture_success(
        self, obj_name: str, target_fixture, require_gripper_far: bool = False
    ) -> bool:
        """Check if pick and place task is successful where the target location is a fixture"""
        is_contact = OU.check_obj_fixture_contact(self, obj_name, target_fixture)
        if require_gripper_far:
            return is_contact and OU.gripper_obj_far(self, obj_name=obj_name)
        return is_contact

    def _check_place_on_receptacle_success(
        self,
        obj_name: str,
        target_receptacle,
        receptacle_fixture,
        require_gripper_far: bool = False,
    ) -> bool:
        """Check if place task is successful where the target location is an object"""
        is_in = OU.check_obj_in_receptacle(self, obj_name, target_receptacle)
        is_tar_contact = OU.check_obj_fixture_contact(
            self, target_receptacle, receptacle_fixture
        )
        if require_gripper_far:
            return (
                is_in and is_tar_contact and OU.gripper_obj_far(self, obj_name=obj_name)
            )
        return is_in and is_tar_contact


class PnPSinkToRightCounterPlate(SinkEnvForPlay):
    @property
    def fruit_container_counter_loc(self):
        return "right"

    @property
    def bread_container_counter_loc(self):
        return "left"

    def _check_success(self):
        obj_name = "vegetable"
        tar_name = "fruit_container"
        return self._check_place_on_receptacle_success(
            obj_name, tar_name, self.counter_sink, require_gripper_far=True
        )

    def set_ep_meta(self, ep_meta):
        ## here we will overrite the object_cfgs if present in the ep_meta. specifically overrite the fruit_container_counter_loc and bread_container_counter_loc
        if "object_cfgs" in ep_meta:
            for obj_cfg in ep_meta["object_cfgs"]:
                if obj_cfg["name"] == "fruit_container":
                    obj_cfg["placement"]["sample_region_kwargs"][
                        "loc"
                    ] = self.fruit_container_counter_loc
                if obj_cfg["name"] == "bread_container":
                    obj_cfg["placement"]["sample_region_kwargs"][
                        "loc"
                    ] = self.bread_container_counter_loc
        super().set_ep_meta(ep_meta)
        return ep_meta

    def _load_model(self, *args, **kwargs):
        if hasattr(self, "eval_mode") and self.eval_mode == "diff_obj":
            for cfg in self._ep_meta["object_cfgs"]:
                if cfg["name"] == "vegetable":
                    del cfg["info"]
        return super()._load_model(*args, **kwargs)


class CloseRightCabinetDoor(SinkEnvForPlay):
    def _check_success(self):
        return self._check_cab_sink_door_success("right", 0.1, "less")

    def _load_model(self, *args, **kwargs):
        if hasattr(self, "eval_mode") and self.eval_mode == "diff_obj":
            self._ep_meta["style_ids"] = self.TRAIN_STYLE_IDS
            self._ep_meta["style_id"] = np.random.choice(self._ep_meta["style_ids"])
        return super()._load_model(*args, **kwargs)


class CloseLeftCabinetDoor(SinkEnvForPlay):
    def _check_success(self):
        return self._check_cab_sink_door_success("left", 0.1, "less")

    def _load_model(self, *args, **kwargs):
        if hasattr(self, "eval_mode") and self.eval_mode == "diff_obj":
            self._ep_meta["style_ids"] = self.TRAIN_STYLE_IDS
            self._ep_meta["style_id"] = np.random.choice(self._ep_meta["style_ids"])
        return super()._load_model(*args, **kwargs)


class PnPSinkToCabinet(SinkEnvForPlay):
    def _check_success(self):
        obj_name = "vegetable"
        return self._check_place_on_fixture_success(obj_name, self.cab_sink)

    def _load_model(self, *args, **kwargs):
        if hasattr(self, "eval_mode") and self.eval_mode == "diff_obj":
            for cfg in self._ep_meta["object_cfgs"]:
                if cfg["name"] == "vegetable":
                    del cfg["info"]
        return super()._load_model(*args, **kwargs)


class TurnOnFaucet(SinkEnvForPlay):
    def _check_success(self):
        handle_state = self.sink.get_handle_state(self)
        water_on = handle_state["water_on"]
        return water_on

    def _load_model(self, *args, **kwargs):
        if hasattr(self, "eval_mode") and self.eval_mode == "diff_obj":
            self._ep_meta["style_ids"] = self.TRAIN_STYLE_IDS
            self._ep_meta["style_id"] = np.random.choice(self._ep_meta["style_ids"])
        return super()._load_model(*args, **kwargs)


# --- reverse-behavior tasks: same scene as SinkEnvForPlay but the cab_sink door
# starts CLOSED (open task) / the faucet starts ON (turn-off task). Base
# _reset_internal sets door open + faucet off, so we flip the relevant one after. ---
class OpenLeftCabinetDoor(SinkEnvForPlay):
    def _reset_internal(self):
        super()._reset_internal()
        if self.cab_sink is not None:
            self.cab_sink.set_door_state(min=0.0, max=0.1, env=self, rng=self.rng)

    def _check_success(self):
        return self._check_cab_sink_door_success("left", 0.9, "greater")

    def _load_model(self, *args, **kwargs):
        if hasattr(self, "eval_mode") and self.eval_mode == "diff_obj":
            self._ep_meta["style_ids"] = self.TRAIN_STYLE_IDS
            self._ep_meta["style_id"] = np.random.choice(self._ep_meta["style_ids"])
        return super()._load_model(*args, **kwargs)


class OpenRightCabinetDoor(SinkEnvForPlay):
    def _reset_internal(self):
        super()._reset_internal()
        if self.cab_sink is not None:
            self.cab_sink.set_door_state(min=0.0, max=0.1, env=self, rng=self.rng)

    def _check_success(self):
        return self._check_cab_sink_door_success("right", 0.9, "greater")

    def _load_model(self, *args, **kwargs):
        if hasattr(self, "eval_mode") and self.eval_mode == "diff_obj":
            self._ep_meta["style_ids"] = self.TRAIN_STYLE_IDS
            self._ep_meta["style_id"] = np.random.choice(self._ep_meta["style_ids"])
        return super()._load_model(*args, **kwargs)


class TurnOffFaucet(SinkEnvForPlay):
    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(mode="on", env=self, rng=self.rng)

    def _check_success(self):
        handle_state = self.sink.get_handle_state(self)
        return not handle_state["water_on"]

    def _load_model(self, *args, **kwargs):
        if hasattr(self, "eval_mode") and self.eval_mode == "diff_obj":
            self._ep_meta["style_ids"] = self.TRAIN_STYLE_IDS
            self._ep_meta["style_id"] = np.random.choice(self._ep_meta["style_ids"])
        return super()._load_model(*args, **kwargs)


class PnPSinkToRightCounterPlateL2(StyleMixin, PnPSinkToRightCounterPlate):
    REQUIRED_STYLE_ID = "001_l2"

    def set_ep_meta(self, ep_meta):
        if "object_cfgs" in ep_meta:  ## overriting
            for obj_cfg in ep_meta["object_cfgs"]:
                if obj_cfg["name"] == "vegetable":
                    obj_cfg["obj_groups"] = "vegetable_set_test"
        return super().set_ep_meta(ep_meta)


class CloseLeftCabinetDoorL2(StyleMixin, CloseLeftCabinetDoor):
    REQUIRED_STYLE_ID = "002_l2"


class PnPSinkToCabinetL2(StyleMixin, PnPSinkToCabinet):
    REQUIRED_STYLE_ID = "003_l2"

    def set_ep_meta(self, ep_meta):
        if "object_cfgs" in ep_meta:  ## overriting
            for obj_cfg in ep_meta["object_cfgs"]:
                if obj_cfg["name"] == "vegetable":
                    obj_cfg["obj_groups"] = "vegetable_set_test"
        return super().set_ep_meta(ep_meta)


class CloseRightCabinetDoorL2(StyleMixin, CloseRightCabinetDoor):
    REQUIRED_STYLE_ID = "006_l2"


class PnPSinkToRightCounterPlateL3(StyleMixin, LayoutMixin, PnPSinkToRightCounterPlate):
    REQUIRED_STYLE_ID = "014_l3"
    REQUIRED_LAYOUT_ID = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split_type(self):
        return "test"

    def set_ep_meta(
        self, ep_meta
    ):  # taken cared by the @split_type in the meta file generation
        if "object_cfgs" in ep_meta:  ## overriting
            for obj_cfg in ep_meta["object_cfgs"]:
                if obj_cfg["name"] == "vegetable_container":
                    obj_cfg["obj_groups"] = "container_set_train"
                    obj_cfg["placement"]["offset"] = (0.05, 0.0)
                    obj_cfg["info"]["groups_containing_sampled_obj"] = ["plate"]
                    obj_cfg["info"]["cat"] = ["plate"]
                    # use the robocasa base path to get the path to the plate model
                    obj_cfg["info"][
                        "groups"
                    ] = f"{robocasa.models.assets_root}/objects/objaverse/plate/plate_1/model.xml"
                    obj_cfg["info"][
                        "mjcf_path"
                    ] = f"{robocasa.models.assets_root}/objects/objaverse/plate/plate_1/model.xml"
        return super().set_ep_meta(ep_meta)


class PnPSinkToMicrowaveTopL3(StyleMixin, LayoutMixin, SinkEnvForPlay):
    REQUIRED_STYLE_ID = "014_l3"
    REQUIRED_LAYOUT_ID = 10

    def _setup_kitchen_references(self) -> None:
        """Setup kitchen fixture references"""
        super()._setup_kitchen_references()

    def split_type(self):
        return "test"

    def _remove_specific_fixtures(self):
        # does not remove any fixtures
        return

    def set_ep_meta(self, ep_meta):
        ep_meta["layout_id"] = 10
        return super().set_ep_meta(ep_meta)

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        remove_keys = ["fruit", "fruit_container"]
        cfgs = [cfg for cfg in cfgs if cfg["name"] not in remove_keys]
        return cfgs

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if self.cab_sink is not None:
            self.cab_sink.set_door_state(min=0.0, max=0.1, env=self, rng=self.rng)

    def _check_success(self):
        obj_name = "vegetable"
        is_tar_contact = OU.is_on_top_of(
            self, obj_name, "microwave_1_main_group", tol_xy=0.05
        )
        return is_tar_contact and OU.gripper_obj_far(self, obj_name=obj_name)


class CloseLeftCabinetDoorL3(StyleMixin, LayoutMixin, CloseLeftCabinetDoor):
    REQUIRED_STYLE_ID = "014_l3"
    REQUIRED_LAYOUT_ID = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_ep_meta(self, ep_meta):
        return super().set_ep_meta(ep_meta)


class TurnOnFaucetL3(StyleMixin, LayoutMixin, TurnOnFaucet):
    REQUIRED_STYLE_ID = "014_l3"
    REQUIRED_LAYOUT_ID = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split_type(self):
        return "test"

    def set_ep_meta(self, ep_meta):
        return super().set_ep_meta(ep_meta)
