"""Helper sensor for calculating utility costs."""

import asyncio
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import logging
from typing import Any, Final, Literal, cast, override

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityCapabilityAttribute,
    SensorEntityStateAttribute,
    SensorStateClass,
)
from homeassistant.components.sensor.recorder import (  # pylint: disable=home-assistant-component-root-import
    reset_detected,
)
from homeassistant.const import (
    EntityStateAttribute,
    UnitOfEnergy,
    UnitOfPower,
    UnitOfVolume,
)
from homeassistant.core import (
    HomeAssistant,
    State,
    callback,
    split_entity_id,
    valid_entity_id,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util, unit_conversion
from homeassistant.util.unit_system import METRIC_SYSTEM

from .const import DOMAIN
from .data import EnergyManager, PowerConfig, async_get_manager
from .helpers import generate_power_sensor_entity_id, generate_power_sensor_unique_id

SUPPORTED_STATE_CLASSES = {
    SensorStateClass.MEASUREMENT,
    SensorStateClass.TOTAL,
    SensorStateClass.TOTAL_INCREASING,
}
VALID_ENERGY_UNITS: set[str] = set(UnitOfEnergy)

VALID_ENERGY_UNITS_GAS = {
    UnitOfVolume.CENTUM_CUBIC_FEET,
    UnitOfVolume.CUBIC_FEET,
    UnitOfVolume.CUBIC_METERS,
    UnitOfVolume.LITERS,
    UnitOfVolume.MILLE_CUBIC_FEET,
    *VALID_ENERGY_UNITS,
}
VALID_VOLUME_UNITS_WATER: set[str] = {
    UnitOfVolume.CENTUM_CUBIC_FEET,
    UnitOfVolume.CUBIC_FEET,
    UnitOfVolume.CUBIC_METERS,
    UnitOfVolume.GALLONS,
    UnitOfVolume.LITERS,
    UnitOfVolume.MILLE_CUBIC_FEET,
}
_LOGGER = logging.getLogger(__name__)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the energy sensors."""
    sensor_manager = SensorManager(await async_get_manager(hass), async_add_entities)
    await sensor_manager.async_start()


@dataclass(slots=True)
class SourceAdapter:
    """Adapter to allow sources and their flows to be used as sensors."""

    source_type: Literal["grid", "gas", "water", "solar"]
    flow_type: Literal["flow_from", "flow_to"] | None
    stat_energy_key: Literal["stat_energy_from", "stat_energy_to"]
    total_money_key: Literal["stat_cost", "stat_compensation"]
    name_suffix: str
    entity_id_suffix: str


SOURCE_ADAPTERS: Final = (
    # Grid import cost (unified format)
    SourceAdapter(
        "grid",
        None,  # No flow_type - unified format
        "stat_energy_from",
        "stat_cost",
        "Cost",
        "cost",
    ),
    SourceAdapter(
        "gas",
        None,
        "stat_energy_from",
        "stat_cost",
        "Cost",
        "cost",
    ),
    SourceAdapter(
        "water",
        None,
        "stat_energy_from",
        "stat_cost",
        "Cost",
        "cost",
    ),
)

# Separate adapter for grid export compensation (needs different price field)
GRID_EXPORT_ADAPTER: Final = SourceAdapter(
    "grid",
    None,  # No flow_type - unified format
    "stat_energy_to",
    "stat_compensation",
    "Compensation",
    "compensation",
)

# Prices the self-consumed solar sensor, but cost_sensors is keyed by the
# solar production statistic (stat_energy_from), which is what the frontend
# looks up. The tracked energy entity is passed separately and is not
# stat_energy_from.
SOLAR_SAVINGS_ADAPTER: Final = SourceAdapter(
    "solar",
    None,
    "stat_energy_from",
    "stat_cost",
    "Savings",
    "cost",
)


def compute_used_solar(
    *,
    from_grid: float,
    to_grid: float,
    solar: float,
    to_battery: float,
    from_battery: float,
) -> float:
    """Return solar used directly by the home for one period.

    This is ``used_solar`` from the energy dashboard's
    ``computeConsumptionSingle`` (frontend ``src/data/energy.ts``). Core has
    no helper for that split, so the priority order is ported here:

    - Grid import that cannot be consumed is charged into the battery first.
    - Remaining battery charge is filled from solar.
    - Remaining solar covers grid export.
    - What is left of solar, capped by home consumption, is used solar.

    Solar sent to the battery is not counted. The dashboard later attributes
    some battery discharge back to solar with a period LIFO stack; that is
    not applied here because it is not a running total and core has no helper
    for it. Export compensation stays on the grid source.
    """
    to_grid = max(to_grid, 0.0)
    to_battery = max(to_battery, 0.0)
    solar = max(solar, 0.0)
    from_grid = max(from_grid, 0.0)
    from_battery = max(from_battery, 0.0)

    used_total = from_grid + solar + from_battery - to_grid - to_battery
    used_total_remaining = max(used_total, 0.0)

    excess_grid_in_after_consumption = max(
        0.0, min(to_battery, from_grid - used_total_remaining)
    )
    to_battery -= excess_grid_in_after_consumption

    solar -= min(solar, to_battery)
    # Remaining solar covers export. Battery-to-grid and the second
    # grid-to-battery pass do not change used_solar, so they stop here.
    solar -= min(solar, to_grid)

    return min(used_total_remaining, solar)


def solar_self_consumed_entity_id(stat_energy_from: str) -> str:
    """Entity id of the sensor that totals self-consumed solar."""
    return f"{stat_energy_from}_self_consumed"


class EntityNotFoundError(HomeAssistantError):
    """When a referenced entity was not found."""


class SensorManager:
    """Class to handle creation/removal of sensor data."""

    def __init__(
        self, manager: EnergyManager, async_add_entities: AddEntitiesCallback
    ) -> None:
        """Initialize sensor manager."""
        self.manager = manager
        self.async_add_entities = async_add_entities
        self.current_entities: dict[tuple[str, str | None, str], EnergyCostSensor] = {}
        self.current_power_entities: dict[str, EnergyPowerSensor] = {}
        self.current_self_use_entities: dict[str, EnergySolarSelfConsumptionSensor] = {}

    async def async_start(self) -> None:
        """Start."""
        self.manager.async_listen_updates(self._process_manager_data)

        if self.manager.data:
            await self._process_manager_data()

    async def _process_manager_data(self) -> None:
        """Process manager data."""
        to_add: list[
            EnergyCostSensor | EnergyPowerSensor | EnergySolarSelfConsumptionSensor
        ] = []
        to_remove = dict(self.current_entities)
        power_to_remove = dict(self.current_power_entities)
        self_use_to_remove = dict(self.current_self_use_entities)

        async def finish() -> None:
            if to_add:
                self.async_add_entities(to_add)
                await asyncio.wait(ent.add_finished for ent in to_add)

            for key, entity in to_remove.items():
                self.current_entities.pop(key)
                await entity.async_remove()

            for power_key, power_entity in power_to_remove.items():
                self.current_power_entities.pop(power_key)
                await power_entity.async_remove()

            for self_use_key, self_use_entity in self_use_to_remove.items():
                self.current_self_use_entities.pop(self_use_key)
                await self_use_entity.async_remove()

        # This guard is for the optional typing of EnergyManager.data.
        # In practice, data is always set to default preferences in async_update
        # before listeners are called, so this case should never happen.
        if not self.manager.data:
            await finish()
            return

        for energy_source in self.manager.data["energy_sources"]:
            for adapter in SOURCE_ADAPTERS:
                if adapter.source_type != energy_source["type"]:
                    continue

                self._process_sensor_data(
                    adapter,
                    energy_source,
                    to_add,
                    to_remove,
                )

            # Handle grid export compensation
            # (unified format uses different price fields)
            if energy_source["type"] == "grid":
                self._process_grid_export_sensor(
                    energy_source,
                    to_add,
                    to_remove,
                )

            if energy_source["type"] == "solar":
                self._process_solar_savings_sensor(
                    energy_source,
                    to_add,
                    to_remove,
                    self_use_to_remove,
                )

            # Process power sensors for battery and grid sources
            self._process_power_sensor_data(
                energy_source,
                to_add,
                power_to_remove,
            )

        await finish()

    @callback
    def _process_sensor_data(
        self,
        adapter: SourceAdapter,
        config: Mapping[str, Any],
        to_add: list[
            EnergyCostSensor | EnergyPowerSensor | EnergySolarSelfConsumptionSensor
        ],
        to_remove: dict[tuple[str, str | None, str], EnergyCostSensor],
    ) -> None:
        """Process sensor data."""
        # No need to create an entity if we already have a cost stat
        if config.get(adapter.total_money_key) is not None:
            return

        # Skip if the energy stat is not configured
        # (e.g., export-only or power-only grids)
        stat_energy = config.get(adapter.stat_energy_key)
        if not stat_energy:
            return

        key = (adapter.source_type, adapter.flow_type, stat_energy)

        # Make sure the right data is there
        # If the entity existed, we don't pop it from to_remove so it's removed
        if not valid_entity_id(stat_energy) or (
            config.get("entity_energy_price") is None
            and config.get("number_energy_price") is None
        ):
            return

        if current_entity := to_remove.pop(key, None):
            current_entity.update_config(config)
            return

        self.current_entities[key] = EnergyCostSensor(
            adapter,
            config,
        )
        to_add.append(self.current_entities[key])

    @callback
    def _process_solar_savings_sensor(
        self,
        config: Mapping[str, Any],
        to_add: list[
            EnergyCostSensor | EnergyPowerSensor | EnergySolarSelfConsumptionSensor
        ],
        to_remove: dict[tuple[str, str | None, str], EnergyCostSensor],
        self_use_to_remove: dict[str, EnergySolarSelfConsumptionSensor],
    ) -> None:
        """Price solar the home used, not total production and not export.

        A total sensor tracks self-consumed energy for this solar source.
        EnergyCostSensor multiplies that sensor. The generated cost entity is
        registered under the production statistic id.
        """
        stat_energy_from = config.get("stat_energy_from")
        if not stat_energy_from or not valid_entity_id(stat_energy_from):
            return

        # User already totals the savings.
        if config.get("stat_cost") is not None:
            return

        if (
            config.get("entity_energy_price") is None
            and config.get("number_energy_price") is None
        ):
            return

        if current_self_use := self_use_to_remove.pop(stat_energy_from, None):
            current_self_use.update_sources()
        else:
            self_use = EnergySolarSelfConsumptionSensor(
                stat_energy_from,
                self._energy_sources,
            )
            self.current_self_use_entities[stat_energy_from] = self_use
            to_add.append(self_use)

        key = ("solar", None, stat_energy_from)
        if current_entity := to_remove.pop(key, None):
            current_entity.update_config(config)
            return

        self.current_entities[key] = EnergyCostSensor(
            SOLAR_SAVINGS_ADAPTER,
            config,
            tracked_energy_entity=solar_self_consumed_entity_id(stat_energy_from),
        )
        to_add.append(self.current_entities[key])

    def _energy_sources(self) -> list[Mapping[str, Any]]:
        """Return the current energy sources."""
        if not self.manager.data:
            return []
        return self.manager.data["energy_sources"]

    @callback
    def _process_grid_export_sensor(
        self,
        config: Mapping[str, Any],
        to_add: list[
            EnergyCostSensor | EnergyPowerSensor | EnergySolarSelfConsumptionSensor
        ],
        to_remove: dict[tuple[str, str | None, str], EnergyCostSensor],
    ) -> None:
        """Process grid export compensation sensor (unified format).

        The unified grid format uses different field names for export pricing:
        - entity_energy_price_export instead of entity_energy_price
        - number_energy_price_export instead of number_energy_price
        """
        # No export meter configured
        stat_energy_to = config.get("stat_energy_to")
        if stat_energy_to is None:
            return

        # Already have a compensation stat
        if config.get("stat_compensation") is not None:
            return

        key = ("grid", None, stat_energy_to)

        # Check for export pricing fields (different names in unified format)
        if not valid_entity_id(stat_energy_to) or (
            config.get("entity_energy_price_export") is None
            and config.get("number_energy_price_export") is None
        ):
            return

        # Create a config wrapper that maps the sell price fields to standard names
        # so EnergyCostSensor can use them
        export_config: dict[str, Any] = {
            "stat_energy_to": stat_energy_to,
            "stat_compensation": config.get("stat_compensation"),
            "entity_energy_price": config.get("entity_energy_price_export"),
            "number_energy_price": config.get("number_energy_price_export"),
        }

        if current_entity := to_remove.pop(key, None):
            current_entity.update_config(export_config)
            return

        self.current_entities[key] = EnergyCostSensor(
            GRID_EXPORT_ADAPTER,
            export_config,
        )
        to_add.append(self.current_entities[key])

    @callback
    def _process_power_sensor_data(
        self,
        energy_source: Mapping[str, Any],
        to_add: list[
            EnergyCostSensor | EnergyPowerSensor | EnergySolarSelfConsumptionSensor
        ],
        to_remove: dict[str, EnergyPowerSensor],
    ) -> None:
        """Process power sensor data for battery and grid sources."""
        source_type = energy_source.get("type")

        if source_type in ("battery", "grid"):
            # Both battery and grid now use unified format
            # with power_config at top level
            power_config = energy_source.get("power_config")
            if power_config and self._needs_power_sensor(power_config):
                self._create_or_keep_power_sensor(
                    source_type, power_config, to_add, to_remove
                )

    @staticmethod
    def _needs_power_sensor(power_config: PowerConfig) -> bool:
        """Check if power_config needs a transform sensor."""
        # Only create sensors for inverted or two-sensor configs
        # Standard stat_rate configs don't need a transform sensor
        return "stat_rate_inverted" in power_config or (
            "stat_rate_from" in power_config and "stat_rate_to" in power_config
        )

    def _create_or_keep_power_sensor(
        self,
        source_type: str,
        power_config: PowerConfig,
        to_add: list[
            EnergyCostSensor | EnergyPowerSensor | EnergySolarSelfConsumptionSensor
        ],
        to_remove: dict[str, EnergyPowerSensor],
    ) -> None:
        """Create a power sensor or keep an existing one."""
        unique_id = generate_power_sensor_unique_id(source_type, power_config)

        # If entity already exists, keep it
        if unique_id in to_remove:
            to_remove.pop(unique_id)
            return

        sensor = EnergyPowerSensor(
            source_type,
            power_config,
            unique_id,
            generate_power_sensor_entity_id(source_type, power_config),
        )
        self.current_power_entities[unique_id] = sensor
        to_add.append(sensor)


def _set_result_unless_done(future: asyncio.Future[None]) -> None:
    """Set the result of a future unless it is done."""
    if not future.done():
        future.set_result(None)


class EnergyCostSensor(SensorEntity):
    """Calculate costs incurred by consuming energy.

    This is intended as a fallback for when no specific cost sensor is available for the
    utility.

    Expected config fields (from adapter or export_config wrapper):
    - stat_energy_key (via adapter): Key to get the energy statistic ID
    - total_money_key (via adapter): Key to get the existing cost/compensation stat
    - entity_energy_price: Entity ID providing price per unit (e.g., $/kWh)
    - number_energy_price: Fixed price per unit

    Note: For grid export compensation, the unified format uses
    different field names (entity_energy_price_export,
    number_energy_price_export). The _process_grid_export_sensor
    method in SensorManager creates a wrapper config that maps
    these to the standard field names (entity_energy_price,
    number_energy_price) so this class can use them.
    """

    _attr_entity_registry_visible_default = False
    _attr_should_poll = False

    _wrong_state_class_reported = False
    _wrong_unit_reported = False

    def __init__(
        self,
        adapter: SourceAdapter,
        config: Mapping[str, Any],
        tracked_energy_entity: str | None = None,
    ) -> None:
        """Initialize the sensor.

        tracked_energy_entity overrides the statistic named by stat_energy_key.
        Solar savings track the self-consumed sensor, while cost_sensors and the
        entity id stay keyed by production (stat_energy_from).
        """
        super().__init__()

        self._adapter = adapter
        self.entity_id = f"{config[adapter.stat_energy_key]}_{adapter.entity_id_suffix}"
        self._attr_device_class = SensorDeviceClass.MONETARY
        self._attr_state_class = SensorStateClass.TOTAL
        self._config = config
        self._tracked_energy_entity = tracked_energy_entity
        self._last_energy_sensor_state: State | None = None
        # SensorManager awaits add_finished; async_on_remove resolves it on the
        # abort path too, since add_to_platform_abort fires on-remove callbacks.
        self.add_finished: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self.async_on_remove(lambda: _set_result_unless_done(self.add_finished))

    @property
    def _energy_entity_id(self) -> str:
        """Entity whose growth is multiplied by the price."""
        if self._tracked_energy_entity is not None:
            return self._tracked_energy_entity
        return cast(str, self._config[self._adapter.stat_energy_key])

    def _reset(self, energy_state: State) -> None:
        """Reset the cost sensor."""
        self._attr_native_value = 0.0
        self._attr_last_reset = dt_util.utcnow()
        self._last_energy_sensor_state = energy_state
        self.async_write_ha_state()

    @callback
    def _update_cost(self) -> None:
        """Update incurred costs."""
        if self._adapter.source_type in ("grid", "solar"):
            valid_units = VALID_ENERGY_UNITS
            default_price_unit: str | None = UnitOfEnergy.KILO_WATT_HOUR

        elif self._adapter.source_type == "gas":
            valid_units = VALID_ENERGY_UNITS_GAS
            # No conversion for gas.
            default_price_unit = None

        elif self._adapter.source_type == "water":
            valid_units = VALID_VOLUME_UNITS_WATER
            if self.hass.config.units is METRIC_SYSTEM:
                default_price_unit = UnitOfVolume.CUBIC_METERS
            else:
                default_price_unit = UnitOfVolume.GALLONS

        energy_state = self.hass.states.get(self._energy_entity_id)

        if energy_state is None:
            return

        state_class = energy_state.attributes.get(
            SensorEntityCapabilityAttribute.STATE_CLASS
        )
        if state_class not in SUPPORTED_STATE_CLASSES:
            if not self._wrong_state_class_reported:
                self._wrong_state_class_reported = True
                _LOGGER.warning(
                    "Found unexpected state_class %s for %s",
                    state_class,
                    energy_state.entity_id,
                )
            return

        # last_reset must be set if the sensor is SensorStateClass.MEASUREMENT
        if (
            state_class == SensorStateClass.MEASUREMENT
            and SensorEntityStateAttribute.LAST_RESET not in energy_state.attributes
        ):
            return

        try:
            energy = float(energy_state.state)
        except ValueError:
            return

        try:
            energy_price, energy_price_unit = self._get_energy_price(
                valid_units, default_price_unit
            )
        except EntityNotFoundError:
            return
        except ValueError:
            energy_price = None

        if self._last_energy_sensor_state is None:
            # Initialize as it's the first time all required entities are in place or
            # only the price is missing. In the later case, cost will update the first
            # time the energy is updated after the price entity is in place.
            self._reset(energy_state)
            return

        if energy_price is None:
            return

        energy_unit: str | None = energy_state.attributes.get(
            EntityStateAttribute.UNIT_OF_MEASUREMENT
        )

        if energy_unit is None or energy_unit not in valid_units:
            if not self._wrong_unit_reported:
                self._wrong_unit_reported = True
                _LOGGER.warning(
                    "Found unexpected unit %s for %s",
                    energy_state.attributes.get(
                        EntityStateAttribute.UNIT_OF_MEASUREMENT
                    ),
                    energy_state.entity_id,
                )
            return

        if (
            state_class != SensorStateClass.TOTAL_INCREASING
            and energy_state.attributes.get(SensorEntityStateAttribute.LAST_RESET)
            != self._last_energy_sensor_state.attributes.get(
                SensorEntityStateAttribute.LAST_RESET
            )
        ) or (
            state_class == SensorStateClass.TOTAL_INCREASING
            and reset_detected(
                self.hass,
                self._energy_entity_id,
                energy,
                float(self._last_energy_sensor_state.state),
                self._last_energy_sensor_state,
            )
        ):
            # Energy meter was reset, reset cost sensor too
            energy_state_copy = copy.copy(energy_state)
            energy_state_copy.state = "0.0"
            self._reset(energy_state_copy)

        # Update with newly incurred cost
        old_energy_value = float(self._last_energy_sensor_state.state)
        cur_value = cast(float, self._attr_native_value)

        converted_energy_price = self._convert_energy_price(
            energy_price, energy_price_unit, energy_unit
        )

        self._attr_native_value = (
            cur_value + (energy - old_energy_value) * converted_energy_price
        )

        self._last_energy_sensor_state = energy_state

    def _get_energy_price(
        self, valid_units: set[str], default_unit: str | None
    ) -> tuple[float, str | None]:
        """Get the energy price.

        Raises:
            EntityNotFoundError: When the energy price entity is not found.
            ValueError: When the entity state is not a valid float.

        """

        if self._config.get("entity_energy_price") is None:
            return cast(float, self._config.get("number_energy_price")), default_unit

        energy_price_state = self.hass.states.get(self._config["entity_energy_price"])
        if energy_price_state is None:
            raise EntityNotFoundError

        energy_price = float(energy_price_state.state)

        energy_price_unit: str | None = energy_price_state.attributes.get(
            EntityStateAttribute.UNIT_OF_MEASUREMENT, ""
        ).partition("/")[2]

        # For backwards compatibility we don't validate the unit of the price
        # If it is not valid, we assume it's our default price unit.
        if energy_price_unit not in valid_units:
            energy_price_unit = default_unit

        return energy_price, energy_price_unit

    def _convert_energy_price(
        self, energy_price: float, energy_price_unit: str | None, energy_unit: str
    ) -> float:
        """Convert the energy price to the correct unit."""
        if energy_price_unit is None:
            return energy_price

        converter: Callable[[float, str, str], float]
        if energy_unit in VALID_ENERGY_UNITS:
            converter = unit_conversion.EnergyConverter.convert
        else:
            converter = unit_conversion.VolumeConverter.convert

        return converter(energy_price, energy_unit, energy_price_unit)

    @override
    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        # Name follows the configured statistic (solar production), not the
        # self-consumed sensor that solar savings actually multiplies.
        name_stat = self._config[self._adapter.stat_energy_key]
        energy_state = self.hass.states.get(name_stat)
        if energy_state:
            name = energy_state.name
        else:
            name = split_entity_id(name_stat)[0].replace("_", " ")

        self._attr_name = f"{name} {self._adapter.name_suffix}"

        self._update_cost()

        # Store stat ID in hass.data so frontend can look it up
        self.hass.data[DOMAIN]["cost_sensors"][
            self._config[self._adapter.stat_energy_key]
        ] = self.entity_id

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                self._energy_entity_id,
                self._async_state_changed_listener,
            )
        )
        _set_result_unless_done(self.add_finished)

    @callback
    def _async_state_changed_listener(self, *_: Any) -> None:
        """Handle child updates."""
        self._update_cost()
        self.async_write_ha_state()

    @override
    async def async_will_remove_from_hass(self) -> None:
        """Handle removing from hass."""
        self.hass.data[DOMAIN]["cost_sensors"].pop(
            self._config[self._adapter.stat_energy_key]
        )
        await super().async_will_remove_from_hass()

    @callback
    def update_config(self, config: Mapping[str, Any]) -> None:
        """Update the config."""
        self._config = config

    @property
    @override
    def native_unit_of_measurement(self) -> str | None:
        """Return the units of measurement."""
        return self.hass.config.currency

    @property
    @override
    def unique_id(self) -> str | None:
        """Return the unique ID of the sensor."""
        entity_registry = er.async_get(self.hass)
        if registry_entry := entity_registry.async_get(
            self._config[self._adapter.stat_energy_key]
        ):
            prefix = registry_entry.id
        else:
            prefix = self._config[self._adapter.stat_energy_key]

        return f"{prefix}_{self._adapter.source_type}_{self._adapter.entity_id_suffix}"


class EnergyPowerSensor(SensorEntity):
    """Transform power sensor values (invert or combine two sensors).

    This sensor handles non-standard power sensor configurations for the energy
    dashboard by either inverting polarity or combining two positive sensors.
    """

    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.POWER
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_has_entity_name = True

    def __init__(
        self,
        source_type: str,
        config: PowerConfig,
        unique_id: str,
        entity_id: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__()
        self._source_type = source_type
        self._config: PowerConfig = config
        self._attr_unique_id = unique_id
        self.entity_id = entity_id
        self._source_sensors: list[str] = []
        self._is_inverted = "stat_rate_inverted" in config
        self._is_combined = "stat_rate_from" in config and "stat_rate_to" in config

        # Combined mode always emits Watts because _update_state converts
        # heterogeneous source units to W internally. Inverted mode copies
        # the source unit in _update_state to track source changes.
        if self._is_combined:
            self._attr_native_unit_of_measurement = UnitOfPower.WATT

        # Determine source sensors
        if self._is_inverted:
            self._source_sensors = [config["stat_rate_inverted"]]
        elif self._is_combined:
            self._source_sensors = [
                config["stat_rate_from"],
                config["stat_rate_to"],
            ]

        # SensorManager awaits add_finished; async_on_remove resolves it on the
        # abort path too, since add_to_platform_abort fires on-remove callbacks.
        self.add_finished: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self.async_on_remove(lambda: _set_result_unless_done(self.add_finished))

    @property
    @override
    def available(self) -> bool:
        """Return if entity is available."""
        if self._is_inverted:
            source = self.hass.states.get(self._source_sensors[0])
            return source is not None and source.state not in (
                "unknown",
                "unavailable",
            )
        if self._is_combined:
            discharge = self.hass.states.get(self._source_sensors[0])
            charge = self.hass.states.get(self._source_sensors[1])
            return (
                discharge is not None
                and charge is not None
                and discharge.state not in ("unknown", "unavailable")
                and charge.state not in ("unknown", "unavailable")
            )
        return True

    @callback
    def _update_state(self) -> None:
        """Update the sensor state based on source sensors."""
        if self._is_inverted:
            source_state = self.hass.states.get(self._source_sensors[0])
            if source_state is None or source_state.state in ("unknown", "unavailable"):
                self._attr_native_value = None
                return
            try:
                value = float(source_state.state)
            except ValueError:
                self._attr_native_value = None
                return

            self._attr_native_unit_of_measurement = source_state.attributes.get(
                EntityStateAttribute.UNIT_OF_MEASUREMENT
            )
            self._attr_native_value = value * -1

        elif self._is_combined:
            discharge_state = self.hass.states.get(self._source_sensors[0])
            charge_state = self.hass.states.get(self._source_sensors[1])

            if (
                discharge_state is None
                or charge_state is None
                or discharge_state.state in ("unknown", "unavailable")
                or charge_state.state in ("unknown", "unavailable")
            ):
                self._attr_native_value = None
                return

            try:
                discharge = float(discharge_state.state)
                charge = float(charge_state.state)
            except ValueError:
                self._attr_native_value = None
                return

            # Get units from state attributes
            discharge_unit = discharge_state.attributes.get(
                EntityStateAttribute.UNIT_OF_MEASUREMENT
            )
            charge_unit = charge_state.attributes.get(
                EntityStateAttribute.UNIT_OF_MEASUREMENT
            )

            # Convert to Watts if units are present
            if discharge_unit:
                discharge = unit_conversion.PowerConverter.convert(
                    discharge, discharge_unit, UnitOfPower.WATT
                )
            if charge_unit:
                charge = unit_conversion.PowerConverter.convert(
                    charge, charge_unit, UnitOfPower.WATT
                )

            self._attr_native_value = discharge - charge

    @override
    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        # Set name based on source sensor(s)
        if self._source_sensors:
            entity_reg = er.async_get(self.hass)
            device_id = None
            source_name = None
            # Check first sensor
            if source_entry := entity_reg.async_get(self._source_sensors[0]):
                device_id = source_entry.device_id
                # Get source name from registry
                source_name = source_entry.name or source_entry.original_name
            # Assign power sensor to same device as source sensor(s)
            # Note: We use manual entity registry update instead of _attr_device_info
            # because device assignment depends on runtime information from the entity
            # registry (which source sensor has a device). This information isn't
            # available during __init__, and the entity is already registered before
            # async_added_to_hass runs, making the standard _attr_device_info pattern
            # incompatible with this use case.
            # If first sensor has no device and we have a second sensor, check it
            if not device_id and len(self._source_sensors) > 1:
                if source_entry := entity_reg.async_get(self._source_sensors[1]):
                    device_id = source_entry.device_id
            # Update entity registry entry with device_id
            if device_id and (power_entry := entity_reg.async_get(self.entity_id)):
                entity_reg.async_update_entity(
                    power_entry.entity_id, device_id=device_id
                )
            else:
                self._attr_has_entity_name = False

            # Set name for inverted mode
            if self._is_inverted:
                if source_name:
                    self._attr_name = f"{source_name} Inverted"
                else:
                    # Fall back to entity_id if no name in registry
                    sensor_name = split_entity_id(self._source_sensors[0])[1].replace(
                        "_", " "
                    )
                    self._attr_name = f"{sensor_name.title()} Inverted"

        # Set name for combined mode
        if self._is_combined:
            self._attr_name = f"{self._source_type.title()} Power"

        self._update_state()

        # Track state changes on all source sensors
        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                self._source_sensors,
                self._async_state_changed_listener,
            )
        )
        _set_result_unless_done(self.add_finished)

    @callback
    def _async_state_changed_listener(self, *_: Any) -> None:
        """Handle source sensor state changes."""
        self._update_state()
        self.async_write_ha_state()


@dataclass(slots=True)
class _MeterTrack:
    """Growth of one energy meter since this sensor started watching it."""

    baseline: float | None = None
    last: float | None = None
    accumulated: float = 0.0


class EnergySolarSelfConsumptionSensor(SensorEntity):
    """Total energy from one solar source that the home used itself.

    The state is the dashboard split applied to growth since this sensor was
    created, not total production. It can decrease when export or battery
    charge is reported after production in the same running total; the cost
    sensor follows that correction.
    """

    _attr_entity_registry_visible_default = False
    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR

    def __init__(
        self,
        solar_stat_id: str,
        get_sources: Callable[[], list[Mapping[str, Any]]],
    ) -> None:
        """Initialize the sensor."""
        super().__init__()
        self._solar_stat_id = solar_stat_id
        self._get_sources = get_sources
        self.entity_id = solar_self_consumed_entity_id(solar_stat_id)
        self._tracks: dict[str, _MeterTrack] = {}
        self._watched: set[str] = set()
        self._unsub_state: Callable[[], None] | None = None
        self._warned_external_export = False
        self._listening = False
        self._attr_native_value = 0.0
        # SensorManager awaits add_finished; async_on_remove resolves it on the
        # abort path too, since add_to_platform_abort fires on-remove callbacks.
        self.add_finished: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self.async_on_remove(lambda: _set_result_unless_done(self.add_finished))

    @callback
    def update_sources(self) -> None:
        """Refresh watched meters after energy preferences change."""
        if not self._listening:
            return
        self._watched = self._watched_entity_ids()
        self._resubscribe()
        self._recalculate()
        self.async_write_ha_state()

    def _watched_entity_ids(self) -> set[str]:
        """Entity ids that can change the self-consumption split."""
        entity_ids: set[str] = set()
        for source in self._get_sources():
            source_type = source.get("type")
            if source_type == "solar":
                self._add_entity(entity_ids, source.get("stat_energy_from"))
            elif source_type == "grid":
                self._add_entity(entity_ids, source.get("stat_energy_from"))
                export_stat = source.get("stat_energy_to")
                if export_stat and not valid_entity_id(export_stat):
                    if not self._warned_external_export:
                        self._warned_external_export = True
                        _LOGGER.warning(
                            "Solar savings cannot see external grid export"
                            " statistic %s, so that export is not subtracted",
                            export_stat,
                        )
                else:
                    self._add_entity(entity_ids, export_stat)
            elif source_type == "battery":
                self._add_entity(entity_ids, source.get("stat_energy_from"))
                self._add_entity(entity_ids, source.get("stat_energy_to"))
        return entity_ids

    @staticmethod
    def _add_entity(entity_ids: set[str], stat_id: Any) -> None:
        """Add a statistic id when it is an entity that can be tracked live."""
        if isinstance(stat_id, str) and valid_entity_id(stat_id):
            entity_ids.add(stat_id)

    def _resubscribe(self) -> None:
        """Track the current set of meter entities."""
        if self._unsub_state is not None:
            self._unsub_state()
            self._unsub_state = None
        if not self._watched:
            return
        self._unsub_state = async_track_state_change_event(
            self.hass,
            list(self._watched),
            self._async_state_changed_listener,
        )

    def _unsubscribe(self) -> None:
        """Stop tracking meter entities."""
        if self._unsub_state is not None:
            self._unsub_state()
            self._unsub_state = None

    def _reading_kwh(self, entity_id: str) -> float | None:
        """Return the meter state in kWh, or None when it cannot be used."""
        state = self.hass.states.get(entity_id)
        if state is None or state.state in ("unknown", "unavailable"):
            return None
        try:
            value = float(state.state)
        except ValueError:
            return None
        unit = state.attributes.get(EntityStateAttribute.UNIT_OF_MEASUREMENT)
        if unit not in VALID_ENERGY_UNITS:
            return None
        if unit == UnitOfEnergy.KILO_WATT_HOUR:
            return value
        try:
            return unit_conversion.EnergyConverter.convert(
                value, unit, UnitOfEnergy.KILO_WATT_HOUR
            )
        except HomeAssistantError:
            return None

    def _increase_kwh(self, entity_id: str) -> float:
        """Growth in kWh since this sensor first saw the meter.

        A drop is treated as a meter reset. Energy before the reset is kept,
        and the new state counts as growth of a fresh segment.
        """
        current = self._reading_kwh(entity_id)
        track = self._tracks.setdefault(entity_id, _MeterTrack())
        if current is None:
            if track.baseline is None or track.last is None:
                return 0.0
            return track.accumulated + (track.last - track.baseline)
        if track.baseline is None:
            track.baseline = current
            track.last = current
            return 0.0
        if track.last is not None and current < track.last:
            track.accumulated += max(track.last - track.baseline, 0.0)
            track.baseline = 0.0
        track.last = current
        return track.accumulated + (current - track.baseline)

    def _recalculate(self) -> None:
        """Set state to self-consumed solar for this source since startup."""
        solar_increases: dict[str, float] = {}
        from_grid = 0.0
        to_grid = 0.0
        to_battery = 0.0
        from_battery = 0.0
        seen: set[str] = set()

        for source in self._get_sources():
            source_type = source.get("type")
            if source_type == "solar":
                stat_id = source.get("stat_energy_from")
                if not isinstance(stat_id, str) or not valid_entity_id(stat_id):
                    continue
                if stat_id in seen:
                    continue
                seen.add(stat_id)
                solar_increases[stat_id] = self._increase_kwh(stat_id)
            elif source_type == "grid":
                for key, bucket in (
                    ("stat_energy_from", "from_grid"),
                    ("stat_energy_to", "to_grid"),
                ):
                    stat_id = source.get(key)
                    if (
                        not isinstance(stat_id, str)
                        or not valid_entity_id(stat_id)
                        or stat_id in seen
                    ):
                        continue
                    seen.add(stat_id)
                    increase = self._increase_kwh(stat_id)
                    if bucket == "from_grid":
                        from_grid += increase
                    else:
                        to_grid += increase
            elif source_type == "battery":
                for key, bucket in (
                    ("stat_energy_from", "from_battery"),
                    ("stat_energy_to", "to_battery"),
                ):
                    stat_id = source.get(key)
                    if (
                        not isinstance(stat_id, str)
                        or not valid_entity_id(stat_id)
                        or stat_id in seen
                    ):
                        continue
                    seen.add(stat_id)
                    increase = self._increase_kwh(stat_id)
                    if bucket == "from_battery":
                        from_battery += increase
                    else:
                        to_battery += increase

        total_solar = sum(max(value, 0.0) for value in solar_increases.values())
        if total_solar <= 0:
            self._attr_native_value = 0.0
            return
        used = compute_used_solar(
            from_grid=from_grid,
            to_grid=to_grid,
            solar=total_solar,
            to_battery=to_battery,
            from_battery=from_battery,
        )
        mine = max(solar_increases.get(self._solar_stat_id, 0.0), 0.0)
        self._attr_native_value = used * (mine / total_solar)

    @override
    async def async_added_to_hass(self) -> None:
        """Register callbacks and publish the initial total."""
        energy_state = self.hass.states.get(self._solar_stat_id)
        if energy_state:
            name = energy_state.name
        else:
            name = split_entity_id(self._solar_stat_id)[1].replace("_", " ")
        self._attr_name = f"{name} Self consumed"

        self._watched = self._watched_entity_ids()
        self._recalculate()
        self._resubscribe()
        self._listening = True
        self.async_on_remove(self._unsubscribe)
        _set_result_unless_done(self.add_finished)

    @callback
    def _async_state_changed_listener(self, *_: Any) -> None:
        """Recompute self-consumed solar after a meter changes."""
        self._recalculate()
        self.async_write_ha_state()

    @property
    @override
    def unique_id(self) -> str:
        """Return the unique ID of the sensor."""
        entity_registry = er.async_get(self.hass)
        if registry_entry := entity_registry.async_get(self._solar_stat_id):
            prefix = registry_entry.id
        else:
            prefix = self._solar_stat_id
        return f"{prefix}_solar_self_consumed"
