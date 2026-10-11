"""Main facade class integrating the hardware."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from zulf_nmr.core.connection import DEFAULT_SYSTEM_CONFIG, Message, SCPIConnection
from zulf_nmr.hardware.acquisition import Acquisition
from zulf_nmr.hardware.awg import Awg
from zulf_nmr.hardware.digital import Digital
from zulf_nmr.hardware.motor import Motor
from zulf_nmr.hardware.sequencer import Sequencer
from zulf_nmr.hardware.shim import Shim
from zulf_nmr.hardware.system import System


class Spectrometer:
    """ZULF spectrometer: one SCPI connection shared by all subsystem drivers.

    Usage:
        with Spectrometer.from_config() as spec:
            print(spec.system.identify())
            spec.motor.move("MAGNET", wait=True)
    """

    def __init__(
        self,
        connection: SCPIConnection,
        hardware_limits: dict | None = None,
        data_output: str | Path = "data/",
    ) -> None:
        limits = hardware_limits or {}
        self.connection = connection
        self.data_output = Path(data_output)

        self.system = System(connection)
        self.motor = Motor(connection, timeout_s=limits.get("motor_timeout_s", 15.0))
        self.shim = Shim(connection, max_current_mA=limits.get("max_shim_current_mA", 100.0))
        self.awg = Awg(
            connection,
            max_sample_rate_hz=limits.get("max_awg_sample_rate_hz", 10000),
            max_amplitude_A=limits.get("max_awg_amplitude_A", 1.0),
        )
        self.acquisition = Acquisition(connection)
        self.sequencer = Sequencer(connection)
        self.digital = Digital(connection)

    @classmethod
    def from_config(
        cls,
        path: str | Path = DEFAULT_SYSTEM_CONFIG,
        on_message: Callable[[Message], None] | None = None,
    ) -> Spectrometer:
        """Build the spectrometer from system.json (connection, limits, directories)."""
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(
            connection=SCPIConnection.from_config(path, on_message=on_message),
            hardware_limits=cfg.get("hardware_limits"),
            data_output=cfg.get("directories", {}).get("data_output", "data/"),
        )

    @property
    def is_connected(self) -> bool:
        return self.connection.is_connected

    def connect(self) -> None:
        self.connection.connect()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Spectrometer:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
