import psutil

class GreenAI:
    """
    Hardware-Aware Threshold Shifting.
    Monitors OS system battery (if running on edge devices like drones/phones).
    If power drops below a critical point, dynamically forces the complexity
    threshold up to ensure the model runs heavily on the power-saving Shallow Local path.
    """
    def __init__(self, critical_battery_level: int = 20):
        self.critical_battery_level = critical_battery_level

    def get_system_status(self) -> dict:
        """Returns the host battery percentage. Fakes 100% on battery-less devices like Cloud VMs."""
        battery = psutil.sensors_battery()
        if battery is None:
            # Running on a cloud server/desktop without battery.
            return {"battery_percent": 100, "power_plugged": True}
        return {"battery_percent": battery.percent, "power_plugged": battery.power_plugged}

    def adjust_thresholds(self, base_thresholds: dict) -> dict:
        """
        Dynamically limits deep-compute if the device lacks sufficient power.
        """
        status = self.get_system_status()
        battery_pct = status["battery_percent"]
        plugged_in = status["power_plugged"]

        adjusted = base_thresholds.copy()

        # If we are below the critical battery line and not plugged in
        if battery_pct <= self.critical_battery_level and not plugged_in:
            print(f"[!] GREEN AI ALERT: Battery critical ({battery_pct}%). Forcing Power-Saving Route...")
            adjusted["variance"] *= 4.0
            adjusted["entropy"] *= 4.0
            adjusted["sparsity"] /= 4.0

        return adjusted
