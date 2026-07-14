"""Read-only environment inspection for TinyLLM-System."""

from tinyllm.doctor.collector import DoctorCollector
from tinyllm.doctor.schema import DoctorReport

__all__ = ["DoctorCollector", "DoctorReport"]
