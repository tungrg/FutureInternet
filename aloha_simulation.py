"""One-attempt pure and slotted ALOHA simulation helpers."""

from collections.abc import Iterable

import numpy as np

TransmissionTimeSpec = float | tuple[float, float]


def _sample_durations(
    num_messages: int,
    transmission_time: TransmissionTimeSpec,
    simulation_time: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of per-message transmission durations."""
    if isinstance(transmission_time, tuple):
        low, high = transmission_time
        if low < 0 or high <= 0 or low > high:
            raise ValueError("transmission_time range must satisfy 0 <= low <= high and high > 0")
        if high > simulation_time:
            raise ValueError("maximum transmission_time cannot exceed simulation_time")
        return rng.uniform(low, high, size=num_messages)

    if transmission_time <= 0:
        raise ValueError("transmission_time and simulation_time must be positive")
    if transmission_time > simulation_time:
        raise ValueError("transmission_time cannot exceed simulation_time")
    return np.full(num_messages, float(transmission_time), dtype=float)


def _count_interval_collisions(starts: np.ndarray, durations: np.ndarray) -> int:
    """Count unique frames whose intervals [start, start + duration) overlap."""
    num_messages = len(starts)
    if num_messages <= 1:
        return 0

    order = np.argsort(starts, kind="mergesort")
    sorted_starts = starts[order]
    sorted_ends = sorted_starts + durations[order]
    collided = np.zeros(num_messages, dtype=bool)

    for i in range(num_messages):
        j = i + 1
        while j < num_messages and (
            sorted_starts[j] < sorted_ends[i] or sorted_starts[j] == sorted_starts[i]
        ):
            collided[i] = True
            collided[j] = True
            j += 1

    return int(np.count_nonzero(collided))


def theoretical_pure_aloha_collision_prob(
    num_messages: int,
    transmission_time: TransmissionTimeSpec = 0.999,
    simulation_time: float = 3600,
) -> float:
    """Compute theoretical Pure ALOHA collision probability: p = 1 - prod(1 - (Ts + Ta) / T)."""
    if num_messages <= 1:
        return 0.0
    mean_tx = (
        0.5 * (transmission_time[0] + transmission_time[1])
        if isinstance(transmission_time, tuple)
        else float(transmission_time)
    )
    return float(1.0 - (1.0 - (2.0 * mean_tx) / simulation_time) ** (num_messages - 1))


def theoretical_slotted_aloha_collision_prob(
    num_messages: int,
    simulation_time: float = 3600,
    time_slot: float = 1.0,
) -> float:
    """Compute theoretical Slotted ALOHA collision probability (when frames fit in one slot)."""
    if num_messages <= 1:
        return 0.0
    num_slots = int(simulation_time // time_slot)
    return float(1.0 - (1.0 - 1.0 / num_slots) ** (num_messages - 1))


def simulate_pure_aloha(
    num_messages: int = 1000,
    transmission_time: TransmissionTimeSpec = 0.999,
    simulation_time: float = 3600,
    rng: np.random.Generator | None = None,
) -> int:
    """Return the number of unique frames involved in a pure-ALOHA collision."""
    if num_messages < 0:
        raise ValueError("num_messages must be non-negative")
    if simulation_time <= 0:
        raise ValueError("transmission_time and simulation_time must be positive")

    rng = np.random.default_rng() if rng is None else rng
    durations = _sample_durations(num_messages, transmission_time, simulation_time, rng)
    starts = rng.uniform(0, simulation_time, num_messages)
    return _count_interval_collisions(starts, durations)


def simulate_slotted_aloha(
    num_messages: int = 1000,
    transmission_time: TransmissionTimeSpec = 0.999,
    simulation_time: float = 3600,
    time_slot: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Return the number of unique frames involved in a slotted-ALOHA collision.

    Supports both fixed transmission times (<= time_slot) and variable transmission
    times that may exceed time_slot and spill over into subsequent slots (Exercise 1.2).
    """
    if num_messages < 0:
        raise ValueError("num_messages must be non-negative")
    if simulation_time <= 0 or time_slot <= 0:
        raise ValueError("transmission_time, simulation_time, and time_slot must be positive")

    num_slots = int(simulation_time // time_slot)
    if num_slots == 0:
        raise ValueError("simulation_time must contain at least one full time slot")

    rng = np.random.default_rng() if rng is None else rng
    durations = _sample_durations(num_messages, transmission_time, simulation_time, rng)
    slot_ids = rng.integers(0, num_slots, size=num_messages)

    # Fast path when every frame fits within a single slot
    if np.all(durations <= time_slot):
        messages_per_slot = np.bincount(slot_ids, minlength=num_slots)
        return int(messages_per_slot[messages_per_slot > 1].sum())

    # General path when frames can exceed time_slot and spill into subsequent slots
    starts = slot_ids.astype(float) * time_slot
    return _count_interval_collisions(starts, durations)


def run_sweep(
    message_counts: Iterable[int] = range(100, 1001, 100),
    simulation_time: float = 3600,
    transmission_time: TransmissionTimeSpec = 0.999,
    time_slot: float = 1.0,
    trials: int = 100,
    seed: int = 42,
) -> tuple[list[dict[str, int | float]], list[dict[str, int | float]]]:
    """Average collision and successful-airtime metrics over repeated trials."""
    if trials <= 0:
        raise ValueError("trials must be positive")

    counts = tuple(int(n) for n in message_counts)
    if any(n <= 0 for n in counts):
        raise ValueError("each message count must be positive")

    rng = np.random.default_rng(seed)
    pure_results: list[dict[str, int | float]] = []
    slotted_results: list[dict[str, int | float]] = []
    num_slots = int(simulation_time // time_slot)
    mean_tx_time = (
        0.5 * (transmission_time[0] + transmission_time[1])
        if isinstance(transmission_time, tuple)
        else float(transmission_time)
    )

    for num_messages in counts:
        pure_collisions = np.array([
            simulate_pure_aloha(num_messages, transmission_time, simulation_time, rng)
            for _ in range(trials)
        ])
        slotted_collisions = np.array([
            simulate_slotted_aloha(
                num_messages, transmission_time, simulation_time, time_slot, rng
            )
            for _ in range(trials)
        ])

        pure_collision_probability = float(np.mean(pure_collisions / num_messages))
        slotted_collision_probability = float(np.mean(slotted_collisions / num_messages))
        pure_successes = num_messages * (1 - pure_collision_probability)
        slotted_successes = num_messages * (1 - slotted_collision_probability)

        pure_results.append({
            "messages": num_messages,
            "offered_load": num_messages * mean_tx_time / simulation_time,
            "mean_collisions": float(np.mean(pure_collisions)),
            "std_collisions": float(np.std(pure_collisions)),
            "collision_probability": pure_collision_probability,
            "std_collision_probability": float(np.std(pure_collisions / num_messages)),
            "theoretical_collision_probability": theoretical_pure_aloha_collision_prob(
                num_messages, transmission_time, simulation_time
            ),
            "success_probability": 1 - pure_collision_probability,
            "throughput_packets_per_second": pure_successes / simulation_time,
            "channel_utilization": pure_successes * mean_tx_time / simulation_time,
        })
        slotted_results.append({
            "messages": num_messages,
            "offered_load": num_messages / num_slots,
            "mean_collisions": float(np.mean(slotted_collisions)),
            "std_collisions": float(np.std(slotted_collisions)),
            "collision_probability": slotted_collision_probability,
            "std_collision_probability": float(np.std(slotted_collisions / num_messages)),
            "theoretical_collision_probability": theoretical_slotted_aloha_collision_prob(
                num_messages, simulation_time, time_slot
            ),
            "success_probability": 1 - slotted_collision_probability,
            "throughput_packets_per_second": slotted_successes / simulation_time,
            "channel_utilization": slotted_successes * mean_tx_time / simulation_time,
        })

    return pure_results, slotted_results


if __name__ == "__main__":
    pure, slotted = run_sweep()
    for protocol, results in (("Pure ALOHA", pure), ("Slotted ALOHA", slotted)):
        print(f"\n{protocol} (one attempt per message; averaged over 100 trials)")
        for result in results:
            print(
                f"N={result['messages']:4d}, G={result['offered_load']:.3f}, "
                f"collision prob={result['collision_probability']:.3f} "
                f"(theory={result['theoretical_collision_probability']:.3f}), "
                f"throughput={result['throughput_packets_per_second']:.4f} packets/s, "
                f"utilization={100 * result['channel_utilization']:.2f}%"
            )
